from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import fitz
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .ir_builder import DEFAULT_EXCLUDE_PATTERNS, build_document_ir, dump_json, find_mineru_file
from .llm_client import build_chat_request, call_openai_compatible, extract_json_content, response_content
from .mineru_runner import run_mineru
from .models import DocumentIR, document_ir_from_dict, to_jsonable
from .payload import DEFAULT_TASK, SYSTEM_PROMPT, response_by_block_id
from .protect import restore_text
from .style import block_rect

RUN_ROOT = Path(os.environ.get("PATRA_WEB_RUNS", "runs/web"))
STATIC_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Patra PDF Section Translator")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class TranslateRequest(BaseModel):
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    json_mode: bool = False
    task: str | None = None


def job_dir(job_id: str) -> Path:
    return RUN_ROOT / job_id


def status_path(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"


def write_status(job_id: str, status: str, **extra: Any) -> None:
    path = status_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"job_id": job_id, "status": status, **extra}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_status(job_id: str) -> dict[str, Any]:
    path = status_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown job id")
    return json.loads(path.read_text(encoding="utf-8"))


def require_job_file(job_id: str, name: str) -> Path:
    path = job_dir(job_id) / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing job file: {name}")
    return path


def load_ir(job_id: str) -> DocumentIR:
    path = require_job_file(job_id, "document_ir.json")
    return document_ir_from_dict(json.loads(path.read_text(encoding="utf-8")))


def process_upload_job(
    job_id: str,
    *,
    task: str,
    protected_terms: str,
    exclude_regex: str,
    run_parse: bool,
    mineru_output: str | None,
) -> None:
    try:
        base = job_dir(job_id)
        pdf_path = base / "input.pdf"
        mineru_dir = base / "mineru"
        write_status(job_id, "processing", message="Preparing MinerU output")

        if run_parse:
            write_status(job_id, "processing", message="Running MinerU without GPU")
            run_mineru(pdf_path, mineru_dir)
            mineru_source: str | Path = mineru_dir
        elif mineru_output:
            mineru_source = Path(mineru_output).expanduser()
        else:
            mineru_source = mineru_dir

        find_mineru_file(mineru_source, "_content_list.json")
        write_status(job_id, "processing", message="Building document blocks")
        terms = [term.strip() for term in protected_terms.split(",") if term.strip()]
        patterns = DEFAULT_EXCLUDE_PATTERNS + [pattern.strip() for pattern in exclude_regex.split(",") if pattern.strip()]
        document = build_document_ir(pdf_path, mineru_source, exclude_patterns=patterns, protected_terms=terms)
        payload = _build_web_payload(document, task)
        dump_json(base / "document_ir.json", to_jsonable(document))
        dump_json(base / "llm_payload.json", payload)
        write_status(job_id, "ready", message="Ready", block_count=len(payload["blocks"]))
    except Exception as exc:  # noqa: BLE001 - background tasks need serialized failures.
        write_status(job_id, "error", message=str(exc))


def _build_web_payload(document: DocumentIR, task: str) -> dict[str, Any]:
    from .payload import build_payload

    return build_payload(document, task=task)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/jobs")
def create_job(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    task: str = Form(DEFAULT_TASK),
    protected_terms: str = Form("attention,Transformer,softmax"),
    exclude_regex: str = Form(""),
    run_parse: bool = Form(True),
    mineru_output: str | None = Form(None),
) -> dict[str, str]:
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    job_id = uuid.uuid4().hex[:12]
    base = job_dir(job_id)
    base.mkdir(parents=True, exist_ok=True)
    pdf_path = base / "input.pdf"
    with pdf_path.open("wb") as handle:
        shutil.copyfileobj(pdf.file, handle)
    write_status(job_id, "queued", message="Queued")
    background_tasks.add_task(
        process_upload_job,
        job_id,
        task=task,
        protected_terms=protected_terms,
        exclude_regex=exclude_regex,
        run_parse=run_parse,
        mineru_output=mineru_output,
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}/status")
def get_status(job_id: str) -> dict[str, Any]:
    return read_status(job_id)


@app.get("/api/jobs/{job_id}/pdf")
def get_pdf(job_id: str) -> FileResponse:
    return FileResponse(require_job_file(job_id, "input.pdf"), media_type="application/pdf")


@app.get("/api/jobs/{job_id}/document")
def get_document(job_id: str) -> dict[str, Any]:
    status = read_status(job_id)
    if status.get("status") != "ready":
        raise HTTPException(status_code=409, detail=status)
    document = load_ir(job_id)
    doc = fitz.open(document.source_pdf)
    try:
        pages = [{"page_idx": i, "width": page.rect.width, "height": page.rect.height} for i, page in enumerate(doc)]
    finally:
        doc.close()
    blocks = []
    for block in document.blocks:
        if block.excluded or block.type not in {"text", "list", "code"} or not block.text.strip():
            continue
        blocks.append(
            {
                "id": block.id,
                "type": block.type,
                "page_idx": block.page_idx,
                "bbox": block.bbox,
                "text": block.text,
                "section_id": block.section_id,
                "text_level": block.text_level,
                "rich_text": [to_jsonable(span) for span in block.rich_text],
            }
        )
    return {"job_id": job_id, "pages": pages, "sections": to_jsonable(document.sections), "blocks": blocks}


@app.get("/api/jobs/{job_id}/blocks/{block_id}/crop.png")
def get_block_crop(job_id: str, block_id: str, zoom: float = 2.0) -> Response:
    document = load_ir(job_id)
    block = next((item for item in document.blocks if item.id == block_id), None)
    if block is None:
        raise HTTPException(status_code=404, detail="Unknown block id")
    doc = fitz.open(document.source_pdf)
    try:
        page = doc[block.page_idx]
        clip = block_rect(page, block.bbox) & page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
        return Response(content=pix.tobytes("png"), media_type="image/png")
    finally:
        doc.close()


@app.post("/api/jobs/{job_id}/blocks/{block_id}/translate")
def translate_block(job_id: str, block_id: str, request: TranslateRequest) -> dict[str, Any]:
    base = job_dir(job_id)
    payload_path = require_job_file(job_id, "llm_payload.json")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    block = next((item for item in payload.get("blocks", []) if item.get("id") == block_id), None)
    if block is None:
        raise HTTPException(status_code=404, detail="Unknown or non-translatable block id")

    single_payload = {
        "system_prompt": payload.get("system_prompt", SYSTEM_PROMPT),
        "task": request.task or payload.get("task", "Translate the selected block."),
        "response_schema": payload.get("response_schema"),
        "blocks": [block],
    }
    body = build_chat_request(
        single_payload,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        json_mode=request.json_mode,
    )
    raw = call_openai_compatible(
        base_url=request.base_url,
        api_key=request.api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        request_body=body,
    )
    try:
        parsed = extract_json_content(response_content(raw))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    response_block = response_by_block_id(parsed).get(block_id)
    if not response_block:
        raise HTTPException(status_code=502, detail="Model response did not include the requested block id")

    document = load_ir(job_id)
    source_block = next(item for item in document.blocks if item.id == block_id)
    if source_block.protection_map:
        response_block = dict(response_block)
        response_block["text"] = restore_text(str(response_block.get("text", "")), source_block.protection_map)
        rich = []
        for span in response_block.get("rich_text", []) or []:
            span = dict(span)
            span["text"] = restore_text(str(span.get("text", "")), source_block.protection_map)
            rich.append(span)
        response_block["rich_text"] = rich

    out_dir = base / "translations"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{block_id}.json").write_text(json.dumps(response_block, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"block": response_block}


@app.get("/api/jobs/{job_id}/blocks/{block_id}/translation")
def get_cached_translation(job_id: str, block_id: str) -> dict[str, Any]:
    path = job_dir(job_id) / "translations" / f"{block_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No cached translation")
    return {"block": json.loads(path.read_text(encoding="utf-8"))}


def main() -> None:
    import uvicorn

    uvicorn.run("patra_mineru.web_app:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
