from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ir_builder import DEFAULT_EXCLUDE_PATTERNS, build_document_ir, dump_json
from .llm_client import translate_payload
from .mineru_runner import (
    BACKENDS,
    DEFAULT_API_URL,
    DEFAULT_BACKEND,
    DEFAULT_DEVICE,
    DEFAULT_LANG,
    DEFAULT_SERVER_URL,
    run_mineru,
)
from .models import to_jsonable
from .payload import (
    DEFAULT_BATCH_BLOCKS,
    DEFAULT_BATCH_CHARS,
    DEFAULT_TASK,
    build_mock_response,
    build_payload,
    load_response,
)
from .reconstruct import load_document_ir, render_html_preview, render_pdf


def _split_csv(values: str | None) -> list[str]:
    if not values:
        return []
    return [value.strip() for value in values.split(",") if value.strip()]


def cmd_parse(args: argparse.Namespace) -> None:
    run_mineru(
        args.pdf,
        args.output_dir,
        backend=args.backend,
        method=args.method,
        start=args.start,
        end=args.end,
        device=args.device,
        lang=args.lang,
        effort=args.effort,
        server_url=args.server_url,
        api_url=args.api_url,
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    exclude_patterns = DEFAULT_EXCLUDE_PATTERNS + _split_csv(args.exclude_regex)
    protected_terms = _split_csv(args.protected_terms)
    document = build_document_ir(
        args.pdf,
        args.mineru_output,
        exclude_patterns=exclude_patterns,
        protected_terms=protected_terms,
        translate_headings=args.translate_headings,
    )
    payload = build_payload(document, task=args.task)
    dump_json(args.ir_out, to_jsonable(document))
    dump_json(args.payload_out, payload)
    print(f"wrote IR: {args.ir_out}")
    print(f"wrote LLM payload: {args.payload_out}")
    print(f"translatable blocks: {len(payload['blocks'])}")
    if not args.translate_headings:
        headings = sum(1 for block in document.blocks if block.text_level)
        print(f"headings kept in the source language: {headings}")



def cmd_translate(args: argparse.Namespace) -> None:
    def report(index: int, total: int, blocks: int) -> None:
        print(f"batch {index}/{total}: {blocks} blocks", flush=True)

    def report_warning(message: str) -> None:
        print(f"warning: {message}", flush=True)

    result = translate_payload(
        payload_path=args.payload,
        output_path=args.output,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        json_mode=args.json_mode,
        no_think=args.no_think,
        batch_blocks=args.batch_blocks,
        batch_chars=args.batch_chars,
        raw_response_out=args.raw_response_out,
        dry_run_request_out=args.dry_run_request_out,
        progress=report,
        warn=report_warning,
    )
    if args.dry_run_request_out:
        print(f"wrote dry-run request: {args.dry_run_request_out}")
        return
    block_count = len(result.get("blocks", [])) if result else 0
    print(f"wrote model response: {args.output}")
    print(f"response blocks: {block_count}")


def cmd_mock_response(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    response = build_mock_response(payload, prefix=args.prefix)
    dump_json(args.output, response)
    print(f"wrote mock response: {args.output}")


def cmd_render(args: argparse.Namespace) -> None:
    document = load_document_ir(args.ir)
    response = load_response(args.response)
    warnings = render_pdf(
        args.pdf,
        document,
        response,
        args.output_pdf,
        draw_debug_boxes=args.debug_boxes,
    )
    if args.output_html:
        render_html_preview(args.pdf, document, response, args.output_html)
    print(f"wrote PDF: {args.output_pdf}")
    if args.output_html:
        print(f"wrote HTML preview: {args.output_html}")
    for warning in warnings:
        print(f"warning: {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patra-mineru")
    sub = parser.add_subparsers(dest="command", required=True)

    parse = sub.add_parser("parse", help="Run the MinerU CLI, locally or against a running MinerU server.")
    parse.add_argument("--pdf", required=True)
    parse.add_argument("--output-dir", required=True)
    parse.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=BACKENDS,
        help="MinerU 3.x backend. The *-http-client backends need --server-url.",
    )
    parse.add_argument("--method", default="auto", choices=["auto", "txt", "ocr"])
    parse.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="auto (let MinerU pick, uses CUDA when available), cpu, cuda, or cuda:N.",
    )
    parse.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        help="OCR language for the pipeline backend. MinerU defaults to 'ch'; use 'en' for English papers.",
    )
    parse.add_argument("--effort", choices=["medium", "high"], help="Hybrid backend effort.")
    parse.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help="Running MinerU VLM server, e.g. http://127.0.0.1:30000. Required by *-http-client backends.",
    )
    parse.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Running mineru-api service, e.g. http://127.0.0.1:8010. Hands the whole parse to that service.",
    )
    parse.add_argument("--start", type=int)
    parse.add_argument("--end", type=int)
    parse.set_defaults(func=cmd_parse)

    prepare = sub.add_parser("prepare", help="Build DocumentIR and LLM payload from MinerU output.")
    prepare.add_argument("--pdf", required=True)
    prepare.add_argument("--mineru-output", required=True)
    prepare.add_argument("--ir-out", required=True)
    prepare.add_argument("--payload-out", required=True)
    prepare.add_argument("--task", default=DEFAULT_TASK)
    prepare.add_argument("--exclude-regex", help="Comma-separated extra section-heading regexes to exclude.")
    prepare.add_argument(
        "--translate-headings",
        action="store_true",
        help="Also translate section headings. Off by default so headings keep their original typography.",
    )
    prepare.add_argument(
        "--protected-terms",
        default="attention,Transformer,softmax,FlashAttention",
        help="Comma-separated terms to keep in the original language.",
    )
    prepare.set_defaults(func=cmd_prepare)

    translate = sub.add_parser("translate", help="Send an LLM payload to an OpenAI-compatible chat completions server.")
    translate.add_argument("--payload", required=True)
    translate.add_argument("--output", required=True)
    translate.add_argument("--base-url", required=True, help="Base URL such as http://localhost:8000/v1")
    translate.add_argument("--model", required=True)
    translate.add_argument("--api-key", help="API key. If omitted, --api-key-env is used, then dummy.")
    translate.add_argument("--api-key-env", default="OPENAI_API_KEY")
    translate.add_argument("--temperature", type=float, default=0.0)
    translate.add_argument("--max-tokens", type=int)
    translate.add_argument("--timeout", type=float, default=300.0)
    translate.add_argument("--json-mode", action="store_true", help="Send response_format={type: json_object}; use only if your server supports it.")
    translate.add_argument(
        "--no-think",
        action="store_true",
        help="Send chat_template_kwargs={enable_thinking: false} (vLLM + Qwen3) to skip the reasoning block.",
    )
    translate.add_argument(
        "--batch-blocks",
        type=int,
        default=DEFAULT_BATCH_BLOCKS,
        help=f"Max blocks per request (default {DEFAULT_BATCH_BLOCKS}). 0 disables the limit.",
    )
    translate.add_argument(
        "--batch-chars",
        type=int,
        default=DEFAULT_BATCH_CHARS,
        help=f"Max block JSON characters per request (default {DEFAULT_BATCH_CHARS}). 0 disables the limit.",
    )
    translate.add_argument("--raw-response-out", help="Optional path for the full chat completions response.")
    translate.add_argument("--dry-run-request-out", help="Write the request body without calling the server.")
    translate.set_defaults(func=cmd_translate)

    mock = sub.add_parser("mock-response", help="Create a response-shaped JSON file without calling a model.")
    mock.add_argument("--payload", required=True)
    mock.add_argument("--output", required=True)
    mock.add_argument("--prefix", default="[MOCK] ")
    mock.set_defaults(func=cmd_mock_response)

    render = sub.add_parser("render", help="Render a response JSON back onto the original PDF layout.")
    render.add_argument("--pdf", required=True)
    render.add_argument("--ir", required=True)
    render.add_argument("--response", required=True)
    render.add_argument("--output-pdf", required=True)
    render.add_argument("--output-html")
    render.add_argument("--debug-boxes", action="store_true")
    render.set_defaults(func=cmd_render)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
