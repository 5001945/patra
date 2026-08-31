from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .payload import DEFAULT_BATCH_BLOCKS, DEFAULT_BATCH_CHARS, split_payload


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build_user_content(payload: dict[str, Any]) -> str:
    # The schema is sent under a name the model is unlikely to echo back. Some
    # models mirror the input envelope and nest their answer under the key they
    # were given, which used to produce {"response_schema": {"blocks": [...]}}.
    user_payload = {
        "task": payload.get("task", "Transform the blocks."),
        "output_format_example": payload.get("response_schema"),
        "blocks": payload.get("blocks", []),
    }
    return _compact_json(user_payload)


def build_chat_request(
    payload: dict[str, Any],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    json_mode: bool = False,
    no_think: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": payload.get("system_prompt", "Return strict JSON only.")},
            {"role": "user", "content": build_user_content(payload)},
        ],
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if no_think:
        # vLLM passes this to the chat template; Qwen3 uses it to skip the <think>
        # block, which otherwise spends a large share of the completion budget.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


THINK_BLOCK_RE = re.compile(r"^\s*<(think|thinking|reasoning)>[\s\S]*?</\1>", re.IGNORECASE)

# Keys a model may nest its answer under when it mirrors the input envelope.
NESTED_ANSWER_KEYS = ("output_format_example", "response_schema", "response", "result", "output", "data")


def strip_reasoning_prefix(text: str) -> str:
    """Drop a leading <think>...</think> block emitted by reasoning models."""
    return THINK_BLOCK_RE.sub("", text, count=1).strip()


def _iter_json_values(text: str):
    """Every JSON value in `text`, skipping whatever sits between them."""
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        if text[index] not in "{[":
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        yield value
        index = end


def salvage_blocks(text: str) -> list[dict[str, Any]]:
    """Collect block objects out of malformed model output.

    Seen from Qwen3: the closing `]}` is repeated after every block, so the content
    reads as `{"blocks":[B1]},B2]},B3]}...` and `json.loads` stops after the first
    object with "Extra data". Each block is still valid JSON on its own.
    """
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in _iter_json_values(text):
        if isinstance(value, dict) and isinstance(value.get("blocks"), list):
            candidates = value["blocks"]
        elif isinstance(value, dict) and "id" in value:
            candidates = [value]
        elif isinstance(value, list):
            candidates = value
        else:
            continue
        for block in candidates:
            if not isinstance(block, dict):
                continue
            block_id = block.get("id")
            if isinstance(block_id, str) and block_id not in seen:
                seen.add(block_id)
                blocks.append(block)
    return blocks


def extract_json_content(content: str, notes: list[str] | None = None) -> dict[str, Any]:
    text = strip_reasoning_prefix(content.strip())
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        parsed = None
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            recovered = salvage_blocks(text)
            if not recovered:
                raise
            if notes is not None:
                notes.append(f"recovered {len(recovered)} block(s) from malformed JSON")
            return {"blocks": recovered}
    if not isinstance(parsed, dict):
        raise ValueError("Model content must decode to a JSON object.")
    if not isinstance(parsed.get("blocks"), list):
        for key in NESTED_ANSWER_KEYS:
            nested = parsed.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("blocks"), list):
                return nested
        raise ValueError("Model JSON must contain a blocks list.")
    return parsed


def response_content(raw: dict[str, Any]) -> str:
    """The assistant message, refusing a completion the server cut short.

    A truncated response otherwise surfaces as `json.JSONDecodeError: Unterminated
    string`, which says nothing about the actual cause.
    """
    choices = raw.get("choices") or [{}]
    choice = choices[0]
    content = (choice.get("message") or {}).get("content", "") or ""
    if choice.get("finish_reason") == "length":
        usage = raw.get("usage") or {}
        raise RuntimeError(
            "The server stopped the completion at the token limit, so the JSON is "
            "truncated (finish_reason=length, "
            f"prompt_tokens={usage.get('prompt_tokens')}, "
            f"completion_tokens={usage.get('completion_tokens')}, "
            f"total_tokens={usage.get('total_tokens')}). "
            "The request has to leave room for a translation about as long as its input: "
            "send fewer blocks per request (--batch-blocks / --batch-chars), raise the "
            "server's context length, drop --max-tokens if it is set too low, or spend "
            "less of the budget on reasoning (--no-think)."
        )
    return content


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    request_body: dict[str, Any],
    timeout: float = 300.0,
) -> dict[str, Any]:
    data = _compact_json(request_body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(chat_completions_url(base_url), data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from LLM server: {detail}") from exc


def translate_payload(
    *,
    payload_path: str | Path,
    output_path: str | Path,
    base_url: str,
    model: str,
    api_key: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: float = 300.0,
    json_mode: bool = False,
    no_think: bool = False,
    batch_blocks: int = DEFAULT_BATCH_BLOCKS,
    batch_chars: int = DEFAULT_BATCH_CHARS,
    raw_response_out: str | Path | None = None,
    dry_run_request_out: str | Path | None = None,
    progress: Callable[[int, int, int], None] | None = None,
    warn: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    batches = split_payload(payload, max_blocks=batch_blocks, max_chars=batch_chars)
    requests = [
        build_chat_request(
            batch,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            no_think=no_think,
        )
        for batch in batches
    ]

    if dry_run_request_out:
        Path(dry_run_request_out).parent.mkdir(parents=True, exist_ok=True)
        body = requests[0] if len(requests) == 1 else requests
        Path(dry_run_request_out).write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        return None

    key = api_key if api_key is not None else os.environ.get(api_key_env, "dummy")
    raws: list[dict[str, Any]] = []
    blocks: list[Any] = []
    for index, (batch, request_body) in enumerate(zip(batches, requests), start=1):
        if progress:
            progress(index, len(batches), len(batch.get("blocks", [])))
        raw = call_openai_compatible(base_url=base_url, api_key=key, request_body=request_body, timeout=timeout)
        raws.append(raw)
        if raw_response_out:
            # Written before parsing, and rewritten after every batch, so a failure
            # later in the run still leaves the completions that did arrive.
            Path(raw_response_out).parent.mkdir(parents=True, exist_ok=True)
            body = raws[0] if len(batches) == 1 else raws
            Path(raw_response_out).write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        notes: list[str] = []
        try:
            parsed = extract_json_content(response_content(raw), notes=notes)
        except Exception as exc:
            raise RuntimeError(f"batch {index}/{len(batches)}: {exc}") from exc
        received = parsed.get("blocks", [])
        if warn:
            for note in notes:
                warn(f"batch {index}/{len(batches)}: {note}")
            returned = {block.get("id") for block in received if isinstance(block, dict)}
            missing = [block["id"] for block in batch.get("blocks", []) if block["id"] not in returned]
            if missing:
                warn(
                    f"batch {index}/{len(batches)}: {len(missing)} block(s) missing from the response: "
                    + ", ".join(missing[:5])
                )
        blocks.extend(received)

    result = {"blocks": blocks}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
