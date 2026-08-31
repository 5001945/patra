from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

from .ir_builder import TRANSLATABLE_TYPES
from .models import Block, DocumentIR, RichSpan, rich_span_from_dict
from .protect import restore_text


SYSTEM_PROMPT = """You transform academic-paper text while preserving structure.

Return strict JSON only. The top-level object must be exactly {"blocks": [...]}.
Open it once at the start and close it once at the very end: every block is one
element of that one array, separated by a comma, and the closing "]}" appears only
after the last block. Do not echo the task or output_format_example keys, and do not
nest the blocks list inside any other key. Keep every block id unchanged. Do not translate or alter
placeholders such as <KEEP_EQ_0/> and <KEEP_TERM_1/>. Do not translate LaTeX, citations,
model names, dataset names, and technical keywords that are protected by
placeholders. If rich_text is provided, preserve bold/italic emphasis in the
translated rich_text output. Every rich_text span must carry the transformed text;
never copy the source text into rich_text while transforming only the text field.
Apply the task to every sentence of every block. A response sentence that is identical
to its source sentence, or that keeps the source word order with only function words
adjusted, is invalid. Only placeholders, proper nouns, identifiers, citations, and terms
the task marks as protected may survive unchanged.
Blocks flagged continues_next and continues_previous are fragments of one sentence that
the page layout split across boxes. Read such a run of blocks as a single sentence,
transform it as a whole, then distribute the result back over the same ids, splitting at
a natural boundary in the target language. Never merge ids, never leave an id empty, and
keep each id's share roughly proportional to the length of its source fragment.
You may adjust span boundaries to fit the translated
text, but do not split spans unless style changes.
""".replace("\n", " ")

DEFAULT_TASK = (
    "Translate the academic-paper text into Korean. Rewrite every sentence with natural "
    "Korean sentence structure and Korean predicates. Never leave a sentence in the source "
    "language, but you may keep source words with only Korean particles attached. Keep "
    "these in English: protected placeholders, section headings, author names, proper nouns, "
    "model, dataset and hardware names, citations, units, and the technical terms that Korean "
    "papers conventionally write in English."
)

# A surviving source sentence at least this long is treated as untranslated.
UNTRANSLATED_SENTENCE_MIN_CHARS = 40

# Below this similarity to the response text, rich_text is only kept when it is
# still closer to the response than to the untransformed source block.
RICH_TEXT_MIN_SIMILARITY = 0.6


def compact_rich_span(span: RichSpan) -> dict[str, Any]:
    item: dict[str, Any] = {"text": span.text}
    if span.bold:
        item["bold"] = True
    if span.italic:
        item["italic"] = True
    if span.font_size is not None:
        item["font_size"] = round(float(span.font_size), 1)
    if span.font_name:
        item["font_name"] = span.font_name
    if span.color:
        item["color"] = span.color
    return item


SENTENCE_END_RE = re.compile(r"""[.!?;:]["'’”\)\]]*$""")


def continues_into_next(current_text: str, next_text: str) -> bool:
    """Whether one sentence was split across two layout boxes.

    Multi-column and multi-page layouts routinely break a sentence mid-clause. The
    model translates each block in isolation, so without a hint it produces two
    half-sentences that do not join up.
    """
    left = normalize_whitespace(current_text)
    right = normalize_whitespace(next_text)
    if not left or not right or SENTENCE_END_RE.search(left):
        return False
    first = right[0]
    return first.islower() or first.isdigit() or first in "(,-"


def build_payload(document: DocumentIR, task: str = DEFAULT_TASK) -> dict[str, Any]:
    translatable = [block for block in document.blocks if is_translatable(block)]
    blocks = []
    for block in translatable:
        entry: dict[str, Any] = {
            "id": block.id,
            "type": block.type,
            "text": block.protected_text or block.text,
        }
        if block.rich_text:
            entry["rich_text"] = [compact_rich_span(span) for span in block.rich_text if span.text]
        blocks.append(entry)

    for index in range(len(translatable) - 1):
        if continues_into_next(translatable[index].text, translatable[index + 1].text):
            blocks[index]["continues_next"] = True
            blocks[index + 1]["continues_previous"] = True

    return {
        "system_prompt": SYSTEM_PROMPT,
        "task": task,
        "response_schema": {
            "blocks": [
                {
                    "id": "same id as input",
                    "text": "transformed plain text with placeholders preserved, no line breaks",
                    "rich_text": [
                        {
                            "text": "transformed span text",
                            "bold": True,
                            "italic": True,
                            "font_size": 9.0,
                            "font_name": "optional font name",
                            "color": "#000000",
                        }
                    ],
                }
            ]
        },
        "blocks": blocks,
    }


# One request per document does not survive a real paper: a 100-block payload is
# ~37k prompt tokens, and the translation of it needs about as many again, so the
# completion is cut off mid-JSON. Smaller requests also keep the model compliant,
# which degrades noticeably over a long single generation.
DEFAULT_BATCH_BLOCKS = 8
DEFAULT_BATCH_CHARS = 12000


def _block_cost(block: dict[str, Any]) -> int:
    return len(json.dumps(block, ensure_ascii=False, separators=(",", ":")))


def split_payload(
    payload: dict[str, Any],
    *,
    max_blocks: int = DEFAULT_BATCH_BLOCKS,
    max_chars: int = DEFAULT_BATCH_CHARS,
) -> list[dict[str, Any]]:
    """Split one payload into request-sized payloads.

    Each keeps the same system prompt, task, and schema; only the block list is
    sliced. A block always stays whole, even when it alone exceeds `max_chars`.
    """
    blocks = payload.get("blocks", [])
    if not blocks or (max_blocks <= 0 and max_chars <= 0):
        return [payload]

    groups: list[list[dict[str, Any]]] = []
    current: list[tuple[dict[str, Any], int]] = []
    size = 0
    for block in blocks:
        cost = _block_cost(block)
        too_many = max_blocks > 0 and len(current) >= max_blocks
        too_long = max_chars > 0 and bool(current) and size + cost > max_chars
        if current and (too_many or too_long):
            # Never end a request on a block whose sentence runs into the next one;
            # the model is told to transform such a run as a whole.
            carry: list[tuple[dict[str, Any], int]] = []
            while len(current) > 1 and current[-1][0].get("continues_next"):
                carry.insert(0, current.pop())
            groups.append([item for item, _ in current])
            current = carry
            size = sum(item_cost for _, item_cost in carry)
        current.append((block, cost))
        size += cost
    if current:
        groups.append([item for item, _ in current])

    return [{**payload, "blocks": group} for group in groups]


def is_translatable(block: Block) -> bool:
    return block.type in TRANSLATABLE_TYPES and bool(block.text.strip()) and not block.excluded


def load_response(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "choices" in data:
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    return data


def coerce_rich_spans(data: list[dict[str, Any]] | None) -> list[RichSpan]:
    if not data:
        return []
    return [rich_span_from_dict(span) for span in data]


def response_by_block_id(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(block["id"]): block for block in response.get("blocks", [])}


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def strip_all_whitespace(value: str) -> str:
    """Whitespace-insensitive form, for deciding whether a block changed at all.

    Models sometimes echo the source but re-break lines to mirror the rich_text
    span split, e.g. "III. EVALUATION" comes back as "III. E\\nVALUATION".
    """
    return re.sub(r"\s+", "", value)


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def rich_text_matches_response(block: Block, response_text: str, spans: list[RichSpan]) -> bool:
    """Detect the model transforming `text` while echoing the source into `rich_text`.

    Rendering prefers rich_text, so an echoed source silently overwrites a correct
    transformation. Comparing against both the response text and the original block
    keeps this language-agnostic.
    """
    joined = normalize_whitespace("".join(span.text for span in spans))
    target = normalize_whitespace(response_text)
    if not joined or not target:
        return False
    to_response = _similarity(joined, target)
    if to_response >= RICH_TEXT_MIN_SIMILARITY:
        return True
    return to_response >= _similarity(joined, normalize_whitespace(block.text))


def untranslated_source_sentences(
    source_text: str,
    rendered_text: str,
    min_chars: int = UNTRANSLATED_SENTENCE_MIN_CHARS,
) -> list[str]:
    """Source sentences that survived verbatim into the rendered output.

    Protected terms, citations, and hardware or model names are expected to stay in
    the source language, so this deliberately works at sentence granularity: a whole
    surviving sentence means the model skipped it, while surviving fragments do not.
    """
    rendered = normalize_whitespace(rendered_text)
    if not rendered:
        return []
    survivors = []
    for sentence in re.split(r"(?<=[.!?])\s+", normalize_whitespace(source_text)):
        candidate = sentence.strip()
        if len(candidate) >= min_chars and candidate in rendered:
            survivors.append(candidate)
    return survivors


def restored_response_block(
    block: Block,
    response_block: dict[str, Any],
    notes: list[str] | None = None,
) -> tuple[str, list[RichSpan]]:
    text = restore_text(str(response_block.get("text", "")), block.protection_map)
    rich = []
    for span in coerce_rich_spans(response_block.get("rich_text")):
        rich.append(
            RichSpan(
                text=restore_text(span.text, block.protection_map),
                bold=span.bold,
                italic=span.italic,
                font_size=span.font_size,
                font_name=span.font_name,
                color=span.color,
            )
        )
    if rich and not rich_text_matches_response(block, text, rich):
        if notes is not None:
            notes.append(f"{block.id}: rich_text looked untransformed; rendered the plain text instead")
        rich = []
    if not rich:
        template = block.rich_text[0] if block.rich_text else RichSpan(text="")
        rich = [
            RichSpan(
                text=text,
                bold=template.bold,
                italic=template.italic,
                font_size=template.font_size,
                font_name=template.font_name,
                color=template.color,
            )
        ]
    return text, rich


def build_mock_response(payload: dict[str, Any], prefix: str = "[MOCK] ") -> dict[str, Any]:
    blocks = []
    for block in payload.get("blocks", []):
        text = prefix + block.get("text", "")
        rich_text = [dict(span) for span in block.get("rich_text", [])]
        if rich_text:
            rich_text[0]["text"] = prefix + str(rich_text[0].get("text", ""))
        else:
            rich_text = [{"text": text}]
        blocks.append({"id": block["id"], "text": text, "rich_text": rich_text})
    return {"blocks": blocks}
