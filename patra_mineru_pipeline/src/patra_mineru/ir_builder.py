from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz

from .models import Block, BlockRegion, DocumentIR, Section, block_regions
from .protect import DEFAULT_PROTECTED_TERMS, protect_text
from .style import compare_key, extract_rich_spans, region_text

TRANSLATABLE_TYPES = {"text", "list", "code"}

# How many earlier text blocks a continuation region may belong to. MinerU puts
# figures and headers of the intervening pages between the two, so the owner is
# not necessarily the immediately preceding block.
FRAGMENT_LOOKBACK = 5
# Shorter leftovers are not distinctive enough to match on.
MIN_FRAGMENT_MATCH_CHARS = 20
FRAGMENT_MATCH_MIN_RATIO = 0.9
DEFAULT_EXCLUDE_PATTERNS = [
    r"^\s*references?\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*acknowledg(e)?ments?\s*$",
    r"^\s*appendix\b",
]


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_mineru_file(output_dir: str | Path, suffix: str) -> Path:
    root = Path(output_dir)
    if root.is_file():
        if root.name.endswith(suffix) or root.name.endswith(suffix.lstrip("_")):
            return root
        raise FileNotFoundError(f"MinerU path is a file, but it does not look like {suffix!r}: {root}")
    if not root.exists():
        raise FileNotFoundError(
            f"MinerU output path does not exist: {root}\n"
            "Run `patra-mineru parse --pdf ... --output-dir ...` first, or pass the actual MinerU output "
            "directory/file to `--mineru-output`."
        )
    matches = sorted(root.rglob(f"*{suffix}"))
    if not matches:
        matches = sorted(root.rglob(f"*{suffix.lstrip('_')}"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find a MinerU content list ending with {suffix!r} under {root}\n"
            "Expected something like `example_content_list.json` or `content_list.json`. "
            "Check where MinerU wrote its output, then pass that directory or JSON file to `--mineru-output`."
        )
    return matches[0]


def _item_text(item: dict[str, Any]) -> str:
    if "text" in item:
        return str(item.get("text") or "")
    if item.get("type") == "list":
        # MinerU 3.x emits list blocks as a list_items array with no text field.
        # Without this branch the whole block reads as empty and is silently dropped
        # from the payload, losing body paragraphs that MinerU grouped as a list.
        return "\n".join(str(part) for part in item.get("list_items", []) if part)
    if item.get("type") == "table":
        parts = item.get("table_caption", []) + [item.get("table_body", "")] + item.get("table_footnote", [])
        return "\n".join(str(part) for part in parts if part)
    if item.get("type") == "image":
        parts = item.get("image_caption", []) + item.get("image_footnote", [])
        return "\n".join(str(part) for part in parts if part)
    return ""


def _is_heading(item: dict[str, Any]) -> bool:
    return item.get("type") == "text" and int(item.get("text_level") or 0) > 0


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _fragment_owner(candidates: list[Block], fragment_text: str) -> Block | None:
    """The block whose text ends with `fragment_text`, if any."""
    key = compare_key(fragment_text)
    if len(key) < MIN_FRAGMENT_MATCH_CHARS:
        return None
    for owner in reversed(candidates):
        owner_key = compare_key(owner.text)
        if len(owner_key) <= len(key):
            continue
        if owner_key.endswith(key):
            return owner
        tail = owner_key[-len(key) :]
        if SequenceMatcher(None, tail, key, autojunk=False).ratio() >= FRAGMENT_MATCH_MIN_RATIO:
            return owner
    return None


def attach_continuation_regions(pdf_path: str | Path, blocks: list[Block]) -> None:
    """Give a block every rectangle its text occupies.

    When a column or page break interrupts a paragraph, MinerU folds the whole
    paragraph into the block for the first rectangle and emits an empty text block
    for the leftover rectangle. Rendering then crammed the entire translation into
    the first box and, seeing nothing to do for the empty block, left the source
    text of the second one on the page.
    """
    if not any(block.type in TRANSLATABLE_TYPES and not block.text.strip() for block in blocks):
        return
    doc = fitz.open(str(pdf_path))
    try:
        owners: list[Block] = []
        for block in blocks:
            if block.type not in TRANSLATABLE_TYPES:
                continue
            if block.text.strip():
                owners.append(block)
                continue
            region = BlockRegion(page_idx=block.page_idx, bbox=list(block.bbox))
            owner = _fragment_owner(owners[-FRAGMENT_LOOKBACK:], region_text(doc, region))
            if owner is None:
                continue
            owner.regions = block_regions(owner) + [region]
            block.continuation_of = owner.id
    finally:
        doc.close()


def build_document_ir(
    pdf_path: str | Path,
    mineru_output_dir: str | Path,
    exclude_patterns: list[str] | None = None,
    protected_terms: list[str] | None = None,
    translate_headings: bool = False,
) -> DocumentIR:
    """Build the document IR.

    Headings are left in the source language by default. Beyond matching the usual
    convention for Korean papers, it keeps the renderer from covering small-caps
    headings such as "III. EVALUATION" with a sans-serif approximation.
    """
    content_path = find_mineru_file(mineru_output_dir, "_content_list.json")
    content_items = load_json(content_path)
    patterns = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS
    terms = list(DEFAULT_PROTECTED_TERMS)
    if protected_terms:
        terms.extend(protected_terms)

    blocks: list[Block] = []
    sections: list[Section] = []
    current_section: Section | None = None
    excluded_stack: list[tuple[int, bool]] = []

    for index, item in enumerate(content_items):
        block_id = f"b_{index:05d}"
        text = _item_text(item)
        text_level = item.get("text_level")
        is_heading = _is_heading(item)

        if is_heading:
            level = int(text_level)
            while excluded_stack and excluded_stack[-1][0] >= level:
                excluded_stack.pop()
            section = Section(
                id=f"s_{len(sections):04d}",
                title=text.strip(),
                level=level,
                start_block_id=block_id,
                excluded=_matches_any(text, patterns),
            )
            sections.append(section)
            current_section = section
            excluded_stack.append((level, section.excluded or (excluded_stack[-1][1] if excluded_stack else False)))

        inherited_excluded = excluded_stack[-1][1] if excluded_stack else False
        if is_heading and not translate_headings:
            inherited_excluded = True
        protected_text, mapping = protect_text(text, terms)
        page_idx = int(item.get("page_idx", 0))
        bbox = [float(v) for v in item.get("bbox", [0, 0, 1000, 1000])]

        blocks.append(
            Block(
                id=block_id,
                type=str(item.get("type", "unknown")),
                page_idx=page_idx,
                bbox=bbox,
                text=text,
                text_level=int(text_level) if text_level is not None else None,
                section_id=current_section.id if current_section else None,
                excluded=inherited_excluded,
                protected_text=protected_text,
                protection_map=mapping,
                source=item,
            )
        )

    # Regions come first: a block split across pages must have both rectangles
    # before its styled spans are read out of the PDF.
    attach_continuation_regions(pdf_path, blocks)
    for block in blocks:
        if block.type in TRANSLATABLE_TYPES and block.text:
            block.rich_text = extract_rich_spans(pdf_path, block_regions(block), block.text)

    return DocumentIR(
        source_pdf=str(pdf_path),
        mineru_output_dir=str(mineru_output_dir),
        blocks=blocks,
        sections=sections,
        metadata={"content_list": str(content_path), "translatable_types": sorted(TRANSLATABLE_TYPES)},
    )
