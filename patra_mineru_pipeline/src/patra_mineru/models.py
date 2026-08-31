from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RichSpan:
    text: str
    bold: bool = False
    italic: bool = False
    font_size: float | None = None
    font_name: str | None = None
    color: str | None = None


@dataclass
class BlockRegion:
    """One rectangle a block occupies on one page.

    A paragraph interrupted by a column or page break occupies several, and its
    text has to flow through all of them.
    """

    page_idx: int
    bbox: list[float]


@dataclass
class Block:
    id: str
    type: str
    page_idx: int
    bbox: list[float]
    text: str = ""
    text_level: int | None = None
    section_id: str | None = None
    excluded: bool = False
    rich_text: list[RichSpan] = field(default_factory=list)
    protected_text: str = ""
    protection_map: dict[str, str] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    # Only set when the block spans more than one rectangle; `page_idx`/`bbox`
    # stay the first one so older tooling keeps working.
    regions: list[BlockRegion] = field(default_factory=list)
    # Set on the empty placeholder block MinerU emits for a continuation region
    # whose text it folded into an earlier block.
    continuation_of: str | None = None


def block_regions(block: Block) -> list[BlockRegion]:
    return block.regions or [BlockRegion(page_idx=block.page_idx, bbox=list(block.bbox))]


@dataclass
class Section:
    id: str
    title: str
    level: int
    start_block_id: str
    excluded: bool = False


@dataclass
class DocumentIR:
    source_pdf: str
    mineru_output_dir: str
    blocks: list[Block]
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def rich_span_from_dict(data: dict[str, Any]) -> RichSpan:
    return RichSpan(
        text=str(data.get("text", "")),
        bold=bool(data.get("bold", False)),
        italic=bool(data.get("italic", False)),
        font_size=data.get("font_size"),
        font_name=data.get("font_name"),
        color=data.get("color"),
    )


def block_from_dict(data: dict[str, Any]) -> Block:
    return Block(
        id=str(data["id"]),
        type=str(data["type"]),
        page_idx=int(data["page_idx"]),
        bbox=[float(v) for v in data["bbox"]],
        text=str(data.get("text", "")),
        text_level=data.get("text_level"),
        section_id=data.get("section_id"),
        excluded=bool(data.get("excluded", False)),
        rich_text=[rich_span_from_dict(s) for s in data.get("rich_text", [])],
        protected_text=str(data.get("protected_text", "")),
        protection_map={str(k): str(v) for k, v in data.get("protection_map", {}).items()},
        source=data.get("source", {}),
        regions=[
            BlockRegion(page_idx=int(r["page_idx"]), bbox=[float(v) for v in r["bbox"]])
            for r in data.get("regions", [])
        ],
        continuation_of=data.get("continuation_of"),
    )


def section_from_dict(data: dict[str, Any]) -> Section:
    return Section(
        id=str(data["id"]),
        title=str(data["title"]),
        level=int(data["level"]),
        start_block_id=str(data["start_block_id"]),
        excluded=bool(data.get("excluded", False)),
    )


def document_ir_from_dict(data: dict[str, Any]) -> DocumentIR:
    return DocumentIR(
        source_pdf=str(data["source_pdf"]),
        mineru_output_dir=str(data["mineru_output_dir"]),
        blocks=[block_from_dict(b) for b in data.get("blocks", [])],
        sections=[section_from_dict(s) for s in data.get("sections", [])],
        metadata=data.get("metadata", {}),
    )
