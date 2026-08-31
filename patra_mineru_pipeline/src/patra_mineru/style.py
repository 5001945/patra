from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re

import fitz

from .models import BlockRegion, RichSpan


def _color_to_hex(value: int | None) -> str | None:
    if value is None:
        return None
    return f"#{value:06x}"


def _font_has(font_name: str, token: str) -> bool:
    compact = font_name.replace("-", "").replace(" ", "").lower()
    return token in compact


def _style_key(span: RichSpan) -> tuple[bool, bool, float | None, str | None, str | None]:
    size = round(span.font_size, 1) if span.font_size is not None else None
    return (span.bold, span.italic, size, span.font_name, span.color)


def _append_text(left: str, right: str, separator: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if separator == "\n":
        if left.endswith("\n") or right.startswith("\n"):
            return left + right
        return left + "\n" + right
    if separator == " " and not left.endswith((" ", "\n")) and not right.startswith((" ", "\n")):
        return left + " " + right
    return left + right


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\\_", "_")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("_", "")
    return re.sub(r"[^a-z0-9]+", "", text)


def _joined_text(spans: list[RichSpan]) -> str:
    return "".join(span.text for span in spans)


def _rich_text_score(spans: list[RichSpan], source_text: str) -> float:
    source = _normalize_text(source_text)
    candidate = _normalize_text(_joined_text(spans))
    if not source or not candidate:
        return 0.0
    ratio = SequenceMatcher(None, source, candidate).ratio()
    extra_ratio = max(0, len(candidate) - len(source)) / max(len(source), 1)
    missing_ratio = max(0, len(source) - len(candidate)) / max(len(source), 1)
    if source in candidate:
        ratio -= min(extra_ratio, 1.0) * 0.75
    return ratio - (extra_ratio * 0.5) - (missing_ratio * 0.35)


def _rich_text_is_usable(spans: list[RichSpan], source_text: str) -> bool:
    source = _normalize_text(source_text)
    candidate = _normalize_text(_joined_text(spans))
    if not source or not candidate:
        return False
    extra_ratio = max(0, len(candidate) - len(source)) / max(len(source), 1)
    missing_ratio = max(0, len(source) - len(candidate)) / max(len(source), 1)
    ratio = SequenceMatcher(None, source, candidate).ratio()
    return ratio >= 0.82 and extra_ratio <= 0.08 and missing_ratio <= 0.18


def merge_rich_spans(spans: list[tuple[float, float, float, RichSpan]]) -> list[RichSpan]:
    if not spans:
        return []
    spans.sort(key=lambda item: (round(item[0], 1), item[1]))
    merged: list[RichSpan] = []
    last_y: float | None = None
    last_x1: float | None = None
    last_size: float = 9.0

    for y0, x0, x1, span in spans:
        if not merged:
            merged.append(span)
            last_y = y0
            last_x1 = x1
            last_size = span.font_size or last_size
            continue

        same_style = _style_key(merged[-1]) == _style_key(span)
        line_changed = last_y is not None and abs(y0 - last_y) > max(2.0, last_size * 0.35)
        gap = 0.0 if last_x1 is None else x0 - last_x1
        separator = "\n" if line_changed else (" " if gap > max(1.5, last_size * 0.25) else "")

        if same_style:
            merged[-1].text = _append_text(merged[-1].text, span.text, separator)
        else:
            if line_changed and span.text and not span.text.startswith("\n"):
                span.text = "\n" + span.text
            elif separator == " " and span.text and not span.text.startswith(" "):
                span.text = " " + span.text
            merged.append(span)

        last_y = y0
        last_x1 = x1
        last_size = span.font_size or last_size

    return merged


def block_rect(page: fitz.Page, bbox: list[float]) -> fitz.Rect:
    """Convert MinerU content_list 0-1000 bbox coordinates into page coordinates."""
    width = page.rect.width
    height = page.rect.height
    x0, y0, x1, y1 = bbox
    return fitz.Rect(x0 / 1000 * width, y0 / 1000 * height, x1 / 1000 * width, y1 / 1000 * height)


def _collect_rich_spans(page: fitz.Page, target: fitz.Rect, *, min_overlap_ratio: float) -> list[RichSpan]:
    spans: list[tuple[float, float, float, RichSpan]] = []
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                span_rect = fitz.Rect(span.get("bbox"))
                overlap = target & span_rect
                if overlap.is_empty:
                    continue
                span_area = span_rect.get_area()
                if span_area <= 0:
                    continue
                center = fitz.Point((span_rect.x0 + span_rect.x1) / 2, (span_rect.y0 + span_rect.y1) / 2)
                center_inside = target.contains(center)
                mostly_inside = overlap.get_area() / span_area >= min_overlap_ratio
                if not center_inside and not mostly_inside:
                    continue
                font_name = span.get("font", "")
                flags = int(span.get("flags", 0))
                spans.append(
                    (
                        span_rect.y0,
                        span_rect.x0,
                        span_rect.x1,
                        RichSpan(
                            text=text,
                            bold=bool(flags & 16) or _font_has(font_name, "bold"),
                            italic=bool(flags & 2) or _font_has(font_name, "italic") or _font_has(font_name, "oblique"),
                            font_size=float(span.get("size")) if span.get("size") else None,
                            font_name=font_name or None,
                            color=_color_to_hex(span.get("color")),
                        ),
                    )
                )
    return merge_rich_spans(spans)


def compare_key(text: str) -> str:
    """Punctuation- and case-insensitive form for comparing two renderings of the
    same text, e.g. MinerU's block text against PyMuPDF's extraction."""
    return _normalize_text(text)


def region_text(doc: fitz.Document, region: BlockRegion) -> str:
    if region.page_idx < 0 or region.page_idx >= len(doc):
        return ""
    page = doc[region.page_idx]
    return page.get_text("text", clip=block_rect(page, region.bbox))


def _collect_over_regions(
    doc: fitz.Document,
    regions: list[BlockRegion],
    *,
    min_overlap_ratio: float,
) -> list[RichSpan]:
    collected: list[RichSpan] = []
    for region in regions:
        if region.page_idx < 0 or region.page_idx >= len(doc):
            continue
        page = doc[region.page_idx]
        part = _collect_rich_spans(page, block_rect(page, region.bbox), min_overlap_ratio=min_overlap_ratio)
        # The break between two regions is a word boundary in the source layout.
        if collected and part and not part[0].text.startswith((" ", "\n")):
            part[0].text = " " + part[0].text
        collected.extend(part)
    return collected


def extract_rich_spans(pdf_path: str | Path, regions: list[BlockRegion], fallback_text: str) -> list[RichSpan]:
    doc = fitz.open(str(pdf_path))
    try:
        strict_spans = _collect_over_regions(doc, regions, min_overlap_ratio=0.55)
        if _rich_text_is_usable(strict_spans, fallback_text):
            return strict_spans

        relaxed_spans = _collect_over_regions(doc, regions, min_overlap_ratio=0.30)
        strict_score = _rich_text_score(strict_spans, fallback_text)
        relaxed_score = _rich_text_score(relaxed_spans, fallback_text)
        if _rich_text_is_usable(relaxed_spans, fallback_text) and relaxed_score >= strict_score:
            return relaxed_spans

        return [RichSpan(text=fallback_text)]
    finally:
        doc.close()
