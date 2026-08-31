from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import fitz

from .models import Block, DocumentIR, RichSpan, block_regions, document_ir_from_dict
from .payload import (
    is_translatable,
    response_by_block_id,
    restored_response_block,
    strip_all_whitespace,
    untranslated_source_sentences,
)
from .style import block_rect


def _inline_text(value: str, *, preserve_newlines: bool) -> str:
    """Prepare span text for the HTML box.

    Span text carries the source PDF's line wrapping. Emitting it as <br/> replays
    the source language's break positions inside the transformed text, which is what
    produced stray mid-sentence breaks. Collapse it instead and let the box re-wrap,
    which also lets justified alignment work. Line-end hyphens are rejoined without a
    space, since hyphenated compounds are far more common than soft hyphenation in
    the technical papers this targets.
    """
    if preserve_newlines:
        return html.escape(value).replace("\n", "<br/>")
    collapsed = re.sub(r"-[ \t]*\n[ \t]*", "-", value)
    collapsed = re.sub(r"\s+", " ", collapsed)
    return html.escape(collapsed)


def _span_to_html(span: RichSpan, default_size: float = 9.0, *, preserve_newlines: bool = False) -> str:
    text = _inline_text(span.text, preserve_newlines=preserve_newlines)
    styles = []
    size = span.font_size or default_size
    styles.append(f"font-size:{size:.2f}pt")
    if span.color:
        styles.append(f"color:{span.color}")
    body = f'<span style="{";".join(styles)}">{text}</span>'
    if span.italic:
        body = f"<i>{body}</i>"
    if span.bold:
        body = f"<b>{body}</b>"
    return body


def rich_spans_to_html(spans: list[RichSpan], fallback_text: str, *, preserve_newlines: bool = False) -> str:
    if not spans:
        spans = [RichSpan(text=fallback_text)]
    body = "".join(_span_to_html(span, preserve_newlines=preserve_newlines) for span in spans)
    return f"<p>{body}</p>"


# Code keeps its own line structure; list blocks join their items with newlines, and
# collapsing those would run separate items together into one paragraph.
NEWLINE_PRESERVING_TYPES = ("code", "list")

# Korean wraps per character rather than per word, so justified text fills the box
# evenly instead of leaving the ragged left-heavy look of left alignment.
BLOCK_CSS = """
p {
  margin: 0;
  line-height: 1.12;
  font-family: sans-serif;
  text-align: justify;
}
"""


MIN_SCALE = 0.55
# Binary search steps for the scale shared by all of a block's rectangles.
SCALE_SEARCH_STEPS = 12
_NO_OVERFLOW = fitz.mupdf.FZ_PLACE_STORY_FLAG_NO_OVERFLOW


def _block_rects(
    doc: fitz.Document,
    block: Block,
    padding: float,
    warnings: list[str],
) -> list[tuple[fitz.Page, fitz.Rect]]:
    rects: list[tuple[fitz.Page, fitz.Rect]] = []
    for region in block_regions(block):
        if region.page_idx < 0 or region.page_idx >= len(doc):
            warnings.append(f"{block.id}: page index out of range")
            continue
        page = doc[region.page_idx]
        rect = block_rect(page, region.bbox)
        rect = fitz.Rect(rect.x0 - padding, rect.y0 - padding, rect.x1 + padding, rect.y1 + padding)
        rect &= page.rect
        if rect.is_empty or rect.width < 4 or rect.height < 4:
            warnings.append(f"{block.id}: empty or tiny bbox")
            continue
        rects.append((page, rect))
    return rects


def _story_fits(story: fitz.Story, rects: list[tuple[fitz.Page, fitz.Rect]], scale: float) -> bool:
    """Whether the story runs out of text before it runs out of rectangles.

    Placing into a rectangle enlarged by 1/scale is how insert_htmlbox models
    shrinking the text; doing it here keeps one scale across all rectangles.
    """
    story.reset()
    for _, rect in rects:
        more, _ = story.place(fitz.Rect(0, 0, rect.width / scale, rect.height / scale), _NO_OVERFLOW)
        # Only drawing advances the story; without this every rectangle would be
        # measured against the start of the text again.
        story.draw(None)
        if not more:
            return True
    return False


def _insert_flowed_html(
    rects: list[tuple[fitz.Page, fitz.Rect]],
    block_html: str,
    *,
    css: str,
    scale_low: float,
) -> tuple[float, float]:
    """Pour one block's HTML through its rectangles, continuing across pages.

    insert_htmlbox only fills a single rectangle, which is all a block that MinerU
    kept on one page needs, so that path is left untouched.
    """
    if len(rects) == 1:
        page, rect = rects[0]
        return page.insert_htmlbox(rect, block_html, css=css, scale_low=scale_low, overlay=True)

    story = fitz.Story(html=block_html, user_css="body {margin:1px;}" + css)
    scale = 1.0
    fits = _story_fits(story, rects, scale)
    if not fits:
        low = max(scale_low, 0.05)
        if _story_fits(story, rects, low):
            fits = True
            high = 1.0
            for _ in range(SCALE_SEARCH_STEPS):
                middle = (low + high) / 2
                if _story_fits(story, rects, middle):
                    low = middle
                else:
                    high = middle
        # Drawing at the smallest allowed scale keeps the text on the page even
        # when it cannot fully fit; only the tail is clipped.
        scale = low

    def rect_function(rect_num, filled):
        if rect_num < len(rects):
            _, rect = rects[rect_num]
            mediabox = fitz.Rect(0, 0, rect.width / scale, rect.height / scale)
        else:
            # Out of real rectangles. A tall spare page ends Story.write instead of
            # letting it ask for rectangles forever; its content is dropped.
            _, rect = rects[-1]
            mediabox = fitz.Rect(0, 0, rect.width / scale, rect.height * 100 / scale)
        return mediabox, mediabox, None

    story.reset()
    flowed = story.write_with_links(rect_function)
    try:
        for index in range(min(len(flowed), len(rects))):
            page, rect = rects[index]
            page.show_pdf_page(rect, flowed, index, overlay=True)
    finally:
        flowed.close()
    return (0.0 if fits else -1.0), scale


def render_pdf(
    original_pdf: str | Path,
    document: DocumentIR,
    response: dict[str, Any],
    output_pdf: str | Path,
    *,
    padding: float = 1.5,
    draw_debug_boxes: bool = False,
) -> list[str]:
    updates = response_by_block_id(response)
    warnings: list[str] = []
    doc = fitz.open(str(original_pdf))
    try:
        for block in document.blocks:
            if not is_translatable(block):
                continue
            response_block = updates.get(block.id)
            if not response_block:
                warnings.append(f"{block.id}: missing response block")
                continue
            rects = _block_rects(doc, block, padding, warnings)
            if not rects:
                continue
            text, rich_spans = restored_response_block(block, response_block, notes=warnings)
            rendered_text = "".join(span.text for span in rich_spans) or text
            if strip_all_whitespace(rendered_text) == strip_all_whitespace(block.text):
                warnings.append(f"{block.id}: model returned the source text unchanged")
            else:
                survivors = untranslated_source_sentences(block.text, rendered_text)
                if survivors:
                    total = sum(len(sentence) for sentence in survivors)
                    warnings.append(
                        f"{block.id}: {len(survivors)} source sentence(s) left untransformed "
                        f"({total} chars), first: {survivors[0][:60]!r}"
                    )
            for page, rect in rects:
                page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)
                if draw_debug_boxes:
                    page.draw_rect(rect, color=(1, 0, 0), width=0.3, overlay=True)

            block_html = rich_spans_to_html(rich_spans, text, preserve_newlines=block.type in NEWLINE_PRESERVING_TYPES)
            spare_height, scale = _insert_flowed_html(rects, block_html, css=BLOCK_CSS, scale_low=MIN_SCALE)
            if spare_height < 0:
                warnings.append(f"{block.id}: text did not fully fit bbox, scale={scale:.3f}")

        Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_pdf), garbage=4, deflate=True)
    finally:
        doc.close()
    return warnings


def render_html_preview(
    original_pdf: str | Path,
    document: DocumentIR,
    response: dict[str, Any],
    output_html: str | Path,
) -> None:
    updates = response_by_block_id(response)
    doc = fitz.open(str(original_pdf))
    try:
        pages = []
        for page_idx, page in enumerate(doc):
            divs = []
            for block in document.blocks:
                if block.page_idx != page_idx or not is_translatable(block) or block.id not in updates:
                    continue
                text, rich_spans = restored_response_block(block, updates[block.id])
                rect = block_rect(page, block.bbox)
                divs.append(
                    f'<div class="block" style="left:{rect.x0}px;top:{rect.y0}px;width:{rect.width}px;height:{rect.height}px">'
                    f"{rich_spans_to_html(rich_spans, text, preserve_newlines=block.type in NEWLINE_PRESERVING_TYPES)}</div>"
                )
            pages.append(
                f'<section class="page" style="width:{page.rect.width}px;height:{page.rect.height}px">'
                + "".join(divs)
                + "</section>"
            )
    finally:
        doc.close()

    html_doc = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body { margin: 24px; background: #ddd; }
    .page { position: relative; margin: 0 auto 24px; background: white; box-shadow: 0 1px 8px rgba(0,0,0,.2); }
    .block { position: absolute; overflow: hidden; background: white; line-height: 1.12; font-family: sans-serif; text-align: justify; }
    .block p { margin: 0; }
  </style>
</head>
<body>
""" + "\n".join(pages) + """
</body>
</html>
"""
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    Path(output_html).write_text(html_doc, encoding="utf-8")


def load_document_ir(path: str | Path) -> DocumentIR:
    return document_ir_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
