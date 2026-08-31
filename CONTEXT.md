# Project Context

## Goal

Build a local paper-PDF transformation tool around MinerU and an OpenAI-compatible
LLM server.

The intended workflow is:

1. Upload or provide a paper PDF.
2. Parse it with MinerU without using GPU.
3. Exclude sections such as References.
4. Preserve equations and protected technical terms with placeholders.
5. Preserve rich text by extracting PDF font/style spans.
6. Send selected text blocks, with merged rich-text runs, to an OpenAI-compatible
   server.
7. Show translated results for selected blocks in a web UI.
8. Render model responses back onto the original PDF layout when needed.

## Current Implementation

The main code lives in `patra_mineru_pipeline/`.

Important files:

- `src/patra_mineru/cli.py`
  - Provides `patra-mineru parse`, `prepare`, `translate`, `mock-response`, and
    `render`.
- `src/patra_mineru/mineru_runner.py`
  - Wraps the MinerU 3.4.1 CLI.
  - `build_mineru_command` / `build_mineru_env` are split out so they can be tested
    without invoking MinerU.
  - `device`: `auto` (default) leaves selection to MinerU, which uses CUDA when
    available; `cpu` hides GPUs; `cuda` / `cuda:N` set `MINERU_DEVICE_MODE`.
  - Remote parsing has two separate forms, see "MinerU Servers" below.
  - Defaults can be set by environment: `PATRA_MINERU_BACKEND`,
    `PATRA_MINERU_DEVICE`, `PATRA_MINERU_LANG`, `PATRA_MINERU_SERVER_URL`,
    `PATRA_MINERU_API_URL`. The web app picks these up without a UI change.
- `src/patra_mineru/ir_builder.py`
  - Reads MinerU `*_content_list.json` or `content_list.json`.
  - Builds `DocumentIR`.
  - Detects/excludes References/Bibliography/Acknowledgements/Appendix sections.
  - Keeps headings in the source language by default; `translate_headings=True`
    (`prepare --translate-headings`) opts out.
  - `_item_text` must cover every MinerU content-list shape. MinerU 3.x puts `list`
    block content in a `list_items` array with no `text` field; a missing branch
    reads as empty text and the block is silently dropped from the payload.
  - `attach_continuation_regions` reattaches paragraphs that a column or page break
    split. See "Blocks Split Across Pages" below.
- `src/patra_mineru/style.py`
  - Uses PyMuPDF to extract rich text spans from the original PDF.
  - `extract_rich_spans` takes a list of `BlockRegion`s, so a block split across
    pages contributes spans from every rectangle it occupies.
  - Converts MinerU normalized bbox coordinates into page coordinates.
  - Merges adjacent spans with identical font/style/color/size.
  - Validates extracted rich text against MinerU block text.
  - If strict bbox extraction is too incomplete, retries relaxed extraction.
  - If extraction still looks wrong, falls back to plain block text to avoid
    sending neighboring block text to the model.
- `src/patra_mineru/protect.py`
  - Protects equations and configured terms with placeholders such as
    `<KEEP_EQ_0/>` and `<KEEP_TERM_1/>`.
  - Restores placeholders after model responses.
- `src/patra_mineru/payload.py`
  - Builds the LLM payload.
  - Keeps renderer-only data in `document_ir.json`, not in the request payload.
  - Includes merged `rich_text` runs for the model.
  - `DEFAULT_TASK` is the shared Korean instruction used by the CLI and the web UI.
  - `continues_into_next` marks block pairs where the layout split one sentence, and
    tags them `continues_next` / `continues_previous` in the payload.
  - `untranslated_source_sentences` reports whole source sentences that survived into
    the rendered output.
  - `split_payload` slices the payload into request-sized batches
    (`DEFAULT_BATCH_BLOCKS`, `DEFAULT_BATCH_CHARS`).
- `src/patra_mineru/llm_client.py`
  - Implements OpenAI-compatible `/chat/completions` requests using the standard
    library.
  - Supports `--json-mode` only when the target server supports
    `response_format`.
  - Sends the schema as `output_format_example`, not `response_schema`, so models
    are less likely to echo the input envelope key.
  - `response_content` rejects a completion the server cut short
    (`finish_reason == "length"`) with the token counts, so a truncated response does
    not surface as a JSON decode error.
  - `translate_payload` sends one request per batch and merges the `blocks` lists.
    See "Batched Translation" below.
  - `--no-think` sends `chat_template_kwargs={"enable_thinking": false}`.
  - `extract_json_content` strips a leading `<think>`/`<thinking>`/`<reasoning>`
    block, then accepts a `blocks` list nested under any of
    `output_format_example`, `response_schema`, `response`, `result`, `output`,
    `data`.
  - `salvage_blocks` is the last resort when the content is not valid JSON at all.
    It walks the text with `JSONDecoder.raw_decode`, skipping whatever sits between
    values, and collects every object that has an `id`.
- `src/patra_mineru/reconstruct.py`
  - Renders response JSON back onto the original PDF with PyMuPDF
    `insert_htmlbox` (single-rectangle blocks) or `_insert_flowed_html`
    (blocks with more than one region).
  - Can also write an HTML preview.
  - Warns when a block's `rich_text` was distrusted, when the model returned the
    source text unchanged, and when whole source sentences survived into the output.
  - Collapses the source PDF's line wrapping instead of emitting `<br/>`, so the box
    re-wraps for the target language. `code` blocks still keep their newlines.
  - Renders paragraphs justified (`text-align: justify`).
- `src/patra_mineru/payload.py` (response side)
  - `rich_text_matches_response` drops `rich_text` when it is closer to the
    original source block than to the response `text`. Models sometimes transform
    `text` but echo the source into `rich_text`; since rendering prefers
    `rich_text`, that silently discarded a correct translation.
- `src/patra_mineru/web_app.py`
  - FastAPI local web app.
  - Handles PDF upload, background MinerU/prepare jobs, document metadata,
    block crops, and block-by-block translation.
- `src/patra_mineru/web/`
  - Static PDF.js frontend.
  - Renders the PDF to canvas and overlays clickable block boxes as HTML.

## Current Web UI

Run:

```bash
conda run -n patra patra-mineru-web
```

Open:

```text
http://127.0.0.1:8765/
```

Behavior:

- Drag and drop a PDF.
- By default, the backend runs MinerU parse in the background.
- If MinerU output already exists, turn off "run MinerU parse" in the UI and
  provide a server-side MinerU output path such as `runs/example/mineru`.
- The UI renders the PDF with PDF.js, not the browser's built-in PDF viewer.
- Clickable boxes are HTML overlays aligned to MinerU block bboxes.
- Clicking a box shows original text and a crop image of that region.
- Clicking translate sends only the selected block to the configured
  OpenAI-compatible server.

## MinerU Servers

MinerU 3.4.1 exposes two different services. They are not interchangeable.

1. `mineru-api` — the full parsing service (FastAPI). The client uploads the PDF and
   the service does everything. Reach it with `parse --api-url http://host:port`.
2. `mineru-openai-server` / `mineru-vllm-server` / `mineru-lmdeploy-server` — a VLM
   inference server only. Layout work still runs locally and only model inference is
   remote. Reach it with `parse --backend vlm-http-client --server-url http://host:port`.

The two compose: `build_request_form_data` sends `backend` and `server_url` with every
request, so one long-lived `mineru-api` can serve `vlm-http-client` jobs that point at
a separate VLM server.

Without `--api-url`, `run_orchestrated_cli` starts a throwaway `mineru-api` for every
single run (`LocalAPIServer`) and stops it afterwards. That startup cost is why a
long-lived service is worth running.

Bind `mineru-api` to `127.0.0.1`, not `0.0.0.0`. On a public bind it rejects
`*-http-client` backends and any `server_url` with HTTP 400 as an SSRF guard, which
breaks exactly the combination above. `--allow-public-http-client` overrides it, but
only accept that on a trusted network.

Pick the port deliberately. `mineru-api` defaults to 8000; `pipeline.sh` currently runs
the VLM server on 50020 and the vLLM translation server on 50019.

Note that `mineru`'s own `-l/--lang` defaults to `ch`. For English papers on the
pipeline backend, pass `--lang en`.

## Translation Quality Requirements

The user's standard is sentence-level: every sentence must read as target-language
prose, not source word order with target particles attached.

- Section headings, author names, proper nouns, model/dataset/hardware names,
  citations, units, and technical terms Korean papers normally write in English are
  expected to stay in English. These are not defects and must not be flagged.
- A whole source sentence surviving into the output is a defect. That is what
  `untranslated_source_sentences` detects, and why it works at sentence granularity
  rather than on fragments.
- Charts and figure captions are deliberately left untouched. MinerU 3.x `chart`
  blocks carry `chart_caption` and `chart_footnote`, and `chart` is absent from
  `TRANSLATABLE_TYPES`; both are intentional. Do not add caption translation.
- Headings are no longer sent to the model at all. Besides matching the convention,
  this stops the renderer from covering small-caps headings with a sans-serif
  approximation, and it avoids the `III. E\nVALUATION` echo artifact entirely.

## Layout Requirements

- Do not force line breaks inside a block. Span text carries the source PDF's wrap
  positions; replaying them as `<br/>` reproduced English break points inside Korean
  text, which read as random mid-sentence breaks.
- Use justified alignment. Korean wraps per character rather than per word, so left
  alignment leaves the text visibly bunched to the left.
- Line-end hyphens are rejoined without a space (`IR-\nlevel` becomes `IR-level`),
  since hyphenated compounds outnumber soft hyphenation in these papers.

## Blocks Split Across Pages

- MinerU reports a paragraph interrupted by a column or page break as *two* items:
  the first carries the whole paragraph text with only the first rectangle's bbox,
  and the leftover rectangle arrives as a separate item with `"text": ""`.
- Symptom before the fix (`runs/1786021447.011`, `b_00011` / empty `b_00019`): the
  entire translation was shrunk into the first box, and the second box kept showing
  the untranslated source, because an empty block is not translatable and was never
  even covered.
- `ir_builder.attach_continuation_regions` reads the leftover rectangle's text out of
  the PDF and, when it is the tail of an earlier block's text
  (`style.compare_key`, exact suffix or >= 0.9 ratio, <= 5 blocks back), records it as
  an extra `Block.regions` entry on that block and sets `continuation_of` on the empty
  one. Regions are attached before `extract_rich_spans` runs, so span extraction also
  covers both rectangles.
- `Block.regions` is only written when a block has more than one rectangle;
  `models.block_regions()` falls back to `[page_idx, bbox]`, so old
  `document_ir.json` files still load.
- `reconstruct._insert_flowed_html` pours one PyMuPDF `Story` through every rectangle,
  under a single scale chosen as the largest in `[MIN_SCALE, 1.0]` that fits across
  all of them. `Story.place()` alone does not advance the story, so `_story_fits`
  calls `story.draw(None)` after each placement; without that every rectangle is
  measured against the start of the text and the search shrinks the block until it
  fits the first box alone, which is the original symptom.
- Single-rectangle blocks still go through `insert_htmlbox` unchanged.

## Batched Translation

- `translate` sends several requests, not one. Defaults: `--batch-blocks 8` and
  `--batch-chars 12000` (block JSON characters); either can be set to `0` to drop
  that limit, and setting both to `0` restores the old single-request behaviour.
- Why: a whole paper does not fit. `runs/1786024296.450` has 100 blocks / 136k block
  characters, which is ~37.4k prompt tokens against a 40,960-token context, leaving
  3,574 completion tokens. The model spent 2,785 characters on `<think>`, emitted 9
  blocks, and was cut off, so `translate` died on
  `json.decoder.JSONDecodeError: Unterminated string`. A request must leave room for
  a translation roughly as long as its input. The same payload now splits into 13
  requests of at most ~3.2k prompt tokens.
- `payload.split_payload` does the slicing. Each batch reuses the same system prompt,
  task, and schema. A block is never cut in half, even when it alone exceeds
  `--batch-chars`, and a batch never ends on a block flagged `continues_next`, since
  the model is told to transform such a run as one sentence.
- `llm_client.response_content` raises on `finish_reason == "length"` with the token
  counts, instead of letting a truncated completion fail as a JSON decode error.
- With more than one batch, `--raw-response-out` holds a JSON *list* of completions
  (one per batch) and is rewritten after each; a single batch still writes the bare
  completion object. `--dry-run-request-out` follows the same rule for requests.
- Block ids are unchanged by batching, so the merged `response.json` is identical in
  shape to the single-request one.
- `--no-think` sends `chat_template_kwargs={"enable_thinking": false}`, which vLLM
  passes to the Qwen3 chat template. Other servers may reject an unknown field.
- A failed batch aborts the run and nothing is written to `--output`. The completions
  that did arrive stay in `--raw-response-out`.
- `translate` warns (it does not fail) when a batch answers with fewer blocks than it
  was sent, naming the missing ids, and when a response had to be salvaged.

## Malformed JSON From The Model

- Seen in `runs/1786025241.103`, batch 2 of 13: Qwen3 repeated the closing `]}` after
  every block, so the content read as `{"blocks":[B1]},B2]},B3]}...` and `json.loads`
  failed with `Extra data`. Each block was individually valid and fully translated;
  only the envelope was wrong.
- `salvage_blocks` recovers these. Verified on that file: batch 2 yields all 8 blocks
  with `rich_text` intact. Recovery is reported as a `translate` warning, since it
  means the model is not following the output contract.
- The system prompt now also spells out that the envelope opens once and closes once,
  after the last block.
- Salvage never masks a truncated completion: a cut-off response has no complete block
  objects to recover, so the original decode error is re-raised, and
  `finish_reason == "length"` is caught before parsing anyway.

## System Prompt Strategy

The structural system prompt is in `patra_mineru.payload.SYSTEM_PROMPT`.

It should specify invariants, not a one-off task:

- Return strict JSON only.
- Keep every block id unchanged.
- Do not alter placeholders such as `<KEEP_EQ_0/>` and `<KEEP_TERM_1/>`.
- Preserve LaTeX, citations, model names, dataset names, and protected terms.
- Preserve bold/italic emphasis in `rich_text`.
- Do not split rich-text spans unless style changes.
- Transform every sentence; an unchanged or word-order-preserving sentence is invalid.
- Treat `continues_next` / `continues_previous` runs as one sentence, then redistribute
  the result across the same ids.

The actual operation, such as translating to Korean, belongs in `prepare --task`
or the web UI task field.

## Important Design Decisions

- Do not send full `document_ir.json` to the LLM. It contains bbox, source, and
  restoration metadata that the model does not need.
- Do send merged `rich_text` runs. The model cannot preserve emphasis it never
  sees.
- Keep `protection_map` only in `document_ir.json`; do not send it to the model.
- Placeholder tokens are generated automatically. Users only provide protected
  terms, for example `attention,Transformer,softmax`.
- Response parsing is deliberately tolerant. Local OSS models often wrap the
  answer or emit reasoning text, and the run is expensive enough that discarding
  a correct translation over an envelope mismatch is the wrong trade.
- If rich-text extraction is suspicious, prefer plain-text fallback over sending
  wrong neighboring text.
- The renderer overlays translated blocks on the original PDF. Equations, images,
  tables, and excluded sections are left untouched unless later code explicitly
  targets them.

## Known Caveats

- MinerU itself may not be installed in the environment. The project wraps the
  `mineru` CLI if available.
- The web UI currently loads PDF.js from a CDN. For offline use, vendor PDF.js
  into `src/patra_mineru/web/`.
- Block overlay is currently block-level, not full section-level aggregation.
  Section grouping can be added by combining all block bboxes with the same
  `section_id`.
- The web translation endpoint translates one selected block at a time. A future
  batch mode could translate all visible/selected blocks.
- Rich-text preservation after translation depends on model compliance. The
  renderer can fall back to plain translated text.
- There is no authentication in the local web app.
- `render_html_preview` still draws a block only in its first region, so the preview
  clips the tail of a page-split block. The PDF path is the accurate one.
- Model compliance degrades over long generations. In `runs/example2`, blocks
  late in the document (`b_00028`-`b_00032`, Evaluation and Conclusion) came back
  with a translated `text` but an untranslated `rich_text`. The renderer recovers
  these, but emphasis is lost for those blocks. Batched requests are now the
  default, which should keep this from recurring.
- Section headings rendered in small caps (`III. EVALUATION`) reach the model as
  a split rich_text span and come back echoed as `III. E\nVALUATION`, untranslated.
  Render warnings now flag this; no code works around it yet.
- Reasoning models such as Qwen3 spend a large share of the completion budget on
  the `<think>` block (8,419 completion tokens for `runs/example2`, roughly half
  of it reasoning). `--no-think` suppresses it on vLLM + Qwen3; `--json-mode` also
  suppresses it on servers that support `response_format`. Neither is on by default.
- Always pass `--raw-response-out`. It is written before parsing and rewritten after
  every batch, so a failure still leaves every completion that did arrive on disk.

## Useful Commands

Prepare existing MinerU output:

```bash
conda run -n patra patra-mineru prepare \
  --pdf example.pdf \
  --mineru-output runs/example/mineru \
  --ir-out runs/example/document_ir.json \
  --payload-out runs/example/llm_payload.json \
  --protected-terms attention,Transformer,softmax
```

Translate a full payload:

```bash
conda run -n patra patra-mineru translate \
  --payload runs/example/llm_payload.json \
  --output runs/example/response.json \
  --base-url http://localhost:8000/v1 \
  --model your-model \
  --temperature 0
```

Render a response:

```bash
conda run -n patra patra-mineru render \
  --pdf example.pdf \
  --ir runs/example/document_ir.json \
  --response runs/example/response.json \
  --output-pdf runs/example/rendered.pdf \
  --output-html runs/example/rendered.html
```

Run tests:

```bash
conda run -n patra python -m unittest discover -s patra_mineru_pipeline/tests
```

Run the web app:

```bash
conda run -n patra patra-mineru-web
```

## Verification So Far

The latest test run passed:

```text
Ran 32 tests
OK
```

The page-split fix was verified against the real `runs/1786021447.011` artifacts by
rebuilding the IR from the existing MinerU output and re-rendering with the existing
`response.json` into a scratch directory (no MinerU run, no model call): `b_00011`
gained a second region on page 1, its `rich_text` went from one plain fallback span to
four styled spans covering both rectangles, and the rendered page-1 rectangle now ends
with the Korean tail instead of the English source. The run directory itself was left
as it was.

`patra2` is the environment to use, matching `pipeline.sh`. It has MinerU 3.4.4;
`patra` has 3.4.1 and is stale. Both expose the same backend names.

The MinerU options and the local-server behaviour were read from the installed package
source rather than guessed. No MinerU run was performed while making these changes, so
the long-lived `mineru-api` path is verified only at the command-construction level:
`PATRA_MINERU_API_URL=http://127.0.0.1:50021` was confirmed to produce
`mineru ... -b vlm-http-client -l en -u http://127.0.0.1:50020 --api-url http://127.0.0.1:50021`.
Actually starting `mineru-api` and parsing through it is still unverified.

The local web app was started once and responded with HTTP 200 at
`http://127.0.0.1:8765/`.

A real run against a vLLM server (`Qwen/Qwen3-32B-AWQ`) produced
`runs/example2/raw_response.json`. All 20 blocks were translated correctly with
ids intact, but `translate` originally failed with
`ValueError: Model JSON must contain a blocks list.` because the model returned
`{"task": ..., "response_schema": {"blocks": [...]}}` after a `<think>` block.
The hardened `extract_json_content` now parses that same response into 20 blocks,
saved as `runs/example2/response.json`.

Rendering that response first produced English Evaluation and Conclusion sections
with zero warnings. After the `rich_text` trust check, the same response renders
Korean for those sections (page 1 hangul count 425 to 968).

Rebuilding the IR with heading preservation dropped translatable blocks from 20 to 15
and found one cross-block continuation (`b_00011` to `b_00019`, split across a column
boundary with a header, footer, and four charts between them). Re-rendering shows
justified paragraphs with no forced mid-sentence breaks and headings still in their
original small-caps typography. `render` now reports the one genuinely untranslated
sentence in `b_00023` instead of passing silently.

Note that `runs/example2/response.json` predates these payload changes, so it was
produced without the continuation hints and the stricter task text. Judge those two
features on a fresh `translate` run, not on this response.

## Suggested Next Steps

- Vendor PDF.js locally so the web UI works without internet.
- Add section-level overlays in addition to block-level overlays.
- Give the web UI the same batching as the CLI; `/translate-block` is still one
  block per request.
- Consider retrying a failed batch instead of aborting the whole run.
- Consider a `--strict` render mode that fails when any block comes back
  untransformed, instead of only warning.
- Add a rendered-PDF download button in the web UI.
- Add endpoint tests for `web_app.py` with FastAPI's test client.
- Add persistent job cleanup controls for `runs/web/`.
