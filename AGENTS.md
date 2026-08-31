# Agent Instructions

This repository contains a local MinerU + OpenAI-compatible PDF translation
prototype. Continue work carefully and keep the handoff files current.

## Required Ongoing Updates

- Update `commands.txt` whenever you run terminal commands.
  - Include the exact command when practical.
  - Add a short comment explaining intent.
  - Include failed commands if they influenced the work.
- Update `CONTEXT.md` after meaningful changes.
  - Record new files, behavior changes, known caveats, and verification results.
  - Keep it concise enough that a new agent can resume without rereading the
    whole conversation.

## Environment

- Use the `patra2` conda environment. This is the environment `pipeline.sh` activates.
  (`patra` still exists with an older MinerU; do not use it.)
- Prefer commands in this form:

```bash
conda run -n patra2 ...
```

- GPU use is allowed for MinerU. `parse --device` selects it: `auto` (default,
  MinerU picks CUDA when available), `cpu`, `cuda`, or `cuda:N`. Only `cpu` sets
  `CUDA_VISIBLE_DEVICES=""`. This reverses the earlier CPU-only rule.
- Translation still targets a separate OpenAI-compatible server; nothing in the
  translate path allocates a GPU locally.
- The workspace root is `/nfs/home/mapae/private/patra`.

## Project Layout

- Main package: `patra_mineru_pipeline/`
- CLI entrypoint: `patra-mineru`
- Local web app entrypoint: `patra-mineru-web`
- Test suite: `patra_mineru_pipeline/tests`
- Example PDF: `example.pdf`
- Example run artifacts: `runs/example/`

## Development Notes

- Use PDF.js for the browser viewer. Do not rely on Chrome's built-in PDF viewer
  because it does not allow reliable dynamic HTML overlays.
- MinerU in `patra2` is 3.4.4, whose CLI differs from 2.x. Backends are `pipeline`,
  `vlm-engine`, `hybrid-engine`, `vlm-http-client`, `hybrid-http-client`. Check
  the installed CLI before assuming 2.x flags.
- Do not edit `note.txt` or `pipeline.sh`. The user maintains those.
- Keep renderer-only metadata in `document_ir.json`; do not send it to the LLM.
- Send merged `rich_text` runs to the LLM so bold/italic can be preserved without
  token-heavy word-level spans.
- Keep `protection_map` out of LLM payloads. It is used locally to restore
  placeholders.
- Prefer plain-text fallback over sending wrong neighboring text when rich-text
  bbox extraction is uncertain.
- For OpenAI-compatible servers, use `/v1/chat/completions` style URLs. Enable
  `--json-mode` only if the server supports OpenAI `response_format`.

## Verification

After code changes, normally run:

```bash
conda run -n patra2 python -m unittest discover -s patra_mineru_pipeline/tests
```

For web app sanity:

```bash
conda run -n patra2 python -c "from patra_mineru.web_app import app; print(app.title)"
```

Then start locally when useful:

```bash
conda run -n patra2 patra-mineru-web
```
