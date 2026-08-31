# Patra MinerU Pipeline

Prototype pipeline for:

1. Taking MinerU `*_content_list.json` output.
2. Excluding sections such as References.
3. Building a rich-text JSON payload immediately before an OpenAI-compatible model call.
4. Rendering an assumed model response back onto the original PDF layout.

The current implementation does not call the OpenAI-compatible server. It writes the
payload you would send, and accepts a response-shaped JSON file for reconstruction.

## Install

```bash
conda run -n patra python -m pip install -e patra_mineru_pipeline
```

## MinerU Parse

If MinerU is installed in the same environment:

```bash
conda run -n patra patra-mineru parse \
  --pdf example.pdf \
  --output-dir runs/example/mineru \
  --backend pipeline --lang en --device auto
```

`--device auto` lets MinerU use the GPU when one is available. Use `--device cpu` to
hide GPUs, or `--device cuda:1` to pin one.

### Using long-lived MinerU servers

Without `--api-url`, every `parse` starts and stops its own throwaway `mineru-api`.
Run one yourself and point at it to skip that startup on each run:

```bash
# once, in its own shell
conda run -n patra2 mineru-api --host 127.0.0.1 --port 50021
```

Keep the bind on `127.0.0.1`. On `0.0.0.0` the service refuses `*-http-client`
backends and `server_url` with HTTP 400 as an SSRF guard, unless you also pass
`--allow-public-http-client`.

A VLM inference server is a separate thing, and the two compose:

```bash
# once, in its own shell
conda run -n patra2 mineru-openai-server --engine vllm --model <path> --port 50020

# per run: orchestration on the persistent api, inference on the VLM server
patra-mineru parse --pdf example.pdf --output-dir runs/example/mineru \
  --backend vlm-http-client --server-url http://127.0.0.1:50020 \
  --api-url http://127.0.0.1:50021 --lang en
```

To avoid editing every call site, export the default instead:

```bash
export PATRA_MINERU_API_URL=http://127.0.0.1:50021
```

## Prepare LLM Payload

```bash
conda run -n patra patra-mineru prepare \
  --pdf example.pdf \
  --mineru-output runs/example/mineru \
  --ir-out runs/example/document_ir.json \
  --payload-out runs/example/llm_payload.json \
  --protected-terms attention,Transformer,softmax
```

The payload contains:

- block ids and page positions,
- protected placeholders for equations and terms,
- `rich_text` spans with bold/italic/font-size when PyMuPDF can recover them,
- `continues_next` / `continues_previous` flags where the page layout split one
  sentence across two boxes.

Section headings stay in the source language and are not sent to the model. This
matches the usual convention for Korean papers and keeps the renderer from covering
small-caps headings with a sans-serif approximation. Pass `--translate-headings` to
translate them anyway.

Send `llm_payload.json` to your OpenAI-compatible server with the included
`system_prompt`, then save the model response as JSON:

```json
{
  "blocks": [
    {
      "id": "b_00001",
      "text": "translated text with <KEEP_TERM_0/> preserved",
      "rich_text": [
        {"text": "translated text", "bold": false, "italic": false}
      ]
    }
  ]
}
```

## Translate With OpenAI-Compatible Server

Preview the exact chat-completions request without calling the server:

```bash
conda run -n patra patra-mineru translate \
  --payload runs/example/llm_payload.json \
  --output runs/example/response.json \
  --base-url http://localhost:8000/v1 \
  --model your-model \
  --dry-run-request-out runs/example/chat_request.json
```

Call the server and save only the parsed model JSON response:

```bash
conda run -n patra patra-mineru translate \
  --payload runs/example/llm_payload.json \
  --output runs/example/response.json \
  --base-url http://localhost:8000/v1 \
  --model your-model \
  --temperature 0 \
  --raw-response-out runs/example/raw_response.json
```

Use `--json-mode` only when the server supports OpenAI `response_format`. For
local compatible servers, the API key defaults to `OPENAI_API_KEY`, then
`dummy`; pass `--api-key` if your server requires a real key.

### Batching

A whole paper does not fit in one request: the payload and the translation of it are
about the same size, so both have to share the server's context window. A 100-block
paper is roughly 37k prompt tokens, which leaves nothing to answer with in a 41k
window, and the completion is cut off mid-JSON.

`translate` therefore sends several requests and merges the responses. Defaults are
`--batch-blocks 8` and `--batch-chars 12000` (block JSON characters per request);
either can be `0` to drop that limit, and both `0` restores a single request. Block
ids are unchanged, so `response.json` looks the same either way.

```bash
conda run -n patra2 patra-mineru translate \
  --payload runs/example/llm_payload.json \
  --output runs/example/response.json \
  --base-url http://localhost:8000/v1 \
  --model your-model \
  --temperature 0 \
  --batch-blocks 6 \
  --no-think \
  --raw-response-out runs/example/raw_response.json
```

`--no-think` sends `chat_template_kwargs={"enable_thinking": false}`, which vLLM
passes to the Qwen3 chat template; reasoning otherwise takes a large share of the
completion budget. Other servers may reject the field.

With more than one batch, `--raw-response-out` and `--dry-run-request-out` hold a JSON
list, one entry per batch. The raw file is rewritten after every batch, so a failure
partway through still leaves the completions that did arrive. If the server stops a
completion at the token limit, `translate` now says so with the token counts instead
of failing on a truncated JSON string.

Local models do not always close the JSON envelope correctly; one seen failure repeats
`]}` after every block, which decodes as `Extra data`. `translate` recovers the blocks
from such a response and warns that it had to. It also warns, without failing, when a
batch answers with fewer blocks than it was sent.

The system prompt is generated in `patra_mineru.payload.SYSTEM_PROMPT`. It should
define structural invariants: return strict JSON, keep block ids, preserve
placeholders, preserve equations and protected technical terms, and keep
bold/italic rich-text emphasis. The concrete instruction, such as translating to
Korean, belongs in `prepare --task ...` so you can reuse the same structural
system prompt for translation, polishing, or summarization.

## Render Assumed Response

For a dry run without a model:

```bash
conda run -n patra patra-mineru mock-response \
  --payload runs/example/llm_payload.json \
  --output runs/example/mock_response.json
```

Then render:

```bash
conda run -n patra patra-mineru render \
  --pdf example.pdf \
  --ir runs/example/document_ir.json \
  --response runs/example/mock_response.json \
  --output-pdf runs/example/rendered.pdf \
  --output-html runs/example/rendered.html
```

The PDF renderer overlays transformed text blocks on top of the original PDF.
Excluded sections, equations, tables, and images are left untouched.

A paragraph interrupted by a column or page break occupies more than one rectangle.
MinerU reports it as the full text on the first rectangle plus an empty block for the
leftover one, so `prepare` reattaches the leftover as an extra `regions` entry on the
owning block (`continuation_of` marks the empty one), and `render` flows the
transformed text through every rectangle at one shared scale. Re-run `prepare` to pick
this up on an IR built by an older version; the response JSON stays valid, since block
ids do not change.

## Rich Text Note

Yes: rich text should be part of the model payload if you want the translated
result to keep emphasis. The model cannot preserve bold/italic it never sees.
This project sends both plain protected text and span-level rich text. The
response schema also allows `rich_text`, so the renderer can reinsert styled spans.
