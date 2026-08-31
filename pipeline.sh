#!/usr/bin/env bash

INPUT_PDF=$1
if [[ -z "$INPUT_PDF" ]]; then
  echo "Usage: $0 <input_pdf>"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate patra

UNIX_TIMESTAMP_INCLUDING_MS=$(date +%s.%3N)
OUTPUT_DIR="runs/$UNIX_TIMESTAMP_INCLUDING_MS/$INPUT_PDF"
MINERU_OUTPUT_DIR="$OUTPUT_DIR/mineru"

# 다른 창에서 mineru-api --host 127.0.0.1 --port 50019

# 다른 창에서 mineru-openai-server --engine vllm --model opendatalab/MinerU2.5-Pro-2605-1.2B --served-model-name "mineru" --port 50020 --gpu-memory-utilization 0.35
patra-mineru parse --pdf "$INPUT_PDF" --output-dir "$MINERU_OUTPUT_DIR" --backend hybrid-http-client --api-url http://127.0.0.1:50019 --server-url http://127.0.0.1:50020 --lang en

patra-mineru prepare --pdf "$INPUT_PDF" --mineru-output "$MINERU_OUTPUT_DIR" --ir-out "$OUTPUT_DIR/document_ir.json" --payload-out "$OUTPUT_DIR/llm_payload.json" --protected-terms attention,Transformer,softmax

# 다른 창에서 vllm serve Qwen/Qwen3-32B-AWQ --served-model-name "Qwen/Qwen3-32B-AWQ" --port 50021 --gpu-memory-utilization 0.50 --reasoning-parser deepseek_r1 --structured-outputs-config.backend xgrammar --structured-outputs-config.disable_any_whitespace true
patra-mineru translate --payload "$OUTPUT_DIR/llm_payload.json" --output "$OUTPUT_DIR/response.json" --base-url http://localhost:50021/v1 --model Qwen/Qwen3.8-27B --temperature 0 --json-mode --raw-response-out "$OUTPUT_DIR/raw_response.json"

patra-mineru render --pdf "$INPUT_PDF" --ir "$OUTPUT_DIR/document_ir.json" --response "$OUTPUT_DIR/response.json" --output-pdf "$OUTPUT_DIR/rendered.pdf" --output-html "$OUTPUT_DIR/rendered.html" --debug-boxes


# 이 스크립트 돌리기 전 요약:
# mineru-api --host 127.0.0.1 --port 50019
# mineru-openai-server --engine vllm --model opendatalab/MinerU2.5-Pro-2605-1.2B --served-model-name "mineru" --port 50020 --gpu-memory-utilization 0.1
# vllm serve Qwen/Qwen3.8-27B --port 50021 --gpu-memory-utilization 0.6 --reasoning-parser deepseek_r1 --structured-outputs-config.backend xgrammar --structured-outputs-config.disable_any_whitespace true
# 또는 OLLAMA_HOST=127.0.0.1:50021 OLLAMA_GPU_OVERHEAD=68719476736 ollama serve
