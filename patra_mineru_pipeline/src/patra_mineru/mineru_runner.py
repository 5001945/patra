from __future__ import annotations

import os
import subprocess
from pathlib import Path

# MinerU 3.x backends. The *-http-client ones offload inference to an already
# running VLM server and require --url; the others run locally.
LOCAL_BACKENDS = ("pipeline", "vlm-engine", "hybrid-engine")
HTTP_CLIENT_BACKENDS = ("vlm-http-client", "hybrid-http-client")
BACKENDS = LOCAL_BACKENDS + HTTP_CLIENT_BACKENDS

# MinerU ignores --method outside these backends.
METHOD_AWARE_BACKENDS = ("pipeline", "hybrid-engine", "hybrid-http-client")

DEFAULT_BACKEND = os.environ.get("PATRA_MINERU_BACKEND", "pipeline")
DEFAULT_DEVICE = os.environ.get("PATRA_MINERU_DEVICE", "auto")
DEFAULT_LANG = os.environ.get("PATRA_MINERU_LANG") or None
DEFAULT_SERVER_URL = os.environ.get("PATRA_MINERU_SERVER_URL") or None
DEFAULT_API_URL = os.environ.get("PATRA_MINERU_API_URL") or None


def build_mineru_command(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    method: str = "auto",
    start: int | None = None,
    end: int | None = None,
    lang: str | None = DEFAULT_LANG,
    effort: str | None = None,
    server_url: str | None = DEFAULT_SERVER_URL,
    api_url: str | None = DEFAULT_API_URL,
) -> list[str]:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown MinerU backend {backend!r}. Expected one of: {', '.join(BACKENDS)}")
    if backend in HTTP_CLIENT_BACKENDS and not server_url:
        raise ValueError(
            f"Backend {backend!r} sends inference to a running VLM server, so --server-url is required, "
            "for example http://127.0.0.1:30000. Start one with `mineru-vllm-server --port 30000`."
        )

    command = ["mineru", "-p", str(input_path), "-o", str(output_dir), "-b", backend]
    if backend in METHOD_AWARE_BACKENDS:
        command += ["-m", method]
    if lang:
        command += ["-l", lang]
    if effort and backend.startswith("hybrid"):
        command += ["--effort", effort]
    if server_url:
        command += ["-u", server_url]
    if api_url:
        command += ["--api-url", api_url]
    if start is not None:
        command += ["--start", str(start)]
    if end is not None:
        command += ["--end", str(end)]
    return command


def build_mineru_env(device: str = DEFAULT_DEVICE) -> dict[str, str]:
    """Environment for the MinerU subprocess.

    `auto` leaves device selection to MinerU, which picks CUDA when it is available.
    `cpu` hides the GPUs. Anything else is passed through as MINERU_DEVICE_MODE, so
    values such as `cuda` or `cuda:1` select a specific device.
    """
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["MINERU_DEVICE_MODE"] = "cpu"
    elif device and device != "auto":
        env["MINERU_DEVICE_MODE"] = device
    return env


def run_mineru(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    method: str = "auto",
    start: int | None = None,
    end: int | None = None,
    device: str = DEFAULT_DEVICE,
    lang: str | None = DEFAULT_LANG,
    effort: str | None = None,
    server_url: str | None = DEFAULT_SERVER_URL,
    api_url: str | None = DEFAULT_API_URL,
) -> None:
    """Run the MinerU CLI.

    With `api_url` the whole parse is handed to a running `mineru-api` service and no
    local model runs. With a `*-http-client` backend plus `server_url`, layout work
    stays local while VLM inference goes to a running MinerU VLM server.
    """
    command = build_mineru_command(
        input_path,
        output_dir,
        backend=backend,
        method=method,
        start=start,
        end=end,
        lang=lang,
        effort=effort,
        server_url=server_url,
        api_url=api_url,
    )
    subprocess.run(command, check=True, env=build_mineru_env(device))
