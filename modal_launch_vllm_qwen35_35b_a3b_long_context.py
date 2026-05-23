
# Example:
# MODAL_VLLM_MODEL_NAME=Qwen/Qwen3.5-35B-A3B \
# MODAL_VLLM_APP_NAME=lateral-vllm-qwen3-5-35b-a3b-long-context-262k \
# MODAL_VLLM_WEB_LABEL=lateral-vllm-qwen3-5-35b-a3b-long-context-262k \
# MODAL_VLLM_REQUIRES_PROXY_AUTH=0 \
# MODAL_VLLM_MIN_CONTAINERS=1 \
# MODAL_VLLM_MAX_CONTAINERS=1 \
# MODAL_VLLM_N_GPU=8 \
# MODAL_VLLM_TENSOR_PARALLEL_SIZE=8 \
# MODAL_VLLM_DATA_PARALLEL_SIZE=1 \
# MODAL_VLLM_MAX_MODEL_LEN=262144 \
# modal deploy --name lateral-vllm-qwen3-5-35b-a3b-long-context-262k /iris/u/asap7772/asmr_private/modal_launch_vllm_qwen35_35b_a3b_long_context.py


# Then, to call it:
# curl -X POST https://<your-app>--<your-endpoint>.modal.run/v1/chat/completions \
#      -H "Content-Type: application/json" \
#      -d '{"model": "llm", "messages": [{"role": "user", "content": "Hello!"}]}'

"""Launch Qwen3.5-35B-A3B on Modal with a 262k-token vLLM context."""

import json
import os
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request

import modal

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


def _get_env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw_value!r}")


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _get_env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return float(raw_value)


def _get_env_csv(name: str, default: list[str]) -> list[str]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return list(default)

    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one comma-separated value")
    return list(dict.fromkeys(values))


def _get_env_shell_args(name: str) -> list[str]:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return []
    return shlex.split(raw_value)


DEFAULT_APP_NAME = "lateral-vllm-qwen3-5-35b-a3b-long-context-262k"

APP_NAME = os.environ.get("MODAL_VLLM_APP_NAME", DEFAULT_APP_NAME)
MODEL_NAME = os.environ.get("MODAL_VLLM_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B")
SERVED_MODEL_NAMES = _get_env_csv("MODAL_VLLM_SERVED_MODEL_NAMES", [MODEL_NAME, "llm"])

STARTUP_TIMEOUT = _get_env_int("MODAL_VLLM_STARTUP_TIMEOUT", 30 * MINUTES)
SESSION_TIMEOUT = _get_env_int("MODAL_VLLM_SESSION_TIMEOUT", 24 * HOURS)
VLLM_PORT = _get_env_int("MODAL_VLLM_PORT", 8000)
VLLM_ENGINE_TIMEOUT = _get_env_int("MODAL_VLLM_ENGINE_TIMEOUT", STARTUP_TIMEOUT)

WEB_ENDPOINT_LABEL = os.environ.get(
    "MODAL_VLLM_WEB_LABEL",
    APP_NAME,
)
WEB_MIN_CONTAINERS = _get_env_int("MODAL_VLLM_MIN_CONTAINERS", 1)
WEB_MAX_CONTAINERS = _get_env_int("MODAL_VLLM_MAX_CONTAINERS", 1)
WEB_SCALEDOWN_WINDOW = _get_env_int("MODAL_VLLM_SCALEDOWN_WINDOW", 20 * MINUTES)
WEB_REQUIRES_PROXY_AUTH = _get_env_bool("MODAL_VLLM_REQUIRES_PROXY_AUTH", False)
CONCURRENT_MAX_INPUTS = _get_env_int("MODAL_VLLM_CONCURRENT_MAX_INPUTS", 250)

GPU_TYPE = os.environ.get("MODAL_VLLM_GPU_TYPE", "H100")
N_GPU = _get_env_int("MODAL_VLLM_N_GPU", 8)
TENSOR_PARALLEL_SIZE = _get_env_int("MODAL_VLLM_TENSOR_PARALLEL_SIZE", 8)
DATA_PARALLEL_SIZE = _get_env_int(
    "MODAL_VLLM_DATA_PARALLEL_SIZE", N_GPU // TENSOR_PARALLEL_SIZE
)
D_TYPE = os.environ.get("MODAL_VLLM_DTYPE", "bfloat16")
GPU_MEMORY_UTILIZATION = _get_env_float("MODAL_VLLM_GPU_MEMORY_UTILIZATION", 0.92)

MAX_MODEL_LEN = _get_env_int("MODAL_VLLM_MAX_MODEL_LEN", 262144)
MAX_NUM_BATCHED_TOKENS = _get_env_int("MODAL_VLLM_MAX_NUM_BATCHED_TOKENS", 32768)
MAX_NUM_SEQS = _get_env_int("MODAL_VLLM_MAX_NUM_SEQS", 128)

ENABLE_PREFIX_CACHING = _get_env_bool("MODAL_VLLM_ENABLE_PREFIX_CACHING", True)
ENABLE_CHUNKED_PREFILL = _get_env_bool("MODAL_VLLM_ENABLE_CHUNKED_PREFILL", True)
ENABLE_ASYNC_SCHEDULING = _get_env_bool("MODAL_VLLM_ENABLE_ASYNC_SCHEDULING", True)
ENFORCE_EAGER = _get_env_bool("MODAL_VLLM_ENFORCE_EAGER", False)
VLLM_EXTRA_ARGS = _get_env_shell_args("MODAL_VLLM_EXTRA_ARGS")


def _config_env() -> dict[str, str]:
    return {
        "MODAL_VLLM_APP_NAME": APP_NAME,
        "MODAL_VLLM_MODEL_NAME": MODEL_NAME,
        "MODAL_VLLM_SERVED_MODEL_NAMES": ",".join(SERVED_MODEL_NAMES),
        "MODAL_VLLM_STARTUP_TIMEOUT": str(STARTUP_TIMEOUT),
        "MODAL_VLLM_SESSION_TIMEOUT": str(SESSION_TIMEOUT),
        "MODAL_VLLM_PORT": str(VLLM_PORT),
        "MODAL_VLLM_ENGINE_TIMEOUT": str(VLLM_ENGINE_TIMEOUT),
        "MODAL_VLLM_WEB_LABEL": WEB_ENDPOINT_LABEL,
        "MODAL_VLLM_MIN_CONTAINERS": str(WEB_MIN_CONTAINERS),
        "MODAL_VLLM_MAX_CONTAINERS": str(WEB_MAX_CONTAINERS),
        "MODAL_VLLM_SCALEDOWN_WINDOW": str(WEB_SCALEDOWN_WINDOW),
        "MODAL_VLLM_REQUIRES_PROXY_AUTH": "1" if WEB_REQUIRES_PROXY_AUTH else "0",
        "MODAL_VLLM_CONCURRENT_MAX_INPUTS": str(CONCURRENT_MAX_INPUTS),
        "MODAL_VLLM_GPU_TYPE": GPU_TYPE,
        "MODAL_VLLM_N_GPU": str(N_GPU),
        "MODAL_VLLM_TENSOR_PARALLEL_SIZE": str(TENSOR_PARALLEL_SIZE),
        "MODAL_VLLM_DATA_PARALLEL_SIZE": str(DATA_PARALLEL_SIZE),
        "MODAL_VLLM_DTYPE": D_TYPE,
        "MODAL_VLLM_GPU_MEMORY_UTILIZATION": str(GPU_MEMORY_UTILIZATION),
        "MODAL_VLLM_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "MODAL_VLLM_MAX_NUM_BATCHED_TOKENS": str(MAX_NUM_BATCHED_TOKENS),
        "MODAL_VLLM_MAX_NUM_SEQS": str(MAX_NUM_SEQS),
        "MODAL_VLLM_ENABLE_PREFIX_CACHING": "1" if ENABLE_PREFIX_CACHING else "0",
        "MODAL_VLLM_ENABLE_CHUNKED_PREFILL": "1" if ENABLE_CHUNKED_PREFILL else "0",
        "MODAL_VLLM_ENABLE_ASYNC_SCHEDULING": "1" if ENABLE_ASYNC_SCHEDULING else "0",
        "MODAL_VLLM_ENFORCE_EAGER": "1" if ENFORCE_EAGER else "0",
        "MODAL_VLLM_EXTRA_ARGS": " ".join(shlex.quote(arg) for arg in VLLM_EXTRA_ARGS),
    }


vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.19.0",
    )
    .env(
        {
            **_config_env(),
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster model transfers
            # Large Qwen3-Next warmup/compile can exceed vLLM's default 60s/300s
            # worker timeouts, producing shm_broadcast warnings and RPC failures.
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": str(VLLM_ENGINE_TIMEOUT),
            "VLLM_ENGINE_READY_TIMEOUT_S": str(VLLM_ENGINE_TIMEOUT),
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": str(VLLM_ENGINE_TIMEOUT),
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
HF_TOKEN_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN")


def _get_local_hf_token() -> str | None:
    for key in HF_TOKEN_ENV_KEYS:
        token = os.environ.get(key)
        if token:
            return token
    return None


local_hf_token = _get_local_hf_token()
if local_hf_token:
    hf_token_secret = modal.Secret.from_dict(
        {key: local_hf_token for key in HF_TOKEN_ENV_KEYS}
    )
else:
    hf_token_secret = modal.Secret.from_name(
        os.environ.get("MODAL_HUGGINGFACE_SECRET_NAME", "huggingface-secret"),
        required_keys=["HF_TOKEN"],
    )


app = modal.App(APP_NAME)


def _build_vllm_cmd() -> list[str]:
    if DATA_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE != N_GPU:
        raise ValueError(
            "Expected DATA_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE to match N_GPU, "
            f"got {DATA_PARALLEL_SIZE} * {TENSOR_PARALLEL_SIZE} != {N_GPU}"
        )

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        *SERVED_MODEL_NAMES,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--data-parallel-size",
        str(DATA_PARALLEL_SIZE),
        "--dtype",
        D_TYPE,
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--uvicorn-log-level",
        "info",
        "--disable-uvicorn-access-log",
        "--aggregate-engine-logging",
    ]

    if ENABLE_PREFIX_CACHING:
        cmd.append("--enable-prefix-caching")
    if ENABLE_CHUNKED_PREFILL:
        cmd.append("--enable-chunked-prefill")
    if ENABLE_ASYNC_SCHEDULING:
        cmd.append("--async-scheduling")
    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")
    cmd.extend(VLLM_EXTRA_ARGS)
    return cmd


def _check_process(process: subprocess.Popen) -> None:
    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(f"vLLM exited before becoming ready with status code {return_code}")


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Timed out waiting for vLLM to become ready")
    return remaining


def _wait_for_port(host: str, port: int, timeout: float, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _check_process(process)
        try:
            with socket.create_connection((host, port), timeout=5.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for vLLM to listen on {host}:{port}")


def _wait_for_health(base_url: str, timeout: float, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"{base_url}/health"
    last_error = ""

    while time.monotonic() < deadline:
        _check_process(process)
        try:
            with urllib.request.urlopen(health_url, timeout=5.0) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.5)

    raise TimeoutError(f"Timed out waiting for vLLM health endpoint {health_url}: {last_error}")


def _get_served_models(base_url: str, timeout: float) -> list[str]:
    models_url = f"{base_url}/v1/models"
    with urllib.request.urlopen(models_url, timeout=max(1.0, timeout)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    served_models = [str(model["id"]) for model in payload.get("data", []) if "id" in model]
    if not served_models:
        raise RuntimeError(f"No served models were returned by {models_url}: {payload}")
    return served_models


def _smoke_test_chat(base_url: str, model_name: str, timeout: float) -> None:
    request_url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0.0,
        "max_tokens": 8,
    }
    request = urllib.request.Request(
        request_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=max(1.0, timeout)) as response:
            if response.status != 200:
                raise RuntimeError(f"Unexpected HTTP {response.status} from {request_url}")
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if "chat template" in error_body.lower():
            raise RuntimeError(
                "vLLM became healthy, but /v1/chat/completions failed because the model has no chat template. "
                "Either provide --chat-template or use /v1/completions instead. "
                f"Response body: {error_body[:512]}"
            ) from exc
        raise RuntimeError(
            f"Smoke test against {request_url} failed with HTTP {exc.code}: {error_body[:512]}"
        ) from exc
    except TimeoutError as exc:
        raise TimeoutError(
            f"Smoke test against {request_url} did not finish within {timeout:.0f}s"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Smoke test against {request_url} failed: {exc}") from exc

    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Smoke test against {request_url} returned no choices: {response_payload}")


def _ensure_hf_token_env() -> dict[str, str]:
    env = os.environ.copy()
    token = next((env.get(key) for key in HF_TOKEN_ENV_KEYS if env.get(key)), None)
    if not token:
        raise RuntimeError(
            "HF_TOKEN was not available inside the Modal container. "
            "Export HF_TOKEN before deploying, or create a Modal secret "
            "named `huggingface-secret` with an `HF_TOKEN` key."
        )

    for key in HF_TOKEN_ENV_KEYS:
        env[key] = token
        os.environ[key] = token
    return env


def _start_vllm_process() -> subprocess.Popen:
    env = _ensure_hf_token_env()
    cmd = _build_vllm_cmd()
    print("HF token is available in the Modal container.")
    print(*cmd)
    return subprocess.Popen(cmd, env=env)


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    min_containers=WEB_MIN_CONTAINERS,
    max_containers=WEB_MAX_CONTAINERS,
    scaledown_window=WEB_SCALEDOWN_WINDOW,
    timeout=SESSION_TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[hf_token_secret],
)
@modal.concurrent(max_inputs=CONCURRENT_MAX_INPUTS)
@modal.web_server(
    port=VLLM_PORT,
    startup_timeout=STARTUP_TIMEOUT,
    label=WEB_ENDPOINT_LABEL,
    requires_proxy_auth=WEB_REQUIRES_PROXY_AUTH,
)
def serve_web():
    process = _start_vllm_process()
    ready_deadline = time.monotonic() + STARTUP_TIMEOUT
    local_base_url = f"http://127.0.0.1:{VLLM_PORT}"

    _wait_for_port("127.0.0.1", VLLM_PORT, _remaining_timeout(ready_deadline), process)
    _wait_for_health(local_base_url, _remaining_timeout(ready_deadline), process)
    served_models = _get_served_models(local_base_url, _remaining_timeout(ready_deadline))
    smoke_test_model = "llm" if "llm" in served_models else served_models[0]
    _smoke_test_chat(local_base_url, smoke_test_model, _remaining_timeout(ready_deadline))
    print(f"vLLM web endpoint ready on port {VLLM_PORT}")
