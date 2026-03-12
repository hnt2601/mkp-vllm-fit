import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.errors import EntryNotFoundError


def is_gguf_model(model_id: str) -> bool:
    """Check if model is a GGUF model by looking for GGUF suffix or quantization patterns."""
    repo_id = model_id.split(":")[0].upper()
    model_upper = model_id.upper()
    return "-GGUF" in repo_id or "_GGUF" in repo_id or ":Q" in model_upper


def extract_repo_id(model_id: str) -> str:
    return model_id.split(":")[0]


def try_extract_base_model(repo_id: str) -> list:
    base_candidates = [
        repo_id,
        repo_id.replace("-GGUF", ""),
        repo_id.replace("-GGML", ""),
        repo_id.replace("-gguf", ""),
        repo_id.replace("-ggml", ""),
        repo_id.replace("_GGUF", ""),
        repo_id.replace("_GGML", ""),
    ]
    return base_candidates


def try_find_non_gguf_base(repo_id: str) -> Optional[str]:
    """Try to find a non-GGUF base repo for tokenizer files."""
    base_candidates = [
        repo_id.replace("-GGUF", ""),
        repo_id.replace("-GGML", ""),
        repo_id.replace("-gguf", ""),
        repo_id.replace("-ggml", ""),
        repo_id.replace("_GGUF", ""),
        repo_id.replace("_GGML", ""),
    ]

    for candidate in base_candidates:
        if candidate != repo_id:
            try:
                hf_hub_download(
                    repo_id=candidate,
                    filename="config.json",
                    force_download=False,
                )
                return candidate
            except EntryNotFoundError:
                continue

    return None


def gguf_repo_has_config(repo_id: str) -> bool:
    """Check if the GGUF repo has its own config.json."""
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            force_download=False,
        )
        return True
    except EntryNotFoundError:
        return False


def _is_local_path(model_id: str) -> bool:
    """Return True if *model_id* refers to a local filesystem path.

    A bare quantization suffix (e.g. ``/path/to/repo:Q4_K_M``) is stripped before
    the existence check so that GGUF-style local paths are handled correctly.

    Paths that start with ``/``, ``./``, or ``../`` are unconditionally treated as
    local regardless of whether they are accessible at detection time. This prevents
    inaccessible-but-valid absolute paths from falling through to HuggingFace Hub
    code and raising an ``HFValidationError``.
    """
    base = model_id.split(":")[0]
    if base.startswith("/") or base.startswith("./") or base.startswith("../"):
        return True
    p = Path(base)
    return p.exists() and p.is_dir()


def get_model_config(model_id: str) -> Tuple[Dict[str, Any], str]:
    # --- Local filesystem path ---------------------------------------------------
    if _is_local_path(model_id):
        base = model_id.split(":")[0]
        local_dir = Path(base).resolve()
        if not local_dir.exists():
            raise ValueError(
                f"Local path '{local_dir}' does not exist. "
                f"Please verify the path and try again."
            )
        if not local_dir.is_dir():
            raise ValueError(
                f"Local path '{local_dir}' is not a directory. "
                f"Please provide the model directory path."
            )
        config_file = local_dir / "config.json"
        if not config_file.exists():
            raise ValueError(
                f"Local path '{local_dir}' does not contain a 'config.json' file. "
                f"Ensure the directory is a valid HuggingFace model directory."
            )
        with open(config_file, "r") as f:
            config = json.load(f)
        return config, str(local_dir)
    # -----------------------------------------------------------------------------

    repo_id = extract_repo_id(model_id)

    if is_gguf_model(model_id):
        has_config = gguf_repo_has_config(repo_id)

        for candidate in try_extract_base_model(repo_id):
            try:
                config_path = hf_hub_download(
                    repo_id=candidate,
                    filename="config.json",
                    force_download=False,
                )
                with open(config_path, "r") as f:
                    config = json.load(f)

                    if has_config and candidate == repo_id:
                        return config, repo_id

                    if not has_config and candidate != repo_id:
                        return config, candidate
            except EntryNotFoundError:
                continue

    for candidate in try_extract_base_model(repo_id):
        try:
            config_path = hf_hub_download(
                repo_id=candidate,
                filename="config.json",
                force_download=False,
            )
            with open(config_path, "r") as f:
                config = json.load(f)
                return config, candidate
        except EntryNotFoundError:
            continue

    raise ValueError(
        f"Could not find config.json for model '{model_id}'. "
        f"Tried: {', '.join(try_extract_base_model(repo_id))}.\n"
        f"The model may not be compatible or may be a GGUF-only format without standard config.\n"
        f"Try specifying the base model directly (e.g., 'Qwen/Qwen1.5-1.8B-Chat' instead of 'Qwen/Qwen1.5-1.8B-Chat-GGUF')."
    )
