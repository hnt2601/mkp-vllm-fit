# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`vllm-fit` is a CLI tool that recommends and profiles optimal vLLM engine arguments for HuggingFace models based on available GPU hardware. It does **not** bundle vLLM — vLLM must be installed separately.

## Development Setup

```bash
uv venv --seed --python 3.10
source .venv/bin/activate
pip install -e .[dev]
```

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_estimator.py

# Run a single test function
pytest tests/test_estimator.py::test_estimate_parameters_basic

# Install package in editable mode
pip install -e .

# Build package
python -m build
```

## Architecture

The tool follows a two-phase approach: **static estimation** then optionally **dynamic profiling**.

### Module Responsibilities

- **[cli.py](src/vllm_fit/cli.py)** — Typer-based CLI with three commands: `recommend`, `profile`, and `serve`. Orchestrates calls to the other modules and formats output using Rich.

- **[estimator.py](src/vllm_fit/estimator.py)** — Pure-Python parameter estimation from model config. Computes `gpu_memory_utilization`, `max_model_len`, `tensor_parallel_size`, and `max_num_seqs` based on VRAM budget, model architecture (hidden size, layers, heads), and quantization type. No GPU required; no network calls.

- **[engine_tester.py](src/vllm_fit/engine_tester.py)** — Dynamic profiling via subprocess. Spawns isolated `multiprocessing.Process` workers (using `spawn` context) to test actual vLLM engine configurations without polluting the parent process. Uses binary search to maximize `max_num_seqs` and `max_model_len` after finding a baseline. Each worker suppresses vLLM stdout/stderr via `os.dup2`.

- **[registry.py](src/vllm_fit/registry.py)** — Fetches `config.json` from HuggingFace Hub. Handles GGUF models by stripping `-GGUF`/`-GGML` suffixes to find the base model config. Returns both the config dict and the `config_repo_id` (which may differ from `model_id` for GGUF models — used for `--hf-config-path` and `--tokenizer` flags).

- **[hardware.py](src/vllm_fit/hardware.py)** — GPU detection via `pynvml` (falls back to PyTorch). Returns VRAM per GPU as `Dict[int, float]` in GB.

### Key Data Flow

```
model_id → registry.get_model_config() → config dict
                                              ↓
hardware.get_vram_info() → total_vram → estimator.estimate_parameters()
                                              ↓
                                    initial_params dict
                                              ↓
                              engine_tester.profile_parameters()
                                              ↓
                                    optimized_params dict → vllm serve command
```

### GGUF Model Handling

GGUF models use `model_id` format `repo/name-GGUF:Q4_0`. The registry strips the `:Q4_0` suffix for the repo ID, then tries base model names (removing `-GGUF`) to find `config.json`. When a base config is found, `--hf-config-path` and `--tokenizer` flags point to the base repo in the generated command.

### Quantization Detection

`estimator.py` detects quantization from `config.json`'s `quantization_config` field, or from model ID patterns (`:Q4_0`, `UD-IQ`, `UD-Q`). Bytes-per-param is inferred from quant method/bits to estimate weight memory. AWQ on ≤4GB GPUs automatically enables `--enforce-eager` and adds ~0.2GB marlin kernel overhead.

### Profiling Strategy

`profile_parameters()` in `engine_tester.py`:
1. Tests initial estimated config
2. On failure: enables `enforce_eager`, retries
3. On continued failure: iteratively reduces `max_num_seqs` → `max_model_len` → `gpu_memory_utilization` to find a baseline
4. On baseline success: binary-searches upward on `max_num_seqs` then `max_model_len`

Each test spawns a subprocess (180s timeout) to load `vllm.LLM` — a real vLLM engine initialization that loads model weights.

## Version Management

Version is defined in [src/vllm_fit/__init__.py](src/vllm_fit/__init__.py) and read by hatchling. Releases use `python-semantic-release` triggered by conventional commits on `main`.
