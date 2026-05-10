# Mafia Game Neurosymbolic Backend Prototype

This repository contains a pure-Python backend prototype for an 8-12 player social deduction game inspired by Mafia.

## Architecture

The engine uses a strict **Fast Path / Slow Path** split:

- **Fast Path (CPU FSM):** deterministic rules, voting, state matrices, night phases, role logic, deaths, role reveals, and win checks.
- **Slow Path (GPU LLM):** `llama-cpp-python` with local Qwen JSON-schema inference converts chat text into bounded psychological deltas. The LLM never owns global state and never decides who dies.

## Main modules

- `mafia_engine.py` defines bot state, matrices, deterministic event routing, role target selection, asynchronous chat batching, and the `LlamaJSONEvaluator`.
- `play_cli.py` runs a terminal game loop with Player 1 as the human user.

## Local model requirement

Place the Qwen GGUF model at:

```bash
./qwen2.5-3b-instruct-q4_k_m.gguf
```

Install `llama-cpp-python` with CUDA support for GPU acceleration, then run:

```bash
python play_cli.py
```

## Development checks

```bash
python -m py_compile mafia_engine.py play_cli.py tests/test_mafia_engine.py
python -m unittest discover -s tests -v
```
