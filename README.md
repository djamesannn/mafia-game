# Mafia Game Neurosymbolic Backend Prototype

This repository contains a pure-Python backend prototype for an 8-player social deduction game inspired by Mafia, with a real-time FastAPI/WebSocket UI.

## Architecture

The engine uses a strict **Fast Path / Slow Path** split:

- **Fast Path (CPU FSM):** deterministic rules, voting, two-part day phases, channel-aware matrix routing, night phases, role logic, deaths, role reveals, and win checks.
- **Slow Path (GPU LLM):** `llama-cpp-python` with local Qwen JSON-schema inference converts chat text into bounded psychological deltas. The LLM never owns global state and never decides who dies, who is nominated, or who is exiled.

## Main modules

- `mafia_engine.py` defines bot state, persistent profile loading, matrices, deterministic event routing, role target selection, asynchronous channel chat batching, trial-phase voting, and the `LlamaJSONEvaluator`.
- `server.py` runs the FastAPI/WebSocket game server with non-blocking timers, multi-room session management, human-like bot action delays, phase suspense buffers, night intent coordination, general/mafia chat routing, and broadcast state snapshots.
- `templates/index.html` is the vanilla JS/CSS browser UI with production-style top bar, central system banners/action panel, right player sidebar, chat tabs, and night-intent badges.
- `profiles.json` contains persistent Russian/English bot identities, avatar bases, genders, and psychotypes sampled into each game.
- `play_cli.py` keeps a simple terminal loop for quick local experiments.

## Local model requirement

The target deployment expects 4GB VRAM and 16GB system RAM. Place the Qwen GGUF model at:

```bash
./qwen2.5-3b-instruct-q4_k_m.gguf
```

`LlamaJSONEvaluator` loads it with `n_ctx=4096` and `n_gpu_layers=-1` for CUDA offload.

## Run the Web UI

Install dependencies (build `llama-cpp-python` with CUDA support for the target GPU), then run:

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000> for the default `lobby` room, or pass `?room=my-room` to create/join another independent room. Player 1 is the human player in the prototype.

## Development checks

```bash
python -m py_compile mafia_engine.py server.py play_cli.py tests/test_mafia_engine.py
python -m unittest discover -s tests -v
python mafia_engine.py
```
