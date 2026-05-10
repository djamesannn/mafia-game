# Mafia Game Neurosymbolic Backend Prototype

This repository contains a pure-Python backend prototype for an 8-12 player social deduction game inspired by Mafia.

## Architecture

The engine uses a strict **Fast Path / Slow Path** split:

- **Fast Path (CPU FSM):** deterministic rules, voting, state matrices, night phases, role logic, deaths, role reveals, and win checks.
- **Slow Path (GPU LLM):** optional `llama-cpp-python` adapter that converts chat text into bounded JSON deltas. The LLM never owns global state and never decides who dies.

## Main module

- `mafia_engine.py` defines bot state, matrices, deterministic event routing, role target selection, asynchronous chat batching, and a runnable demo round.

## Quick smoke test

```bash
python mafia_engine.py
```
