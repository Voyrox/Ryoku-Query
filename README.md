<div align="center">

<img src="https://raw.githubusercontent.com/neur0map/ryoku-arch/main/ryoku/assets/brand/logo-mark.png" alt="Ryoku" width="160" />

# Nero

**The Ryoku support bot for Discord** &middot; *reviewed answers, safety prompts, and source-backed replies.*

Nero is the private Discord bot for answering reviewed Ryoku support questions.
It serves curated responses from `data/support.json`, adds
safety context for risky actions, and can search a local stable Ryoku checkout
for source-backed answers through Prowl.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-E2342A?style=for-the-badge)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-E2342A?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![discord.py](https://img.shields.io/badge/discord.py-2.6%2B-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discordpy.readthedocs.io)
[![Ryoku Docs](https://img.shields.io/badge/docs-ryoku.dev-E2342A?style=for-the-badge)](https://docs.ryoku.dev)

<kbd>[Docs](https://docs.ryoku.dev)</kbd> &middot; <kbd>[Support catalog](data/support.json)</kbd> &middot; <kbd>[Benchmarks](data/benchmark.json)</kbd> &middot; <kbd>[Discord](https://discord.gg/8KjBmUEyKA)</kbd>

</div>

---

## What it does

- Answers reviewed Ryoku support questions in Discord

## Quick start

### 1. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Clone the stable Ryoku repo

The bot uses a local stable Ryoku checkout for source-backed answers.

```bash
git clone https://github.com/neur0map/ryoku-arch.git /path/to/ryoku
```

Point `RYOKU_REPO_PATH` in `.env` at that checkout.

### 3. Initialize the Ryoku checkout for Prowl

Run this once inside the cloned Ryoku repo:

```bash
cd /path/to/ryoku
prowl-agent init --no-input --integrations none
```

### 4. Configure `.env`

Set:

- `TOKEN` to Nero's Discord bot token
- `RYOKU_REPO_PATH` to your cloned Ryoku checkout
- `PROWL_AGENT_PATH` to the absolute Prowl executable, e.g. `/home/neur0map/workspace/prowl-agent/prowl-agent`. This prevents service `PATH` drift.
- `SUPPORT_CHANNEL_ID` to the one Discord channel where ordinary messages should be answered. Leave it as `0` to require a mention/reply everywhere.
- `FEEDBACK_DB_PATH` to a local SQLite file for Correct/Incorrect signals. It defaults to `runtime/nero-feedback.sqlite3` and must be backed up separately from Git.
- `OLLAMA_HOST`, `GEMMA_MODEL=gemma4:e4b`, and `LFM_MODEL=lfm2.5:latest` for local answers
- optionally `MODEL_NAME`, `PROWL_TIMEOUT_SECONDS`, `PROWL_RESULT_LIMIT`, and `OLLAMA_TIMEOUT_SECONDS`

### Local model behavior

Nero retrieves the reviewed support card and/or Prowl citations **before** invoking a model. The model only turns that evidence into a concise Discord reply; it is not permitted to invent a command or a source.

- **Gemma (`gemma4:e4b`)** handles normal, reviewed support answers with thinking disabled.
- **LFM (`lfm2.5:latest`)** handles replies backed by verified Prowl evidence, including diagnostic and contributor/source questions. Only its final response is sent to Discord.
- Both routes receive the same evidence-only contract: no invented commands, URLs, paths, system state, or citations; ask one focused question when the evidence cannot identify the symptom; and lead with a read-only check before a state-changing recovery action unless the user explicitly asks to perform it.
- The bot serializes local model calls and sends `keep_alive: 0`. This is deliberate: on this 14 GiB CPU-first host, keeping both models resident can cause model eviction or OOM. Additional Discord requests wait their turn and fall back to the reviewed answer if Ollama is unavailable.


## Run

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  --env-file .env \
  python -B bot.py
```

In `SUPPORT_CHANNEL_ID`, users can ask Nero normal Ryoku/Arch support questions without mentioning it. In every other channel, call Nero with `@Nero` or reply to one of Nero's messages. Mentions and replies work in all channels.

Every Nero reply has **Correct** and **Incorrect** buttons. Only the person who asked can submit or revise a rating. Nero stores the rating and the Discord message IDs needed to join it to the future cleaned message archive; it does not add question or answer text to this feedback journal.

## Test

Run the unit test suite:

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B -m unittest discover -s tests -v
```

Run the regression benchmark:

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B benchmark.py --mode hybrid \
  --benchmark data/regressions.json \
  --ryoku-repo /path/to/ryoku \
  --check
```

Run the local LLM benchmark when Ollama is available:

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  --env-file .env \
  python -B benchmark_llm.py --check
```

The LLM benchmark uses `data/llm-evals.json` to exercise both routes: Gemma for a reviewed support answer and LFM for a source-cited answer. It records only final Discord-visible text, latency, pass/fail, model, and route metadata; hidden reasoning markers such as `<think>`, `thinking`, or `hidden-reasoning` cause a failure and are redacted from serialized output.

## Benchmark

Run the full benchmark with replay sampling:

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B benchmark.py --mode hybrid \
  --ryoku-repo /path/to/ryoku \
  --replays 3 --check
```
