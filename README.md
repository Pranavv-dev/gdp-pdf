# GDP.pdf Harness

This repo is an eval harness for the [GDP.pdf benchmark](https://surgehq.ai/benchmarks/gdp-pdf), 
published by [Surge AI](https://www.surgehq.ai). 

The full dataset is published on [Hugging Face](https://huggingface.co/datasets/surgeai/GDP.pdf). 
You can use this harness to run GDP.pdf on your own models, or to reproduce the published leaderboard. 

## Running An Eval

This harness is a `uv` package, built with the [Inspect AI](https://inspect.aisi.org.uk) framework. 

To run an eval: set your provider API keys, then run an `inspect eval` command: 

```bash
# can also set these in `.env`
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=... 

# evaluate all 100 held-out tasks, no tools
uv run inspect eval src/gdp_pdf/task.py --model anthropic/claude-opus-4-8

# pick the judge, cap tasks, and raise concurrency
uv run inspect eval src/gdp_pdf/task.py \
  --model anthropic/claude-opus-4-20250514 \
  -T judge_model=google/gemini-3.5-flash \
  -T limit=10 \
  --max-connections 8
```

The scorer reports two metrics per model: `all_pass/mean` (the headline
leaderboard number — every rubric criterion satisfied) and
`mean_criteria/mean` (average fraction of criteria satisfied), comparable to the
[public leaderboard](https://surgehq.ai/benchmarks/gdp-pdf).

## Pass@K / Pass^K

Run multiple epochs with Inspect's score reducers to compute Pass@K
and Pass^K in the same run:

```bash
uv run inspect eval src/gdp_pdf/task.py --model ... \
  --epochs 5 --epochs-reducer "mean,pass_at_5,at_least_5"
```

This reports, per metric key: `all_pass/mean` (Pass@1), `all_pass/pass_at_5`
(Pass@5, the unbiased [Codex-paper estimator](https://arxiv.org/pdf/2107.03374)),
and `all_pass/at_least_5` (Pass^5 — all 5 runs pass). 

## Leaderboard Configuration

To match the configuration used in the leaderboard, run the complete task set
with five runs per task and Gemini 3.5 Flash as the judge:

```bash
uv run inspect eval src/gdp_pdf/task.py \
  --model ... \
  --epochs 5 \
  --epochs-reducer "mean,pass_at_5,at_least_5" \
  -T judge_model=google/gemini-3.5-flash
```

For supported models, optionally set the reasoning effort with
`--reasoning-effort ...`.

## Offline single-task check

To smoke-test without keys, run the task against a local pack with Inspect's
mock model (no network, no judge calls):

```bash
# Inspect chdirs into the task file's directory, so pass task_pack as an
# absolute path ($PWD expands before the CLI runs).
uv run inspect eval src/gdp_pdf/task.py \
  -T task_pack="$PWD/task_packs/sample_pdf" \
  --model mockllm/model -T judge_model=mockllm/model
```

The committed `task_packs/sample_pdf/` is a synthetic placeholder task. 

Structural tests (no network/keys) run via:

```bash
uv run python tests/test_offline.py
```
