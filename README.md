# Claude Tool-Use Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeeFlannery/claude-tool-use-lab/blob/main/lab/claude-tool-use-lab.ipynb)

A hands-on, 45 to 60 minute lab: build a read-only, tool-calling agent with the Anthropic Python SDK, lock its scope down, then test it with an LLM judge. Learners build an inventory assistant for a fake soda company's shop staff.

## What it teaches

By the end, a learner can:

1. **Read** a Messages API response by its content blocks and `stop_reason`.
2. **Define** tools with JSON Schema that Claude can call reliably.
3. **Implement** the agent loop: run each `tool_use`, return `tool_result`, repeat until `end_turn`, with a max-turns guard.
4. **Enforce and test** read-only scope with an allowlist, a call log, and an LLM-judge eval.

## Files

| Path | What it is |
| --- | --- |
| `lab/claude-tool-use-lab.ipynb` | Learner notebook, 35 cells, 5 `# TODO` exercises |
| `lab/claude-tool-use-lab-solution.ipynb` | Same notebook, every exercise filled in |
| `scripts/build_notebooks.py` | Single source for both notebooks. Edit this, not the `.ipynb` files |
| `tests/test_keyfree.py` | Fast checks on the tool schemas, the guard, and the eval fixtures. No key needed |
| `tests/test_notebook_live.py` | Runs the solution notebook headless against the real API. Skipped without a key |

## Run it

**In Colab:** click the badge, add `ANTHROPIC_API_KEY` under Secrets (key icon, left sidebar), allow notebook access, and run top to bottom.

**Locally** with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
export ANTHROPIC_API_KEY=...   # or inject it: op run --env-file=op.env -- <cmd>
uv run --with jupyterlab jupyter lab lab/
```

**Tests:**

```bash
uv run pytest                                   # key-free tests; the live run skips itself
ANTHROPIC_API_KEY=... uv run pytest -v          # adds the full headless notebook run
```

**After editing content:**

```bash
uv run python scripts/build_notebooks.py && uv run pytest
```

## Curriculum design notes

**Audience.** Developers comfortable with Python functions, dicts, and JSON, who have called an LLM API at most once and never built tool calling. No framework knowledge assumed. The lab teaches the raw loop on purpose: once learners have built it, SDK helpers and MCP stop being magic.

**Structure.** Five sections, each running concept, worked example, exercise, check. The worked example shows the move once; the exercise asks for the same move one step further; the check cell gives instant feedback with a pass line or a hint naming what to fix. Every hint points at a cause ("Append the assistant reply before the tool results"), not just a symptom.

**Exercise to objective map.**

| Exercise | Learner builds | Objective |
| --- | --- | --- |
| 1 | `text_of()`: pull text out of mixed content blocks | 1: read responses |
| 2 | `check_stock`: a strict tool schema plus its function | 2: define tools |
| 3 | `run_agent()`: the full loop with a max-turns guard | 3: agent loop |
| 4 | `guarded_execute()`: allowlist plus call log | 4: enforce scope |
| 5 | `run_evals()`: deterministic checks plus a judge, pass rate | 4: test it |

**Design choices.**

- *One running domain.* The soda inventory carries through every section, so learners spend attention on tool calling, not on a new dataset each time.
- *The guard earns its place.* Section 4 ships a write tool into the registry as "drift", shows the plain executor running it, then has the guard block it live. Learners see the failure before they build the fix.
- *Judge calibration is taught, not assumed.* The judge is tested on a known-good and a known-bad answer before it grades the agent.
- *Cut list.* Streaming, the SDK tool runner, MCP, and parallel-call tuning are named in "Go further" rather than taught. Each is a module of its own and none serves the four objectives.

**Keeping it current.**

- The model ID lives in one constant (`MODEL`). A key-free test asserts it appears exactly once, so a model swap is a one-line change.
- Both notebooks come from `scripts/build_notebooks.py`, so the learner and solution versions can't drift apart. A test checks that every non-exercise cell is identical across them.
- The live test runs the whole solution notebook against the real API. Run it on each SDK release and each new model; a failing check cell names the exact section that broke.
- When the API changes shape (new block types, new stop reasons), update Section 1's concept cell first. Everything else builds on how learners read a response.
