# Gate invocation

## The prefix — every gate command in this project

```sh
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" uv run --no-sync <cmd>
```

Both halves are load-bearing. `PYTHONPATH` selects the primary tree for the non-pytest half; unsetting `LD_LIBRARY_PATH` keeps the host OpenVINO build out of the loader path. Without them `tests/conftest.py` imports `pose_estimation` → `openvino` and dies at `ImportError … GLIBC_2.43 not found` **before collecting one test** — in the primary tree as well as in a worktree. A bare `uv run --no-sync pytest` reproduces it. Every teammate brief carries this prefix verbatim; a partially-remembered prefix costs a window (`scripts/gate.sh` port → `.agent/polish.md`).

- **Full suite ≈ 17-25 min (1720 tests).** Always `run_in_background: true` + redirect to a log, then work while it runs — a background command piped through `tail` buffers everything until exit.
- **Run the decisive gate alone.** `tests/test_r_timebase_truth.py::test_c8_08` runs the WHOLE suite in a subprocess (`pytest -q --maxfail=1 -k "not test_c8_08"`, cwd = project root), asserts rc=0 and reconciles the pass count against a separate collection. Five concurrent worktree gates plus MAIN's on 8 cores drove it past its 900 s timeout — that is the whole difference between `1701 passed / 1 failed` and green. Never run it beside a decode, an inference sweep or a mutation campaign; brief reviewers to targeted files and keep the full-suite slot for MAIN.
- **One red file anywhere under `tests/` takes `test_c8_08` down** and with it the decisive gate for the whole unit. Measured: red file present → 59 failed / 1643 passed; absent → 1642 passed. So a diff-blind or reviewer red suite lives on its `archive/*` tag (→ `retention.md`), and the implementing unit restores it with `git show <tag>:tests/<file> > tests/<file>`, drives it green, and commits it green. The generic "put the red suite in the primary tree" instruction does not hold here.
- **A green suite measures nothing about pinning.** `30280c3` shipped 12 review fixes with 96 tests green and the first mutation campaign showed 18 of 72 mutants surviving. Fix-plus-test is not fix-plus-*pinning*-test.

## The accelerator run recipe — mutually exclusive with the gate prefix

The gate prefix is exactly what strips the accelerator: it reports `['CPU']`, compiles, runs and produces correct output with no warning anywhere, because CPU is a supported device. **Only the wall clock tells.** Measured on the same two events, `--tracking body`: **1015.5 s / 927.4 s under the gate prefix vs 638.5 s / 497.2 s** under the run recipe = **1.6-1.9×**, which is 8 h against 15 h over the corpus.

```sh
source /var/home/eturkes/.local/app/intel-accel/env.sh
PYTHONPATH="$PWD/src:$PYTHONPATH" .venv/bin/python scripts/<driver>.py
```

- Verify placement **before** funding hours: `openvino.Core().available_devices` must read `['CPU', 'GPU', 'NPU']`, and the event's own `run.log` must open `pose-device=NPU` + "loaded … with the openvino/NPU backend".
- The inherited `PYTHONPATH` selects the *host* OpenVINO build, which needs glibc 2.43 — a loud failure in the container, never a silent CPU fallback. Stripping the env entirely falls back to the `.venv` pip wheel at `['CPU']`.
- Enablement, device preference + self-test → `CLAUDE.local.md` → `~/agents/docs/openvino.md`.

## Worktree gate recipe (`.scratch/worktrees/<name>`)

Every teammate worktree runs the full Python gate concurrently off the one primary environment, read-only:

```sh
export UV_PROJECT_ENVIRONMENT=<primary-tree>/.venv PYTHONPATH="$PWD/src"
uv run --no-sync ruff check && uv run --no-sync ruff format --check \
  && uv run --no-sync ty check && uv run --no-sync pytest
```

- **Two resolution paths with opposite defaults — the most expensive trap in this project.** `pyproject.toml` sets `pythonpath = ["src", "tests"]`, which pytest resolves against **rootdir** and inserts at `sys.path[0]`: `pytest` launched from a worktree imports **that worktree's** `pose_estimation`, with or without `PYTHONPATH`. Everything else — `python -c`, `python -m pose_estimation.<mod>`, `ty check` — resolves through the hatchling editable install to the **primary** tree's `src/`. So a reviewer running its red suite from its own worktree tests its own copy and every primary-tree fix reads as still-broken. **Print `<module>.__file__` before believing a red**, and copy a test file into the primary `tests/` to exercise primary code.
- `UV_PROJECT_ENVIRONMENT` must be exported on **every** call: `uv run` inside a worktree otherwise creates and selects a worktree-local `.venv`, which `--no-sync` leaves empty → `No module named pytest`.
- `--no-sync` keeps the shared environment unmutated, which is what makes concurrent worktree gating safe.
- Tool caches (`.ruff_cache`, `.pytest_cache`, `.ty_cache`, `.coverage`) are cwd-relative → already private per worktree.
- **Three gitignored trees must be symlinked in read-only, or the gate is not equivalent**: `ln -sfn <primary>/{videos,renv/library,inventory} <worktree>/`. Without `videos` every real-data command fails at a missing path; without `renv/library` every R case SKIPs (with it: 469 passed / 0 skipped, `tests/test_r_pipeline.py` 25 passed / 0 skipped); without `inventory` `tests/test_sessions.py::test_p05_real_corpus_headline_counts` SKIPs and that single skip fails C8.08, whose A32 reconciliation demands zero skipped. Never write through any of the three — `inventory/` carries source paths and is as sensitive as the corpus, and R must never `renv::install`/`renv::snapshot` through the link.

## Scratch validators pending port

A gate backing a durable claim must rerun from committed state, so a scratch-local validator is a temporary encoding: its regeneration path is recorded here and its port is scheduled in `.agent/polish.md`.

- `.scratch/steq.py` — ASD-STE100 register scan over the human-facing surface (inventory: `docs/technical/conventions.md` → *Text register*). Drops fences/tables/headings/frontmatter, joins wrapped lines into blocks so a sentence is measured whole, splits on `.!?`, flags `LONG` (> `--max`; 20 for instructions, 25 for descriptions), `FILLER`, `CONTRACTION` (also fires on possessive `'s`), `PASSIVE` (be-verb + participle heuristic). Code-file mode samples quoted `help=`/`description=`/`title=` strings only. Measured at `--max 20`: `README.md` 14 → 2, `docs/capture_protocol.md` 20 → 7; residual flags are 21-25-word descriptions, which the rule allows.
- `.scratch/fidelity.sh <base-ref> <file>…` — pairs with it: diffs the multiset of format specifiers, `--flags`, backticked spans, file names and numbers between a base ref and the working tree. A register-only edit must show no delta; every delta needs an explanation. Caught the p-value reformat (`p<.05` → `p < 0.05`) and confirmed 14 R files invariant.

## Committed report grader

`scripts/check_review_report.py` — 9 predicates over the two-tier report shape, so a MILESTONE-REVIEW wave's report-shape claim reruns from committed state: three required sections, parseable `| <ID> |` rows, unique ids, a verdict cell per row, **zero `unknown`** (P03 — what makes an all-`unknown` seed grade nonzero), every `pass` row stating what was checked, a `### <id>` detail section per `fail` row carrying `file:line` + `predicate` + `impact` + `acceptance`, an acceptance check on every register entry. Seed each reviewer's skeleton with all rows `unknown`, grade it both ways before dispatch, and name the command in the brief.
