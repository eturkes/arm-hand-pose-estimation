# M2 review ledger

MILESTONE-REVIEW state for M2. One row per check-set row; the ledger is the resume point, so a
row's evidence must be readable here without the reviewer's worktree, which does not survive close.

**Scope, set by the roadmap's carry-over rule.** M2.1-M2.7.2 closed under the earlier regime that ran
`rev`/`rev2` inside WORK-UNIT; their check sets are adjudicated and those rulings bind — only new
evidence reopens one. This ledger covers **M2.7.3 onward** plus the two milestone-scoped lenses no
unit could run: cross-unit integration and the `audit-m2` claim replay.

**Completion counter.** `rows adjudicated / rows enumerated`, per track below. The milestone reaches
REVIEWED when every track reads `enumerated` and no row reads `open`.

## Tracks

| track | reviewer | units | tier | rows | adjudicated | status |
| ----- | -------- | ----- | ---- | ---- | ----------- | ------ |
| T1 | `rev-m2u82` | M2.8.1, M2.8.2 | kernel | unenumerated | 0 | dispatched |
| T2 | `rev-m2u84` | M2.8.4 | kernel | unenumerated | 0 | dispatched |
| T3 | `rev-m2u85` | M2.8.3, M2.8.5 | kernel | unenumerated | 0 | dispatched |
| T4 | `audit-m2` | M2.7.3-M2.8.5 | claim replay | unenumerated | 0 | dispatched |
| T5 | `rev-m2-cross` | cross-unit + M2.7.3, M2.7.4 | integration / docs / CLAUDE.md | unenumerated | 0 | dispatched |

Wave 1 base `384c15f`. Worktrees `.scratch/worktrees/<name>`, branches `wt/<name>`, reports
`.scratch/worktrees/<name>/.scratch/agents/<name>.md`, archived at close to
`.agent/archive/review-m2-<name>.md` so a later session adjudicates from committed state.

Report shape graded by `.scratch/check_review_report.py` — 9 predicates over the two-tier shape;
seed grades `FAIL P03` (12 rows unknown) and a filled report grades `PASS`, both measured at seed.

## Rulings

MAIN's ruling on a row binds for the milestone; new evidence is what reopens it. A fix earns one
re-review round scoped to its own acceptance check; a second `fail` on one row ⇒ MAIN rules and the
ruling enters the unit's contract.

| row | track | verdict | ruling | evidence / acceptance check |
| --- | ----- | ------- | ------ | --------------------------- |
| — | — | — | — | awaiting wave-1 harvest |

## Register carry-out

Out-of-contract observations that survive as `.agent/polish.md` rows. Seeded empty.
