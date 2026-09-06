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

| track | reviewer | units | rows | fail | status |
| ----- | -------- | ----- | ---- | ---- | ------ |
| T1 | `rev-m2u82` | M2.8.1, M2.8.2 | 34 | 10 | harvested |
| T2 | `rev-m2u84` | M2.8.4 | 28 | 7 | harvested |
| T3 | `rev-m2u85` | M2.8.3, M2.8.5 | 42 | 15 | harvested |
| T4 | `audit-m2` | M2.7.3-M2.8.5 | 50 | 12 | harvested |
| T5 | `rev-m2-cross` | cross-unit + M2.7.3, M2.7.4 | 39 | 11 | harvested |
Wave 1 base `384c15f`. Worktrees `.scratch/worktrees/<name>`, branches `wt/<name>`, reports
`.scratch/worktrees/<name>/.scratch/agents/<name>.md`, archived at close to
`.agent/archive/review-m2-<name>.md` so a later session adjudicates from committed state.

Report shape graded by `.scratch/check_review_report.py` — 9 predicates over the two-tier shape;
seed grades `FAIL P03` (12 rows unknown) and a filled report grades `PASS`, both measured at seed.


MAIN's ruling on a row binds for the milestone; new evidence is what reopens it. A fix earns one
re-review round scoped to its own acceptance check. `pass` rows are ruled ACCEPTED-as-checked on
harvest; `open` fail rows await MAIN's ruling.

## Rows

### T1 `rev-m2u82` — M2.8.1, M2.8.2

34 rows, 10 fail. Report `.agent/archive/review-m2-rev-m2u82.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (24): R01 R02 R03 R04 R05 R06 R07 R08 R09 R10 R11 R12 R13 R16 R17 R18 R19 R22 R23 R24 R26 R31 R33 R34

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R14 | HIGH — raw, unvalidated qualification cells become both report strata and allowed strings, so an edited identifier-shaped label is emitted under all-green verdicts. | open | see report detail `### R14` |
| R15 | MEDIUM — a fresh pilot rerun never clears its event output, so stale diagnostics/clinical files can be credited when the current source produces none. | open | see report detail `### R15` |
| R20 | MEDIUM — inference is skipped, but the shipped driver rewrites `run_report.json` with invocation-local attempt/time fields, breaking P05 byte idempotence. | open | see report detail `### R20` |
| R21 | HIGH — `process_source` swallows `KeyboardInterrupt`; a child-only interruption exits 0, so the driver can run R and mark a partial source complete. | open | see report detail `### R21` |
| R25 | MEDIUM — `_artifacts` accepts a one-row diagnostics file whose `video` belongs to another event/camera, then attributes its counters to this `ok` asset. | open | see report detail `### R25` |
| R27 | CRITICAL — `--out` can create logs inside the session tree before validation, and `--report` can write there after both digest snapshots, leaving the input mutated or a green witness stale. | open | see report detail `### R27` |
| R28 | MEDIUM — `abs(NaN - derived) > 1e-9` is false, so a nonfinite stored rate passes `stored_rate_equals_its_derivation`. | open | see report detail `### R28` |
| R29 | HIGH — shipped key validation remains `allowed OR regex`, while the alleged membership oracle walks only local fixtures; an unknown identifier-shaped key passes production. | open | see report detail `### R29` |
| R30 | HIGH — clinical-failed event wall enters `_corpus_wall`, its frames are excluded from `_artifacts`, yet the report labels the mixed 100/300 population `sample: corpus`. | open | see report detail `### R30` |
| R32 | HIGH — P05/P06, P09, P11, P13, and P14 chiefly grade `_Driver`, local verdict walkers, or source/prose regexes; 105 green cases miss nine shipped-code reds. | open | see report detail `### R32` |

### T2 `rev-m2u84` — M2.8.4

28 rows, 7 fail. Report `.agent/archive/review-m2-rev-m2u84.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (21): R01 R02 R03 R04 R05 R06 R07 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R22 R25 R26 R27

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R08 | HIGH — deleting the report key leaves P08 green, so N7 and the artifact-identity conjunct are not witnessed | open | see report detail `### R08` |
| R09 | MEDIUM — replacing both helper calls with duplicated `max` formulas leaves P09 green; A07's spy was not built | open | see report detail `### R09` |
| R20 | MEDIUM — three unqualified `[0,1]` claims are false: a finite off-frame point exports as `(1.1,-0.05)` | open | see report detail `### R20` |
| R21 | HIGH — `VideoCapture.set(...,1)` return/read-back is ignored; a rejected request returns an unreleased usable capture | open | see report detail `### R21` |
| R23 | HIGH — unreplayable JSON pins sample IDs only; it carries no script, registry-generation, run-tree, manifest, or config digest | open | see report detail `### R23` |
| R24 | HIGH — code, tests, 19 descriptors, and docs still publish the pre-fix anisotropic token/claim | open | see report detail `### R24` |
| R28 | MEDIUM — roadmap still says body restores 17/92 and its schema record freezes pre-fix anisotropy, contradicting later 14/89/isotropic state | open | see report detail `### R28` |

### T3 `rev-m2u85` — M2.8.3, M2.8.5

42 rows, 15 fail. Report `.agent/archive/review-m2-rev-m2u85.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (27): R01 R02 R03 R04 R08 R09 R11 R12 R13 R14 R15 R16 R17 R18 R19 R21 R22 R23 R24 R26 R27 R28 R33 R34 R37 R38 R40

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R05 | HIGH: 19 angle rows and the technical claim retain the pre-M2.8.4 anisotropy metadata | open | see report detail `### R05` |
| R06 | HIGH: manifest `event_id`/`camera_name` are unvalidated yet trusted; the contract exposes no non-`ok` exclusion census | open | see report detail `### R06` |
| R07 | MEDIUM: duplicate source headers publish duplicate census entries while the feature-table set check passes | open | see report detail `### R07` |
| R10 | HIGH: event grouping keys come from forgeable manifest fields rather than the validated sessions placement | open | see report detail `### R10` |
| R20 | LOW: literal YAML NEL (`U+0085`) is normalized to a space, so the JSON-scalar renderer is not round-trip total | open | see report detail `### R20` |
| R25 | MEDIUM: an arbitrary fifth file is outside the digest yet `validate_generation()` accepts the changed set | open | see report detail `### R25` |
| R29 | HIGH: current-PID staging/retiring paths are recursively removed before `_publish`, before any swap | open | see report detail `### R29` |
| R30 | HIGH: PID reuse lets the next failed attempt delete the only complete staged generation before replacement | open | see report detail `### R30` |
| R31 | HIGH: post-swap sweep recursively deletes every prefix-matching sibling without reading an ownership marker | open | see report detail `### R31` |
| R32 | HIGH: only two sessions digests are rechecked; concurrent inventory and clinical-file edits publish successfully | open | see report detail `### R32` |
| R35 | HIGH: committed campaign evidence has a stale source digest, so the documented command refuses at base before any sweep | open | see report detail `### R35` |
| R36 | HIGH: digest binds only `cohort.py` + `test_cohort.py`, omitting campaign logic, direct modules, fixture helpers/goldens and external schema | open | see report detail `### R36` |
| R39 | HIGH: all required sections exist, but the angle boundary falsely says the corrected pipeline remains anisotropic with 9.9° error | open | see report detail `### R39` |
| R41 | MEDIUM: invalid inventory, sessions and missing manifest leak three different exception types across the public publisher boundary | open | see report detail `### R41` |
| R42 | HIGH: shipped 60/60 suite is green while 15 adversarial cases fail, and one P14 test asserts foreign recursive deletion as success | open | see report detail `### R42` |

### T4 `audit-m2` — M2.7.3-M2.8.5

50 rows, 12 fail. Report `.agent/archive/review-m2-audit-m2.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (38): R01 R02 R03 R04 R05 R06 R07 R08 R09 R10 R11 R12 R13 R16 R17 R18 R19 R20 R21 R22 R25 R26 R27 R29 R30 R32 R33 R34 R35 R36 R38 R39 R42 R43 R45 R46 R48 R49

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R14 | 8/9 fire: deleting one of 13 local-decision labels leaves P05 green. | open | see report detail `### R14` |
| R15 | UNREPRODUCIBLE: the cited review ledger is absent from committed state. | open | see report detail `### R15` |
| R23 | Six components reproduce, but the only retained run gives 540 windows, not 572. | open | see report detail `### R23` |
| R24 | UNREPRODUCIBLE: pre-fix logs/report/tree are absent; current retained output is post-fix. | open | see report detail `### R24` |
| R28 | Partition/CFR/540 replay; historical wall and latency range do not because its report/logs are absent. | open | see report detail `### R28` |
| R31 | UNREPRODUCIBLE: pre-fix run tree/report is deleted; current marker carries only the 7.828 h NPU run. | open | see report detail `### R31` |
| R37 | Marker retains 219/0 collisions, but not the external descriptor bytes, digest, composition or prefix census. | open | see report detail `### R37` |
| R40 | UNREPRODUCIBLE: checker rc=1 because its pre-fix input tree was deleted; JSON only records the values. | open | see report detail `### R40` |
| R41 | Run is isotropic, but cohort code/docs still publish `deg_image_plane_uncalibrated` and claim anisotropy. | open | see report detail `### R41` |
| R44 | All census values and ~180 KB reproduce, but elapsed was 30.59 s, not 14.4 s. | open | see report detail `### R44` |
| R47 | Stale-green barrier fires: recorded source digest differs from current source, so checker exits 1 before sweeps. | open | see report detail `### R47` |
| R50 | Label census reproduces 75; the 14-missing-font claim has no consumer font bytes or digest in this repo. | open | see report detail `### R50` |

### T5 `rev-m2-cross` — cross-unit + M2.7.3, M2.7.4

39 rows, 11 fail. Report `.agent/archive/review-m2-rev-m2-cross.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (28): R01 R02 R03 R04 R05 R06 R07 R10 R11 R12 R13 R18 R19 R20 R21 R22 R24 R26 R27 R28 R29 R30 R31 R32

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R08 | LOW: P16 independently enumerates all 5 group-QC goldens, falsifying the six-place budget | open | see report detail `### R08` |
| R09 | HIGH: inventory evidence has no source tuple/digests and records an intrinsically stale head SHA | open | see report detail `### R09` |
| R14 | HIGH: digest covers only `cohort.py` + its test while output reads/rendering depend on four omitted modules | open | see report detail `### R14` |
| R15 | HIGH: inventory and sessions follow marker symlinks; the other four reject non-regular markers | open | see report detail `### R15` |
| R16 | HIGH: inventory and sessions accept duplicate marker keys; the other four use rejecting hooks | open | see report detail `### R16` |
| R17 | HIGH: sessions excludes the whole marker; measure excludes the whole generation/upstream block | open | see report detail `### R17` |
| R23 | MEDIUM: roadmap still says 17 restored columns → 92 published beside the measured 89/1068 result | open | see report detail `### R23` |
| R25 | MEDIUM: memory cites old `export.py:250` anisotropy; code now uses one max-dimension scale and cohort docs/tests pin 9.9° | open | see report detail `### R25` |
| R33 | LOW: shipped code comments retain contract/ruling provenance (`A02`, `A09`, `P17`) instead of standalone constraints | open | see report detail `### R33` |
| R34 | LOW: `steq.py` found 24 flags across four human surfaces, including 22 passive constructions in the two docs units | open | see report detail `### R34` |
| R35 | MEDIUM: duplicated publisher trust-root machinery drifted in exactly the four ways exposed by R14-R17 | open | see report detail `### R35` |

## Register carry-out

Out-of-contract observations that survive as `.agent/polish.md` rows. Filled at ruling time.
