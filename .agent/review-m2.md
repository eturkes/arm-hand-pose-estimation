# M2 review ledger

MILESTONE-REVIEW state for M2. One row per check-set row; the ledger is the resume point, so a
row's evidence must be readable here without the reviewer's worktree, which does not survive close.

**Scope, set by the roadmap's carry-over rule.** M2.1-M2.7.2 closed under the earlier regime that ran
`rev`/`rev2` inside WORK-UNIT; their check sets are adjudicated and those rulings bind — only new
evidence reopens one. This ledger covers **M2.7.3 onward** plus the two milestone-scoped lenses no
unit could run: cross-unit integration and the `audit-m2` claim replay.

**Completion counter.** `rows adjudicated / rows enumerated`, per track below. The milestone reaches
REVIEWED when every track reads `enumerated`, no row reads `open`, **and every ACCEPT-FIX row has
closed on its own acceptance check under MAIN's rerun** — adjudicated is not fixed.

**Wave 1 state: 193 rows enumerated, 55 fail, 55 + 2 MAIN rows adjudicated, 0 open.**
**43 ACCEPT-FIX (open work) · 9 ACCEPT-LIMIT · 2 POLISH · 3 REJECT.** The 2 POLISH rows are filed in
`.agent/polish.md`; the ACCEPT-LIMIT and REJECT rows are closed here. The next MILESTONE-REVIEW
session takes the 43 ACCEPT-FIX rows in dependency order — the isotropy token cluster first, since it
is the only row where shipped clinical bytes carry a false claim.

## Tracks

| track | reviewer | units | rows | fail | status |
| ----- | -------- | ----- | ---- | ---- | ------ |
| T1 | `rev-m2u82` | M2.8.1, M2.8.2 | 34 | 10 | harvested |
| T2 | `rev-m2u84` | M2.8.4 | 28 | 7 | harvested |
| T3 | `rev-m2u85` | M2.8.3, M2.8.5 | 42 | 15 | harvested |
| T4 | `audit-m2` | M2.7.3-M2.8.5 | 50 | 12 | harvested |
| T5 | `rev-m2-cross` | cross-unit + M2.7.3, M2.7.4 | 39 | 11 | harvested |
| T6 | MAIN | rows MAIN measured itself | 2 | 2 | adjudicated |
Wave 1 base `384c15f`. Worktrees `.scratch/worktrees/<name>`, branches `wt/<name>`, reports
`.scratch/worktrees/<name>/.scratch/agents/<name>.md`, archived at close to
`.agent/archive/review-m2-<name>.md` so a later session adjudicates from committed state.

**Decisive gate at wave close, run alone from committed state: `1702 passed in 1467.46s`, rc=0.**
The concurrent run during the wave read 1701 passed / 1 failed on `test_c8_08`'s 900 s subprocess
timeout — five worktree gates plus MAIN's on 8 cores. Run the decisive gate alone.

Report shape graded by **`scripts/check_review_report.py`** — 9 predicates over the two-tier shape,
ported out of `.scratch/` at wave close so the grading claim reruns from committed state. Graded both
ways twice: at seed (12 `unknown` rows → `FAIL P03`; filled → `PASS`) and after the port (T3's
archived report → `PASS`, 42 rows / 15 fail, rc=0; one verdict flipped to `unknown` → `FAIL P03`,
rc=1). Every future MILESTONE-REVIEW wave seeds its report skeletons against it.


MAIN's ruling on a row binds for the milestone; new evidence is what reopens it. A fix earns one
re-review round scoped to its own acceptance check. `pass` rows are ruled ACCEPTED-as-checked on
harvest; `open` fail rows await MAIN's ruling.

## Rows

### T1 `rev-m2u82` — M2.8.1, M2.8.2

34 rows, 10 fail. Report `.agent/archive/review-m2-rev-m2u82.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (24): R01 R02 R03 R04 R05 R06 R07 R08 R09 R10 R11 R12 R13 R16 R17 R18 R19 R22 R23 R24 R26 R31 R33 R34

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R14 | HIGH — raw, unvalidated qualification cells become both report strata and allowed strings, so an edited identifier-shaped label is emitted under all-green verdicts. | **ACCEPT-FIX** | pilot validates every source table before its cells become allowlist strings |
| R15 | MEDIUM — a fresh pilot rerun never clears its event output, so stale diagnostics/clinical files can be credited when the current source produces none. | **ACCEPT-FIX** | a pilot rerun clears the selected event tree before launch |
| R20 | MEDIUM — inference is skipped, but the shipped driver rewrites `run_report.json` with invocation-local attempt/time fields, breaking P05 byte idempotence. | **ACCEPT-FIX** | a resumed pass leaves every published byte unmoved, or P05 is amended to name the invocation-local fields |
| R21 | HIGH — `process_source` swallows `KeyboardInterrupt`; a child-only interruption exits 0, so the driver can run R and mark a partial source complete. | **ACCEPT-FIX** | KeyboardInterrupt propagates out of `process_source`; no partial source is marked complete |
| R25 | MEDIUM — `_artifacts` accepts a one-row diagnostics file whose `video` belongs to another event/camera, then attributes its counters to this `ok` asset. | **ACCEPT-FIX** | `_artifacts` refuses a diagnostics row whose `video` names another source |
| R27 | CRITICAL — `--out` can create logs inside the session tree before validation, and `--report` can write there after both digest snapshots, leaving the input mutated or a green witness stale. | **ACCEPT-FIX** | CRITICAL — log and report sinks refuse any path inside a published session tree, before validation and after both digest snapshots |
| R28 | MEDIUM — `abs(NaN - derived) > 1e-9` is false, so a nonfinite stored rate passes `stored_rate_equals_its_derivation`. | **ACCEPT-FIX** | a nonfinite stored rate fails `stored_rate_equals_its_derivation` |
| R29 | HIGH — shipped key validation remains `allowed OR regex`, while the alleged membership oracle walks only local fixtures; an unknown identifier-shaped key passes production. | **ACCEPT-FIX** | the shipped key guard is membership in the schema/label union, and the oracle walks the shipped guard |
| R30 | HIGH — clinical-failed event wall enters `_corpus_wall`, its frames are excluded from `_artifacts`, yet the report labels the mixed 100/300 population `sample: corpus`. | **ACCEPT-FIX** | `sample: corpus` requires one event population behind numerator and denominator |
| R32 | HIGH — P05/P06, P09, P11, P13, and P14 chiefly grade `_Driver`, local verdict walkers, or source/prose regexes; 105 green cases miss nine shipped-code reds. | **ACCEPT-FIX** | every contract predicate grades a shipped symbol; the nine reds go green against `src/` and `scripts/` |

### T2 `rev-m2u84` — M2.8.4

28 rows, 7 fail. Report `.agent/archive/review-m2-rev-m2u84.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (21): R01 R02 R03 R04 R05 R06 R07 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R22 R25 R26 R27

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R08 | HIGH — deleting the report key leaves P08 green, so N7 and the artifact-identity conjunct are not witnessed | **ACCEPT-FIX** | deleting `configuration.coord_normalization` fails P08 |
| R09 | MEDIUM — replacing both helper calls with duplicated `max` formulas leaves P09 green; A07's spy was not built | **ACCEPT-FIX** | bypassing `coord_scale` with a duplicated formula fails P09 |
| R20 | MEDIUM — three unqualified `[0,1]` claims are false: a finite off-frame point exports as `(1.1,-0.05)` | **ACCEPT-FIX** | every `[0,1]` claim is scoped to in-frame points |
| R21 | HIGH — `VideoCapture.set(...,1)` return/read-back is ignored; a rejected request returns an unreleased usable capture | **ACCEPT-FIX** | a refused orientation request raises and releases the capture |
| R23 | HIGH — unreplayable JSON pins sample IDs only; it carries no script, registry-generation, run-tree, manifest, or config digest | **ACCEPT-FIX** | the fidelity JSON binds to script, registry generation, run tree and config digests |
| R24 | HIGH — code, tests, 19 descriptors, and docs still publish the pre-fix anisotropic token/claim | **ACCEPT-FIX** | published angle columns carry `deg_image_plane`; no consumer text claims anisotropic normalization |
| R28 | MEDIUM — roadmap still says body restores 17/92 and its schema record freezes pre-fix anisotropy, contradicting later 14/89/isotropic state | **ACCEPT-FIX** | roadmap and contracts state 14 restored / 89 published / isotropic, with no pre-fix figure left live |

### T3 `rev-m2u85` — M2.8.3, M2.8.5

42 rows, 15 fail. Report `.agent/archive/review-m2-rev-m2u85.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (27): R01 R02 R03 R04 R08 R09 R11 R12 R13 R14 R15 R16 R17 R18 R19 R21 R22 R23 R24 R26 R27 R28 R33 R34 R37 R38 R40

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R05 | HIGH: 19 angle rows and the technical claim retain the pre-M2.8.4 anisotropy metadata | **ACCEPT-FIX** | see T2 R24 — one fix closes both |
| R06 | HIGH: manifest `event_id`/`camera_name` are unvalidated yet trusted; the contract exposes no non-`ok` exclusion census | **ACCEPT-FIX** | manifest `event_id`/`camera_name` are validated against the sessions placement before use |
| R07 | MEDIUM: duplicate source headers publish duplicate census entries while the feature-table set check passes | **ACCEPT-FIX** | a duplicate source header fails the census rather than publishing twice |
| R10 | HIGH: event grouping keys come from forgeable manifest fields rather than the validated sessions placement | **ACCEPT-FIX** | event grouping keys come from the validated sessions tree |
| R20 | LOW: literal YAML NEL (`U+0085`) is normalized to a space, so the JSON-scalar renderer is not round-trip total | **POLISH** | LOW — YAML NEL round-trip; register row with the reproducer |
| R25 | MEDIUM: an arbitrary fifth file is outside the digest yet `validate_generation()` accepts the changed set | **ACCEPT-FIX** | an extra file in the generation fails `validate_generation` |
| R29 | HIGH: current-PID staging/retiring paths are recursively removed before `_publish`, before any swap | **ACCEPT-FIX** | staging and retiring debris is swept only after the swap lands |
| R30 | HIGH: PID reuse lets the next failed attempt delete the only complete staged generation before replacement | **ACCEPT-FIX** | no complete staged generation is deleted by a later attempt under PID reuse |
| R31 | HIGH: post-swap sweep recursively deletes every prefix-matching sibling without reading an ownership marker | **ACCEPT-FIX** | the orphan sweep reads an ownership marker before any recursive delete |
| R32 | HIGH: only two sessions digests are rechecked; concurrent inventory and clinical-file edits publish successfully | **ACCEPT-FIX** | a recursive non-following pre/post snapshot covers inventory, sessions and run |
| R35 | HIGH: committed campaign evidence has a stale source digest, so the documented command refuses at base before any sweep | **ACCEPT-FIX** | the committed campaign evidence regenerates green from committed source |
| R36 | HIGH: digest binds only `cohort.py` + `test_cohort.py`, omitting campaign logic, direct modules, fixture helpers/goldens and external schema | **ACCEPT-FIX** | `source_digest` covers every source that shapes published bytes or acceptance |
| R39 | HIGH: all required sections exist, but the angle boundary falsely says the corrected pipeline remains anisotropic with 9.9° error | **ACCEPT-FIX** | see T2 R24 — the angle boundary states the corrected isotropic pipeline |
| R41 | MEDIUM: invalid inventory, sessions and missing manifest leak three different exception types across the public publisher boundary | **ACCEPT-FIX** | every public publisher entry point raises `CohortError` alone |
| R42 | HIGH: shipped 60/60 suite is green while 15 adversarial cases fail, and one P14 test asserts foreign recursive deletion as success | **ACCEPT-FIX** | the 15 adversarial cases go green and no shipped case asserts foreign recursive deletion as success |

### T4 `audit-m2` — M2.7.3-M2.8.5

50 rows, 12 fail. Report `.agent/archive/review-m2-audit-m2.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (38): R01 R02 R03 R04 R05 R06 R07 R08 R09 R10 R11 R12 R13 R16 R17 R18 R19 R20 R21 R22 R25 R26 R27 R29 R30 R32 R33 R34 R35 R36 R38 R39 R42 R43 R45 R46 R48 R49

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R14 | 8/9 fire: deleting one of 13 local-decision labels leaves P05 green. | **ACCEPT-FIX** | deleting any one local-decision label fails P05 |
| R15 | UNREPRODUCIBLE: the cited review ledger is absent from committed state. | **ACCEPT-LIMIT** | the roadmap cites no `.scratch/` report as evidence for a durable claim; cited reports are archived or the citation drops |
| R23 | Six components reproduce, but the only retained run gives 540 windows, not 572. | **ACCEPT-LIMIT** | MAIN re-derived it: 572 is the M2.8.1 PRE-fix pilot population and 540 the post-fix one, both recorded (`roadmap.md:801,944`); the pre-fix tree is deleted, so the figure is historical |
| R24 | UNREPRODUCIBLE: pre-fix logs/report/tree are absent; current retained output is post-fix. | **ACCEPT-LIMIT** | the record labels the pre-fix measurement historical and unreproducible, never re-derivable |
| R28 | Partition/CFR/540 replay; historical wall and latency range do not because its report/logs are absent. | **ACCEPT-LIMIT** | as R24 |
| R31 | UNREPRODUCIBLE: pre-fix run tree/report is deleted; current marker carries only the 7.828 h NPU run. | **ACCEPT-LIMIT** | as R24 |
| R37 | Marker retains 219/0 collisions, but not the external descriptor bytes, digest, composition or prefix census. | **ACCEPT-FIX** | the census records the external descriptor digest, so `descriptor_collision` is re-checkable |
| R40 | UNREPRODUCIBLE: checker rc=1 because its pre-fix input tree was deleted; JSON only records the values. | **ACCEPT-LIMIT** | as R24, and closed by T2 R23's source binding |
| R41 | Run is isotropic, but cohort code/docs still publish `deg_image_plane_uncalibrated` and claim anisotropy. | **ACCEPT-FIX** | see T2 R24 |
| R44 | All census values and ~180 KB reproduce, but elapsed was 30.59 s, not 14.4 s. | **REJECT** | MAIN re-measured under quiescence: **13.26 s** against the recorded 14.4 s, so the claim reproduces; 30.59 s was five reviewers sharing 8 cores |
| R47 | Stale-green barrier fires: recorded source digest differs from current source, so checker exits 1 before sweeps. | **ACCEPT-FIX** | see T3 R35 — one regeneration closes both |
| R50 | Label census reproduces 75; the 14-missing-font claim has no consumer font bytes or digest in this repo. | **ACCEPT-LIMIT** | the glyph claim states its measurement is against the consumer repo, outside this boundary |

### T5 `rev-m2-cross` — cross-unit + M2.7.3, M2.7.4

39 rows, 11 fail. Report `.agent/archive/review-m2-rev-m2-cross.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (28): R01 R02 R03 R04 R05 R06 R07 R10 R11 R12 R13 R18 R19 R20 R21 R22 R24 R26 R27 R28 R29 R30 R31 R32

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R08 | LOW: P16 independently enumerates all 5 group-QC goldens, falsifying the six-place budget | **ACCEPT-LIMIT** | memory names seven enumerators, measured |
| R09 | HIGH: inventory evidence has no source tuple/digests and records an intrinsically stale head SHA | **ACCEPT-FIX** | inventory evidence binds to a source tuple and drops the head SHA |
| R14 | HIGH: digest covers only `cohort.py` + its test while output reads/rendering depend on four omitted modules | **ACCEPT-FIX** | duplicate of T3 R36 |
| R15 | HIGH: inventory and sessions follow marker symlinks; the other four reject non-regular markers | **ACCEPT-FIX** | every publisher rejects a non-regular marker |
| R16 | HIGH: inventory and sessions accept duplicate marker keys; the other four use rejecting hooks | **ACCEPT-FIX** | every publisher rejects duplicate marker keys |
| R17 | HIGH: sessions excludes the whole marker; measure excludes the whole generation/upstream block | **ACCEPT-FIX** | every census digest covers its provenance block minus its own self-referential key |
| R23 | MEDIUM: roadmap still says 17 restored columns → 92 published beside the measured 89/1068 result | **ACCEPT-FIX** | duplicate of T2 R28 |
| R25 | MEDIUM: memory cites old `export.py:250` anisotropy; code now uses one max-dimension scale and cohort docs/tests pin 9.9° | **ACCEPT-FIX** | memory states the max-dimension scale, not the retired per-axis one |
| R33 | LOW: shipped code comments retain contract/ruling provenance (`A02`, `A09`, `P17`) instead of standalone constraints | **REJECT** | an amendment id beside a stated constraint is a pointer to binding law, not provenance narrative; the Authoring rule prunes origin stories, and `# A02: the partition is measured, never declared` is the constraint |
| R34 | LOW: `steq.py` found 24 flags across four human surfaces, including 22 passive constructions in the two docs units | **POLISH** | LOW — 22 passive constructions on two human-facing documents; register row carrying the `steq.py` command |
| R35 | MEDIUM: duplicated publisher trust-root machinery drifted in exactly the four ways exposed by R14-R17 | **ACCEPT-LIMIT** | publisher extraction stays DECLINED by the standing ruling; the measured four-way drift is recorded on that polish row as new evidence |

### T6 MAIN — rows MAIN measured itself

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| X01 | MEDIUM — `pyproject.toml` declares 11 console scripts and `.venv/bin` carried 10: `pose-estimation-cohort` was absent because the environment was never re-synced after M2.8.5 added the entry point, so every operator-facing invocation in `docs/technical/cohort.md` failed in the shipped environment. Module invocation worked throughout. | **ACCEPT-FIX — CLOSED this session** | `uv sync --frozen` reinstalled the project alone (no third-party churn, `--dry-run` verified first); `.venv/bin` now carries **11/11**, and `env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" .venv/bin/pose-estimation-cohort --help` runs rc=0. A **bare** invocation still dies at `ImportError … GLIBC_2.43 not found` — the inherited-`PYTHONPATH` host-OpenVINO leak memory already records, which hits every entry point in this container and is not this row |
| X02 | Decisive gate under five concurrent reviewers read 1701 passed / 1 failed — `test_c8_08` `subprocess.TimeoutExpired` at 900 s, the contention failure memory already records. Collection 1702 confirms the recorded count. | **REJECT — confirmed by rerun** | gate rerun **alone**: **1702 passed in 1467.46 s (24:27), rc=0**, `test_c8_08` included. Contention, not a defect |

## Register carry-out

Out-of-contract observations that survive as `.agent/polish.md` rows. Filled at ruling time.
