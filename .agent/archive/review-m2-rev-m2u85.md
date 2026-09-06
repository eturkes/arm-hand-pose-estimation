# rev-m2u85 — M2 MILESTONE-REVIEW report

Scope: M2.8.3 + M2.8.5 — the cohort publisher: schema half and compute/publication half.
Base `384c15f`. Marker `REV-M2U85-DONE-1`. Validator: `python /run/host/home/eturkes/Projects/pose-estimation/.scratch/check_review_report.py .scratch/agents/rev-m2u85.md`

## Verdict table

Fixed check set: 42 rows. Verdicts bind to contract D01-D10, P01-P18, A01-A24 and the named scope sources.

| row | unit | check (predicate / surface) | verdict | finding <=1 line | evidence pointer |
| --- | ---- | --------------------------- | ------- | ---------------- | ---------------- |
| R01 | M2.8.3 | `FEATURES` is a unique `(level,column)` table equal to the run-measured finite-column partition; composition yields every and only the 89 published columns | pass | 89 entries = 89 unique keys = real-run finite partition; publication cross-check returned 89/3 | `cohort.py:242-287`; real publish census |
| R02 | M2.8.3 | feature construction uses the 20-family rule without a transcribed row copy; `describe()` projects the frozen six-field schema deterministically | pass | inspected 20-family decomposition, aliases, side and derivation rules; no second row table exists | `cohort.py:65-287` |
| R03 | M2.8.3 | every unit belongs to the closed seven-token vocabulary and family/suffix assignment implements A09+A20 | pass | 7/7 frozen tokens exercised; all 89 assignments independently pinned by family/suffix tests | `cohort.py:40-59,65-235`; `test_cohort.py:1492-1515` |
| R04 | M2.8.3 | every range is an admissible measurement domain derived by family/suffix rule, never a corpus min/max | pass | all 89 ranges derive from 20 family bounds plus suffix transforms; independent oracle covers each row | `cohort.py:61-235`; `test_cohort.py:1406-1421` |
| R05 | both | schema units and angle claim boundary remain truthful after M2.8.4 isotropic normalization; no stale anisotropy claim survives publication | fail | HIGH: 19 angle rows and the technical claim retain the pre-M2.8.4 anisotropy metadata | `cohort.py:41-44`; `cohort.md:66`; `test_cohort_review.py:12-18` |
| R06 | M2.8.5 | inventory, sessions, manifest and clinical headers are validated before row use; non-`ok` manifest assets contribute nothing and are counted excluded | fail | HIGH: manifest `event_id`/`camera_name` are unvalidated yet trusted; the contract exposes no non-`ok` exclusion census | `cohort.py:838-869`; `corpus_run.py:123-151`; `test_cohort_review.py:24-40` |
| R07 | M2.8.5 | run-level finite scan partitions every source column exactly once into published or `structurally_absent`, cross-checking `FEATURES` both directions | fail | MEDIUM: duplicate source headers publish duplicate census entries while the feature-table set check passes | `cohort.py:526-534`; `test_cohort_review.py:42-60` |
| R08 | M2.8.5 | cell rows are exactly the registry-derived `(task,side)` set, unique and canonical, with cell counts over manifest-`ok` assets independent of feature finiteness | pass | independent oracle and real publish confirm 12 unique canonical cells with 379 manifest-`ok` assets and feature-independent counts | `cohort.py:483-544`; `test_cohort.py:856-877`; real publish census |
| R09 | M2.8.5 | feature rows are exactly cells × `FEATURES`, unique and canonically ordered; each contributor count is bounded by its cell count | pass | 1068 unique rows = 12×89; independent product/count-bound oracle passes and order follows `FEATURES` then cells | `cohort.py:546-610`; `test_cohort.py:879-914`; real publish census |
| R10 | M2.8.5 | four-stage estimand groups by asset → event → subject → cohort without leakage or accidental pooling | fail | HIGH: event grouping keys come from forgeable manifest fields rather than the validated sessions placement | `cohort.py:503-524,838-869`; `test_cohort_review.py:24-40` |
| R11 | M2.8.5 | median, type-7 quartiles, arithmetic mean, sample SD, singleton SD and nine-decimal `_cell` serialization match A03/A19 | pass | independent byte oracle covers all six statistics; singleton and 9-decimal rules agree with A03/A19 | `cohort.py:408-436,570-601`; `test_cohort.py:1004-1054` |
| R12 | M2.8.5 | duplicating leaf rows changes leaf counts but not distribution statistics; whole-subject cloning changes all named contributor counts | pass | row-replication and subject-clone metamorphic tests move exactly the contracted count/statistic partitions | `test_cohort.py:1056-1168`; targeted pytest 11/11 |
| R13 | M2.8.5 | `NA`/`NaN`/`Inf`/`-Inf` are excluded; a run-published feature empty in one cell remains present with zero counts and empty statistics | pass | all four nonfinite tokens and one-cell emptiness are independently re-derived; product membership and zero counts remain intact | `cohort.py:416-423,510-525`; `test_cohort.py:1334-1403` |
| R14 | M2.8.5 | fewer than five contributing subjects empties all six distribution fields, including `view_dispersion`, while preserving every count | pass | four-subject fixture preserves five counts and empties all six distributions; A23 oracle carries the same floor | `cohort.py:584-605`; `test_cohort.py:1334-1361,1527-1555` |
| R15 | M2.8.5 | view dispersion uses events with ≥2 finite asset medians, population SD/abs(mean), excludes zero-mean events, and counts exactly that population | pass | independent CV oracle plus per-feature finite, zero-mean and single-view controls cover formula and population | `cohort.py:558-569`; `test_cohort.py:1170-1279`; targeted pytest 11/11 |
| R16 | M2.8.5 | no/single eligible multiview event and missing values in one camera yield the contract’s empty-or-finite result without changing event grain | pass | finite-per-feature, zero-mean and all-single-view fixtures cover reduced populations and empty serialization | `test_cohort.py:1203-1279`; targeted pytest 11/11 |
| R17 | M2.8.5 | census `rows_zero_values`, `rows_without_multiview`, and `rows_below_subject_floor` count feature rows with their documented meanings, including synthetic nonzero paths | pass | adversarial probe produced actual `(1,980,1)` = direct row-count oracle `(1,980,1)` | `cohort.py:602-626`; review synthetic census probe |
| R18 | M2.8.5 | population, estimand, column-census, collision, input-digest and generation child schemas exactly implement A04/A10/A22 | pass | generated marker keys, child sets, types and frozen literals are exact; real marker validates all domains | `cohort.py:358-374,608-626,888-910`; `test_cohort.py:751-851` |
| R19 | M2.8.5 | descriptor YAML has `{columns:[…]}`, canonical `(level,column)` order and one exact eight-field projection per feature | pass | parsed 89-row fragment is an exact unique projection in canonical order with the required root and key set | `cohort.py:645-678`; `test_cohort.py:1423-1490,1973-2002` |
| R20 | M2.8.5 | handwritten YAML quoting/escaping round-trips every current and adversarial label/range through the consumer-compatible parser | fail | LOW: literal YAML NEL (`U+0085`) is normalized to a space, so the JSON-scalar renderer is not round-trip total | `cohort.py:665-677`; `test_cohort_review.py:62-77` |
| R21 | M2.8.5 | descriptor `raw` values are unique, correctly namespaced, and external collision checking reports checked/source/counts without silent skip | pass | 89/89 internal raws unique and prefixed; live consumer expands to 219 unique raws, reports checked, and intersects at 0 | `cohort.py:680-727`; live consumer census |
| R22 | both | published CSV/YAML/JSON strings and keys satisfy the closed typed domains and contain no subject/event/camera/path/media identifier channel | pass | typed allowlist checks every field/key and input-derived needle scan is empty on all four generated artifacts | `test_cohort.py:751-851,1517-1580`; real publication |
| R23 | M2.8.5 | tree digest covers all three non-marker files plus the whole canonical marker payload minus only `generation.tree_digest` | pass | digest names and hashes each payload, then hashes canonical full marker body with only the self-reference removed | `cohort.py:729-757`; `test_cohort.py:1672-1719` |
| R24 | M2.8.5 | marker validation uses `lstat`, requires a regular non-symlink file, rejects duplicate JSON keys, and pins generator plus version | pass | direct regular-file, duplicate-key, symlink and both ownership-component controls cover the unverified marker | `cohort.py:785-829`; `test_cohort.py:1642-1697,1775-1840` |
| R25 | M2.8.5 | `validate_generation` rejects every missing/extra/half-published file and malformed schema using exactly `CohortError` | fail | MEDIUM: an arbitrary fifth file is outside the digest yet `validate_generation()` accepts the changed set | `cohort.py:970-1001`; `test_cohort_review.py:79-89` |
| R26 | M2.8.5 | independent edits to each CSV, descriptor YAML, census, provenance and digest are detected without one artifact masking another | pass | each of three payloads, both census directions, estimand/provenance and marker identity edits independently refuse as exact `CohortError` | `test_cohort.py:1699-1719,2019-2080`; campaign tamper matrix |
| R27 | M2.8.5 | output overlap is rejected in both ancestor directions against every input before publication or deletion | pass | realpath containment guard covers equal/inside/contains for inventory, sessions and run before validation or cleanup | `cohort.py:765-783,929-940`; `test_cohort.py:1721-1759` |
| R28 | M2.8.5 | a non-empty destination is replaceable only when its own regular marker proves this generator and exact version | pass | unmarked, foreign generator/version, directory marker and symlink marker controls all preserve destination bytes | `cohort.py:785-829`; `test_cohort.py:1761-1840` |
| R29 | M2.8.5 | publication writes a complete sibling staging tree, verifies ownership, swaps atomically, and sweeps debris only after the swap lands | fail | HIGH: current-PID staging/retiring paths are recursively removed before `_publish`, before any swap | `cohort.py:945-966`; `test_cohort_review.py:92-110` |
| R30 | M2.8.5 | every crash/kill window preserves either the prior complete generation or a recoverable complete staged generation; no complete generation is swept early | fail | HIGH: PID reuse lets the next failed attempt delete the only complete staged generation before replacement | `cohort.py:949-966`; `test_cohort_review.py:113-130` |
| R31 | M2.8.5 | orphan sweeping deletes only staging/backup trees whose own marker proves ownership; foreign dirs and symlinks cannot license recursive deletion | fail | HIGH: post-swap sweep recursively deletes every prefix-matching sibling without reading an ownership marker | `cohort.py:831-835,963`; `test_cohort_review.py:135-149` |
| R32 | M2.8.5 | recursive non-following pre/post snapshots prove inventory, sessions and run bytes/kinds/link targets unmoved, separately from semantic validation | fail | HIGH: only two sessions digests are rechecked; concurrent inventory and clinical-file edits publish successfully | `cohort.py:929-946`; `test_cohort_review.py:152-173` |
| R33 | M2.8.5 | output is byte-deterministic across hash seed, four locale states, timezone, umask, `-O`, output name, cwd and transient pid | pass | four subprocess environments plus moved cwd/pid/rebuilt corpus produce identical bytes; targeted P15/P16 set 3/3 green | `test_cohort.py:1922-1971`; `check_cohort_determinism.py:67-93` |
| R34 | M2.8.5 | republication is byte-idempotent and canonical order is stable rather than merely self-consistent | pass | owned-tree republication preserves every byte/kind/path and descriptor order differs from both label orders | `test_cohort.py:1973-2017`; targeted P15/P16 set 3/3 green |
| R35 | M2.8.5 | determinism campaign exercises six publication sweeps and 15 distinct consumer-boundary tamper classes with meaningful negative controls | fail | HIGH: committed campaign evidence has a stale source digest, so the documented command refuses at base before any sweep | `cohort_determinism_results.json:2`; `test_cohort_review.py:176-186` |
| R36 | M2.8.5 | campaign `source_digest` covers every source that can shape published bytes or acceptance, not only the main module and suite | fail | HIGH: digest binds only `cohort.py` + `test_cohort.py`, omitting campaign logic, direct modules, fixture helpers/goldens and external schema | `check_cohort_determinism.py:37-47`; `test_cohort_review.py:192-207` |
| R37 | M2.8.5 | CLI exposes and accepts exactly `--inventory --sessions --run --out`, and the pyproject entry point resolves to the reviewed worktree implementation | pass | worktree module path confirmed; real-corpus module CLI and synthetic four-flag smoke both publish and validate; entry point is exact | `cohort.py:1003-1024`; `pyproject.toml:59`; real publish |
| R38 | M2.8.5 | `.gitignore` covers `cohort` and all sibling staging/backup names without hiding unrelated tracked source | pass | `cohort` plus `cohort.*/` match output, staging and retiring probes; tracked module/tests/docs remain visible | `.gitignore:100-103`; `test_cohort.py:2159-2185` |
| R39 | M2.8.5 | technical documentation states the four-stage estimand, view population/limit, measured exclusion census, small-cell floor and claim boundary accurately | fail | HIGH: all required sections exist, but the angle boundary falsely says the corrected pipeline remains anisotropic with 9.9° error | `cohort.md:25-75`; R05 red test |
| R40 | M2.8.5 | entrypoints, architecture, tests and conventions indexes each register the publisher with correct command/module/campaign counts | pass | all four exhaustive indexes name the CLI/module/suite/campaign and the entry-point count is eleven | `entrypoints.md:3,17`; `architecture.md:28`; `tests.md:62`; `conventions.md:33` |
| R41 | both | all expected refusal paths normalize malformed input, parse, filesystem and arithmetic failures to `CohortError` without leaking incidental exceptions | fail | MEDIUM: invalid inventory, sessions and missing manifest leak three different exception types across the public publisher boundary | `cohort.py:852-855,927-931`; `test_cohort_review.py:211-227` |
| R42 | both | touched tests independently prove each contract branch, especially all-zero census counters, below-floor cells, YAML hazards, digest coverage and crash ordering | fail | HIGH: shipped 60/60 suite is green while 15 adversarial cases fail, and one P14 test asserts foreign recursive deletion as success | `test_cohort.py:1903-1920`; `test_cohort_review.py` |

## Detail

One `### R<nn>` section per `fail` row: `file:line`, divergence, breached predicate, impact,
acceptance check, red-test pointer.

### R05

- **Severity:** HIGH.
- **Evidence:** `src/pose_estimation/export.py:21,147-161` fixes `image-isotropic-maxdim` and one `max(frame_w, frame_h)` divisor. `src/pose_estimation/cohort.py:41-44` still assigns `deg_image_plane_uncalibrated`; `docs/technical/cohort.md:66-68` still says x/y use different divisors and retain a 9.9-degree error.
- **Divergence / breached predicate:** M2.8.4 D03+D08 reversed the mechanism behind M2.8.3 A09: the similarity transform preserves the true image-plane angle and requires `deg_image_plane`. R05/P10/P18 require schema and claim text to track that shipped coordinate identity.
- **Impact:** all 19 angle-family descriptors carry a false unit qualifier, and the human-facing claim reports a retired systematic error as present. Downstream interpretation is wrong although the numerical values are corrected.
- **Acceptance check:** keep seven unit tokens while replacing the retired angle token with `deg_image_plane`; update A09/A20 dependants and `cohort.md` to state one scalar, preserved image-plane angles, and the remaining non-anatomical/lens-calibration boundary.
- **Red test:** `tests/test_cohort_review.py:12-18` fails at base (`deg_image_plane_uncalibrated != deg_image_plane`) and then checks the documentation correction.

### R06

- **Severity:** HIGH.
- **Evidence:** `src/pose_estimation/corpus_run.py:123-151` validates only manifest asset ids, row count, uniqueness and disposition. `src/pose_estimation/cohort.py:846-869` builds the canonical placement map but uses it only as an id set, then trusts each manifest row’s `event_id` and `camera_name`. Contract §2 also requires every non-`ok` asset to be “counted as excluded”, while A22’s exact marker schema provides no such field.
- **Divergence / breached predicate:** R06 and the manifest trust-root clause require validated artifact bindings before any row is read. A total manifest can redirect an `ok` asset to another existing event/camera pair and still pass every current check. The excluded-asset conjunct is separately unrepresentable under the frozen output schema.
- **Impact:** a valid-looking manifest can select the wrong clinical artifact or escape the intended event grouping; the synthetic seed changed the published feature bytes while publication and validation stayed green. Real partial runs would also lose their exclusion census at this boundary.
- **Acceptance check:** cross-check each contributing manifest `(asset_id,event_id,camera_name)` against the validated sessions placement before artifact access. Rule and freeze an explicit non-`ok` disposition census, or amend §2 to name the upstream manifest as the retained count.
- **Red test:** `tests/test_cohort_review.py:24-40` substitutes another valid event/camera pair and fails because `cohort.run()` does not raise.

### R07

- **Severity:** MEDIUM.
- **Evidence:** `src/pose_estimation/cohort.py:526-534` preserves duplicate header names in `source_columns`/`published`, then reduces them to a `frozenset` for the `FEATURES` cross-check. A duplicate known frame column produced 90 `columns.published` rows but only 89 unique keys.
- **Divergence / breached predicate:** R07/P03 require a total disjoint partition in which every source column appears exactly once. Equal duplicated headers across all artifacts pass header equality and publish a duplicate census entry.
- **Impact:** the marker violates its own column-census key semantics, while all 89 feature rows remain apparently valid; downstream count/set checks can disagree silently.
- **Acceptance check:** reject any clinical header with duplicate field names before building value maps, and assert census list cardinality equals its key-set cardinality.
- **Red test:** `tests/test_cohort_review.py:42-60` duplicates one source header in every frame artifact and fails because publication succeeds.

### R10

- **Severity:** HIGH.
- **Evidence:** `src/pose_estimation/cohort.py:503-524` groups asset medians by `_Contributor.event_id`; `src/pose_estimation/cohort.py:867-868` sources that key and its artifact camera from the unvalidated manifest rather than the validated placement.
- **Divergence / breached predicate:** D02/R10 require the event stage to be the canonical sessions event. A manifest row can alias another event, causing two assets to enter the wrong event median and `subjects_of_event` attribution.
- **Impact:** the synthetic forged binding changed `cohort_features.csv` bytes with all validations green, so the publisher can silently change both weighting and values without changing the canonical registry or sessions tree.
- **Acceptance check:** derive event/camera from the validated sessions placement, require exact equality with the manifest for every `ok` row, and keep the independent four-stage oracle green.
- **Red test:** `tests/test_cohort_review.py:24-40` is red at base and green once the binding is checked before `_aggregate`.

### R20

- **Severity:** LOW.
- **Evidence:** `src/pose_estimation/cohort.py:665-677` emits every scalar through `json.dumps(..., ensure_ascii=False)`. PyYAML treats a literal `U+0085` inside the resulting YAML double-quoted scalar as a line break and loads it as an ASCII space.
- **Divergence / breached predicate:** R20/A08 require the hand-rendered YAML projection to preserve label bytes under the consumer-compatible parser. JSON syntax is a YAML subset only after accounting for YAML’s additional line-break code points.
- **Impact:** current labels are unaffected, but a valid future bilingual label containing NEL silently changes value at the consumer boundary while the generated descriptor still appears parseable.
- **Acceptance check:** escape YAML line-break code points (`U+0085`, and retain probes for `U+2028/U+2029`) while leaving ordinary Japanese glyphs literal; require `safe_load(render_descriptors(rows))` to equal the source projection.
- **Red test:** `tests/test_cohort_review.py:62-77` fails at base with `"before after" != "before\\x85after"`.

### R25

- **Severity:** MEDIUM.
- **Evidence:** `src/pose_estimation/cohort.py:970-1001` confirms the marker and three named payloads but never enumerates the output directory. Adding a regular fifth file leaves the recomputed digest unchanged and returns the marker successfully.
- **Divergence / breached predicate:** §4 says the generation holds four files, P17 requires half-published sets to refuse, and the standing trust-root rule permits exactly one internally unverified file: the marker. R25 requires both missing and extra members to fail.
- **Impact:** an edited generation can carry arbitrary bytes—including an identifier-bearing file—outside every digest while the prescribed consumer validation reports success. This weakens both integrity and redaction claims at tree scope.
- **Acceptance check:** require the non-following directory entry set to equal `{cohort_cells.csv, cohort_features.csv, descriptors.yaml, cohort.json}`, with all four regular and non-symlink, before digest verification.
- **Red test:** `tests/test_cohort_review.py:79-89` adds `unverified.txt` and fails because `validate_generation()` does not raise.

### R29

- **Severity:** HIGH.
- **Evidence:** `src/pose_estimation/cohort.py:945-952` derives siblings from `os.getpid()` and calls `_remove(staging)` plus `_remove(retiring)` before `_publish`; only the broad sweep at `:961-964` follows a successful promotion.
- **Divergence / breached predicate:** P14/R29 explicitly freezes “crash debris is swept only after the swap lands”. The exact-PID cleanup is an earlier recursive sweep and makes a filename sufficient authority to delete.
- **Impact:** a failed attempt can erase an unrelated directory before producing one byte. A PID collision/reuse also erases recoverable publisher debris before a replacement exists.
- **Acceptance check:** create a fresh collision-resistant staging sibling without deleting an existing path; treat any collision as foreign/refusal. Move all cleanup behind successful promotion and apply the same ownership proof as destination retirement.
- **Red test:** `tests/test_cohort_review.py:92-110` creates current-PID foreign staging debris, injects `_publish` failure, and fails because the sentinel is gone.

### R30

- **Severity:** HIGH.
- **Evidence:** the same pre-publish `_remove(staging)` at `src/pose_estimation/cohort.py:951` runs even when `out` is absent and that sibling is a fully valid cohort generation.
- **Divergence / breached predicate:** R30/P14 requires every kill window to leave the old destination or a recoverable complete staged/retiring generation until a new swap lands. PID reuse turns a prior complete stage into preflight garbage.
- **Impact:** if a crash left the complete stage as the sole generation, the next attempt deletes it and a subsequent pre-swap failure leaves no complete generation anywhere.
- **Acceptance check:** preserve every existing complete staged/retiring generation until a replacement is durably promoted; sweep only owned debris afterward, or recover/promote the valid stage first.
- **Red test:** `tests/test_cohort_review.py:113-130` creates and validates the exact PID-named stage, injects the next publish failure, and fails because the complete tree was deleted.

### R31

- **Severity:** HIGH.
- **Evidence:** `src/pose_estimation/cohort.py:831-835` iterates every sibling whose name starts with `cohort.staging.` or `cohort.retiring.` and passes it directly to recursive `_remove`; it never reads a marker. The shipped suite at `tests/test_cohort.py:1903-1920` positively expects unmarked directories to disappear.
- **Divergence / breached predicate:** P14/R31 and A14 make ownership a prerequisite for recursive deletion. A filename prefix is not ownership, and the suite has encoded the known-bad side of the contract.
- **Impact:** any foreign directory sharing that prefix is recursively deleted after an otherwise successful publish. A symlink itself is unlinked rather than followed, but an ordinary foreign directory loses all contents.
- **Acceptance check:** inspect each candidate non-followingly and delete only a regular-directory generation whose own regular, non-symlink marker proves this generator/version; leave unmarked/incomplete/foreign siblings intact. Rewrite the existing positive sweep fixture to carry valid owned markers.
- **Red test:** `tests/test_cohort_review.py:135-149` fails because a foreign sentinel directory is deleted after the swap.

### R32

- **Severity:** HIGH.
- **Evidence:** `src/pose_estimation/cohort.py:929-939` records and rechecks `sessions.tree_digest` plus `sessions.generation_digest` only. Inventory is validated once; the manifest is validated once; no recursive run/inventory snapshot exists. `input_digests` at `:941-946` is computed after aggregation and cannot witness pre/post equality.
- **Divergence / breached predicate:** A11/P12a require recursive non-following snapshots of all three upstream trees, and P12b requires inventory, sessions and manifest validation before and after. R32 freezes that split.
- **Impact:** inventory rows or already-read clinical artifacts can change mid-run; the publisher then emits a valid aggregate describing bytes that no longer exist, with provenance and consumer validation green.
- **Acceptance check:** snapshot relative path, kind, symlink target and regular-file digest for inventory, sessions and run before access and after aggregation; compare exactly. Repeat all semantic validators afterward and refuse before staging on any movement.
- **Red test:** `tests/test_cohort_review.py:152-173` mutates an inventory file and a clinical artifact after `_aggregate`; both parameter arms fail because publication succeeds.

### R35

- **Severity:** HIGH.
- **Evidence:** the default documented command returned rc=1 with `cohort determinism: source bytes moved` at base `384c15f`; `tests/cohort_determinism_results.json:2` does not equal the SHA-256 over the script’s declared source tuple. Removing the stale evidence lets the campaign itself pass 6 sweeps / 15 tamper classes in 2.2 s.
- **Divergence / breached predicate:** D06/P15 and R35 make this committed campaign the gitignored tree’s byte oracle. A stale source binding means the shipped oracle refuses before running and cannot support the recorded PASS claim.
- **Impact:** the milestone has no runnable committed determinism evidence at its own reviewed SHA; documentation and roadmap report a pass that the prescribed command cannot reproduce without deleting and rewriting evidence.
- **Acceptance check:** regenerate `tests/cohort_determinism_results.json` from the final committed source inputs, then assert its `source_digest` in the suite and rerun the unmodified documented command at committed state with rc=0.
- **Red test:** `tests/test_cohort_review.py:176-186` independently recomputes the declared digest and fails at base.

### R36

- **Severity:** HIGH.
- **Evidence:** `scripts/check_cohort_determinism.py:37-47` hashes only `src/pose_estimation/cohort.py` and `tests/test_cohort.py`. The campaign’s own code, `inventory.render_csv/render_json`, sessions/corpus validation and digest code, duplicate-key helper, `test_sessions` fixture helpers, two header goldens, and resolved `../rehab/schema/columns.yaml` all shape baseline bytes or tamper acceptance without joining the digest.
- **Divergence / breached predicate:** R36 and the standing committed-evidence rule require the evidence binding to cover its producer and all byte-shaping fixture dependencies. The current tuple is not the dependency closure it claims.
- **Impact:** an omitted source can change every generated byte or weaken a tamper class while the recorded `source_digest` remains valid. The script then overwrites `published_digests` with a new PASS instead of detecting drift from the prior oracle.
- **Acceptance check:** derive and hash a closed, explicit dependency set including the campaign, publisher dependency modules, fixture helpers/data and external schema state; assert the set in tests. Before rewriting evidence, compare regenerated published digests with the committed baseline and require an intentional regeneration mode for drift.
- **Red test:** `tests/test_cohort_review.py:192-207` requires the internal producer/fixture closure and fails because only two paths are declared.

### R39

- **Severity:** HIGH.
- **Evidence:** `docs/technical/cohort.md:66-68` says the pipeline divides x/y by different dimensions and retains a measured 9.9-degree error; `src/pose_estimation/export.py:21,147-161` ships one max-dimension scalar.
- **Divergence / breached predicate:** P18/R39 require an accurate claim boundary, not mere phrase presence. M2.8.4 D03+D08 explicitly reversed this text and assigned M2.8.3 the metadata amendment.
- **Impact:** the primary human-facing interpretation guide tells consumers that corrected image-plane angles carry a systematic distortion that no longer exists.
- **Acceptance check:** state isotropic max-dimension normalization, preservation of true image-plane angles/ratios, and the remaining projection, lens-calibration and non-anatomical limitations; remove the retired 9.9-degree current-error claim.
- **Red test:** `tests/test_cohort_review.py:17-27` shares R05’s regression and checks the documentation arm after the unit token.

### R41

- **Severity:** MEDIUM.
- **Evidence:** `src/pose_estimation/cohort.py:927-931` calls inventory and sessions validators directly; `_contributors` catches only `ManifestError` after `read_manifest`. Removing the three trust-root files raises `InventoryError`, `SessionsError`, and `FileNotFoundError`, respectively.
- **Divergence / breached predicate:** A16/R41 define `CohortError` as the exact public refusal class for every expected publisher and consumer rejection. Upstream-invalid and missing-manifest paths are routine refusals, not unexpected process faults.
- **Impact:** API consumers and the CLI must catch dependency-specific/private exceptions despite the public boundary, and one missing-file path escapes as raw filesystem detail. Error handling differs by which trust root failed.
- **Acceptance check:** wrap inventory/sessions validation failures, manifest read/validation failures, clinical decode/CSV failures and external-schema parse failures at their cohort boundary; preserve causes while raising exactly `CohortError` for expected invalid input.
- **Red test:** `tests/test_cohort_review.py:211-227` has three red parameter arms naming the leaked classes.

### R42

- **Severity:** HIGH.
- **Evidence:** the shipped `tests/test_cohort.py` remains 60/60 green, while `tests/test_cohort_review.py` produces 15/15 failures in 1.04 s. `tests/test_cohort.py:1903-1920` explicitly creates unmarked sibling directories and requires their recursive deletion—the inverse of A14/P14 ownership.
- **Divergence / breached predicate:** R42 and the kernel-tier contract require independent negative controls for synthetic-only paths and publication crash states. Current coverage both omits contract-bearing branches and positively blesses one destructive violation.
- **Impact:** the decisive suite certified stale metadata, forged event binding, duplicate census keys, extra unverified files, three data-loss windows, incomplete upstream witnesses, stale/incomplete determinism evidence and exception leakage.
- **Acceptance check:** land the review regressions with the implementation fixes; replace the unowned-debris expectation with owned-marker fixtures; add direct nonzero oracles for all three row censuses; require original 60 plus all review controls green before re-closing the units.
- **Red test:** `tests/test_cohort_review.py` is the committed 15-case red battery; each failing row above names its narrower pointer.

## Register

- **X01 — external descriptor state is an unbound implicit input.** Evidence: identical explicit inventory/sessions/run inputs published byte-identical CSV/YAML artifacts but different `cohort.json` from two identical source checkouts; worktree module location reported `{checked:false,n_external:0}`, primary location reported `{checked:true,n_external:219}`. `src/pose_estimation/cohort.py:680-697` resolves `../rehab` from `__file__`, while §2/P15 omit that state from inputs/provenance. **Acceptance check:** either make the consumer schema an explicit digested input under a CLI-contract amendment, or remove it from generation bytes and run collision checking as a separate hermetic integration gate; add a checkout-location differential control.
