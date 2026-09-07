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
**43 ACCEPT-FIX · 9 ACCEPT-LIMIT · 2 POLISH · 3 REJECT.** The 2 POLISH rows are filed in
`.agent/polish.md`; the ACCEPT-LIMIT and REJECT rows are closed here.

**Wave 2 — fix application, session 3 of N. 40 of the 43 ACCEPT-FIX rows are CLOSED; 3 stay open.**
Per track: T1 **10/10** · T2 6/7 · T3 **14/14** · T4 3/4 · T5 6/7 · T6 **1/1**.
The three reviewer tips carry committed red batteries, and those are the acceptance checks in
executable form — restore them with `git show archive/m2-review-<name>:tests/<file> > tests/<file>`
and drive them green. Measured at restore: **27 red** over
`test_review_m2u82.py` (10) · `test_m2u84_review.py` (2) · `test_cohort_review.py` (15).
**All 27 are green now** — session 2 restored the parked `tests/test_review_m2u82.py` from
`.scratch/review-reds/`, drove its 10 reds green and committed it in `tests/`, so nothing stays
parked and the decisive gate carries the whole battery.

**A restored red is evidence, not law — three of the 17 needed correcting before they could grade
anything**, all the M2.8.2 A10 class "a case that grades a stand-in grades nothing".
`test_open_capture_refuses_failed_orientation_request` (T2 R21) contradicted its own ledger wording
and the *test* was right; `test_r35_committed_campaign_evidence_matches_its_declared_sources`
transcribed a two-file digest list that R36's own `SOURCES` expansion falsifies, so it now re-derives
from the campaign's `_source_digest()` through `runpy`; `test_r32_concurrent_upstream_mutation_is_
detected` parametrized two of the three inputs its acceptance check names, so `sessions` was added.
Read each red against the shipped contract *and* against its own ledger row before crediting it.

**Remaining 3 — one shape: an evidence document that does not bind to the digests of what produced
it.** All three force a republish or a regeneration, which is why they were deferred as a cohort.
- **T2 R23** — `tests/isotropy_angle_fidelity_results.json` pins 8 sample ids and nothing else; it
  needs script, validated-registry generation, run-tree and effective-config digests.
- **T4 R37** — the cohort census must record the external descriptor digest.
- **T5 R17** — a census digest must cover its provenance block minus its own self-referential key.
  **Design settled, session 3, unimplemented.** Sessions half = a NEW self-digest key in the marker,
  joining `GENERATION_KEYS`, following `cohort.tree_digest(out_dir, marker)`'s A24 idiom; never a
  changed `tree_digest` signature, because `scripts/corpus_run_2d.py:513,549` and
  `scripts/pilot_corpus_run.py:650,668` call it with one argument and `corpus_run_2d` computes it
  *before* `validate_generation`, so a signature change moves the error path. Measure half =
  `measure.manifest_digest` includes `generation` minus `generation.manifest`. Blast radius measured:
  sessions republish **0.5 s**; `sessions.tree_digest` excludes the marker, so `cohort`'s recorded
  value is unaffected; `output/corpus-2d/run_report.json` records only booleans, so the 7.828 h
  corpus-run record is not falsified. The measure half needs a manifest-only rewrite (read each axis
  CSV, `write_axis` the same rows and provenance → byte-identical CSVs, fresh digest, **no decode**),
  then republish qualification (~30 s), calibration_qc and cohort (~14 s).

**Session 2 datum — one reviewer's own two reds contradicted each other, and the fixture lost.**
R20's byte-idempotence red wrote a synthetic diagnostics row with no `video` column while R25's red
demands exactly that column be checked for ownership; applying both left R20 red on
`artifacts_owned`, not on idempotence. The R20 fixture gained the column, because
`write_source_diagnostics` always emits it and a fixture omitting it grades `wrong_diag` rather than
the predicate its case exists for. **Reds from one reviewer are no more mutually consistent than
reds from two** — restore the whole battery and run it as a set before crediting any single row.

**Session 2 datum — a source-digest tripwire is the real cost of touching a shared module.** Editing
`multicam.py` for the T1 containment guard fired
`test_calibration_qc.py::test_committed_determinism_evidence_matches_its_exact_source_tripwire`, and
both `check_calibration_qc_determinism.py` and `check_qualify_determinism.py` then REFUSED to
regenerate over a stale result — by design. `rm` the two JSON files, rerun both campaigns, and do it
**last**, after every source and test edit: regeneration is a fixpoint. Regenerated green at
**21 sweeps / 18 tampers / 39 PASS / 0 FAIL** and **40 sweeps / 40 passed / 19 tamper classes /
0 tamper failures**.

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

**Decisive gate at wave 1 close, run alone from committed state: `1702 passed in 1467.46s`, rc=0.**
The concurrent run during the wave read 1701 passed / 1 failed on `test_c8_08`'s 900 s subprocess
timeout — five worktree gates plus MAIN's on 8 cores. Run the decisive gate alone.
**Wave 2 session 1 close, again alone: `1720 passed in 1307.40s (21:47)`, rc=0** — 1702 plus the 18
restored review cases now in `tests/`.

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

All ten CLOSED in wave-2 session 2; the acceptance checks are `tests/test_review_m2u82.py`, restored
from `archive/m2-review-rev-m2u82` into `tests/` and now **10/10 green** in the decisive gate.

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R14 | HIGH — raw, unvalidated qualification cells become both report strata and allowed strings, so an edited identifier-shaped label is emitted under all-green verdicts. | **ACCEPT-FIX — CLOSED** | new `pilot._assert_sources_validated` calls `inventory.validate_generation` + `qualify.validate_generation(…, inventory_dir=…)` before `_load_assets`, so no cell reaches a stratum label or the allowlist until its own generation proves out. `test_pilot_refuses_unvalidated_source_tables_before_publication` green |
| R15 | MEDIUM — a fresh pilot rerun never clears its event output, so stale diagnostics/clinical files can be credited when the current source produces none. | **ACCEPT-FIX — CLOSED** | `_run_event` `rmtree`s `out/<event>` before launch, matching the driver's `_attempt_event`. `test_pilot_rerun_clears_the_selected_event_before_launch` green |
| R20 | MEDIUM — inference is skipped, but the shipped driver rewrites `run_report.json` with invocation-local attempt/time fields, breaking P05 byte idempotence. | **ACCEPT-FIX — CLOSED** | took the first arm — the report is now a function of STATE, never of the pass. `run` publishes `events_complete` / `events_failed` / `events_unattempted` read from the markers instead of `events_attempted` / `attempts_*`, and `throughput.invocation_wall_s` is deleted; the invocation's own counters print to stdout. `test_complete_resume_keeps_every_published_output_byte_identical` green. **Its fixture needed one correction** — see the session datum above |
| R21 | HIGH — `process_source` swallows `KeyboardInterrupt`; a child-only interruption exits 0, so the driver can run R and mark a partial source complete. | **ACCEPT-FIX — CLOSED** | `run.py`'s handler prints then `raise`s; the `finally` arm still releases the capture and writes the decoded-prefix diagnostics. `test_interrupted_source_propagates_after_writing_diagnostics` green. Blast radius = one shipped case, `test_corpus_run_preconditions.py::test_p11_interrupted_source_still_publishes_its_decoded_prefix`, which asserted the swallow and **halted the whole pytest session** once the interrupt propagated; rewritten around `pytest.raises`, keeping every P11 diagnostics assertion |
| R25 | MEDIUM — `_artifacts` accepts a one-row diagnostics file whose `video` belongs to another event/camera, then attributes its counters to this `ok` asset. | **ACCEPT-FIX — CLOSED** | `_artifacts` now requires `diagnostics[0]["video"] == f"{event_id}/{camera_name}"`. **Measured on the shipped corpus before shipping, because a wrong key refuses every asset: 379/379 diagnostics files match, 0 mismatch, 0 malformed.** `test_artifact_validator_rejects_a_diagnostic_owned_by_another_source` green |
| R27 | CRITICAL — `--out` can create logs inside the session tree before validation, and `--report` can write there after both digest snapshots, leaving the input mutated or a green witness stale. | **ACCEPT-FIX — CLOSED** | `multicam._published_root`'s containment test is extracted as public `multicam.published_overlap(target, tree)` and `_resolve_session_output` now calls it, so there is one rule rather than a fifth copy. Both scripts gain `_assert_sinks_outside(sessions, out, report)` — run before the first mkdir and re-run immediately before the report write. `test_driver_refuses_every_sink_overlapping_the_published_sessions[out,report]` green, both write-free |
| R28 | MEDIUM — `abs(NaN - derived) > 1e-9` is false, so a nonfinite stored rate passes `stored_rate_equals_its_derivation`. | **ACCEPT-FIX — CLOSED** | the comparison is inverted to `not abs(...) <= 1e-9`, so NaN falls on the mismatch side. `test_cfr_derivation_rejects_a_nonfinite_stored_rate` green |
| R29 | HIGH — shipped key validation remains `allowed OR regex`, while the alleged membership oracle walks only local fixtures; an unknown identifier-shaped key passes production. | **ACCEPT-FIX — CLOSED** | `FIELD_NAME` deleted. `_assert_redacted(payload, allowed, fields)` now raises from a new `redaction_violations`, whose key clause is membership in `fields | allowed`; each script declares a frozen `REPORT_FIELDS` (pilot 65 names, driver 66), spelled rather than harvested from the payload so it cannot self-authorise. **The suite's `_violations` oracle is deleted and now delegates to `redaction_violations`**, so guard and oracle are one implementation. `test_shipped_redaction_guard_rejects_an_unknown_identifier_shaped_key` green |
| R30 | HIGH — clinical-failed event wall enters `_corpus_wall`, its frames are excluded from `_artifacts`, yet the report labels the mixed 100/300 population `sample: corpus`. | **ACCEPT-FIX — CLOSED** | `_corpus_wall` sums markers of COMPLETE events alone, so `events_measured == events_total` now means one population behind numerator and denominator. `test_throughput_never_labels_mixed_frame_and_wall_populations_as_corpus` green |
| R32 | HIGH — P05/P06, P09, P11, P13, and P14 chiefly grade `_Driver`, local verdict walkers, or source/prose regexes; 105 green cases miss nine shipped-code reds. | **ACCEPT-FIX — CLOSED** | the nine reds all execute shipped symbols (`pilot.main`, `pilot._run_event`, `driver.main`, `driver._artifacts`, `driver._cfr`, `run.process_source`, `pilot._assert_redacted`) and are green against `src/` + `scripts/`. The P13 stand-in is gone with R29; `test_p13_the_analog_declares_the_shape_backstop_its_runtime_guard_applies` is **deleted**, since the backstop it pinned no longer exists |

### T2 `rev-m2u84` — M2.8.4

28 rows, 7 fail. Report `.agent/archive/review-m2-rev-m2u84.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (21): R01 R02 R03 R04 R05 R06 R07 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R22 R25 R26 R27

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R08 | HIGH — deleting the report key leaves P08 green, so N7 and the artifact-identity conjunct are not witnessed | **ACCEPT-FIX — CLOSED** | new `_load_driver()` imports `scripts/corpus_run_2d.py` as a module and `test_p08_report_payload_carries_the_normalisation_identity` runs `driver.main()` in analyse-only mode over a synthetic tree, asserting the EMITTED payload carries the token — the prior case compared module constants. Mutant verified: deleting `"coord_normalization": COORD_NORMALIZATION,` (`scripts/corpus_run_2d.py:605`) FAILS it |
| R09 | MEDIUM — replacing both helper calls with duplicated `max` formulas leaves P09 green; A07's spy was not built | **ACCEPT-FIX — CLOSED** | `_four_coordinate_paths(point, frame_h, frame_w)` extracted from the P09 body (body / matched-hand / fallback-hand / hands-only), then `test_p09_every_coordinate_path_routes_through_the_coord_scale_helper` monkeypatches `export.coord_scale` to a sentinel divisor of 7.0 and requires every emitted coordinate to reflect it. Mutant verified: both `scale = coord_scale(...)` sites replaced by `float(max(frame_w, frame_h))` FAILS it |
| R20 | MEDIUM — three unqualified `[0,1]` claims are false: a finite off-frame point exports as `(1.1,-0.05)` | **ACCEPT-FIX — CLOSED** | all three claims in `export.py` scoped to in-frame landmarks — the `coord_scale` docstring, the `frame_to_rows` docstring, the `read_csv_keypoints` `kps` description — each stating that a finite off-frame prediction is preserved rather than clipped. `test_p03_finite_off_frame_landmarks_are_preserved_outside_the_unit_range` exports `(110, -5)` in a 100x50 frame as `(1.1, -0.05)` |
| R21 | HIGH — `VideoCapture.set(...,1)` return/read-back is ignored; a rejected request returns an unreleased usable capture | **ACCEPT-FIX — CLOSED** | `video_io.open_capture` now reads the `set` return, warns, releases, returns `None`; `test_m2u84_review.py::test_open_capture_refuses_failed_orientation_request` green. The row said "raises" and the reviewer's own red asserts `None` — **the red is right**: `open_capture`'s contract is "the open capture or `None`", so raising breaks every caller. `probe_container` left alone: it releases in `finally` and publishes `orientation_auto` as a cell, so a refusal is already visible there. **Measured before shipping, because a wrong answer refuses every video**: `set` returns `True` on **382/382** real corpus assets, 0 unopenable, and `open_capture` still returns a capture that reads a frame. Recorded as `contract-m2u84.md` **A16** |
| R23 | HIGH — unreplayable JSON pins sample IDs only; it carries no script, registry-generation, run-tree, manifest, or config digest | **ACCEPT-FIX** | the fidelity JSON binds to script, registry generation, run tree and config digests |
| R24 | HIGH — code, tests, 19 descriptors, and docs still publish the pre-fix anisotropic token/claim | **ACCEPT-FIX — CLOSED** (isotropy cluster: T2 R24 = T3 R05 = T3 R39 = T4 R41, one fix) | `cohort.UNIT_DEG` → `"deg_image_plane"` (`cohort.py:44`); `tests/test_cohort.py` `_UNIT_CONSTANTS` + P18 phrase list follow (`"9.9"` → `"one scalar"`: the retired pipeline's error is provenance); `docs/technical/cohort.md` angle boundary rewritten to one scalar / similarity map / no lens correction. **Republished `cohort/` on the real corpus under the accelerator recipe: 12 cells, 1068 feature rows, 89 published / 3 excluded — reconciles exactly with the recorded census**, and `descriptors.yaml` carries 19 `deg_image_plane`, 0 `deg_image_plane_uncalibrated`. `test_cohort_review.py::test_r05_…` green |
| R28 | MEDIUM — roadmap still says body restores 17/92 and its schema record freezes pre-fix anisotropy, contradicting later 14/89/isotropic state | **ACCEPT-FIX — CLOSED** (= T5 R23) | corrected in place, which is this project's standing handling of a frozen record's refuted premise (roadmap §M2.8.3, A12). Roadmap: the wave-1 finding reads past-tense, A20's inheritance line and A09's determination now name `deg_image_plane` and state that M2.8.4 made the map a similarity. `contract-m2u83.md`: **A25** supersedes A09's token with pointers planted at §5 P10, at A09's head and at A20; A12's `92 → 1104` prediction is marked discharged-and-wrong against the measured **89 published / 1068 rows**, and A18's `92 × 12` restated as `89 × 12`. Verified by sweep — every surviving `deg_image_plane_uncalibrated` sits in a frozen reviewer report or under an explicit supersession note, and `14 of the 17, never all 17` is what the roadmap says in both places |

### T3 `rev-m2u85` — M2.8.3, M2.8.5

42 rows, 15 fail. Report `.agent/archive/review-m2-rev-m2u85.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (27): R01 R02 R03 R04 R08 R09 R11 R12 R13 R14 R15 R16 R17 R18 R19 R21 R22 R23 R24 R26 R27 R28 R33 R34 R37 R38 R40

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R05 | HIGH: 19 angle rows and the technical claim retain the pre-M2.8.4 anisotropy metadata | **ACCEPT-FIX — CLOSED** | see T2 R24 — one fix closed both |
| R06 | HIGH: manifest `event_id`/`camera_name` are unvalidated yet trusted; the contract exposes no non-`ok` exclusion census | **ACCEPT-FIX — CLOSED** | A28: `_contributors` reads `event_id`/`camera_name` from the validated `sessions/placements.csv` and refuses a manifest that disagrees. `test_r06_manifest_event_and_camera_must_match_sessions` green |
| R07 | MEDIUM: duplicate source headers publish duplicate census entries while the feature-table set check passes | **ACCEPT-FIX — CLOSED** | A27: new `_assert_unique_header` wired into `_read_table` + `_read_artifact`. `test_r07_duplicate_source_header_is_refused` green |
| R10 | HIGH: event grouping keys come from forgeable manifest fields rather than the validated sessions placement | **ACCEPT-FIX — CLOSED** | closed by the same A28 change as R06 |
| R20 | LOW: literal YAML NEL (`U+0085`) is normalized to a space, so the JSON-scalar renderer is not round-trip total | **POLISH — CLOSED by fix** | fixed rather than deferred, since the renderer is 3 lines from total: new `_json_scalar` escapes U+0085/U+2028/U+2029 so `render_descriptors` round-trips through PyYAML, source kept ASCII-only via `chr(codepoint)`. `test_r20_descriptor_renderer_round_trips_yaml_line_break_codepoints` green |
| R25 | MEDIUM: an arbitrary fifth file is outside the digest yet `validate_generation()` accepts the changed set | **ACCEPT-FIX — CLOSED** | A26: `tree_digest` folds the sorted entry-name set (marker excluded, so staging and published digest alike) into the digest → a fifth file fails `validate_generation`. `test_r25_extra_publication_entry_is_refused` green |
| R29 | HIGH: current-PID staging/retiring paths are recursively removed before `_publish`, before any swap | **ACCEPT-FIX — CLOSED** | A30, one rewrite closing R29+R30+R31: no pre-run `_remove` of pid-named siblings; `staging`/`retiring` now come from `tempfile.mkdtemp(prefix=…, dir=out.parent)` so nothing else can hold the name; cleanup touches only this invocation's paths; `out.parent.mkdir(parents=True, exist_ok=True)` precedes mkdtemp |
| R30 | HIGH: PID reuse lets the next failed attempt delete the only complete staged generation before replacement | **ACCEPT-FIX — CLOSED** | same A30 rewrite; `test_r30_complete_pid_reused_staging_survives_until_a_new_swap` green |
| R31 | HIGH: post-swap sweep recursively deletes every prefix-matching sibling without reading an ownership marker | **ACCEPT-FIX — CLOSED** | stronger than the acceptance check asked: `_sweep_orphans` **deleted** from `cohort.py` rather than taught to read a marker — a prefix-name sweep is the wrong instrument, since it deleted foreign trees *and* complete generations. mkdtemp names are unguessable, so there is nothing to sweep. `test_r31_successful_swap_preserves_unowned_sibling_directory` green |
| R32 | HIGH: only two sessions digests are rechecked; concurrent inventory and clinical-file edits publish successfully | **ACCEPT-FIX — CLOSED** | A29: new `_snapshot(root)` (recursive, non-following: link target / dir / file digest) over **all three** inputs, taken before aggregation and re-checked after. The sessions digest re-comparison is **deleted, not kept beside it** — a snapshot over raw bytes subsumes a digest that sees only the files its own contract names, which is A26's gap one publisher up. `test_r32_concurrent_upstream_mutation_is_detected[inventory,sessions,run]` green; the reviewer's case parametrized `inventory`/`run` alone, so `sessions` was added to make the case grade the acceptance check as written |
| R35 | HIGH: committed campaign evidence has a stale source digest, so the documented command refuses at base before any sweep | **ACCEPT-FIX — CLOSED** (= T4 R47) | regenerated after every other cohort fix landed: `scripts/check_cohort_determinism.py` reads **PASS — 6 sweeps, 15 tamper classes, 3.0 s**, rc=0 |
| R36 | HIGH: digest binds only `cohort.py` + `test_cohort.py`, omitting campaign logic, direct modules, fixture helpers/goldens and external schema | **ACCEPT-FIX — CLOSED** (= T5 R14) | `SOURCES` expanded 2 → 10: the campaign script itself, `cohort/corpus_run/inventory/qualify/sessions.py`, `test_cohort.py`, `test_sessions.py`, both `2d_idx_clinical{,_windows}.csv` goldens. `test_r36_campaign_digest_covers_its_fixture_and_publisher_dependencies` green |
| R39 | HIGH: all required sections exist, but the angle boundary falsely says the corrected pipeline remains anisotropic with 9.9° error | **ACCEPT-FIX — CLOSED** | see T2 R24 |
| R41 | MEDIUM: invalid inventory, sessions and missing manifest leak three different exception types across the public publisher boundary | **ACCEPT-FIX — CLOSED** | upstream `inventory`/`sessions` validation wrapped so only `CohortError` crosses the boundary; `OSError` from a missing manifest wrapped too. `test_r41_invalid_inputs_raise_exactly_cohort_error[inventory,sessions,manifest]` green |
| R42 | HIGH: shipped 60/60 suite is green while 15 adversarial cases fail, and one P14 test asserts foreign recursive deletion as success | **ACCEPT-FIX — CLOSED** | the 15 cases ship as `tests/test_cohort_review.py`, all green; the shipped `test_p14_successful_swap_sweeps_staging_and_retiring_debris` — which asserted foreign recursive deletion as success — was rewritten as `test_p14_successful_swap_preserves_every_sibling_it_does_not_own` |

### T4 `audit-m2` — M2.7.3-M2.8.5

50 rows, 12 fail. Report `.agent/archive/review-m2-audit-m2.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (38): R01 R02 R03 R04 R05 R06 R07 R08 R09 R10 R11 R12 R13 R16 R17 R18 R19 R20 R21 R22 R25 R26 R27 R29 R30 R32 R33 R34 R35 R36 R38 R39 R42 R43 R45 R46 R48 R49

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R14 | 8/9 fire: deleting one of 13 local-decision labels leaves P05 green. | **ACCEPT-FIX — CLOSED** | P05 now checks both directions like P03: new frozen `ABSENCE_SECTIONS` binds each absence to the sections exercising it, and each of the preamble, the 5 rows and the 7 governed spans is checked separately, with the label total DERIVED as their sum so a stray label fails too. **13/13 sweep**: deleting each label in turn fails P05 with its own message. Nine-control harness rebuilt at `.scratch/nc_m2u74.py` (the original did not survive its worktree) — **9/9 fire**, document restored byte-identical (`55eb769a1768`) |
| R15 | UNREPRODUCIBLE: the cited review ledger is absent from committed state. | **ACCEPT-LIMIT** | the roadmap cites no `.scratch/` report as evidence for a durable claim; cited reports are archived or the citation drops |
| R23 | Six components reproduce, but the only retained run gives 540 windows, not 572. | **ACCEPT-LIMIT** | MAIN re-derived it: 572 is the M2.8.1 PRE-fix pilot population and 540 the post-fix one, both recorded (`roadmap.md:801,944`); the pre-fix tree is deleted, so the figure is historical |
| R24 | UNREPRODUCIBLE: pre-fix logs/report/tree are absent; current retained output is post-fix. | **ACCEPT-LIMIT** | the record labels the pre-fix measurement historical and unreproducible, never re-derivable |
| R28 | Partition/CFR/540 replay; historical wall and latency range do not because its report/logs are absent. | **ACCEPT-LIMIT** | as R24 |
| R31 | UNREPRODUCIBLE: pre-fix run tree/report is deleted; current marker carries only the 7.828 h NPU run. | **ACCEPT-LIMIT** | as R24 |
| R37 | Marker retains 219/0 collisions, but not the external descriptor bytes, digest, composition or prefix census. | **ACCEPT-FIX** | the census records the external descriptor digest, so `descriptor_collision` is re-checkable |
| R40 | UNREPRODUCIBLE: checker rc=1 because its pre-fix input tree was deleted; JSON only records the values. | **ACCEPT-LIMIT** | as R24, and closed by T2 R23's source binding |
| R41 | Run is isotropic, but cohort code/docs still publish `deg_image_plane_uncalibrated` and claim anisotropy. | **ACCEPT-FIX — CLOSED** | see T2 R24 |
| R44 | All census values and ~180 KB reproduce, but elapsed was 30.59 s, not 14.4 s. | **REJECT** | MAIN re-measured under quiescence: **13.26 s** against the recorded 14.4 s, so the claim reproduces; 30.59 s was five reviewers sharing 8 cores |
| R47 | Stale-green barrier fires: recorded source digest differs from current source, so checker exits 1 before sweeps. | **ACCEPT-FIX — CLOSED** | see T3 R35 — one regeneration closed both. Two sibling evidence files were stale for the same reason (`video_io.py` moved) and were regenerated with it: `tests/qualify_determinism_results.json` (**40 sweeps / 40 passed / 19 tamper classes / 0 tamper failures**) and `tests/calibration_qc_determinism_results.json` (**21 sweeps / 18 tampers / 39 PASS / 0 FAIL**) |
| R50 | Label census reproduces 75; the 14-missing-font claim has no consumer font bytes or digest in this repo. | **ACCEPT-LIMIT** | the glyph claim states its measurement is against the consumer repo, outside this boundary |

### T5 `rev-m2-cross` — cross-unit + M2.7.3, M2.7.4

39 rows, 11 fail. Report `.agent/archive/review-m2-rev-m2-cross.md` carries the detail sections.

**`pass` rows, ruled ACCEPTED as checked** (28): R01 R02 R03 R04 R05 R06 R07 R10 R11 R12 R13 R18 R19 R20 R21 R22 R24 R26 R27 R28 R29 R30 R31 R32

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| R08 | LOW: P16 independently enumerates all 5 group-QC goldens, falsifying the six-place budget | **ACCEPT-LIMIT** | `.claude/rules/fixtures.md` names seven enumerators, measured |
| R09 | HIGH: inventory evidence has no source tuple/digests and records an intrinsically stale head SHA | **ACCEPT-FIX — CLOSED** | `check_inventory_determinism.py` gains a spelled-out 4-entry `SOURCE_FILES` (checker, `__init__.py`, `inventory.py`, `video_io.py`), `source_digests()`, `stale_source_mismatches()` and a refusal before any write, **verified end to end: rc=2, the mismatched file named, the stale result unmodified**; `tested_head` is gone and `schema_version` is 2. Two committed cases in `tests/test_inventory_review.py` — one recomputes every digest from the worktree and asserts `tested_head` absent, one grades the refusal on both arms plus the no-`source_sha256` arm. `run_inventory_mutations.py`'s own head SHA rides the pri-1 `.agent/polish.md` regeneration row |
| R14 | HIGH: digest covers only `cohort.py` + its test while output reads/rendering depend on four omitted modules | **ACCEPT-FIX — CLOSED** | duplicate of T3 R36 |
| R15 | HIGH: inventory and sessions follow marker symlinks; the other four reject non-regular markers | **ACCEPT-FIX — CLOSED** | one shared loader `inventory.read_marker` (`lstat` + `stat.S_ISREG` -> `OSError`, then `json.loads` with a duplicate-rejecting hook), placed in `inventory.py` because it is the import-graph root, so the fix adds ZERO import edges and is not the DECLINED `publication.py` extraction (T5 R35). `sessions.py` routes BOTH readers through it — `_assert_owned` (the one with teeth: publication deletes what the marker says it owns) and `validate_generation`; `qualify._read_marker` collapses to a delegation. New `tests/test_publisher_trust_roots.py`, 16 cases: 5 non-regular kinds x 2 modules, each paired with the untouched tree validating |
| R16 | HIGH: inventory and sessions accept duplicate marker keys; the other four use rejecting hooks | **ACCEPT-FIX — CLOSED** | `inventory.reject_duplicate_keys` is the shared `object_pairs_hook`, reached through `read_marker`; top-level and nested duplicate-key cases for both modules. Two deliberate non-migrations: `qualify._reject_duplicate_keys` keeps its NAME as a one-line delegation because `scripts/run_calibration_qc_mutations.py:99,320-321` string-match `object_pairs_hook=qualify._reject_duplicate_keys` under `assert s.count(old)==1`; and `measure._object_pairs` stays, because it raises `MeasureError(reason="manifest_duplicate_key")` and routing it through the shared `ValueError` would break its reason-coded refusal (`_read_manifest` does not catch `ValueError` and already performs the R15 kind-check). That is the ruling on R16's "all six publishers" wording |
| R17 | HIGH: sessions excludes the whole marker; measure excludes the whole generation/upstream block | **ACCEPT-FIX** | every census digest covers its provenance block minus its own self-referential key |
| R23 | MEDIUM: roadmap still says 17 restored columns → 92 published beside the measured 89/1068 result | **ACCEPT-FIX — CLOSED** | duplicate of T2 R28 |
| R25 | MEDIUM: the memory store cited old `export.py:250` anisotropy; code now uses one max-dimension scale and cohort docs/tests pin 9.9° | **ACCEPT-FIX — CLOSED** | the view-dispersion bullet (now `.claude/rules/cohort.md`) cites `export.coord_scale` and `max(frame_w, frame_h)` instead of the retired per-axis `export.py:250`, and the anisotropy bullet is restated past-tense with the standing obligation on live consumers. `cohort.md` and `test_cohort.py` lost the 9.9° with T2 R24 |
| R33 | LOW: shipped code comments retain contract/ruling provenance (`A02`, `A09`, `P17`) instead of standalone constraints | **REJECT** | an amendment id beside a stated constraint is a pointer to binding law, not provenance narrative; the Authoring rule prunes origin stories, and `# A02: the partition is measured, never declared` is the constraint |
| R34 | LOW: `steq.py` found 24 flags across four human surfaces, including 22 passive constructions in the two docs units | **POLISH** | LOW — 22 passive constructions on two human-facing documents; register row carrying the `steq.py` command |
| R35 | MEDIUM: duplicated publisher trust-root machinery drifted in exactly the four ways exposed by R14-R17 | **ACCEPT-LIMIT** | publisher extraction stays DECLINED by the standing ruling; the measured four-way drift is recorded on that polish row as new evidence |

### T6 MAIN — rows MAIN measured itself

| row | severity + finding | ruling | acceptance check |
| --- | ------------------ | ------ | ---------------- |
| X01 | MEDIUM — `pyproject.toml` declares 11 console scripts and `.venv/bin` carried 10: `pose-estimation-cohort` was absent because the environment was never re-synced after M2.8.5 added the entry point, so every operator-facing invocation in `docs/technical/cohort.md` failed in the shipped environment. Module invocation worked throughout. | **ACCEPT-FIX — CLOSED this session** | `uv sync --frozen` reinstalled the project alone (no third-party churn, `--dry-run` verified first); `.venv/bin` now carries **11/11**, and `env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/src" .venv/bin/pose-estimation-cohort --help` runs rc=0. A **bare** invocation still dies at `ImportError … GLIBC_2.43 not found` — the inherited-`PYTHONPATH` host-OpenVINO leak `.claude/rules/gates.md` records, which hits every entry point in this container and is not this row |
| X02 | Decisive gate under five concurrent reviewers read 1701 passed / 1 failed — `test_c8_08` `subprocess.TimeoutExpired` at 900 s, the contention failure `.claude/rules/gates.md` records. Collection 1702 confirms the recorded count. | **REJECT — confirmed by rerun** | gate rerun **alone**: **1702 passed in 1467.46 s (24:27), rc=0**, `test_c8_08` included. Contention, not a defect |

## Register carry-out

Out-of-contract observations that survive as `.agent/polish.md` rows. Filled at ruling time.
