# rev-m2u82 — M2 MILESTONE-REVIEW report

Scope: M2.8.1 + M2.8.2 — corpus-run preconditions, instrumented pilot, disposition manifest, full corpus run.
Base `384c15f`. Marker `REV-M2U82-DONE-1`. Validator: `python /run/host/home/eturkes/Projects/pose-estimation/.scratch/check_review_report.py .scratch/agents/rev-m2u82.md`

## Verdict table

One row per check-set row, ids `R01`.. — ADD rows until the check set is complete, then
adjudicate. Every row is born `unknown`; it flips to `pass` or `fail` when ruled.

| row | unit | check (predicate / surface) | verdict | finding <=1 line | evidence pointer |
| --- | ---- | --------------------------- | ------- | ---------------- | ---------------- |
| R01 | M2.8.1 | `--output-dir` reaches `process_session`; every camera output lands below the requested event root | pass | Traced the shipped closure and session loop; the user base is forwarded once and each camera path is `<base>/<session>/<camera>.csv`. | `src/pose_estimation/run.py:705-735`; `src/pose_estimation/multicam.py:513-555`; P01 test |
| R02 | M2.8.1 | published-tree containment compares canonical final roots symmetrically, catches symlinks/equality, refuses before writes, preserves ad-hoc default | pass | Exercised/read all four overlap classes plus ancestor-base allow and ad-hoc default; canonical final-root check precedes `mkdir` and callbacks. | `src/pose_estimation/multicam.py:462-510`; `tests/test_corpus_run_preconditions.py:170-216` |
| R03 | M2.8.1 | an external-output session run leaves the published generation valid and byte witnesses unmoved | pass | The external dispatch test snapshots every published byte and reruns `validate_generation`; the resolver writes only below the external event root. | `tests/test_corpus_run_preconditions.py:152-167`; `src/pose_estimation/multicam.py:474-510` |
| R04 | M2.8.1 | source diagnostics have the frozen one-row schema, mkdir safely, publish from `finally`, and obey zero/open-failure semantics | pass | Read the open-failure/zero-frame/interruption branches and ran the contract suite; successful opens publish one exact nine-field row from `finally`, while open failure publishes none. | `src/pose_estimation/run.py:371-645`; `tests/test_corpus_run_preconditions.py:264-632` |
| R05 | M2.8.1 | session orchestration forwards diagnostics and announces CSV/diagnostics only when the files exist; `None` stays silent | pass | Traced `output_diag` through the shipped callback and verified filesystem-gated announcements plus the explicit-disabled case. | `src/pose_estimation/run.py:705-735`; `src/pose_estimation/multicam.py:540-557`; P06/P07 tests |
| R06 | M2.8.1 | timestamp dispositions are mutually exclusive and exhaustive on every branch: three counters sum to every live `timestamp()` call | pass | The `if/elif/else` increments one counter after each completed return path; prefix tests reach accepted, index fallback, forced, and live-clock branches. | `src/pose_estimation/video_io.py:27-103`; precondition P08/P09 tests; corpus P12 test |
| R07 | M2.8.1 | instrumentation leaves every timestamp value and strict-monotonic postcondition unchanged | pass | Frozen hexadecimal outputs cover file/live, invalid, duplicate, and regressing timestamps; all values and the strict-increase repair match the pre-change table. | `tests/test_corpus_run_preconditions.py:475-523`; `video_io.py:67-103` |
| R08 | M2.8.1 | diagnostics derive frame total/rate from authoritative counters and print a same-population fallback summary | pass | Writer reads `clock.n_timestamps` and `clock.cfr_fallback_rate` once for the row and prints those same counter values; interruption-prefix test is green. | `src/pose_estimation/run.py:616-645`; P11 tests; 23-case suite rc=0 |
| R09 | M2.8.1 | each of five entry drops plus the zero-emission loop path yields exactly one source-derived reason | pass | Traced all five guarded `next`s and the post-loop zero-yield arm; source vocabulary/site equality plus executed branch cases cover all six one-for-one. | `analysis/clinical_features.R:1340-1594`; `tests/test_corpus_run_preconditions.py:664-701,838-854` |
| R10 | M2.8.1 | every input group reaches exactly one of windowed or dropped, disjointly, in both 2D and 3D | pass | Per-group `ri_before` makes any emitted window exclude a drop and every no-row path records one; both-mode partition case reports 3=1+2 with zero overlap. | `analysis/clinical_features.R:1350-1594`; `tests/test_corpus_run_preconditions.py:720-784` |
| R11 | M2.8.1 | group-QC always publishes in both modes, including empty output, with frozen five-column schema and no consumer bleed | pass | The producer unconditionally writes `group_qc` after both-mode windowing; exact header, no 3D tags, and positive consumer filters keep it outside legacy tables. | `analysis/clinical_features.R:1144-1172,2058-2065`; schema/golden/rescan tests |
| R12 | M2.8.1 | existing goldens stay byte-identical; populated `2d_drop` pins real reasons, ordering, and partition; all artifact enumerators include group-QC | pass | Regeneration and 49-case golden gate are byte-green; `2d_drop` carries two real reasons and 3=1+2 disjoint, while generator/test/rescan enumerators include group-QC. | `scripts/regenerate_r_clinical_goldens.py:333-369`; `tests/test_r_clinical_goldens.py:159-252`; rc=0 |
| R13 | M2.8.1 | pilot samples events by length-free hash rule, covers every required axis, and rejects undercoverage | pass | Selection operates only on event ids and seeded hashes, covers each axis before replication, then compares selected/population value sets and raises on any gap. | `scripts/pilot_corpus_run.py:175-207`; contract A11 |
| R14 | M2.8.1 | pilot redaction is membership-based for emitted aggregates and contains identifier-bearing subprocess output in files | fail | HIGH — raw, unvalidated qualification cells become both report strata and allowed strings, so an edited identifier-shaped label is emitted under all-green verdicts. | `scripts/pilot_corpus_run.py:37,113-164,463-474,577-588`; red `test_pilot_refuses_unvalidated_source_tables_before_publication` |
| R15 | M2.8.1 | pilot is committed/rerunnable and recomputes all claimed precondition, partition, diagnostics, and throughput verdicts | fail | MEDIUM — a fresh pilot rerun never clears its event output, so stale diagnostics/clinical files can be credited when the current source produces none. | `scripts/pilot_corpus_run.py:210-241,481-489`; red `test_pilot_rerun_clears_the_selected_event_before_launch` |
| R16 | M2.8.2 | every shipped rtmlib run-path `PoseTracker` construction disables IoU tracking | pass | Repository census finds one shipped construction; the behavioral fixture captures that real object from `run.main` and observes `tracking is False`. | `src/pose_estimation/run.py:825-846`; `tests/test_corpus_run_2d.py:101-181` |
| R17 | M2.8.2 | committed probe reproduces freeze with tracking, advance without it, and amended cadence sweep including frequency 1 control | pass | Reran the no-corpus probe: all seven verdicts true; shipped-behavior tests sweep miss/no-miss at frequencies 1,2,5,7,13 and retain frequency 1 as positive control. | `scripts/probe_tracker_freeze.py:69-158`; probe rc=0; 105-case suite rc=0 |
| R18 | M2.8.2 | shipped construction never sends a detected-person source through empty-bbox whole-frame pose and calls detection at exact cadence | pass | The captured shipped tracker receives both miss residues: zero whole-frame calls and exactly `ceil(frames/frequency)` detector calls across the amended sweep. | `tests/test_corpus_run_2d.py:155-304`; tracker probe rc=0 |
| R19 | M2.8.2 | completion marker is written only after all event run/clinical outputs and timing metadata are final | pass | `_run_stage` waits and closes each log; only after successful pose then completed R does `_attempt_event` write the final status and both rounded timings. | `scripts/corpus_run_2d.py:140-207`; `src/pose_estimation/corpus_run.py:76-81` |
| R20 | M2.8.2 | a marked-complete event is skipped on resume without inference or byte changes | fail | MEDIUM — inference is skipped, but the shipped driver rewrites `run_report.json` with invocation-local attempt/time fields, breaking P05 byte idempotence. | `scripts/corpus_run_2d.py:408-438,481-547`; red `test_complete_resume_keeps_every_published_output_byte_identical` |
| R21 | M2.8.2 | every unmarked/crash-window state is re-attempted from a clean event tree without crediting partial work or deleting another complete event | fail | HIGH — `process_source` swallows `KeyboardInterrupt`; a child-only interruption exits 0, so the driver can run R and mark a partial source complete. | `src/pose_estimation/run.py:584-599`; `scripts/corpus_run_2d.py:178-206`; red `test_interrupted_source_propagates_after_writing_diagnostics` |
| R22 | M2.8.2 | manifest validation independently enforces nonempty/cardinality, canonical key-set equality, and key uniqueness | pass | The shipped validator refuses emptiness first, then checks row count, uniqueness, and key-set equality separately; five direct mutations drive the shipped symbol. | `src/pose_estimation/corpus_run.py:123-149`; `tests/test_corpus_run_2d.py:568-598` |
| R23 | M2.8.2 | `asset_disposition` is total and single-valued over every stage/result/artifact branch; every canonical asset gets one row | pass | One return chain maps absent, run-failed, clinical-failed, missing-landmark, and success states; `_manifest_rows` appends exactly once per canonical id before validation. | `src/pose_estimation/corpus_run.py:84-102`; `scripts/corpus_run_2d.py:210-233`; stage-mapping tests |
| R24 | M2.8.2 | writer and validator consume one machine-readable frozen disposition vocabulary from `src/` | pass | `ASSET_DISPOSITIONS` is single-defined in `src`; manifest validation, zero-filled census, redaction allowlist, and tests import/read that object rather than transcribing it. | `src/pose_estimation/corpus_run.py:26-34,123-149`; `scripts/corpus_run_2d.py:41-42,124-126,446` |
| R25 | M2.8.2 | every `ok` row owns one landmark CSV and one diagnostic row; no non-`ok` row claims either | fail | MEDIUM — `_artifacts` accepts a one-row diagnostics file whose `video` belongs to another event/camera, then attributes its counters to this `ok` asset. | `scripts/corpus_run_2d.py:264-303`; red `test_artifact_validator_rejects_a_diagnostic_owned_by_another_source` |
| R26 | M2.8.2 | every R-processed landmark publishes group-QC; input-type exits are excluded, while event-scoped clinical failure disposes every event asset | pass | The only pre-write exits classify schema/input type; every processing path reaches the unconditional group-QC write, and a nonzero event R exit maps all its assets to `clinical_failed`. | `analysis/clinical_features.R:1960-2065`; `src/pose_estimation/corpus_run.py:84-102`; P10 tests |
| R27 | M2.8.2 | end-to-end run preserves generation validation, content-tree digest, and marker-byte digest as three independent witnesses | fail | CRITICAL — `--out` can create logs inside the session tree before validation, and `--report` can write there after both digest snapshots, leaving the input mutated or a green witness stale. | `scripts/corpus_run_2d.py:400-407,437-439,545-546`; red parametrized sink test |
| R28 | M2.8.2 | every `ok` asset has exhaustive CFR counters; stored totals/rates equal derivation; pooled rate weights the same frame population | fail | MEDIUM — `abs(NaN - derived) > 1e-9` is false, so a nonfinite stored rate passes `stored_rate_equals_its_derivation`. | `scripts/corpus_run_2d.py:294,305-344`; red `test_cfr_derivation_rejects_a_nonfinite_stored_rate` |
| R29 | M2.8.2 | report redaction decides both key and value placements by membership in the exact shipped schema/label union | fail | HIGH — shipped key validation remains `allowed OR regex`, while the alleged membership oracle walks only local fixtures; an unknown identifier-shaped key passes production. | `scripts/pilot_corpus_run.py:412-435`; `tests/test_corpus_run_2d.py:1589-1776`; red shipped-guard test |
| R30 | M2.8.2 | every throughput/cost rate pairs marker-summed numerators and denominators from one event population; corpus label requires full measured population | fail | HIGH — clinical-failed event wall enters `_corpus_wall`, its frames are excluded from `_artifacts`, yet the report labels the mixed 100/300 population `sample: corpus`. | `scripts/corpus_run_2d.py:235-253,264-344,456,526-541`; red mixed-population test |
| R31 | M2.8.2 | report verdicts recompute the contract invariants and any false verdict makes a partial run non-successful | pass | The eleven declared booleans are rebuilt from current manifest/artifact/partition/counter/witness state and `main` returns 0 only when `all(verdicts.values())`; semantic gaps are isolated in R27-R30. | `scripts/corpus_run_2d.py:458-482,538-547`; partial mixed-population red returns 1 |
| R32 | both | contract suites exercise shipped symbols and meaningful mutations, not local models, comment/name regexes, or vacuous fixtures | fail | HIGH — P05/P06, P09, P11, P13, and P14 chiefly grade `_Driver`, local verdict walkers, or source/prose regexes; 105 green cases miss nine shipped-code reds. | `tests/test_corpus_run_2d.py:313-446,630-689,1214-1241,1589-1900`; review red suite |
| R33 | both | tracker probe, corpus driver, pilot, and R golden regeneration are committed mechanisms runnable from the reviewed state | pass | Probe ran with seven true verdicts; all three script help paths exit 0; precondition, corpus, and golden gates execute from this commit without corpus access except the measurement itself. | `scripts/{probe_tracker_freeze,pilot_corpus_run,corpus_run_2d,regenerate_r_clinical_goldens}.py`; smoke rc=0 |
| R34 | both | technical entrypoint documentation matches actual flags, modes, locations, resume semantics, redaction, and exit behavior | pass | Compared parser/help output and control flow to the corpus section: defaults, due-event limit, analyse-only, report location, manifest codes, log location, all-verdict exit, and patient boundary match. | `docs/technical/entrypoints.md:235-290`; both CLI help rc=0 |

## Detail

One `### R<nn>` section per `fail` row: `file:line`, divergence, breached predicate, impact,
acceptance check, red-test pointer.

### R32

- **Severity:** high.
- **File:line:** `tests/test_corpus_run_2d.py:313-446`, `:630-689`, `:1214-1241`, `:1589-1900`.
- **Divergence:** despite A10's explicit rule, P05/P06 exercise local `_Driver`; P09 exercises `_artifact_verdict`; one P11 case labels itself a stand-in; P13 exercises `_violations` over hand-authored fields; P14 scans source/roadmap prose. The shipped validator is bound only for P07/P08 and narrow helper fragments. The committed 105-case suite is green while the review's shipped `main`, `_artifacts`, `_cfr`, and `_assert_redacted` controls expose nine red predicates (ten failing parametrized cases).
- **Breached predicate:** M2.8.1 A10 and M2.8.2 A10: each case grades the shipped symbol and behavior, never a local reimplementation or spelling regex.
- **Impact:** the suite can remain decisive-green while resume fixity, source ownership, redaction membership, output containment, and rate-population claims are false; its green count materially overstates kernel assurance.
- **Acceptance check:** replace stand-ins with runpy/imported shipped functions or move each judgment into `src/`; retain the current mutations, then require every review red to turn green under the fixes and red again under its named seed.
- **Red test:** `tests/test_review_m2u82.py` (nine review predicates; ten failing parameter instances at base) is the shipped-symbol counterexample set.

### R27

- **Severity:** critical.
- **File:line:** `scripts/corpus_run_2d.py:400-407`, `:437-439`, `:545-546`; `src/pose_estimation/multicam.py:474-510`.
- **Divergence:** the session callback has a symmetric containment guard, but the driver has two independent sinks outside it. With `--out` equal/inside the published tree, `logs.mkdir()` writes before the first `validate_generation`; validation then refuses only after the generation is poisoned. With external `--out` but `--report` inside the tree, both post-run witnesses are captured first and the report is written afterward, so the report can say both digests are unmoved while its own bytes just moved them.
- **Breached predicate:** M2.8.2 P11/A07 and M2.8.1 D01: the run writes nothing inside a published session generation, refuses overlap before every write, and substantiates the claim with all three witnesses.
- **Impact:** an option accepted by the shipped CLI can corrupt the read-only patient-adjacent input generation; the report-only route exits green and leaves a false immutability verdict.
- **Acceptance check:** canonicalize and symmetrically reject **all** sinks (`out`, its logs/manifest/default report, and explicit `report`) against the sessions root before any mkdir/read-side effect; synthetic equal/inside/symlink/ancestor cases must be write-free, and an external run must keep all three digests fixed through the final report write.
- **Red test:** `tests/test_review_m2u82.py::test_driver_refuses_every_sink_overlapping_the_published_sessions` (both `out` and `report` parameters red at `384c15f`).

### R28

- **Severity:** medium.
- **File:line:** `scripts/corpus_run_2d.py:294`, `:305-344`.
- **Divergence:** the stored rate is parsed with `float()`, then mismatch is `abs(stored - derived) > 1e-9`. IEEE `NaN` makes that comparison false, so `_cfr()` returns `assets_rate_mismatch=0` and the published derivation verdict stays true.
- **Breached predicate:** M2.8.2 P12/A08: every stored derived value must equal its finite counter derivation.
- **Impact:** a malformed diagnostics row can publish a nonfinite per-asset rate while the structural validator certifies it and admits its counters to the pooled report.
- **Acceptance check:** require finite, nonnegative integer counters/totals and a finite stored rate before tolerance comparison; seed `nan`, infinities, negatives, and inconsistent sums and require the relevant verdict false with no pooled ingestion.
- **Red test:** `tests/test_review_m2u82.py::test_cfr_derivation_rejects_a_nonfinite_stored_rate` (red at `384c15f`).

### R29

- **Severity:** high.
- **File:line:** `scripts/pilot_corpus_run.py:412-435`; `tests/test_corpus_run_2d.py:1589-1776`, `:1903-1921`.
- **Divergence:** A09 permits `_assert_redacted` as a weaker runtime backstop only because the suite is supposed to be the exact membership oracle. Instead, `_violations()` is a local reimplementation over hand-built payloads/field sets and never walks the shipped corpus payload against a frozen report schema. Production therefore admits any unknown lowercase identifier-shaped key through `FIELD_NAME`; the existing shipped-symbol case exercises only known throughput labels.
- **Breached predicate:** M2.8.2 P13/A09/A10: membership decides both placements, and a case over shipped behavior must grade the shipped symbol rather than a stand-in.
- **Impact:** a newly introduced or dynamically sourced identifier-shaped key passes both runtime and suite, so the report can leak an identifier while retaining a green redaction credential.
- **Acceptance check:** publish/freeze the exact recursive report field set and enforce it in the shipped walker (or run the membership oracle over a real constructed payload); inject an unknown lowercase key at every nesting level and require refusal without echoing its bytes.
- **Red test:** `tests/test_review_m2u82.py::test_shipped_redaction_guard_rejects_an_unknown_identifier_shaped_key` (red at `384c15f`).

### R30

- **Severity:** high.
- **File:line:** `scripts/corpus_run_2d.py:235-253`, `:264-344`, `:456`, `:526-541`.
- **Divergence:** `_corpus_wall()` includes every marker carrying `run_s`, including a clinical-failed event. `_artifacts()` excludes every non-`ok` asset's diagnostics from `frames`; `sample` nevertheless becomes `corpus` when marker timings cover all event ids. Synthetic two-event evidence publishes 100 frames / 300 run-seconds = 0.333 fps and `sample: corpus`, although the 200-frame failed event supplies denominator only.
- **Breached predicate:** M2.8.2 P14/A13 and the memory law: every published rate's numerator, denominator, and sample label must name the same population.
- **Impact:** failed/diagnostically-invalid runs publish a measured corpus rate the pipeline never attained, recreating the exact resumed-rate defect A13 claims closed.
- **Acceptance check:** key frame counts and wall times by event and join one explicit eligible event set before aggregation; otherwise publish `null`/partial. Cover clinical failure, missing/wrong diagnostics, no-landmark, and successful partial/full populations.
- **Red test:** `tests/test_review_m2u82.py::test_throughput_never_labels_mixed_frame_and_wall_populations_as_corpus` (red at `384c15f`: `sample=corpus`, frames=100, run wall=300).

### R21

- **Severity:** high.
- **File:line:** `src/pose_estimation/run.py:584-599`; `scripts/corpus_run_2d.py:178-206`; `tests/test_corpus_run_preconditions.py:601-632`.
- **Divergence:** `process_source()` catches `KeyboardInterrupt`, writes the decoded-prefix diagnostics in `finally`, and returns normally. If the inference child alone receives the interruption (or a backend raises it), `pose_estimation.run` exits 0; `_attempt_event()` therefore runs the clinical stage and writes `status=complete`. The existing M2.8.1 case positively pins the swallow instead of propagation.
- **Breached predicate:** M2.8.2 P06/D05: work interrupted before completion must remain unmarked and be rerun from scratch; M2.8.1 A04 requires the diagnostic prefix, not a false success exit.
- **Impact:** a partial landmark CSV can become the marker-authoritative final artifact and will be skipped forever on resume—the exact existence/row-count ambiguity D05 says the marker prevents.
- **Acceptance check:** preserve the `finally` diagnostics write but re-raise `KeyboardInterrupt`; invoke the source path with an interrupting tracker and assert diagnostics exist, the exception/nonzero reaches the caller, and `_attempt_event` creates no complete marker.
- **Red test:** `tests/test_review_m2u82.py::test_interrupted_source_propagates_after_writing_diagnostics` (red at `384c15f`).

### R25

- **Severity:** medium.
- **File:line:** `scripts/corpus_run_2d.py:256-303`; `src/pose_estimation/run.py:603-645`.
- **Divergence:** `_artifacts()` checks only row count plus five numeric fields. It never checks the frozen nine-column header or `diagnostics[0]["video"] == f"{event_id}/{camera_name}"`; a copied foreign row passes `artifacts_owned`, contributes counters, and leaves every CFR derivation check green.
- **Breached predicate:** M2.8.2 P09 and M2.8.1 P05/A04: an `ok` asset must own its one exact-schema source-diagnostics row.
- **Impact:** the report can certify artifact ownership while attributing another source's frame counts and fallback rate to this asset, corrupting both the per-asset contract and pooled CFR population.
- **Acceptance check:** validate the exact `SOURCE_DIAGNOSTIC_FIELDS` header and expected `video` identity before accepting counters; mutate either while keeping one numeric row and require `wrong_diag > 0`, `artifacts_owned=false`, and no counter ingestion.
- **Red test:** `tests/test_review_m2u82.py::test_artifact_validator_rejects_a_diagnostic_owned_by_another_source` (red at `384c15f`).

### R20

- **Severity:** medium.
- **File:line:** `scripts/corpus_run_2d.py:408-438`, `:481-547`; `tests/test_corpus_run_2d.py:331-369`.
- **Divergence:** the real driver correctly skips a complete event, but it always republishes the report. The first pass records nonzero `run.events_attempted`, `attempts_complete`, and `throughput.invocation_wall_s`; the second records zeros, so `run_report.json` changes. P05's case uses local `_Driver`, snapshots only its event tree, and never executes the shipped publisher.
- **Breached predicate:** M2.8.2 P05: a second driver run over fully complete events performs zero inference **and leaves every output byte-identical**; A10 additionally requires shipped behavior rather than a stand-in.
- **Impact:** a green resume mutates a published output and erases how the completing invocation was recorded, so byte fixity and provenance depend on how many times the operator reruns the command.
- **Acceptance check:** run the shipped `main()` twice over a synthetic one-event complete tree and require a byte-identical whole output tree; either make durable report fields corpus-derived or explicitly amend P05 to exclude/redefine invocation telemetry in a separate non-idempotent artifact.
- **Red test:** `tests/test_review_m2u82.py::test_complete_resume_keeps_every_published_output_byte_identical` (red at `384c15f`; the differing file is `run_report.json`).

### R14

- **Severity:** high.
- **File:line:** `scripts/pilot_corpus_run.py:37`, `:113-164`, `:463-474`, `:577-588`.
- **Divergence:** `_load_assets()` reads raw inventory, qualification, and placement CSVs before the only generation check; the script imports/calls the sessions validator alone and never calls `qualify.validate_generation`. Those unchecked `codec` / `device_config` cells then key `selection.coverage`, enter each per-asset row, and join the runtime allowed-string set. A synthetic table with no generation documents and `device_config=subject90` publishes that value with every verdict true.
- **Breached predicate:** M2.8.1 P17 + D07 and the technical consumer contract: emitted strata must be *published* labels, and every qualification consumer validates its generation before reading a row.
- **Impact:** a stale or edited qualification table can alter the selected sample and turn an identifier-shaped input cell into an allowlisted report value/key; the redaction gate certifies the leak instead of refusing it.
- **Acceptance check:** validate inventory, sessions, and qualification generations with their upstream bindings before `_load_assets()` or any output creation; mutate any source CSV without its marker and assert nonzero/no report, while a valid linked trio remains green.
- **Red test:** `tests/test_review_m2u82.py::test_pilot_refuses_unvalidated_source_tables_before_publication` (red at `384c15f`).

### R15

- **Severity:** medium.
- **File:line:** `scripts/pilot_corpus_run.py:210-241`, `:481-489`.
- **Divergence:** a non-`--reuse-run` attempt launches into an existing `<out>/<event>` without deleting it. `process_source()` treats capture-open failure as a clean return and writes no current CSV/diagnostics, so an older diagnostic and its clinical companions remain discoverable; count-only `diagnostics_complete` can then credit old measurements to the new wall clock. Separately, `--reuse-run` leaves both wall accumulators at zero, overwrites the report with `frames_per_s_incl_startup=null`, and still returns all seven verdicts true.
- **Breached predicate:** M2.8.1 P17/P18 + D07: the rerunnable committed instrument must measure the current pilot attempt, not a union with a prior attempt, and must not erase its throughput credential on analysis-only reuse.
- **Impact:** a fresh rerun can pair old numerator/artifacts with this invocation's denominator; a reuse run can erase the measured startup rate while retaining a green report.
- **Acceptance check:** before each fresh selected event, delete its output tree (preserve it only under `--reuse-run`), require diagnostic asset-key equality/uniqueness, and persist/read event timing for reuse or make missing timing a false verdict. Seed stale files and reuse a measured tree.
- **Red test:** `tests/test_review_m2u82.py::test_pilot_rerun_clears_the_selected_event_before_launch` (red at `384c15f`).

## Register

Out-of-contract observations, one bullet each, carrying an evidence pointer + acceptance check.

- **REG-01 — pilot sinks bypass its own guard.** `scripts/pilot_corpus_run.py:471-474,490-491,590-591` repeats R27's ordering: overlapping `--out` mutates before validation and overlapping `--report` writes after the last digest. **Acceptance check:** apply the same canonical all-sink, pre-write disjointness test to pilot `out`/`report`, including symlink overlap, and prove all session bytes unchanged on refusal.
- **REG-02 — completion markers are not bound to inputs or configuration.** `scripts/corpus_run_2d.py:151-206,391-438,483-505` writes status/timings only, then skips any complete marker while the report echoes the current model/tracking/devices. Reusing one output tree after a valid session-generation or CLI-config change can mix old and new event artifacts under one current configuration claim. **Acceptance check:** store a canonical configuration + upstream-generation fingerprint in every marker; reject or cleanly rerun mismatches, and require one uniform fingerprint before publication.
- **REG-03 — the full driver also consumes qualification without validating it.** `scripts/corpus_run_2d.py:392-407` calls pilot `_load_assets()` first and validates sessions+inventory only; `qualify.validate_generation` is never invoked. **Acceptance check:** validate the qualification generation and its sessions/inventory bindings before reading any row, then mutate `assets_qc.csv` without its marker and require write-free refusal.
