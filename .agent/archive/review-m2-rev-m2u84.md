# rev-m2u84 — M2 MILESTONE-REVIEW report

Scope: M2.8.4 — isotropic coordinates, the refuted orientation repair, the corrected corpus re-run.
Base `384c15f`. Marker `REV-M2U84-DONE-1`. Validator: `python /run/host/home/eturkes/Projects/pose-estimation/.scratch/check_review_report.py .scratch/agents/rev-m2u84.md`

## Verdict table

Fixed check set: 28 rows. Rows R01-R17 adjudicate contract P01-P17. Rows R18-R28 cover the touched integration surface, explicit high-value attacks, evidence claim boundaries, and normative consumer documentation. Out-of-set observations go to the Register.

| row | unit | check (predicate / surface) | verdict | finding <=1 line | evidence pointer |
| --- | ---- | --------------------------- | ------- | ---------------- | ---------------- |
| R01 | P01 | body export uses one `max(frame_w, frame_h)` scalar for x/y/z in both aspects | pass | body x/y/z share `coord_scale=max`; both non-square aspects passed exact-array checks | `src/pose_estimation/export.py:147`; `tests/test_isotropic_coords.py:63` |
| R02 | P02 | identical pixel geometry preserves angles and distance ratios exactly across transposed aspects | pass | same pixel geometry produced exactly equal angle/ratio tuples after six-decimal export | `tests/test_isotropic_coords.py:72`; targeted pytest: 5 passed |
| R03 | P03 | all four in-frame corners remain inside `[0,1]` in landscape and portrait | pass | origin and four inclusive frame-extreme coordinates stayed bounded in both aspects | `tests/test_isotropic_coords.py:91`; targeted pytest: 5 passed |
| R04 | P04 | coordinate scale is invariant to width/height transpose and mid-clip orientation change | pass | transposing 1080×1920 preserved scalar and exported x/y/z exactly | `src/pose_estimation/export.py:147`; `tests/test_isotropic_coords.py:108` |
| R05 | P05 | `process_source` exports with decoded frame shape despite contradictory capture properties | pass | 64×48 pixels overrode synthetic 640×480 properties; export was `(0.5,0.375)` | `src/pose_estimation/run.py:520`; `tests/test_isotropic_coords.py:152` |
| R06 | P06 | this OpenCV/FFmpeg build applies declared 0/90/180/270 display matrices and reports decoded dimensions | pass | four stamped `tkhd` classes decoded as declared; reported dimensions matched returned frames | `tests/test_isotropic_coords.py:276`; targeted pytest: 4 passed |
| R07 | P07 | corpus detector input is display-oriented once, with neither disabled auto-orientation nor double rotation | pass | detector received exactly one display transform for all four classes and read-back showed auto-orientation enabled | `tests/test_isotropic_coords.py:287`; targeted pytest: 4 passed |
| R08 | P08 | symbol, report key, and `image-isotropic-maxdim` token are frozen and token is bound to `max` behavior | fail | HIGH — deleting the report key leaves P08 green, so N7 and the artifact-identity conjunct are not witnessed | `tests/test_isotropic_coords.py:314`; `scripts/corpus_run_2d.py:489` |
| R09 | P09 | body, matched-hand, fallback-hand, and hands-only export paths all route through `coord_scale` | fail | MEDIUM — replacing both helper calls with duplicated `max` formulas leaves P09 green; A07's spy was not built | `tests/test_isotropic_coords.py:333`; `src/pose_estimation/export.py:167` |
| R10 | P10 | every 2D R golden enumerated from `_DATASETS` remains byte-identical | pass | regeneration enumerated 12 unique 2D artifacts from four source datasets and matched every byte | `tests/test_r_clinical_goldens.py:18`; `tests/test_isotropic_coords.py:385` |
| R11 | P11 | body run emits all 17 trunk/posture columns: 14 finite-capable and exactly 3 sagittal NA at correct grains | pass | executable guards plus validated cohort partition yield 14 finite-capable and only 3 sagittal exclusions across two grains | `analysis/clinical_features.R:1000`; `analysis/clinical_features.R:1038`; `cohort/cohort.json` aggregate |
| R12 | P12 | corrected run manifest is registry-total/unique, uses six dispositions, and publishes exactly 11 true verdicts | pass | validated registry/manifests are set-equal at 379 unique rows; six dispositions and exactly 11 source-derived verdicts all hold | `scripts/corpus_run_2d.py:458`; `output/corpus-2d/run_report.json` aggregate |
| R13 | P13 | rerun roots are resolved/disjoint both ways and new root starts with zero completion markers | pass | accepted operator record states resolved bidirectional disjointness/zero-marker launch; close state has 193 markers and no staging root | `.agent/archive/contract-m2u84.md:178`; current bounded marker count |
| R14 | P14 | qualification and calibration determinism evidence exactly covers source-derived control sets and current digests | pass | both generators reran byte-clean: 40 sweeps/19 tamper cases and 21 sweeps/18 tampers, all green | `tests/test_isotropic_coords.py:444`; checker rc=0 ×2 |
| R15 | P15 | static gates, targeted suites, collection delta, rc, and skip claims remain sound at review base | pass | all three static gates and 40 targeted cases passed; 19+2+1 additions minus one replaced node matches recorded +21 | `.agent/archive/contract-m2u84.md:186`; targeted gate logs |
| R16 | P16 | angle-fidelity evidence has correct math, frozen sample rule/digest, non-vacuous anisotropic arm, and passing bounds | pass | validated-registry recomputation matched 8-member digest/bounds; 90° oracle and all three evidence verdicts passed | `scripts/check_isotropy_angle_fidelity.py:65`; `tests/isotropy_angle_fidelity_results.json` |
| R17 | P17 | fusion read-back exactly inverts isotropic export in both aspects and no consumer restates the old inverse | pass | both calibrated aspects reconstructed source geometry; source census found one shared scalar inverse and no per-axis multiplication | `src/pose_estimation/multicam.py:649`; `tests/test_multicam.py:769` |
| R18 | integration | corpus CLI defaults to `--tracking body` and report configuration reflects the effective mode/normalisation | pass | bare parser returns `body`; shipped aggregate reports `body` plus `image-isotropic-maxdim` | `scripts/corpus_run_2d.py:373`; `output/corpus-2d/run_report.json` aggregate |
| R19 | site census | every frame-dimension coordinate normaliser/un-normaliser across `src/` and `analysis/` is compatible with isotropy | pass | only two export calls plus fusion inverse use artifact scale; per-axis hits are distinct model/crop/grid coordinate spaces | `src/pose_estimation/export.py:167`; `src/pose_estimation/multicam.py:649`; source census |
| R20 | range claim | narrative/schema `[0,1]` claims are scoped to in-frame points and do not misdescribe finite off-frame predictions | fail | MEDIUM — three unqualified `[0,1]` claims are false: a finite off-frame point exports as `(1.1,-0.05)` | `src/pose_estimation/export.py:157`; `src/pose_estimation/export.py:275`; `src/pose_estimation/export.py:425` |
| R21 | orientation guard | `open_capture` explicitly requests auto-orientation with failure semantics adequate to call it an assertion | fail | HIGH — `VideoCapture.set(...,1)` return/read-back is ignored; a rejected request returns an unreleased usable capture | `src/pose_estimation/video_io.py:133`; `tests/test_m2u84_review.py:23` RED |
| R22 | refutation | M1/M2/M3 independently bound the no-manual-rotation conclusion and scope it to the measured decode stack/path | pass | direct pixels, dimension agreement, and corpus posture corroboration support no manual rotate; synthetic P06/P07 pin OpenCV 4.13 path | `.agent/archive/contract-m2u84.md:17`; `tests/test_isotropic_coords.py:276` |
| R23 | evidence provenance | durable isotropy JSON is bound to the producing source/config/sample while post-fix replay refusal stays honest | fail | HIGH — unreplayable JSON pins sample IDs only; it carries no script, registry-generation, run-tree, manifest, or config digest | `scripts/check_isotropy_angle_fidelity.py:29`; `tests/isotropy_angle_fidelity_results.json:83` |
| R24 | units | published angle columns use `deg_image_plane` and consumer-facing text states image-plane/lens-distortion limits | fail | HIGH — code, tests, 19 descriptors, and docs still publish the pre-fix anisotropic token/claim | `src/pose_estimation/cohort.py:44`; `tests/test_m2u84_review.py:33` RED; `docs/technical/cohort.md:66` |
| R25 | similarity boundary | claims distinguish preserved angles/ratios from absolute normalized distances and projected clinical magnitudes | pass | source/contract preserve only angles and ratios; raw distances remain max-dimension units and projection limits stay explicit | `analysis/clinical_features.R:813`; `analysis/clinical_features.R:938`; `.agent/archive/contract-m2u84.md:94` |
| R26 | clinical partition | the second 2D sagittal guard is real and the 89-published/3-excluded split follows executable source, not transcription | pass | R source forces one frame sagittal value NA and aggregates its two windows; validated publisher derives 89/3 exactly | `analysis/clinical_features.R:1038`; `analysis/clinical_features.R:1081`; `cohort/cohort.json` |
| R27 | corrected data | bounded structural checks support corrected-corpus totals/configuration without exposing patient-adjacent identifiers | pass | validated aggregates show 193 events/379 unique assets/337090 frames, body+isotropic config, 89 features/1068 rows | `output/corpus-2d/run_report.json`; `cohort/cohort.json`; bounded validators |
| R28 | normative docs | roadmap, memory, and technical docs consistently describe the shipped isotropic corpus and current unit tokens | fail | MEDIUM — roadmap still says body restores 17/92 and its schema record freezes pre-fix anisotropy, contradicting later 14/89/isotropic state | `.agent/roadmap.md:386`; `.agent/roadmap.md:621`; `.agent/memory.md:562` |

## Detail

### R08 — HIGH — report identity deletion survives P08

- **Location/divergence:** `tests/test_isotropic_coords.py:314` checks the exported token, `coord_scale`, and the driver's imported constant. It never checks the emitted payload at `scripts/corpus_run_2d.py:489`. Deleting that dictionary entry left `pytest -k p08` green (1 passed).
- **Breached predicate:** P08 freezes `configuration.coord_normalization`; negative control N7 must fire when the report identity is dropped. D05 says that this report field is the only discriminator between shape-identical pre-fix and post-fix CSV generations.
- **Impact:** A future report-key omission can erase coordinate semantics from a corrected corpus while the claimed gate remains green. Consumers cannot distinguish generations from landmark CSV bytes alone.
- **Acceptance check:** Add a committed test that reaches report-payload construction and asserts `configuration.coord_normalization == export.COORD_NORMALIZATION == "image-isotropic-maxdim"`. Prove that deleting `scripts/corpus_run_2d.py:489` makes that test red.
- **Red-test pointer:** Coverage defect demonstrated by a surviving N7 mutation at base; no runtime red test is honest because the current payload is correct.
- **Surviving-mutant command (run from this worktree root):**
  ```sh
  export UV_PROJECT_ENVIRONMENT=/run/host/home/eturkes/Projects/pose-estimation/.venv PYTHONPATH="$PWD/src"; python3 -c 'from pathlib import Path; p=Path("scripts/corpus_run_2d.py"); s=p.read_text(); old="            \"coord_normalization\": COORD_NORMALIZATION,\n"; assert s.count(old)==1; p.write_text(s.replace(old,""))'; env -u LD_LIBRARY_PATH uv run --no-sync pytest tests/test_isotropic_coords.py -q -k p08; rc=$?; git checkout -- scripts/corpus_run_2d.py; exit "$rc"
  ```
  Current result: rc=0, 1 passed. The accepted predicate must make this mutant fail.

### R09 — MEDIUM — duplicated normalisers survive P09

- **Location/divergence:** `tests/test_isotropic_coords.py:333` compares ordinary values only. Replacing both `coord_scale` calls at `src/pose_estimation/export.py:167` and `src/pose_estimation/export.py:330` with duplicated `float(max(frame_w, frame_h))` formulas left `pytest -k p09` green (1 passed).
- **Breached predicate:** P09 and A07 require a call-path spy so body, matched-hand, fallback-hand, and hands-only paths demonstrably route through the one helper. The delivered case proves equal values, not helper routing.
- **Impact:** One branch can detach from the judgment-bearing helper and silently miss a later semantic change while the suite continues to certify shared routing.
- **Acceptance check:** Monkeypatch `export.coord_scale` to a sentinel divisor, drive all four paths, and require every emitted coordinate to reflect the sentinel. Prove that either helper-bypass mutation makes the test red.
- **Red-test pointer:** Coverage defect demonstrated by the two-call bypass mutant at base; current production routing itself is correct.
- **Surviving-mutant command (run from this worktree root):**
  ```sh
  export UV_PROJECT_ENVIRONMENT=/run/host/home/eturkes/Projects/pose-estimation/.venv PYTHONPATH="$PWD/src"; python3 -c 'from pathlib import Path; p=Path("src/pose_estimation/export.py"); s=p.read_text(); old="scale = coord_scale(frame_h, frame_w)"; assert s.count(old)==2; p.write_text(s.replace(old,"scale = float(max(frame_w, frame_h))"))'; env -u LD_LIBRARY_PATH uv run --no-sync pytest tests/test_isotropic_coords.py -q -k p09; rc=$?; git checkout -- src/pose_estimation/export.py; exit "$rc"
  ```
  Current result: rc=0, 1 passed. The accepted predicate must kill either helper-bypass mutant.

### R20 — MEDIUM — exported coordinates are not universally `[0,1]`

- **Location/divergence:** `src/pose_estimation/export.py:157`, `src/pose_estimation/export.py:275`, and `src/pose_estimation/export.py:425` state that every coordinate/read-back keypoint is in `[0,1]`. A finite synthetic point `(110,-5)` in a 100×50 frame exports unchanged as `(1.1,-0.05)`. The contract's own corpus spot-check also records residual high-confidence off-frame rows.
- **Breached predicate:** P03 is correctly limited to landmarks *inside the frame*, but D03 and the shipped API/schema prose silently strengthen that predicate to every coordinate. The stronger range contract is false.
- **Impact:** A consumer can reject legitimate off-frame predictions as corrupt, or build a validator/fusion precondition that disagrees with bytes the exporter intentionally preserves. Clipping would also change geometry, so the prose must not imply it.
- **Acceptance check:** Qualify all three claims as applying to in-frame landmarks and state that finite off-frame predictions may be outside `[0,1]`. Add an explicit off-frame preservation case beside P03 so future authors do not "repair" the mismatch by clipping.
- **Red-test pointer:** Behavioral counterexample is red against the prose, not the implementation; production correctly preserves rather than clips the finite point.

### R21 — HIGH — orientation request failure is silent

- **Location/divergence:** `src/pose_estimation/video_io.py:133` calls `cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)` but ignores its Boolean result and never reads the property back. A file-backed fake that rejects the write is still returned and is not released.
- **Breached predicate:** D02 says that reliance on the backend default becomes an explicit assertion, and P07 exists so disabled auto-orientation cannot silently enter the corpus path. An unchecked best-effort setter is not that assertion.
- **Impact:** A backend/version that rejects or ignores the property can feed non-display-oriented pixels into inference while the run continues. That recreates the exact 38-asset hazard the unit says it closed.
- **Acceptance check:** For file-backed sources, require a successful set and true read-back; otherwise release the capture and fail loudly through `open_capture`'s existing refusal channel. Cover both `set() == False` and `set() == True` with read-back false. Define the separate live-camera policy explicitly.
- **Red-test pointer:** `tests/test_m2u84_review.py:23` is red at base for the rejected-set branch; current result = 2-test review file fails exactly at the returned capture assertion.

### R23 — HIGH — durable fidelity evidence has no input/source identity

- **Location/divergence:** `scripts/check_isotropy_angle_fidelity.py:29` declares the JSON durable because the pre-fix tree was deleted. `tests/isotropy_angle_fidelity_results.json:83` pins only the eight sample ids by digest; its top-level schema has no script hash, validated-registry generation identity, pre-fix report/manifest/tree digest, effective configuration, or source revision.
- **Breached predicate:** A13 says that selection is a function of the *validated* registry and that the evidence file replaces replay. P16 credits exact numbers from a deleted derived tree. A non-replayable artifact must bind both the computation and the input bytes that produced those numbers.
- **Impact:** The committed values cannot be distinguished from results produced by another pre-fix tree, another script revision, a stale registry, or manual editing. The current sample digest matching the current registry proves membership only, not the measured coordinates.
- **Acceptance check:** Recover the pre-fix tree and regenerate with at least checker digest, validated inventory generation/tree digest, pre-fix report and manifest/tree digests, and effective coordinate/tracking configuration, plus a committed validator. If recovery is impossible, downgrade P16's corpus-number claim to unbound historical evidence and rely only on replayable P01/P02 geometry.
- **Red-test pointer:** No honest numerical red test can reconstruct deleted inputs; the missing provenance fields are the defect and must be fixed before a schema test is meaningful.

### R24 — HIGH — corrected angles retain the pre-fix unit

- **Location/divergence:** M2.8.4 D08 at `.agent/archive/contract-m2u84.md:116` requires `deg_image_plane`. Current `src/pose_estimation/cohort.py:44` and `tests/test_cohort.py:45` freeze `deg_image_plane_uncalibrated`; 19 rows in `cohort/descriptors.yaml` publish it. `docs/technical/cohort.md:66` still says x/y use different dimensions and retain the measured 9.9° anisotropy.
- **Breached predicate:** D08 assigns the unit amendment to M2.8.3 after D03 restores a similarity map. The shipped cohort code/data/docs never consumed that amendment, despite `.agent/memory.md:562` recording the intended token and residual lens-distortion qualification.
- **Impact:** The published clinical descriptor schema falsely labels corrected image-plane angles as carrying a removed, quantified anisotropic error. Consumers receive mutually inconsistent normalization and unit claims from the run report and cohort artifact.
- **Acceptance check:** Amend the M2.8.3 unit vocabulary; set `UNIT_DEG = "deg_image_plane"`; update its tests and consumer text to state “image-plane, not anatomical, no lens-distortion correction”; republish `cohort/`; validate its generation; and prove no descriptor or current technical claim retains `deg_image_plane_uncalibrated`.
- **Red-test pointer:** `tests/test_m2u84_review.py:33` is red at base with the exact intended token and turns green under the required source change.

### R28 — MEDIUM — current roadmap carries mutually exclusive corpus semantics

- **Location/divergence:** `.agent/roadmap.md:386` still says `--tracking body` restores all 17 trunk/posture columns and yields 92 published features. The same roadmap later records the measured 14-restored/3-sagittal-NA/89-published partition. `.agent/roadmap.md:621` and the governing M2.8.3 contract still freeze `deg_image_plane_uncalibrated`, while `.agent/memory.md:562` records the corrected `deg_image_plane` ruling.
- **Breached predicate:** The normative state surfaces must consistently describe the shipped isotropic corpus and current units. M2.8.4 D08 explicitly assigns the schema amendment to M2.8.3; marking the prerequisite met without applying that amendment leaves two authoritative states.
- **Impact:** A later agent can legitimately re-derive either 92 or 89 features and either angle token from attached project state. That ambiguity already propagated into production under R24.
- **Acceptance check:** Change the M2.8.4 summary row to “17 present, 14 finite, 3 sagittal NA, 89 published”; add the post-M2.8.4 unit amendment to the M2.8.3 contract/roadmap; label retained 9.9°/old-token passages explicitly as pre-fix history; then run a consistency search over current-state claims.
- **Red-test pointer:** R24's red token test covers the code consequence; this row is a normative-state repair rather than a second runtime behavior.

## Register

- **LOW — stale reviewer oracle:** the assigned `pytest tests/test_export.py -q` path does not exist at base; the current focused export suite is `tests/test_rtmlib_csv_export.py` (13 passed). Evidence: missing-path rc=4 plus test census. **Acceptance check:** update the M2.8.4 review recipe to the existing file or name the intended replacement.
- **LOW — P17 grep is not the stated zero-match witness:** `.agent/archive/contract-m2u84.md:339` says `rg -n 'resolution"\]' src/pose_estimation/` returns no multiplication site, but it returns four benign resolution accesses, including the corrected inverse. **Acceptance check:** replace it with a multiplication-specific AST/checker predicate that survives formatting and fails only on a per-axis inverse.
