# rev-m2-cross — M2 MILESTONE-REVIEW report

Scope: cross-unit integration + M2.7.3/M2.7.4 docs spot-check + project-`CLAUDE.md` conformance.
Base `384c15f`. Marker `REV-M2-CROSS-DONE-1`. Validator: `python /run/host/home/eturkes/Projects/pose-estimation/.scratch/check_review_report.py .scratch/agents/rev-m2-cross.md`

## Verdict table

Fixed check set: 39 rows. Every row starts `unknown`; adjudication proceeds in batches of at most five.

| row | unit | check (predicate / surface) | verdict | finding <=1 line | evidence pointer |
| --- | ---- | --------------------------- | ------- | ---------------- | ---------------- |
| R01 | integration/artifacts | regenerator `_expected_outputs` enumerates `group_qc` + cohort additions | pass | 5 stems map to all 16 clinical goldens, including both-mode `group_qc`; cohort is a separate publisher domain | `scripts/regenerate_r_clinical_goldens.py:333` |
| R02 | integration/artifacts | golden-test `_DATASETS` enumerates every published dataset | pass | `_DATASETS` names 5 datasets and the same 16 committed files found on disk | `tests/test_r_clinical_goldens.py:18` |
| R03 | integration/artifacts | golden-test `_BASE_WIDTHS` covers every dataset schema | pass | all 4 artifact kinds have exact base widths, including `group_qc=5` | `tests/test_r_clinical_goldens.py:54` |
| R04 | integration/artifacts | timebase `_EXPECTED_GOLDENS` covers every required golden | pass | 16-name set matches the committed golden directory and includes every `group_qc` file | `tests/test_r_timebase_truth.py:81` |
| R05 | integration/artifacts | pipeline `produced ==` oracle names the complete published set | pass | six-file oracle includes 3D `group_qc`, window QC, windows, frames, phases, and source | `tests/test_r_pipeline.py:1147` |
| R06 | integration/artifacts | first determinism result `source_digests` matches its source tuple | pass | recomputed all 20 qualification source hashes; names and digests exactly match committed evidence | `scripts/check_qualify_determinism.py:47`; `tests/qualify_determinism_results.json:19` |
| R07 | integration/artifacts | second determinism result `source_digests` matches its source tuple | pass | recomputed all 23 calibration-QC source hashes; names and digests exactly match committed evidence | `scripts/check_calibration_qc_determinism.py:35`; `tests/calibration_qc_determinism_results.json:8` |
| R08 | integration/artifacts | repository scan finds no seventh enumerator and producer duplicates none | fail | LOW: P16 independently enumerates all 5 group-QC goldens, falsifying the six-place budget | `tests/test_corpus_run_preconditions.py:858` |
| R09 | integration/tripwire | inventory determinism tuple reaches every byte-shaping module | fail | HIGH: inventory evidence has no source tuple/digests and records an intrinsically stale head SHA | `scripts/check_inventory_determinism.py:256` |
| R10 | integration/tripwire | sessions determinism tuple reaches every byte-shaping module | pass | sessions has no stale evidence file: two current-code tests execute the byte oracle across shuffle/env/out variations | `tests/test_sessions.py:599`; `tests/test_sessions.py:625` |
| R11 | integration/tripwire | qualification determinism tuple reaches every byte-shaping module | pass | 20-file tuple closes `qualify` plus inventory/video, sessions/multicam, all measure modules, and oracle tests | `scripts/check_qualify_determinism.py:47`; `tests/test_measure.py:280` |
| R12 | integration/tripwire | measurement determinism tuple reaches every byte-shaping module | pass | package glob is mechanically required to be a subset of qualification `SOURCE_FILES`; all 10 modules are present | `tests/test_measure.py:280` |
| R13 | integration/tripwire | calibration-QC determinism tuple reaches every byte-shaping module | pass | 23-file tuple closes the publisher, both probe inputs, upstream parser graph, and four oracle suites | `scripts/check_calibration_qc_determinism.py:35` |
| R14 | integration/tripwire | cohort determinism tuple reaches every byte-shaping module | fail | HIGH: digest covers only `cohort.py` + its test while output reads/rendering depend on four omitted modules | `scripts/check_cohort_determinism.py:37`; `src/pose_estimation/cohort.py:730` |
| R15 | integration/trust | all six publishers require a regular non-symlink marker | fail | HIGH: inventory and sessions follow marker symlinks; the other four reject non-regular markers | `src/pose_estimation/inventory.py:1075`; `src/pose_estimation/sessions.py:812` |
| R16 | integration/trust | all six publishers reject duplicate provenance keys | fail | HIGH: inventory and sessions accept duplicate marker keys; the other four use rejecting hooks | `src/pose_estimation/inventory.py:1075`; `src/pose_estimation/sessions.py:812` |
| R17 | integration/trust | all six census digests cover provenance except the self digest | fail | HIGH: sessions excludes the whole marker; measure excludes the whole generation/upstream block | `src/pose_estimation/sessions.py:588`; `src/pose_estimation/measure/__init__.py:281` |
| R18 | integration/registration | `pyproject.toml` exposes the complete 11-script CLI surface | pass | parsed 11 script keys; all resolve to shipped modules and include the cohort publisher | `pyproject.toml:48` |
| R19 | integration/registration | `.gitignore` names all generated publication trees | pass | `git check-ignore --no-index` matched six roots plus every staged/retiring sibling family | `.gitignore:60`; `.gitignore:103` |
| R20 | integration/registration | entrypoint docs name every shipped command and output | pass | parsed script keys are a subset of the 11-row table; cohort purpose points to its dedicated schema owner | `docs/technical/entrypoints.md:3`; `docs/technical/entrypoints.md:17` |
| R21 | integration/registration | conventions docs name every shared publication contract | pass | auxiliary campaign inventory names registry, qualification, alignment, calibration-QC, cohort, and both mutation gates | `docs/technical/conventions.md:25` |
| R22 | integration/registration | all four M2.8.5 indexes expose the cohort surface | pass | architecture, entrypoints, tests, and conventions each name the module, CLI, suite, or campaign | `docs/technical/architecture.md:28`; `docs/technical/entrypoints.md:17`; `docs/technical/tests.md:62`; `docs/technical/conventions.md:37` |
| R23 | integration/docs | no count survives after its population changed | fail | MEDIUM: roadmap still says 17 restored columns → 92 published beside the measured 89/1068 result | `.agent/roadmap.md:386` |
| R24 | integration/docs | no projection is carried forward as an authoritative budget | pass | all corpus-run projections are labelled and retired; current sizing explicitly uses measured 7.83 h | `.agent/memory.md:309`; `.agent/memory.md:514` |
| R25 | integration/docs | no stale `file:line` citation survives unit movement | fail | MEDIUM: memory cites old `export.py:250` anisotropy; code now uses one max-dimension scale and cohort docs/tests pin 9.9° | `.agent/memory.md:550`; `src/pose_estimation/export.py:147`; `docs/technical/cohort.md:66` |
| R26 | M2.7.3/docs | `calibration_qc.CLAIMS` remains the 15-claim single source of truth | pass | canonical tuple has 15 claims; both prose predicates iterate that tuple rather than another frozen copy | `src/pose_estimation/calibration_qc.py:200`; `scripts/check_claim_report.py:55` |
| R27 | M2.7.3/docs | sampled claims are byte-faithful in both prose copies | pass | C01/C04/C07/C11/C15 each occur exactly once after Markdown flattening in both documents | `docs/calibration_finding.md:53`; `docs/technical/calibration_qc.md:258` |
| R28 | M2.7.3/docs | claim checker + focused tests pass and checker runs under one second | pass | standalone 9/9 green; shared grade took 0.014 s; focused pair contributed 9 of 22 passing tests | `scripts/check_claim_report.py:232`; `tests/test_claim_report.py:24` |
| R29 | M2.7.4/docs | prospective and capture-protocol scope banners are bidirectional | pass | P10 found the exact banner and reciprocal repo path in both normative documents | `docs/prospective_capture.md:7`; `docs/capture_protocol.md:9` |
| R30 | M2.7.4/docs | all five non-negotiables carry literal `MUST` obligations | pass | P03 bound 5 table rows to 7 named sections and found literal `MUST` in every section body | `scripts/check_prospective_capture.py:184` |
| R31 | M2.7.4/docs | prospective checker + focused tests pass and checker runs under one second | pass | standalone 13/13 green; shared grade took 0.032 s; focused pair contributed 13 of 22 passing tests | `scripts/check_prospective_capture.py:370`; `tests/test_prospective_capture.py:24` |
| R32 | conformance/authoring | register scanner covers every durable file touched by the seven units | pass | 56 durable text files classified; copied scanner ran over all 18 human-register candidates; 38 agent/code/payload files reviewed by register | `docs/technical/conventions.md:53` |
| R33 | conformance/authoring | sampled durable text prunes provenance and states future rules positively | fail | LOW: shipped code comments retain contract/ruling provenance (`A02`, `A09`, `P17`) instead of standalone constraints | `src/pose_estimation/cohort.py:3`; `src/pose_estimation/cohort.py:33` |
| R34 | conformance/authoring | sampled human-facing docs meet ASD-STE100 sentence/register constraints | fail | LOW: `steq.py` found 24 flags across four human surfaces, including 22 passive constructions in the two docs units | `docs/calibration_finding.md:69`; `docs/prospective_capture.md:198` |
| R35 | conformance/engineering | touched code is scoped, deduplicated, agent-legible, and comments explain why | fail | MEDIUM: duplicated publisher trust-root machinery drifted in exactly the four ways exposed by R14-R17 | `src/pose_estimation/qualify.py:1557`; `src/pose_estimation/measure/__init__.py:281`; `src/pose_estimation/cohort.py:737` |
| R36 | conformance/engineering | deterministic checks own every mechanically decidable new rule | unknown | unknown | unknown |
| R37 | conformance/state | roadmap prunes closed-M2 detail into archive with quantified savings | unknown | unknown | unknown |
| R38 | conformance/state | memory prunes obsolete controls and agrees with `.claude/rules/` routing | unknown | unknown | unknown |
| R39 | conformance/state | polish register prunes obsolete/redundant entries with quantified savings | unknown | unknown | unknown |

## Detail

### R08 — LOW — seventh artifact enumerator omitted from the integration law

- Evidence: `tests/test_corpus_run_preconditions.py:858` fixes a five-name `group_qc` tuple independently of the six places listed at `.agent/memory.md:656`.
- Divergence: the repository has seven independent update sites; the durable budget says six. The producer itself constructs suffixes and does not add another fixed set.
- Breached predicate: the artifact-enumerator census must be complete before it can bound publication wiring work.
- Impact: a new dataset or mode can satisfy the six recorded sites while P16 silently retains an obsolete population, or planning can under-budget the seam.
- Acceptance check: replace the duplicated P16 tuple with an import from the canonical golden set, or update `.agent/memory.md` to seven and add a repository check that inventories every independent literal set.
- Red-test pointer: `tests/test_corpus_run_preconditions.py:858` is the missed enumerator itself; a new dataset added only to `_DATASETS` demonstrates the stale-population failure.

### R09 — HIGH — inventory determinism evidence is unbound to its sources

- Evidence: `scripts/check_inventory_determinism.py:256` writes `tested_head`; the script defines no `SOURCE_FILES`, source digest function, or stale-result refusal. `tests/inventory_determinism_results.json:1` consequently carries no source map.
- Divergence: the committed PASS can survive changes to `inventory.py`, `video_io.py`, the checker, or its oracle because the recorded head names the pre-regeneration parent and no test compares source bytes.
- Breached predicate: durable determinism evidence must bind to every module that shapes published bytes through current source digests.
- Impact: the milestone can quote a green campaign that did not execute the code under review; an undetected nondeterministic edit remains certification-green.
- Acceptance check: add a spelled-out `SOURCE_FILES` tuple, record per-file SHA-256 values, refuse regeneration over mismatched evidence, and add a committed test that recomputes every digest from the worktree.
- Red-test pointer: port the stale-source assertions at `tests/test_m2u5_alignment.py:1049`; they fail against the current inventory result schema.

### R14 — HIGH — cohort evidence omits direct byte-shaping dependencies

- Evidence: `scripts/check_cohort_determinism.py:37` hashes only `src/pose_estimation/cohort.py` and `tests/test_cohort.py`. The publisher uses `inventory.render_json` at `src/pose_estimation/cohort.py:730`, `inventory.render_csv` at `src/pose_estimation/cohort.py:883`, `corpus_run.read_manifest` at `src/pose_estimation/cohort.py:852`, and sessions/qualification parsers throughout input loading.
- Divergence: edits to `inventory.py`, `corpus_run.py`, `sessions.py`, or `qualify.py` can change accepted inputs or published bytes while `source_digest` stays unchanged; the checker itself is also outside its evidence digest.
- Breached predicate: the determinism tripwire must reach every module that shapes a published tree and every oracle source that certifies it.
- Impact: committed cohort evidence can remain green after the implementation under review changes its output bytes or trust decisions.
- Acceptance check: replace `SOURCES` with a spelled-out per-file source map covering the checker, `cohort.py`, `corpus_run.py`, `inventory.py`, `sessions.py`, `qualify.py`, and `test_cohort.py`; assert the committed result equals recomputed hashes.
- Red-test pointer: mutate `inventory.render_csv` or `corpus_run.DISPOSITION_OK`; a source-staleness test must fail before the campaign is regenerated.

### R15 — HIGH — two publisher trust roots follow marker symlinks

- Evidence: `src/pose_estimation/inventory.py:1075` calls `read_text` on `census.json` without `lstat`; `src/pose_estimation/sessions.py:812` does the same for `generation.json`. Qualify, measure, calibration-QC, and cohort explicitly reject symlink/non-regular markers.
- Divergence: inventory and sessions accept a marker whose bytes live outside the published tree, so the trust root is not owned by the set it certifies.
- Breached predicate: every publisher marker must be a regular non-symlink file before parsing or ownership decisions.
- Impact: swapping an external marker can alter generation claims without changing the tree entry, and ownership/validation can trust attacker- or peer-controlled bytes.
- Acceptance check: `lstat` each marker, require `stat.S_ISREG`, normalize failures to the publisher error domain, and add symlink, dangling-symlink, directory, FIFO, and device cases for both modules.
- Red-test pointer: mirror `tests/test_qualify.py:1005` and `tests/test_cohort.py:1810` for inventory and sessions; both cases are red now.

### R16 — HIGH — inventory and sessions accept ambiguous marker documents

- Evidence: `src/pose_estimation/inventory.py:1075` and `src/pose_estimation/sessions.py:812` use plain `json.loads`; a synthetic marker with the same key stated twice validated in both modules. Qualify, measure, calibration-QC, and cohort use `object_pairs_hook` rejection.
- Divergence: last-key-wins parsing lets one trust-root document carry two claims while validation reasons over only the latter.
- Breached predicate: publisher marker parsing must reject every duplicate key at every nesting depth.
- Impact: a sessions marker can present one ownership/provenance claim to review and another to the parser, including the claim that licenses replacing a non-empty tree.
- Acceptance check: share a duplicate-key-rejecting JSON loader across all six publishers and add top-level plus nested duplicate cases for inventory and sessions.
- Red-test pointer: mirror `tests/test_measure.py:180`; duplicate-key inventory and sessions cases are red now.

### R17 — HIGH — sessions and measurement self-digests omit upstream provenance

- Evidence: `src/pose_estimation/sessions.py:588` excludes all of `generation.json` from `tree_digest`; `generation_digest` is only an external byte witness. `src/pose_estimation/measure/__init__.py:281` removes the whole `generation` block before `manifest_digest`. Synthetic edits to each marker's `inventory` claim validated without recomputing any digest when the optional upstream directory was omitted.
- Divergence: the two digests cover tables/axes but not the provenance claims that consumers use to connect them to an upstream generation.
- Breached predicate: the marker self-digest must cover the provenance block while excluding only its own self-referential digest key.
- Impact: corruption that stops at an upstream claim is undetected by standalone validation, contradicting the detection guarantee and allowing a sidecar/tree to be relabelled as another registry generation.
- Acceptance check: add a self-digest key to sessions and include the marker minus that key; change measure `manifest_digest` to include `generation` minus `generation.manifest`; add one-byte provenance tamper cases without an upstream argument.
- Red-test pointer: edit only `generation.inventory` in the synthetic builders at `tests/test_sessions.py:168` and `tests/test_measure.py:68`; both validators accept today.

### R23 — MEDIUM — active roadmap preserves a rejected feature census

- Evidence: `.agent/roadmap.md:386` states that body tracking populates all 17 trunk/posture columns and yields 92 published features. The adjacent authoritative rows state 89 published / 3 excluded / 1068 feature rows at `.agent/roadmap.md:385` and `.agent/roadmap.md:387`.
- Divergence: the corrected corpus restored 14 of 17 columns because three sagittal fields remain structurally absent in 2D; 92 was a projection, not the measured published population.
- Breached predicate: an active count must move when its population moves, with rejected projections retained only in archived history or explicitly marked as pre-measurement.
- Impact: a future schema change or review can budget and assert 92 rows from the milestone table while the publisher and census enforce 89.
- Acceptance check: rewrite the M2.8.4 spine result to 89 published / 3 excluded, or label 92 as the rejected pre-run projection and point directly to the measured correction.
- Red-test pointer: the derived Cartesian oracle at `tests/test_cohort.py:880` rejects a stale 92-name `FEATURES` population.

### R25 — MEDIUM — isotropy repair did not reach the cohort claim surfaces

- Evidence: `.agent/memory.md:550` states that current export divides x by width and y by height and cites stale `export.py:250`; current `src/pose_estimation/export.py:147` returns one `max(frame_w, frame_h)` scale. `docs/technical/cohort.md:66`, `src/pose_estimation/cohort.py:41`, and `tests/test_cohort.py:2097` still state or pin the pre-fix 9.9° anisotropy error.
- Divergence: M2.8.4 changed the shipped corpus to `image-isotropic-maxdim`; the 9.9° measurement describes the deleted pre-fix tree, not current cohort inputs.
- Breached predicate: later-unit repairs must update every durable claim, explanatory comment, test oracle, and `file:line` citation they invalidate.
- Impact: consumers are told that a removed normalization defect remains in every angle, and the suite now protects the stale statement against correction.
- Acceptance check: update memory, code comment, cohort documentation, and documentation-token test to distinguish image-plane/projection limits from the removed anisotropic normalization error; cite `coord_scale` by symbol or its current line and assert the isotropic token.
- Red-test pointer: change the doc to the current isotropic claim; `tests/test_cohort.py:2097` fails because it requires the obsolete `9.9` token.

### R33 — LOW — shipped comments retain ruling provenance instead of the rule

- Evidence: `src/pose_estimation/cohort.py:3` names the archive contract and `A02`; `src/pose_estimation/cohort.py:33` defines the exception through `A16/P17`; further comments use `A09`, `A20`, `A21`, and `D05` as sentence subjects.
- Divergence: the comments require readers to recover historical amendment identifiers before they can understand current code, despite the authoring rule to prune provenance and state durable constraints directly.
- Breached predicate: durable text must retain the operative reason and target while omitting discovery history and ruling provenance.
- Impact: fresh-agent read cost rises, identifiers go stale when contracts are archived or amended, and the current false anisotropy rationale survived behind an amendment label.
- Acceptance check: rewrite source comments to state the closed-domain, error-domain, privacy, and measured-schema constraints without A/P/D identifiers; a repository comment scan leaves no ruling ids outside frozen `.agent/archive/` records.
- Red-test pointer: a purpose-built authoring check over `src/`, `scripts/`, and live docs must report the current `cohort.py` ruling-id comments and ignore archived contracts.

### R34 — LOW — human-facing register scan is red

- Evidence: copied `.scratch/steq.py` returned rc=1 over all 18 touched human-register candidates: 12 flags in `docs/calibration_finding.md`, 10 in `docs/prospective_capture.md`, one in `docs/capture_protocol.md`, and one CLI-help flag in `src/pose_estimation/sessions.py`. Clear examples are passive experimental prose at `docs/calibration_finding.md:69` and passive obligations at `docs/prospective_capture.md:198`.
- Divergence: sentence lengths pass, but the ASD-STE100 active-voice rule does not; the prospective checker enforces only the 25-word ceiling.
- Breached predicate: human-facing prose must use active voice, fixed terminology, and the configured filler/contraction bounds as well as sentence length.
- Impact: normative owners and actions become implicit, especially in calibration and consent requirements; the committed checker gives a green result over only part of the declared register.
- Acceptance check: clear every true-positive `steq.py` finding, narrow any scanner false positives explicitly, and wire the same scan to both docs-tier checkers so register conformance cannot drift green.
- Red-test pointer: `.scratch/steq.py --max 25` over the 18-file touched-surface list currently exits 1 with total 24.

### R35 — MEDIUM — repeated publication primitives have already diverged

- Evidence: qualification, measurement, calibration-QC, sessions, inventory, and cohort each implement variants of marker parsing, self-digesting, tree walking, ownership, staging, and removal. Compare `src/pose_estimation/qualify.py:1557`, `src/pose_estimation/measure/__init__.py:281`, and `src/pose_estimation/cohort.py:737`; cohort also imports qualification's private duplicate-key hook.
- Divergence: no shared publication boundary owns the three trust-root properties or source-reach contract, producing the missing guards and incompatible digest scopes in R14-R17.
- Breached predicate: cross-cutting deterministic publication mechanics must be deduplicated into a tightly scoped module or mechanically forced to conform.
- Impact: every new publisher copies a subtly different sibling idiom, so security and determinism fixes land in some trees and silently miss others.
- Acceptance check: extract or generate one marker-reader/self-digest primitive with parameterized schemas, then add a six-publisher conformance matrix for regular-file, duplicate-key, provenance-tamper, and source-reach cases.
- Red-test pointer: the synthetic inventory/sessions marker tests described in R15-R17 fail the desired matrix today.

## Register

Out-of-contract observations carry an evidence pointer + acceptance check.
