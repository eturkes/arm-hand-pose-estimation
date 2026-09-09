# Spec

## Intent

Turn the hospital's uncontrolled three-camera recordings of spinal-cord-injury rehabilitation
tasks into measured movement statistics that feed the `../rehab` clinical dashboard over the
hospital SCI database.

- Corpus = `videos/3-cam/`: 382 hand-held clips, 16 subjects × 6 tasks × 2 sides, three views per
  family at best, no fixed rig, no calibration.
- The pipeline tracks 2D pose per clip, extracts clinical kinematic features, and aggregates to
  cohort level — no per-subject rows, no patient identifier, no join column leaving this repo.
- The shareable review UI is itself an intended deliverable for collaborators, not only a
  prototype step.
- Eventual destination = a subject↔patient join into the hospital SCI database; that mapping does
  not exist yet.

## Artifacts

`P` = the gate prefix; `P` + the mutually-exclusive accelerator recipe → `.claude/rules/gates.md`.

- `prototype/review-ui/` — **the inspectable artifact.** One local web UI, bilingual ja/en
  (`?lang=en`), three views: corpus census · clip player with pose overlay · cohort explorer.
  FastAPI + vanilla JS canvas + vendored Plotly/IBM Plex, own uv project, read-only over the
  published trees, degrading per absent tree. Synthetic fixture = the only committed media; its
  `README.md` = view guide + regeneration + limits.
  `uv run --directory prototype/review-ui python -m review_ui` → `http://127.0.0.1:8791/`.
  Proof → `proof/`: 4 captures + API transcript, by `tools/capture_proof.py`.
- `cohort/` — the `../rehab` export. 12 `(task, side)` cells · 89 features · 1068 rows;
  `descriptors.yaml` = ja/en labels, units, ranges.
  `P pose-estimation-cohort --inventory inventory --sessions sessions --run output/corpus-2d --out cohort`
- `output/corpus-2d/` — per-asset 2D landmarks + clinical features, 379 assets / 193 events /
  331 152 frame rows, 7.828 h under the accelerator recipe. `python scripts/corpus_run_2d.py`.
- `inventory/` `sessions/` `qualification/` `calibration_qc/` — four publishers upstream of the
  run; each `P pose-estimation-<name> … --out <dir>`; `--help` = args.
- Decisive gate — `P pytest`, 1750 tests, 14-25 min, alone.

## Decisions

- **Repo scope = `videos/3-cam/`** — retired data + siblings → `.claude/rules/data-boundary.md`.
- **`src/` + `tests/` + `analysis/` + six publishers = production spine**, gates binding.
- **Claim boundary.** Retrospective 3D feasibility may be claimed from internal geometric + QC
  evidence alone; clinical validity, absolute metric accuracy and marker-based equivalence may not.
  Crossing it needs the prospective calibrated capture in `docs/prospective_capture.md`.
- **3D is closed negative.** Extrinsic recovery is unachievable here — 15-20 px systematic
  cross-view keypoint bias, refused by the shipped estimator and by independent bundle adjustment,
  not transferring across events, so no repair route survives. Publishes through `calibration_qc/`;
  reopens on prospective calibrated capture.
- **Metric scale is unavailable.** Over a stratified 52/379 sample exact dimensional identity
  resolved 0/52, best conditional route floors at ±17.7 %. Angles, angular velocities, timing +
  dimensionless shape survive; metre-valued distance, velocity + jerk do not. Publishes
  `scale_unmeasured` on all 379 rows.
- **Calibration is per recording event at best** — no fixed rig existed; orientation, codec,
  parity + duration spread each measure it independently.
- **M3 (analysis-ready 3D aggregation) DESCOPED + terminal** — M3.1/M3.2/M3.3a ship + keep gates;
  M3.3b + M3.4-M3.6 cut. Revive = user ruling → `.agent/archive/m3.md`.
- **`publication.py` extraction stays DECLINED** — six publishers keep their own
  staging/swap/digest/ownership copies; the measured five-way drift is the evidence.
- **Detector on CPU, pose on NPU** — NPU pads YOLOX's dynamic output with uninitialised memory
  that passes every validity filter as real rows.
- **Acceptance contracts live at `.agent/archive/contract-m<m>u<u>.md`** — 7 files under
  `scripts/ src/ tests/` break if it moves, one a generated data field no gate resolves;
  re-derive → `upstream-sync.md`.
- **Assurance tier = `kernel`** across pipeline, publishers + analysis.

## Deferred

Evidence → `.agent/archive/{polish,review-m2}.md`; regen → `gates.md`. Read rows first.

- **`isotropy_angle_fidelity_results.json` pins sample ids alone** (T2 R23) → binds script,
  validated-registry generation, run-tree + config digests.
- **Cohort census omits the external descriptor digest** (T4 R37) → census records it;
  `descriptor_collision` re-checkable.
- **Two census digests skip their own provenance block** (T5 R17, `sessions.py:588`,
  `measure/__init__.py:281`) → each covers its block minus its self-referential key.
- **`inventory_mutation_results.json` + `measure_mutation_results.json` target moved bytes** (2/2,
  2/3) → each campaign runs from clean checkout, no target-mismatch refusal, reproduces its kill
  list, and exits 1 on a seeded survivor with `M028` the sole allowlisted equivalent — today
  `run_inventory_mutations.py` returns 0 whatever survives (their only firing → `gates.md`).
- **Six legacy R clinical consumers re-implement suffix/read/bind** → one mode-aware reader; a
  mixed 2D+3D tree yields named mode outputs or rejection, 2D goldens unchanged.
- **`make_templates.R` + `validate_metadata.R` skip `_3d` silently** → both route through that
  reader; 2D templates + validation unchanged.
- **Four `../rehab` JP faces miss 14 cohort-label characters** → each reports 0 missing code points
  over the `cohort.FEATURES` `ja` union.
- **Gate prefix recalled by failure** → `scripts/gate.sh pytest -q` collects in primary tree + fresh
  worktree, bare `uv run pytest` still reproduces `GLIBC_2.43`, 0 bare invocations in `.agent/spec.md`
  + `.claude/rules/` + `docs/` + `scripts/` (`.agent/archive/` frozen, keeps its bare forms).
- **12/12 `check_*.py` ship green-only; `run_measure_mutations.py` firing unrecorded** (census +
  acceptance → `gates.md`) → each checker grows a committed seed driving it to its own named
  failure; a runner reports 12/12 refused, rc=1 naming one that stops. Covers the
  `nc_m2u74.py` port: 9/9 from clean, `SEED-INERT` on an inert seed,
  `docs/prospective_capture.md` byte-identical.
- **`steq.py` + `fidelity.sh` port; 24 register flags on four human surfaces** → one committed
  scanner reports 0 `LONG`/`FILLER` over the `conventions.md` inventory, fails a seeded 30-word
  instruction, no specifier/flag/number delta.
- **M2.8.4's two corpus checks port** → `scripts/check_corpus_2d_integrity.py` rc=0 on
  `output/corpus-2d` (379/379 set-equal, 11 verdicts true, breach < 0.1 %), rc=1 named on 3 tampers.
- **`--det-device GPU` unqualified against real detections** (9.7 vs 213 ms → ~7.1 h toward ~1 h) →
  GPU + CPU agree on the detection set, every padded row rejected by value, pilot green; then flip.
- **Detector scores outside `[0,1]` accepted silently** → score-range guard fires on NPU, silent on
  CPU + GPU, under the synthetic zeros probe.
- **`pose-estimation-run --session-dir … --output-dir X` ignores `X`** → rtmlib session artifacts
  land under the requested root; MediaPipe unchanged.
- **`sessions.py` predicates pinned by tests alone** → a campaign mirroring
  `run_inventory_mutations.py` kills every mutant with >=1 committed test, replaying from clean.
- **Review UI JP subset builds from gitignored `cohort/descriptors.yaml`** → `build_assets.py`
  refuses with a named cause when absent; a committed check reports 0 missing code points.
- **Overlay landmark->pixel map proven by eye alone** → headless check drives the fixture's known
  coordinates through the canvas scale math, max deviation < 1 px, failing on a seeded off-by-one.
- **HEVC decode failure reported but never exercised** (123/379 hevc) → an hevc clip on a
  decoder-less build shows the `player.decode_failed` banner, overlay still advancing on rAF.

## Phase

**ITERATE.** `prototype/review-ui/` runs by its recorded command, proof under `proof/`.
IMPLEMENT starts on the user's go.
