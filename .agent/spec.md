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

`P` = the gate prefix; `P` and the mutually-exclusive accelerator recipe → `.claude/rules/gates.md`.

- `prototype/review-ui/` — **PROTOTYPE target.** One local web UI, bilingual ja/en, three views:
  corpus census · clip player with pose overlay · cohort-stats explorer. FastAPI + vanilla JS
  canvas + vendored Plotly/IBM Plex, its own uv project, read-only over the published trees.
  `uv run --directory prototype/review-ui python -m review_ui` → `http://127.0.0.1:8791/`.
  Proof → `prototype/review-ui/proof/`. Synthetic fixture clip = the only committed media.
- `cohort/` — the `../rehab` export. 12 `(task, side)` cells · 89 features · 1068 feature rows;
  `descriptors.yaml` carries ja/en labels, units, ranges.
  `P pose-estimation-cohort --inventory inventory --sessions sessions --run output/corpus-2d --out cohort`
- `output/corpus-2d/` — per-asset 2D landmarks + clinical features, 379 assets / 193 events /
  331 152 frame rows, 7.828 h measured under the accelerator recipe.
  `python scripts/corpus_run_2d.py` (trees defaulted).
- `inventory/` `sessions/` `qualification/` `calibration_qc/` — the four publishers upstream of
  the run; each `P pose-estimation-<name> … --out <dir>`, `--help` gives the argument set.
- Decisive gate — `P pytest`, 1750 tests, 14-25 min, alone.

## Decisions

- **Repo scope = `videos/3-cam/`.** `videos/initial/` = retired preliminary data. Siblings under
  the same data root are out: `harness/` = a capture harness never built, `database/` = the
  hospital SCI records, the eventual join target.
- **Existing `src/` + `tests/` + `analysis/` + the six publishers = production spine**, kept whole
  under PROTOTYPE with their gates binding. The prototype UI is additive; `prototype/` alone
  retires at IMPLEMENT close.
- **Claim boundary.** Retrospective 3D feasibility may be claimed from internal geometric + QC
  evidence alone. Clinical validity, absolute metric accuracy and marker-based equivalence may
  not. Crossing it needs the prospective calibrated capture `docs/prospective_capture.md`
  specifies.
- **3D is closed negative.** Extrinsic recovery is unachievable on this corpus (15-20 px
  systematic cross-view keypoint bias, refused by the shipped estimator and by independent bundle
  adjustment), and the bias does not transfer across events, so no repair route survives. The
  ruling publishes through `calibration_qc/`; it reopens only on prospective calibrated capture.
- **Metric scale is unavailable.** Over a stratified 52/379 sample, exact dimensional identity
  resolved 0/52 and the best conditional route floors at ±17.7 %. Angles, angular velocities,
  timing and dimensionless shape survive; metre-valued distance, velocity and jerk do not. The
  axis publishes `scale_unmeasured` on all 379 rows.
- **Calibration is per recording event at best** — no fixed rig ever existed, which orientation,
  codec, parity and duration spread each measure independently.
- **Cohort publication stops at this repo's boundary**: aggregates only, no subject rows, no
  identifier, no join column, no join to `../rehab`.
- **M3 (analysis-ready 3D aggregation) is DESCOPED and terminal.** M3.1, M3.2 and M3.3a ship and
  keep their gates; M3.3b and M3.4-M3.6 are cut. Reviving any part is a user ruling. Record →
  `.agent/archive/m3.md`.
- **The `publication.py` extraction stays DECLINED.** The six publishers keep their own copies of
  the staging/swap/digest/ownership machinery; the measured five-way drift is the evidence, and
  each drift is fixed where it lands.
- **Detector runs on CPU, pose on NPU.** NPU pads YOLOX's dynamic output with uninitialised
  memory, which passes every validity filter as real rows.
- **Acceptance contracts live at `.agent/archive/contract-m<m>u<u>.md`** — 15 files bind that
  path, one of them a generated data field no gate resolves.
- **Assurance tier = `kernel`** across the shipped pipeline, publishers and analysis surface.

## Deferred

Full evidence for every row, live or retired, sits in `.agent/archive/polish.md` and
`.agent/archive/review-m2.md` — read a row there before acting.

- **`tests/isotropy_angle_fidelity_results.json` pins sample ids alone** (review T2 R23) → the
  JSON binds script, validated-registry generation, run-tree and effective-config digests.
- **The cohort census omits the external descriptor digest** (T4 R37) → the census records it, so
  `descriptor_collision` is re-checkable.
- **Two census digests skip their own provenance block** (T5 R17, `sessions.py:588`,
  `measure/__init__.py:281`; design settled, unimplemented) → every census digest covers its
  provenance block minus its own self-referential key.
- **`tests/inventory_mutation_results.json` targets bytes no longer shipped** → the campaign runs
  from a clean checkout with no target-mismatch refusal and reproduces its kill list.
- **Six legacy R clinical consumers each re-implement suffix/read/bind** → one central mode-aware
  reader; a synthetic mixed 2D+3D directory yields separately named mode outputs or a rejection;
  2D goldens unchanged.
- **`make_templates.R` + `validate_metadata.R` silently skip `_3d`** → both route through that
  reader; 2D templates and validation unchanged.
- **All four `../rehab` JP faces miss the 14 characters the cohort labels use** → each face
  reports zero missing code points over the `cohort.FEATURES` `ja` character union; one sampled
  label renders with no tofu. Work lands in `../rehab`.
- **The gate prefix is recalled by failure** → `scripts/gate.sh pytest -q` collects in the primary
  tree and in a fresh worktree, a bare `uv run pytest` still reproduces the `GLIBC_2.43`
  ImportError, and zero bare invocations remain in `.agent/`, `docs/`, `scripts/`.
- **`.scratch/nc_m2u74.py` is scratch-local behind a durable 9/9 claim** → a committed check
  reports 9 of 9 firing from a clean checkout, leaves `docs/prospective_capture.md`
  byte-identical, and reports `SEED-INERT` on a seeded inert control.
- **`.scratch/steq.py` + `.scratch/fidelity.sh` are scratch-local, and 24 register flags stand on
  four human surfaces** → one committed check reports 0 `LONG`/`FILLER` over the
  `conventions.md` *Text register* inventory, fails a seeded 30-word instruction, and
  `fidelity.sh` shows no specifier/flag/number delta. One pass, one scanner.
- **M2.8.4's two corpus checks are scratch-local** → `scripts/check_corpus_2d_integrity.py` rc=0
  on `output/corpus-2d` (379/379 set-equal, 11 verdicts true, breach rate < 0.1 % both aspects)
  and rc=1 with a named cause on each of three synthetic tampers.
- **`--det-device GPU` is unqualified against real detections** (9.7 ms vs CPU 213 ms → the corpus
  run falls ~7.1 h toward ~1 h) → GPU and CPU agree on the detection set with every padded row
  rejected by value; the pilot reruns green; then the flag flips.
- **A detector returning scores outside `[0,1]` is accepted silently** → a score-range guard fires
  on NPU and stays silent on CPU + GPU under the synthetic zeros probe.
- **`pose-estimation-run --session-dir … --output-dir X` ignores `X`** → rtmlib session artifacts
  land under the requested root; MediaPipe session behaviour unchanged.
- **`sessions.py` shipped with its predicates pinned by tests alone** → a mutation campaign
  mirroring `run_inventory_mutations.py` kills every mutant with ≥1 committed test and replays
  from a clean base.

## Phase

**PROTOTYPE.** The inspectable artifact is `prototype/review-ui/`, unbuilt. It reaches ITERATE
when the UI runs by its recorded command with proof under `prototype/review-ui/proof/`.
