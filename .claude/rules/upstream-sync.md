---
paths:
  - "CLAUDE.md"
  - ".claude/rules/*.md"
  - ".agent/*.md"
  - ".agent/archive/*.md"
---

# Upstream instruction sync

`CLAUDE.md` arrives as an upstream drop-in landing over this repo's local adaptations.
**The clauses below are the only surviving copy.**

**Procedure, every refresh:** diff the refreshed file against its prior commit, re-apply every
clause below, keep every upstream change no clause contradicts, commit `state: …`. Verify with
`rg -l 'archive/contract-' scripts/ src/ tests/` = 7, re-derived rather than trusted — a
whole-tree sweep counts every document that merely mentions the path and drifts on every edit
(18 at this writing). `git log --grep upstream` lists priors. A refresh that retires a constant
also falsifies claims elsewhere → sweep `.agent/` and `.claude/rules/` for the retired term and
correct what depended on it, since a stale sizing datum reaches planning as a budget.

- **Clause 1 — acceptance contracts live at `.agent/archive/contract-m<m>u<u>.md`**, never at
  `.agent/contracts/`. Upstream's requirements — committed, outside the attached set, read on
  demand — are all met at the archive path, and **7 files under `scripts/ src/ tests/` break if
  it moves**, one of them a generated data field: `scripts/make_calibration_qc_fixtures.py` writes
  the path into `tests/fixtures/calibration_qc_set/manifest.json`, and
  `check_calibration_qc_fixtures.py` validates digests without resolving that field, so a rename
  missing the generator leaves a dangling pointer no gate reports.
- **Clause 2 — a review closes on fixes applied, never on rows adjudicated.** An adjudicated row
  whose ruling accepts a fix stays open until that fix closes on its own acceptance check under
  MAIN's rerun. A wave that rules every row and applies none has not closed; it re-enters to
  apply them. Measured: M2's wave 1 adjudicated 193 rows / 55 fails in one pass and needed three
  further sessions to close 40 of its 43 accepted fixes.
- **Clause 3 — mutable state belongs in `.agent/spec.md` + `.agent/deferred.md`, never in
  `.claude/rules/`.** A rule file is read as standing law, so a mid-session ledger parked there
  reads as current long after it stops being true. Anything MAIN rewrites while it works — status,
  open rows, counts — stays in those two; `.claude/rules/` takes only what holds until new
  evidence reverses it, and points at the queue rather than restating a row.
- **Clause 4 — `.agent/archive/` is this repo's detail store and its records are frozen.**
  `roadmap.md`, `polish.md`, `review-m2.md`, the 14 unit contracts and the reviewer reports live
  there and keep their own stale pointers (`/session-roadmap`, `.agent/memory.md`, a `Read()`
  deny list). Read an archive pointer as a citation of its own time. The live surfaces are
  `.agent/spec.md`, `.agent/deferred.md` and `.claude/rules/`.

**Superseded by upstream — never restore.** The `Read()` path-exclusion control (→
`data-boundary.md`). The `N% NK/1M` gauge convention: MAIN's window is whatever
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` clamps it to, so no literal belongs in it. `.agent/roadmap.md`
and `.agent/polish.md` as attached state, and the `/session-roadmap`, `/session-prompt`,
`/session-polish` commands that consumed them — `.agent/spec.md` is now the sole attached state
and the phase flow replaces the MODE dispatch. The **`≤ 8 KB` cap on `.agent/spec.md`**: liveness
replaces it — every line binds current or future work, superseded text dies in the commit that
supersedes it, size is emergent. A byte budget rewards compressing live prose over deleting dead
rows, which is the opposite of the ranking. **The deferral queue as a `spec.md` section**: a queue
is monotonic, so attached it is a permanent growth term; it lives at `.agent/deferred.md` and
`spec.md` `Deferred` keeps the pointer plus whatever blocks the current spine.
`.serena/project.yml` `ignored_paths` and the `read-guard.sh` volume budget: both mechanisms are
deregistered toolchain-wide.

**A `scripts/check_drop_ins.py` gate was weighed and declined**: the user announces every
refresh, and the clauses self-evidence against the tree — 7 shipped files name the archive path,
so a reverted `CLAUDE.md` contradicts them on sight. Rigor concentrates where no re-check exists.
