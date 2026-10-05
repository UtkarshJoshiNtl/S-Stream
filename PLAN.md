# S-Stream Master Product, Correctness, and UX Plan

**Status:** Active master plan — execution in progress
**Audit snapshot:** 2026-09-25
**Execution window:** 2026-08-30 → 2026-10-03 (69 commits, C01–C69)
**Scope:** Product positioning, UI/UX, maneuverability, usability, simulation correctness, analysis, performance, accessibility, testing, documentation, and release trust.

Every item in this document is tracked as a numbered commit in §11. When a commit lands,
its row in §11 is marked `[x]` and an entry is appended to the progress log in §12.

## 1. Product Direction

S-Stream is an interactive, educational fluid-dynamics workbench. It is not intended to replace OpenFOAM, ANSYS Fluent, or other production engineering CFD suites. Its primary promise is:

> Make a fluid concept understandable in under one minute, while making the limits of the result visible.

The product should optimize for this order:

1. Accuracy and honest capability labels
2. Performance and responsiveness
3. Usability, discoverability, and accessibility
4. Additional features

The application must let a user:

1. Start from a meaningful preset or guided story.
2. See a valid result immediately.
3. Understand which controls are safe to change.
4. Edit geometry, probes, and parameters without losing state.
5. Observe stability, convergence, and limitations continuously.
6. Undo mistakes safely.
7. Export a reproducible result with trustworthy metadata.

Every feature exposed in the UI must have a defined state:

- **Verified:** Implemented, automatically tested, and bounded by a documented operating envelope.
- **Experimental:** Available for exploration, visibly labeled, with warnings and limitations.
- **Hidden:** Not exposed as a normal product workflow until its dependencies and validation are complete.

A capability is not Verified merely because it renders or runs without an exception.

## 2. Product Success Criteria

### 2.1 First-use success

A new user can:

- Launch the application and reach a useful simulation without editing source code.
- Understand what the current scene represents.
- Run, pause, step, reset, and inspect a result.
- Find the primary controls without reading the source.
- Place an obstacle and a probe using discoverable interactions.
- Understand whether the simulation is stable and whether the displayed result is trustworthy.
- Export a result and its provenance.

Target: first meaningful visualization in 60 seconds or less on the supported reference machine.

### 2.2 Trust success

The user can always answer:

- What model and engine is running?
- What boundary conditions and forcing assumptions apply?
- Which fields and units are displayed?
- Is the result stable, developing, or invalid?
- Which capabilities are verified, experimental, or hidden?
- What validation evidence supports the current result?

### 2.3 Usability success

- No silent fallback from an unsupported field to smoke.
- No drawing or editing action without visible feedback.
- No state mismatch after preset load, reset, engine switch, or undo.
- No long-running job that can be silently lost when a window closes.
- No primary action exists only in an undocumented shortcut.
- Keyboard, mouse, and visible controls follow the same interaction model.

## 3. Existing Strengths to Preserve

- `SimEngine` provides an engine-facing abstraction in `engines/base.py`.
- The workbench is separated from solver implementations and mostly talks through the engine interface.
- The GUI has a left accordion sidebar, canvas tool strip, bottom transport bar, live fields, probes, outcome panels, presets, recipes, and export.
- Scene serialization and product metadata support reproducible educational workflows.
- `GUI_REDESIGN_PLAN.md` established the visual direction and is now historical context; this document is the consolidated source of truth.
- The educational positioning in `README.md` and the trust matrix in `TRUST.md` are differentiators, provided they stay synchronized with the code.

## 4. Severity Definitions

- **P0:** Breaks a core advertised workflow, causes deterministic runtime failure, corrupts displayed state, or creates a serious trust violation.
- **P1:** Produces misleading data, loses user work, breaks lifecycle guarantees, or makes a supported workflow unreliable.
- **P2:** Material usability, accessibility, performance, discoverability, or maintainability debt.

All findings below were re-verified against the code at the audit snapshot. Every one is
confirmed; the audit also corrected three framings (see §5.3) and surfaced additional
defects (see §5.4), all of which are folded into the commit plan in §11.

## 5. Findings

### 5.1 UI/UX and maneuverability — P0

| ID | Finding | Evidence | Commits |
|----|---------|----------|---------|
| P0-1 | Obstacle overlay rendering has a variable-shadowing defect: the obstacle loop in `_draw_overlay()` rebinds the obstacle mask to a scene object, so quiver/streamline/force-arrow helpers receive a non-array. | `workbench/viewport.py:409-448`, `:628`, `:710-711`, `:953` | C13, C14 |
| P0-2 | Preset and wizard grid dimensions do not rebuild the engine. 7 of 10 presets load against a mismatched solver. | `scene/scene.py:481-493`, `workbench/app.py:502-526`, `:460`, `:471`, `:865`, `:883` | C24, C25 |
| P0-3 | The flagship vortex-shedding recipe does not resolve to a preset: the action table keys on a filename stem but looks up a display name, and `karman_street.json` is named `High-Re Wake`. | `workbench/app.py:1050-1087`, `presets/scenes/karman_street.json:3`, `presets/loader.py:14-32` | C15, C16, C17 |
| P0-4 | Reset and scene changes leave runtime state stale: probe histories, stability trace, demo state, recording, scrubbing flag, timeline, and overlay caches all survive. `_apply_and_refresh()` does not clear the timeline at all, so scrubbing after a preset load writes the previous scene's state into the new solver. | `workbench/app.py:502-526`, `:625-644`, `workbench/viewport.py:146-165` | C26, C27, C28, C29, C33 |
| P0-5 | Undo/redo is split across objects and snapshots occur after mutation, so the first edit is unreachable and the panel can end up owning a different `Scene` object than `MainWindow`. | `workbench/panels/scene_panel.py:1011-1015`, `:1133-1160`, `workbench/app.py:510`, `:723-729` | C30, C31, C32, C34 |
| P0-6 | The field catalog is a static list including phase/temperature/component fields regardless of engine. Three distinct silent-failure modes exist: exception → smoke in the viewport, the same in `export/image.py`, and a flat `0.5` field returned for temperature when thermal is disabled. | `workbench/app.py:51-62`, `:777`, `workbench/viewport.py:364-369`, `export/image.py:34-37`, `engines/lbm2d.py:463-473` | C18, C19, C20 |
| P0-7 | Preset probe fields can be unrecordable: `oil_water_separation.json` requests `component1`, `component2`, `color`, none of which `Probe.record()` supports, and unsupported fields are dropped silently while the history key is pre-seeded. | `presets/scenes/oil_water_separation.json:13-15`, `scene/probe.py:16-41`, `workbench/panels/analysis_panel.py:44-49`, `:67-72` | C21, C22, C23 |

### 5.2 UI/UX and maneuverability — P1

| ID | Finding | Evidence | Commits |
|----|---------|----------|---------|
| P1-1 | Strouhal uses the wrong sample period (error is exactly 2×), and `analysis/regimes.py:42` additionally drops `diameter`/`velocity`, so the St shown in the UI and in every exported report is a raw FFT frequency rather than a Strouhal number. | `workbench/panels/analysis_panel.py:206-217`, `:257-261`, `analysis/physics.py:77-105`, `analysis/regimes.py:42` | C39, C40, C43 |
| P1-2 | Pressure is represented by five incompatible definitions (`rho/3`, gauge, EMA-normalized render, `rho-1`, percentile contour levels), the colormap tooltip describes a quantity that is not what is rendered, and every engine except `LBM2D` renders `rho-1` — a factor-of-3 divergence. | `engines/lbm2d.py:413-419`, `:453-457`, `workbench/viewport.py:749-755`, `resources/colormaps.py:138-144`, `workbench/panels/analysis_panel.py:266-278` | C35, C36, C37, C38, C41, C43 |
| P1-3 | The sanity banner refreshes only on apply/reset, never during a run, and `check_sanity` never inspects the field state — a diverged run with a sane setup produces no warning. Demo progress is configured but `setValue()` is never called. | `workbench/app.py:676-707`, `:644`, `:524`, `analysis/sanity.py:18-119`, `workbench/panels/outcome_panel.py:106-109` | C44, C45 |
| P1-4 | Timeline history is incomplete and memory-heavy (~3.1 GB for 300 frames of the 512×256 flagship), omits smoke, particles, probes, and `step_count`, uses NumPy-specific restore, and silently does nothing on `LBM2DMultiComponent`. | `workbench/app.py:82-84`, `:584-611` | C49, C50 |
| P1-5 | Long-running jobs are neither cancellable nor awaitable: no `requestInterruption` anywhere, the sweep dialog's only button dismisses it while the thread runs (QThread destroyed while running → abort), failures never surface, and there is no `closeEvent` to finalize an active recording. | `workbench/dialogs/sweep_dialog.py:31-52`, `:143-146`, `workbench/app.py:1088-1094`, `:1176-1197` | C47, C48, C67 |
| P1-6 | Engine switching does not rebuild fields, engine-specific parameter groups, param values, viewport caches, or demo state. | `workbench/app.py:969-1000`, `:1005-1042`, `workbench/panels/scene_panel.py:323-331` | C53 |
| P1-7 | The UI boundary is not device-safe: the timeline reaches past the engine API into `sim.f`/`sim.rho`, which raises `TypeError` on CuPy arrays and kills the tick handler. (The cited viewport/export lines are already safe — corrected in §5.3.) | `workbench/app.py:587-588`, `:607-609` | C49, C58 |
| P1-8 | There is no dirty-state or unsaved-work protection: the title carries no modified marker, `closeEvent` is absent, `_file_new`/`_file_open` discard work silently, and `reset()` only confirms when `step_count > 0`. | `workbench/app.py:546-548`, `:460`, `:471`, `:625-635` | C46, C47 |
| P1-9 | Imported geometry previews are placeholders; the image obstacle label is drawn at a hard-coded screen position, and optional dependency failures are silent. | `workbench/viewport.py:571-579`, `workbench/panels/scene_panel.py:1036-1078` | C59, C60 |

### 5.3 Corrections to the original audit framing

1. **`scene.py` never had a `_init_droplet` helper** — the multiphase droplet setup is inlined in each engine's `initialize()`. C10 introduces an engine-owned reset hook rather than calling a nonexistent helper.
2. **`engines/api.py` fails with `NameError`, not `ImportError`** — `BaseModel` is unbound in the failure branch and the first `class RunRequest(BaseModel)` statement raises. C04 fixes the import guard.
3. **Periodic domain mode is not reachable from the product UI** — `_configure_domain_for_scene` only selects channel or cavity. C08 therefore treats it as a latent API hazard, not a user-visible P0.
4. **Outcome-panel sanity pills are not stale** — `outcome_panel.update_outcome` refreshes every 15 steps. Only the banner and the missing field-state checks are defects, which C44 addresses precisely.
5. **The P1-7 evidence was mis-located** — `viewport.py` and `export/image.py` convert through the engine API already; the timeline is the actual GPU boundary break.
6. **`pressure_drop` is not in `analysis/physics.py`** — it is inlined in `scorecard.py:45-46`. C41 promotes it into the analysis layer with guards.

### 5.4 Additional defects surfaced by the audit

All are folded into §11: fused path changes the obstacle scheme silently when a non-Newtonian model is selected (`_can_use_fused` gates on `isinstance(BGKCollision)`, and Guo forcing is likewise BGK-only); the HUD calls `detect_flow_regime` without `step_count` on every `paintGL`, so it can only ever report the first two branches; the outcome path recomputes regime, sanity, and drag twice per update and re-runs at ~30 Hz while paused; `getattr(self.sim, "f", self.sim.get_velocity_view())` evaluates its default eagerly, causing a full device→host copy every tick on GPU; `Ctrl+Q` is bound twice and `Ctrl+Shift+S` collides with Save-As; `"cylinder"` matches two presets so recipe resolution depends on filesystem order; undo stacks survive a scene load; `_file_save` does not update the title; `_scrubbing` is never cleared, so a reset during a slider drag freezes the timeline permanently; parameter handlers silently create dead attributes on the wrong engine; coverage config omits `workbench` and `resources`.

### 5.5 P2 usability and accessibility

- **Information architecture drift** — Start tabs, preset gallery, and recipes are parallel doors to one widget; the AI action exposes "Coming Soon" although `analysis/ai_context.py` is implemented. Commits C63, C68.
- **Select mode is not implemented** — it is the default tool and shortcut `1`, and it is entirely inert: no hit-testing, move, resize, or delete. Commit C51.
- **Shortcut documentation is inconsistent** — four concrete mismatches: `Esc` overpromises, `R` is an undocumented global irreversible binding, `4` activates an invisible tool in Beginner mode, and viewport-scoped shortcuts are advertised as global; plus duplicate `Ctrl+Q` and a `Ctrl+Shift+S` collision. Commit C52.
- **Keyboard and screen-reader path is incomplete** — no accessible names or descriptions anywhere, default tab order, status hints not associated with the canvas. Commit C66.
- **Per-frame analysis is expensive** — full-field speed allocation every tick, drag loops every 5 ticks, timeline copies every tick, no budget anywhere. Commits C54, C55, C56, C57.
- **Overlay caches are not invalidated consistently** — stale while paused indefinitely. Commit C29.
- **GUI tests are too shallow** — construction-only smoke tests, and coverage config excludes `workbench`/`resources`, which is the root cause of the whole P0 cluster. Commits C14, C65.

## 6. Simulation Correctness and Trust Findings

### 6.1 Solver and boundary conditions

| Finding | Evidence | Commits |
|---------|----------|---------|
| Fused obstacle path: the default fused BGK kernel does its own on-node obstacle swap and never reads `obstacle_bc`, so the documented default halfway bounce-back is not what runs. | `engines/lbm2d.py:61-67`, `:138-157`, `:214`, `:285-292`, `:554-568` | C07 |
| Periodic obstacles: periodic mode returns before obstacle handling. | `engines/lbm2d.py:346-351` | C08 |
| Inlet/obstacle ordering: obstacles are applied before inflow, so a solid cell in the inlet column is permanently overwritten with equilibrium populations — in both the fused and non-fused paths, and likewise at the outlet. | `engines/lbm2d.py:351`, `:369-370`, `:570-575`, `:61-73`, `:314` | C09 |
| 3D thermal coupling: `LBM3D.step()` never invokes temperature collision, streaming, or buoyancy, yet the UI reports success and the engine then advertises an inert `temperature` field. | `engines/lbm3d.py:259-270`, `:311-315`, `engines/thermal_mixin.py:270-334` | C11 |
| Scene initialization: `apply_to_sim()` initializes liquid and multicomponent engines with `rho=1.0`, bypassing their intended low-density vapor background and seeding an inflow velocity those closed-domain engines never use. | `scene/scene.py:481-492`, `engines/lbm2d_liquid.py:224-238`, `engines/lbm2d_multicomponent.py:393-412` | C10 |
| Thermal direction: D2Q9 pairing is not inherently reversed; the open issue is coordinate convention and weak validation, and the thermal test accepts either velocity sign. | `engines/thermal_mixin.py:147-176`, `tests/validation/test_benchmarks.py:213-226` | C11 |
| 3D smoke helper: `_advect_smoke_3d_nb` is unused and `SmokeMixin` provides the active path; not a current 3D blocker. | `engines/smoke_mixin.py` | C61 |

### 6.2 Analysis

- Reynolds, drag, and pressure-drop calculations are 2D-oriented: `drag_coefficient` indexes `sim.get_f()[i, y, x]` and hard-codes `LATTICE_2D`/`range(1,9)`, `reynolds_number` returns `0.0` for force-driven and periodic scenes, and `characteristic_length` ignores polygons and uses the first obstacle rather than the dominant one. Commits C54, and gates in C35.
- `detect_flow_regime()` returns "No driven flow" from `u_inflow` alone, so the shipped Lid-Driven Cavity story is permanently mislabeled at 0.95 confidence. Commit C06.
- `wake_strength` is a global standard deviation of speed with no wake region, no baseline, and no UI display; expected-range checks are hard-coded product metadata, not uncertainty-aware validation. Commit C42.
- Strouhal needs step-aware sampling and a reference validation case. Commits C39, C40, C43.
- Sanity warnings are heuristics with no field-state inspection and do not replace grid/time convergence. Commits C44, C63.
- Regime/scorecard run on fixed tick counters with no wall-clock budget, with duplicate work per update. Commit C55.

### 6.3 Optional API and release

- FastAPI model classes are defined even when the optional import is unavailable. Commit C04.
- Version is `0.6.0` in `pyproject.toml`, `0.3.5` in `sstream/__init__.py`, `0.3.5` asserted in a test, `0.3.5` in the newest changelog entry, and `0.3.0` hard-coded twice in the API. Commits C02, C03.
- `Production/Stable` conflicts with the educational positioning. Commit C01.
- `TRUST.md` says thermal is not in the default step path while 2D now steps it; `CHANGELOG.md` advertises 3D thermal buoyancy that never advances; two validation tests exceed the advertised trust level. Commits C03, C11, C12, C68.

## 7. Performance and Scalability Findings

- Full-field analysis is recomputed on arbitrary tick literals with no demonstrated budget. Commits C56, C57.
- `drag_coefficient` recomputes `np.where(~obs)` and seven full-domain temporaries once per direction, i.e. eight times per call. Commit C54.
- Timeline snapshots reach ~3.1 GB for the flagship preset and copy the full distribution array every tick. Commits C49, C50.
- Timeline restore uses NumPy-specific operations and raw attributes, so GPU engines raise and multicomponent silently does nothing. Commits C49, C58.
- Optional CuPy/Lettuce paths are experimental with incomplete UI/test coverage. Commit C58.
- MPI and multi-GPU are explicitly deferred in `TRUST.md`.
- Professional workflows require a performance contract, not only a functional test. Commit C57.

Target budgets for the optimization pass:

- 60 FPS interaction for ordinary 128×128 scenes on the reference machine.
- No visible input latency for drawing, tool switching, and transport controls.
- Analysis updates throttled independently of the simulation step by wall clock.
- Timeline memory capped by megabytes, not only frame count.
- GPU array transfers performed only at explicit engine/UI boundaries.
- Large-grid behavior reported honestly rather than silently degrading.

## 8. Target UX Architecture

### 8.1 Core user loop

**Choose a story → inspect the setup → run → observe → adjust → understand → export.**

1. **Scene:** preset, geometry, emitters, probes, boundary/domain mode.
2. **Run controls:** play, pause, step, reset, speed, and autorun progress.
3. **Visualize:** supported field, colormap, overlays, slice selection, and colorbar metadata.
4. **Analyze:** probes, Re/Cd/St, regime, stability, sanity, and scorecard.
5. **Outcome:** interpretation, limitations, validation status, and next recommended action.

### 8.2 Beginner mode

Hide advanced parameters by default without making them unreachable; explain every primary
control in plain language; use units, expected ranges, and safe defaults; show a pre-run
setup check; prevent unsupported or unsafe combinations where possible; offer "why this
matters" explanations next to Re, Cd, St, and stability; direct users to a guided story
before presenting an empty scene.

### 8.3 Expert mode

Expose domain mode, collision operator, thermal, rheology, boundary, and solver settings;
show raw field definitions and units; allow manual stepping, timeline scrubbing, sweeps, and
custom scenes; display capability badges and warnings without blocking exploration; preserve
the same core navigation and scene lifecycle as Beginner mode.

### 8.4 Canvas interaction model

Every visible tool must have a complete contract:

- **Select:** select, move, resize, duplicate, delete, and show properties.
- **Circle:** drag center-to-radius, show radius/diameter in grid units, report aspect changes.
- **Rectangle:** drag corner-to-corner, show dimensions, support axis lock.
- **Polygon:** click points, show vertex count and live preview, finalize with Enter or double-click, cancel with Escape.
- **Probe:** click a valid cell, show coordinates and selected fields, reject unsupported fields before placement.
- **Pan/zoom:** explicit controls and shortcuts; do not overload drawing gestures.
- **Escape:** cancel only the active operation and explain what was canceled.
- **Undo/redo:** show available actions and preserve scene identity.

The viewport must use one canonical coordinate system and visibly show grid coordinates when precision matters.

### 8.5 Feedback and state model

Every action provides one of: success, warning, error, busy/progress, canceled, or
unsupported capability. Required behavior: no silent field fallback; no silent
optional-dependency failure; no stale stability or progress state; no long job without
cancellation and completion feedback; no close without a decision when unsaved work or
active jobs exist; no error without a recovery action.

### 8.6 Accessibility and keyboard support

Every interactive control has an accessible name and visible focus state; tab order follows
the task flow; drawing and selection have keyboard alternatives or a documented accessible
equivalent; status is not conveyed by color alone; text and controls meet readable contrast
targets; shortcut help is generated from the same binding table used by the application;
error text is associated with the relevant control; reduced-motion and high-contrast
behavior are considered for the viewport and status indicators.

## 9. Architectural Work Plan

1. **One scene owner.** A single session controller owns `Scene` identity, engine
   construction, runtime probes, analysis histories, demo state, recording/export state,
   undo/redo, dirty state, and viewport cache invalidation. All UI mutations emit one change
   signal; no panel silently replaces the scene reference.
2. **One apply/reset path.** A single operation normalizes the scene, creates or resizes the
   engine, applies geometry/probes/parameters, rebuilds supported fields and controls, clears
   runtime histories and caches, updates viewport/analysis/outcome panels, and updates dirty
   state and provenance. Used by initial launch, wizard completion, preset load, engine
   switch, reset, undo/redo, and file load.
3. **Capability and field contract.** Engine metadata describes supported fields, field shape
   and units, pressure reference/definition, boundary and domain modes, supported geometry
   dimensions, capability/trust level, optional dependency requirements, array/device type,
   and state-snapshot support. The UI derives options from this contract, not a static list.
4. **Job and lifecycle model.** Sweeps, video recording, exports, large simulations, and
   report generation share start/progress/cancel/completion/failure/window-close behavior.
5. **Performance architecture.** Simulation cadence is separated from UI analysis cadence;
   immutable scene/field metadata is cached; expensive analysis is downsampled or scheduled;
   timeline snapshots are memory-budgeted; CPU/GPU array operations sit behind the engine
   interface; performance benchmarks precede optimization.

## 10. Phase Map

| Phase | Objective | Commits |
|-------|-----------|---------|
| Phase 0 — Trust freeze | Prevent misleading or broken workflows from being presented as reliable. | C01–C12 |
| Phase 1 — P0 correctness | Make the core workbench reliable before adding features. | C13–C26 |
| Phase 1b — State machine | One scene owner, one apply/reset path, consistent undo. | C27–C34 |
| Phase 2 — Scientific truth | Ensure displayed values mean what the labels say. | C35–C44 |
| Phase 3 — Maneuverability | Easy to operate without hiding expert capability. | C45–C52 |
| Phase 4 — Performance | Preserve responsiveness as grids, fields, and timelines grow. | C53–C58 |
| Phase 5 — 3D / multiphase / release | Promote gated capabilities and harden exported results. | C59–C64 |
| Closeout — Evidence and documentation | Bind the capability matrix to automated evidence. | C65–C69 |

## 11. Commit Execution Plan

69 commits, executed in order, dated 2026-08-30 → 2026-10-03.

### Phase 0 — Trust freeze and immediate containment (C01–C12)

| # | Date | Commit | Scope | Findings |
|---|------|--------|-------|----------|
| [x] | 08-30 | `chore(release): drop Production/Stable classifier for educational scope` | `pyproject.toml` | §6.3 |
| [x] | 08-30 | `chore(release): single-source version 0.6.0 from package metadata` | `sstream/__init__.py`, `engines/api.py`, tests | §6.3 |
| [x] | 08-31 | `docs(changelog): 0.6.0 entry separating shipped from planned work` | `CHANGELOG.md` | §6.3 |
| [x] | 08-31 | `fix(api): import engines.api safely without the fastapi extra` | `engines/api.py` | §6.3 |
| [x] | 09-01 | `fix(rheology): wire BinghamModel yield-stress constructor` | `workbench/panels/scene_panel.py` | §6.3 |
| [x] | 09-01 | `fix(regimes): classify lid-driven cavity as driven flow` | `analysis/regimes.py` | §6.2 |
| [x] | 09-02 | `fix(lbm2d): honor configured obstacle BC on the fused path` | `engines/lbm2d.py` | §6.1 |
| [x] | 09-02 | `fix(lbm2d): apply obstacles under periodic domain mode` | `engines/lbm2d.py` | §6.1 |
| [x] | 09-03 | `fix(lbm2d): stop inlet/outflow overwriting obstacle cells` | `engines/lbm2d.py` | §6.1 |
| [x] | 09-03 | `fix(scene): engine-owned reset for multiphase density init` | `scene/scene.py`, engines | §6.1 |
| [x] | 09-04 | `feat(lbm3d): advance thermal physics in the 3D step path` | `engines/lbm3d.py` | §6.1 |
| [x] | 09-04 | `docs(trust): reconcile TRUST.md with actual step paths and tests` | `TRUST.md` | §6.3 |

Exit criteria: no advertised feature silently falls back or runs a different model; no P0
trust contradiction remains in README, TRUST.md, or CHANGELOG.md.

### Phase 1 — P0 UI and state correctness (C13–C26)

| # | Date | Commit | Scope | Findings |
|---|------|--------|-------|----------|
| [x] | 09-05 | `fix(viewport): stop shadowing the obstacle mask in overlay drawing` | `workbench/viewport.py` | P0-1 |
| [x] | 09-05 | `test(viewport): paint-path regression tests for every overlay with obstacles` | `tests/test_viewport_overlays.py` | P0-1 |
| [x] | 09-06 | `feat(presets): add stable preset ids to schema and loader` | `presets/` | P0-3 |
| [x] | 09-06 | `fix(recipes): resolve recipe actions by preset id with startup validation` | `workbench/app.py` | P0-3 |
| [x] | 09-07 | `test(recipes): every recipe action resolves to a real preset` | `tests/test_recipes.py` | P0-3 |
| [x] | 09-07 | `feat(ui): derive the field catalog from engine capabilities` | `workbench/app.py` | P0-6 |
| [x] | 09-08 | `fix(ui): remove silent field fallbacks in viewport and export` | `workbench/viewport.py`, `export/image.py` | P0-6 |
| [x] | 09-08 | `fix(lbm2d): refuse the temperature field when thermal is disabled` | `engines/lbm2d.py`, `engines/lbm3d.py` | P0-6 |
| [x] | 09-09 | `feat(probe): record engine fields instead of a fixed whitelist` | `scene/probe.py` | P0-7 |
| [x] | 09-09 | `fix(probes): validate probe fields at placement and preset load` | `workbench/` | P0-7 |
| [x] | 09-10 | `test(probes): every preset probe field is recordable` | `tests/test_probe_fields.py` | P0-7 |
| [x] | 09-11 | `fix(engine): rebuild the engine when scene dimensions change` | `workbench/app.py` | P0-2 |
| [x] | 09-11 | `test(scene): scene dimensions always match the engine grid shape` | `tests/test_scene_apply.py` | P0-2 |
| [x] | 09-12 | `feat(lifecycle): single runtime-reset path for all derived state` | `workbench/app.py` | P0-4 |

Exit criteria: preset/template load always yields matching scene and engine dimensions;
reset, load, undo, and redo leave all panels and runtime state synchronized; every supported
preset opens without silent field or probe failures; obstacle overlays render in every mode.

### Phase 1b — One scene owner and one reset path (C27–C34)

| # | Date | Commit | Scope | Findings |
|---|------|--------|-------|----------|
| [x] | 09-12 | `fix(reset): clear probe and stability histories on reset and load` — *delivered by C26* | `workbench/panels/analysis_panel.py` | P0-4 |
| [x] | 09-12 | `fix(reset): clear demo, recorder, and scrub state on reset` — *delivered by C26* | `workbench/app.py` | P0-4 |
| [x] | 09-13 | `fix(viewport): invalidate overlay caches on scene and engine change` | `workbench/viewport.py` | P2 |
| [x] | 09-13 | `refactor(scene): make MainWindow the sole owner of the Scene` | `workbench/panels/scene_panel.py`, `workbench/app.py` | P0-5 |
| [x] | 09-14 | `fix(undo): snapshot before mutation and preserve scene identity` | `workbench/panels/scene_panel.py`, `scene/serializer.py` | P0-5 |
| [x] | 09-14 | `fix(undo): clear undo history when a scene is loaded` | `workbench/app.py`, `workbench/panels/scene_panel.py` | P0-5 |
| [x] | 09-15 | `test(state): reset and load clear all runtime-derived state` | `tests/test_state_reset.py` | P0-4 |
| [x] | 09-15 | `test(undo): property edits round-trip with stable scene identity` | `tests/test_undo_redo.py` | P0-5 |

Exit criteria: one owner for scene identity; every mutation path is reversible; no stale
timeline, probe, demo, or cache state survives a load or reset.

### Phase 2 — Scientific truth and analysis semantics (C35–C44)

| # | Date | Commit | Scope | Findings |
|---|------|--------|-------|----------|
| [x] | 09-16 | `feat(engine): canonical gauge pressure with field unit metadata` | `engines/base.py`, `resources/colormaps.py` | P1-2 |
| [ ] | 09-16 | `refactor(engines): route every pressure field through the canonical quantity` | `engines/*` | P1-2 |
| [ ] | 09-17 | `fix(viewport): contours and colorbar read canonical pressure with units` | `workbench/viewport.py` | P1-2 |
| [ ] | 09-17 | `fix(analysis): field statistics follow the selected field with units` | `workbench/panels/analysis_panel.py` | P1-2 |
| [ ] | 09-18 | `feat(probe): step-aligned probe timestamps and cadence-correct Strouhal` | `scene/probe.py`, `analysis/physics.py` | P1-1 |
| [ ] | 09-18 | `fix(regimes): pass real cadence, diameter, and velocity to Strouhal` | `analysis/regimes.py`, `workbench/` | P1-1 |
| [ ] | 09-19 | `refactor(physics): promote pressure drop into analysis with domain guards` | `analysis/physics.py`, `analysis/scorecard.py` | P1-2 |
| [ ] | 09-19 | `fix(scorecard): region-defined wake strength with an upstream baseline` | `analysis/scorecard.py` | §6.2 |
| [ ] | 09-20 | `test(physics): cross-consumer pressure and Strouhal consistency` | `tests/test_pressure_units.py`, `tests/test_strouhal.py` | P1-1, P1-2 |
| [ ] | 09-20 | `feat(sanity): divergence checks with continuous stability refresh` | `analysis/sanity.py`, `workbench/app.py` | P1-3 |

Exit criteria: the same field/value is consistent across viewport, colorbar, probes,
analysis, reports, and exports; every displayed metric has a documented definition, units,
and operating envelope; every Verified capability has a named CI test and tolerance.

### Phase 3 — Maneuverability and onboarding (C45–C52)

| # | Date | Commit | Scope | Findings |
|---|------|--------|-------|----------|
| [ ] | 09-21 | `feat(outcome): drive demo progress from the real simulation step` | `workbench/panels/outcome_panel.py`, `workbench/app.py` | P1-3 |
| [ ] | 09-21 | `feat(app): dirty-state tracking with a window-title marker` | `workbench/app.py` | P1-8 |
| [ ] | 09-22 | `feat(app): close-event protection for unsaved work and live jobs` | `workbench/app.py` | P1-8, P1-5 |
| [ ] | 09-22 | `fix(jobs): cancellable sweep worker with an explicit failure state` | `workbench/dialogs/sweep_dialog.py`, `analysis/sweep.py` | P1-5 |
| [ ] | 09-23 | `feat(engine): snapshot/restore contract for state history` | `engines/base.py`, `engines/lbm2d.py` | P1-4, P1-7 |
| [ ] | 09-23 | `feat(timeline): memory-budgeted history with explicit restore semantics` | `workbench/app.py` | P1-4 |
| [ ] | 09-24 | `feat(viewport): working select tool with hit-testing and delete` | `workbench/viewport.py` | P2 |
| [ ] | 09-24 | `refactor(shortcuts): one binding registry generating the status-bar help` | `workbench/app.py`, `workbench/viewport.py` | P2 |

Exit criteria: a new user can complete a guided experiment without external instructions;
a user can recover from every error without restarting; all visible tools have documented
mouse and keyboard behavior.

### Phase 4 — Performance and scale (C53–C58)

| # | Date | Commit | Scope | Findings |
|---|------|--------|-------|----------|
| [ ] | 09-25 | `refactor(panel): rebuild engine-specific controls on engine change` | `workbench/panels/scene_panel.py`, `workbench/app.py` | P1-6 |
| [ ] | 09-25 | `perf(physics): hoist drag-loop masks out of the link sweep` | `analysis/physics.py` | §7 |
| [ ] | 09-26 | `perf(analysis): wall-clock throttling and removal of duplicate regime work` | `workbench/panels/outcome_panel.py`, `workbench/app.py` | §7 |
| [ ] | 09-26 | `perf(analysis): documented cadence constants and stability throttle` | `workbench/panels/analysis_panel.py` | §7 |
| [ ] | 09-27 | `test(perf): frame and analysis budgets for representative grid sizes` | `tests/test_performance.py` | §7 |
| [ ] | 09-27 | `test(gpu): optional GPU UI-handoff and export coverage` | `tests/test_gpu_ui.py` | P1-7 |

Exit criteria: interaction latency meets the agreed frame budget; large-grid memory is
bounded and visible; experimental GPU engines fail at the UI boundary with a clear message.

### Phase 5 — 3D, multiphase, interoperability, release (C59–C64)

| # | Date | Commit | Scope | Findings |
|---|------|--------|-------|----------|
| [ ] | 09-28 | `feat(viewport): rasterized previews for imported STL and image obstacles` | `workbench/viewport.py`, `scene/scene.py` | P1-9 |
| [ ] | 09-28 | `fix(scene): actionable errors for missing optional dependencies` | `workbench/panels/scene_panel.py` | P1-9 |
| [ ] | 09-29 | `feat(scene): depth-aware 3D scene, probe, and analysis workflows` | `workbench/`, `scene/` | Phase 5 |
| [ ] | 09-29 | `test(3d): D3Q19 mass, momentum, and boundary reference tests` | `tests/validation/` | Phase 5 |
| [ ] | 09-30 | `feat(export): field units and provenance in exported artifacts` | `export/image.py`, `export/report.py`, `export/data.py` | Phase 6 |
| [ ] | 09-30 | `feat(release): reproducible run manifest for exports and reports` | `export/manifest.py` | Phase 6 |

Exit criteria: a 3D scene can be created, edited, run, probed, analyzed, scrubbed, and
exported without dimension errors; thermal and multiphase claims match stepped behavior;
exported results carry reproducible metadata.

### Closeout — Evidence, accessibility, and documentation (C65–C69)

| # | Date | Commit | Scope | Findings |
|---|------|--------|-------|----------|
| [ ] | 10-01 | `chore(coverage): include workbench and resources in coverage scope` | `pyproject.toml` | P2 |
| [ ] | 10-01 | `feat(a11y): accessible names, focus order, and generated shortcut help` | `workbench/` | §8.6 |
| [ ] | 10-02 | `test(lifecycle): cancel and close behavior for sweeps, exports, recording` | `tests/test_lifecycle.py` | P1-5 |
| [ ] | 10-02 | `docs(trust): capability matrix bound to automated evidence` | `TRUST.md` | §12 |
| [ ] | 10-03 | `docs(plan): final status review and next roadmap` | `PLAN.md`, `README.md` | §15 |

## 12. Testing Strategy

### 12.1 Test layers

1. **Unit tests:** scene, serialization, fields, analysis units, shortcuts, state transitions.
2. **Solver tests:** analytic and reference cases for every engine capability.
3. **Qt offscreen tests:** widget lifecycle, preset loading, field selection, dialogs, viewport painting.
4. **Integration tests:** scene → engine → simulation → analysis → export.
5. **Performance tests:** frame time, analysis cadence, memory budget, optional GPU handoff.
6. **Accessibility tests:** focus order, accessible names, shortcut discoverability, contrast.
7. **Trust tests:** every advertised feature maps to a test, label, and documented limitation.

### 12.2 Required new tests

| Test file | Covers | Commit |
|-----------|--------|--------|
| `tests/test_viewport_overlays.py` | obstacle plus each overlay, no paint exception | C14 |
| `tests/test_recipes.py` | every recipe resolves to a preset or working action | C17 |
| `tests/test_probe_fields.py` | every preset probe field is recordable | C23 |
| `tests/test_scene_apply.py` | scene dimensions and engine grid shape agree after load | C25 |
| `tests/test_state_reset.py` | reset/load clears probes, analysis, caches, demo, jobs | C33 |
| `tests/test_undo_redo.py` | property edits round-trip, scene identity stable | C34 |
| `tests/test_pressure_units.py` | viewport, probes, statistics, reports, export agree | C43 |
| `tests/test_strouhal.py` | cadence-aware Strouhal against a reference value | C43 |
| `tests/test_performance.py` | frame and analysis budgets for representative grids | C57 |
| `tests/test_lifecycle.py` | cancel and close behavior for sweeps, exports, recording | C67 |

### 12.3 Validation commands

```text
pytest
pytest -m "not slow"
pytest tests/validation/ -v
ruff check .
black --check .
```

The full suite must include the GUI offscreen tests before a release is considered
trustworthy. GPU tests remain optional and must skip cleanly when CUDA/CuPy is unavailable.

## 13. Trust and Release Checklist

Before release:

- [ ] Version is consistent across `pyproject.toml`, `sstream/__init__.py`, tests, and changelog. (C02, C03)
- [ ] `Production/Stable` is not claimed unless the product scope and validation support it. (C01)
- [ ] `TRUST.md` matches actual default step paths and UI exposure. (C12, C68)
- [ ] Every Verified item has a CI test and documented tolerance. (C68)
- [ ] Every Experimental item is visibly labeled and bounded. (C53, C68)
- [ ] Hidden/unfinished features are not exposed as normal actions. (C11, C18)
- [ ] Optional dependencies fail with actionable installation guidance. (C04, C60)
- [ ] All preset/template loads are tested end to end. (C17, C23, C25)
- [ ] No field silently falls back to smoke. (C19, C20)
- [ ] No reset/load/undo operation leaves stale state. (C26–C34)
- [ ] No long job is lost silently on close. (C47, C48, C67)
- [ ] GPU and large-grid limitations are documented. (C50, C58)
- [ ] External comparison claims distinguish educational scope from engineering use. (C01, C68)

## 14. Research and Comparison Basis

OpenFOAM user guide and parallel execution documentation; ANSYS Fluent product and workflow
documentation; NASA grid-convergence and Richardson-extrapolation guidance; OpenLB LBM
performance and parallel examples; ParaView and PyVista postprocessing workflows; W3C Web
Content Accessibility Guidelines; Qt accessibility documentation; Shan–Chen and
lattice-Boltzmann model literature.

Professional tools are capability benchmarks, not a reason to reproduce their entire scope.
S-Stream's advantage is a fast, transparent, interactive learning loop; its responsibility is
to make that loop correct, legible, and safe.

## 15. Definition of Done

- A new user can reach a meaningful, understandable result quickly.
- Every visible control and capability has an explicit contract.
- Scene, engine, viewport, analysis, and export state remain synchronized.
- Physics metrics have consistent definitions, units, and validation evidence.
- Unsupported and experimental features cannot masquerade as verified behavior.
- The interface remains responsive under the supported operating envelope.
- Keyboard, mouse, and visible-control workflows are coherent.
- Automated tests cover the critical UI and physics paths.
- Documentation describes the shipped product honestly.
- The project remains focused on education and intuition rather than unsupported production claims.

## 16. Progress Log

Appended in commit order. Each entry records the commit id, what changed, and what remains.

| Commit | Date | Outcome |
|--------|------|---------|
| C01 | 08-30 | **Done.** `pyproject.toml` classifier `5 - Production/Stable` → `4 - Beta`. The 0.6.0 metadata claimed the highest tier while `README.md` disclaims production use and `TRUST.md` labels most capabilities Experimental/Hidden. Plan itself landed here: 69 commits, traceability matrix, progress log. `.gitignore` un-ignores `PLAN.md` so it stays tracked. |
| C02 | 08-30 | **Done.** Version was declared four times and disagreed in all four: `pyproject.toml` 0.6.0, `sstream/__init__.py` 0.3.5, a test pinning 0.3.5, `engines/api.py` hard-coding 0.3.0 twice. `sstream.__version__` now resolves from `pyproject.toml` (installed metadata as fallback), and `engines/api.py` reports it instead of a literal. The pinned assertion is replaced by a pyproject-consistency check plus a scan for hard-coded `version=` literals. |
| C03 | 08-31 | **Done.** `CHANGELOG.md` gains an `[Unreleased]` section that separates implemented, verified, and planned work, states that entries never imply a stronger label than `TRUST.md`, and adds a Corrections block under 0.3.5 recording the overstated 3D thermal claim and the version disagreement. |
| C04 | 08-31 | **Done.** `engines.api` defined three pydantic models at module scope while `BaseModel` was unbound in the missing-extra branch, so a plain `pip install s-stream` died with `NameError` at the first `class RunRequest(BaseModel)` — the intended "install sstream[api]" message was unreachable. Models moved inside `create_app()` behind `_require_fastapi()`, which chains the original `ImportError` for a traceback. Covered by a subprocess test that hides `fastapi`/`pydantic` and asserts the module still imports with `HAS_FASTAPI is False`. |
| C05 | 09-01 | **Done.** `scene_panel._apply_non_newtonian_model()` called `BinghamModel(yield_stress=...)`, but the parameter is `tau_y`, so selecting Bingham raised `TypeError` inside a Qt slot — the engine silently kept its old collision operator and the `parameters_changed` refresh never fired. The call site now matches the signature. Added a parametrized offscreen test over all three rheology models asserting `sim.collision_op` becomes a `NonNewtonianCollision`; verified it fails on the old call and passes on the new one. || C06 | 09-01 | **Done.** `detect_flow_regime()` read `sim.u_inflow` alone, but `MainWindow._configure_domain_for_scene` deliberately forces `u_inflow = 0.0, lid_velocity = 0.1` for the shipped Lid-Driven Cavity — so that preset was permanently reported as "No driven flow" at 0.95 confidence. Added `analysis.physics.drive_velocity()`, which resolves inlet vs. lid the way `reynolds_number` and `drag_coefficient` already did, and used it for the branch test. The check also moved above the full-field vorticity and FFT work it used to perform before discarding the result. Tests cover the cavity preset path and all three drive-resolution cases. |
| C07 | 09-02 | **Done.** `_can_use_fused()` did not consider obstacles, so the default `LBM2D` silently ran the fused kernel for any obstacle scene. That kernel does its own on-node swap and never reads `obstacle_bc`, whose documented default is `halfway` — putting the no-slip surface ~0.5 lattice units from where the validated Cd configuration put it, and changing the obstacle BC whenever a non-BGK operator was selected. The gate now requires an empty obstacle mask, so obstacle scenes take the non-fused path that honours `obstacle_bc`. New `TestFusedPathContract` asserts the gate excludes obstacles, that flipping `use_fused` no longer changes obstacle physics, that halfway and fullway still differ, and that an empty scene keeps the fast path. All 9 validation tests still pass. |
| C08 | 09-02 | **Done.** `apply_boundary_conditions()` returned immediately in `DOMAIN_PERIODIC`, before `apply_obstacles()`, so in periodic mode the obstacle mask was a render-only overlay: the viewport drew solids the solver streamed straight through. Obstacles are interior boundaries and now apply in every domain mode. While testing the kernel I also found a second defect in `_halfway_bb_2d_nb`: it wrote `f[opp[i]] = f[i]` in direction order, so a fluid cell walled on two opposite sides read back its own first write and both populations collapsed toward the same value. All sources are now snapshotted before any write, making the reflection a genuine simultaneous update. Two regression tests pin both behaviours (the aliasing test was confirmed to fail against the old kernel), and `apply_obstacles()` now short-circuits on an empty mask. Full suite: 322 passed, 27 skipped. |
| C09 | 09-03 | **Done.** No domain boundary condition consulted the obstacle mask, so any blockage touching the domain edge had populations rewritten inside it: `apply_inflow` overwrote the whole left column with a fresh equilibrium, `apply_outflow` copied the interior into the outlet column, `_wall_bb_nb` swapped row 0 and the last row, and the cavity lid plus side walls did the same. A solid touching x=0 therefore acted as a mass source and the wall scheme rotated fluid into a no-slip surface, corrupting blockage ratio and drag for exactly the geometries a blockage study measures. Every one of these paths now takes the `[y, x]` obstacle mask and skips solids: the two Zou–He kernels, both moving-wall kernels, `_wall_bb_nb`, and `_bounce_side_walls`. The channel path no longer duplicates the inlet/outlet logic inline and delegates to `apply_inflow()`/`apply_outflow()`, which also removes the dead `use_zou_he` branch that made `apply_outflow`'s Zou-He path unreachable. Eight regression tests in `TestBoundaryConditionsRespectObstacles` cover every path and were each confirmed to fail against the pre-fix engines. Full suite: 330 passed, 27 skipped. |
| C10 | 09-03 | **Done.** `apply_to_sim()` called `initialize(rho=1.0, u=scene.u_inflow, v=0.0)` on every engine, but that IC is only meaningful for a driven open domain. For `LBM2DLiquid` and `LBM2DMultiComponent`, which are closed-domain with bounce-back on all four sides, it set the vapor background to ρ=1.0 instead of 0.01 — collapsing the liquid/vapor density contrast that produces surface tension in the first place — and seeded a uniform u=0.15 (Ma≈0.26) that pushes the droplet across the box before the interface can develop. Added `SimEngine.reset_for_scene(scene)` as the single engine-owned reset entry point, defaulted to the previous behaviour for driven engines and overridden by both Shan-Chen engines to call their own `initialize()`. `apply_to_sim()` now delegates instead of hardcoding an IC, and its docstring/contract makes the hook mandatory for new engines. New `tests/test_scene_reset.py` (10 tests) pins the vapor background, quiescence, droplet seeding, `scene.u_inflow` inertness and idempotence for both multiphase engines; 8 of them were confirmed to fail against the pre-fix code. Full suite: 340 passed, 27 skipped. |
| C11 | 09-04 | **Done.** Three separate defects, all fixed. (1) `LBM3D.step()` never called a single thermal kernel, so `temperature` stayed exactly 0.0 while `get_field_names()` still advertised a `temperature` field — the engine reported success and rendered an inert one. The step path now streams `f_T`, injects buoyancy, collides, then collides the temperature, so the force is relaxed with the populations instead of sitting un-relaxed. (2) `_apply_buoyancy_2d_nb` applied `g_y` to D2Q9 directions 3 and 4, which are `(-1, 0)` and `(0, -1)`, so gravity partly drove the *streamwise* axis; heated from below the 2D engine produced a mean vertical velocity of `-1.1e-3`, i.e. the fluid sank. Both kernels now use the discrete body force `w_i * (F · c_i) / cs2`, which sums to exactly `F` for any direction; the old 3D version touched only the six axis-aligned populations, leaving 12 of 19 D3Q19 directions with no force at all. Buoyancy also now skips obstacles. (3) `set_temperature_boundary` mapped `top` to row 0 and `bottom` to the last row — inverted relative to `apply_moving_wall_top`/`_bottom` — so every heated-floor setup was stratified upside down. New `tests/test_thermal.py` (20 tests) pins the momentum balance exactly, the direction sign, per-direction coverage, obstacle masking, the 3D step path and the region mapping; 15 of them fail against the pre-fix code. `test_phase3.py::test_buoyancy_creates_convection` was rewritten to use the production `step()` path and now asserts the Rayleigh–Bénard sign (floor up, ceiling down) instead of accepting either direction, and `test_benchmarks.py::test_thermal_buoyancy_convection` was reading the two rows the wrong way round. Full suite: 360 passed, 27 skipped. |
| C12 | 09-04 | **Done.** `TRUST.md` had drifted from the code in both directions. It listed 6 of the 9 validation tests, omitting `test_poiseuille_l2_fine`, `test_trt_lid_cavity` and `test_thermal_buoyancy_convection`. It kept thermal buoyancy in **Hidden** with the reason "not default `step()`" — untrue for the 2D engine before this phase and untrue for the 3D engine as of C11; the scene panel has had an "Enable Thermal" control all along. It described halfway bounce-back as the default obstacle BC without saying that obstacle scenes used to bypass it entirely. Rewritten against the current tree: all 9 validation benchmarks, a new boundary-contract table for the C07–C11 invariants, thermal promoted to **Experimental**, halfway bounce-back annotated, and an explicit **Known coverage gaps** section naming the fused-path restriction, the absence of any quantitative multiphase test, the missing rheological accuracy check, the 3D obstacle-BC coverage hole, and the deliberate choice not to apply buoyancy inside solids. Also added a coordinate-convention section, since row 0 = bottom vs. the viewport's flipped render was the root of the C11 `top`/`bottom` mix-up. Every test name cited was verified to exist and pass. Also flipped the C03 checkbox, which was left unchecked when that commit was made. Phase 0 is now complete: all 12 rows checked, working tree clean, 360 tests passing. |
| C13 | 09-05 | **Done.** `for obs in self.scene.obstacles:` in `_draw_overlay()` rebound the `obs` parameter. Because the loop runs before the quiver/streamline/force-arrow calls, every one of those helpers received the last `ObstacleSpec` dataclass instead of the mask, and `obs is not None` was still true, so `_draw_quiver` evaluated `obs[y, x]` on a dataclass. There is no exception guard around `_draw_overlay` in `paintGL`, so turning on quiver or streamlines in any scene containing an obstacle raised `TypeError` out of the GL paint path. Fixed by renaming the loop variable. Found a second defect while verifying the first: `_estimate_obstacle_force` set `interior = obs_region` and summed velocity over the **solid** cells, where bounce-back leaves the frozen populations and so the sum is meaningless — fixing only the shadowing would have left the force arrows dead. It now uses the standard momentum-deficit estimate over the body footprint — free stream minus actual, normalised by `u_inflow` — so an undisturbed body reads 0 and a fully stalled body reads exactly 1.0. Sampling the *fluid* inside the box was the first attempt and is wrong: a `RectObstacle` tiles its own bounding box, so there is no fluid there to find, and the C14 tests are what proved it. A third defect surfaced during C14 and is fixed here too: `painter.drawPolygon(QPointF, QPointF, QPointF)` is not a valid PySide6 overload and raises `TypeError: too many arguments`, so both arrowheads crashed on first draw. Force arrows were broken three independent ways. Full suite: 377 passed, 27 skipped. |
| C14 | 09-05 | **Done.** New `tests/test_viewport_overlays.py` (17 tests). Driving the real overlay code turned out to need a paint-device shim: `_draw_overlay` constructs `QPainter(self)` and QOpenGLWidget is not a valid paint device under `QT_QPA_PLATFORM=offscreen`, so the test swaps the module's `QPainter` symbol for one that paints into a `QImage`. The drawing code under test is unchanged. The suite immediately caught a **second** bug that the C13 reasoning had missed: `painter.drawPolygon(QPointF, QPointF, QPointF)` is not a valid PySide6 overload — it raises `TypeError: too many arguments` — so both arrowheads (quiver at :654 and force arrow at :917) crashed on first draw. Force arrows have therefore been broken three ways over: shadowed mask, meaningless sum, and now an invalid call. 13 of the 17 tests fail against the pre-fix viewport. Also pinned: an undisturbed body reports zero drag, a fully stalled body reports exactly 1.0, the result is scale-invariant, cross-stream momentum becomes lift, a rect that exactly tiles its own bounding box is still measured (the case that ruled out the `~obs_region` reading), a circular footprint is not diluted by fluid in its box corners, and `u_inflow=0` does not divide by zero. Full suite: 377 passed, 27 skipped. |
| C15 | 09-06 | **Done.** The root cause of P0-3 is that presets had no identity at all. `_RECIPE_ACTIONS` keys on filename stems (`karman_street`) while `list_presets()` only exposes `scene.name` (the display name `High-Re Wake`), so `_execute_recipe` had to guess with two substring attempts. Neither can match for that preset, and the underscore-to-space fallback is a silent coupling to how someone happened to name a scene. Added `SceneProductMeta.preset_id`, round-tripped it through the serializer, and stamped an explicit id into all 10 preset JSONs. `list_presets()` now returns `id`, plus `find_preset()` for exact lookup and `validate_presets()` which reports missing ids, duplicate ids and unreadable files instead of letting `list_presets()` skip them silently. The validator earned its place immediately: it caught that the JSON key I had written as `id` did not match the `preset_id` field, which the `f.stem` fallback in `list_presets()` had been quietly papering over. Full suite: 377 passed, 27 skipped. |
| C16 | 09-06 | **Done.** `_execute_recipe` now calls `find_preset()` for an exact id match instead of the two-stage substring guess. Verified the flagship recipe was genuinely dead before the change: replaying the old lookup against the real preset list returns "NOT FOUND" for `karman_street`, because `karman_street` and `karman street` are both absent from the display name `High-Re Wake`. The other three resolved only by accident of naming, and the `cylinder` entry matched on a substring that would also have matched a future "Super Cylinder". Added `_check_recipe_actions()`, called once from `__init__`, which cross-checks every `_RECIPE_ACTIONS` id against the library and folds in `validate_presets()`, logging each problem and surfacing the first in the status bar. The failure message now names the recipe and the id it wanted, instead of the misleading "Check presets/ folder". A module `logger` was added for this; there was none in `app.py`. Full suite: 377 passed, 27 skipped. |
| C17 | 09-07 | **Done.** New `tests/test_recipes.py` (32 tests). The first version asserted that every preset's `product.recipe` tag names a real recipe, and failed immediately: `oil_water_separation.json` is tagged `Explore multiphase flow`, which is not in the action table. Reading the code showed the premise was wrong — `product.recipe` is a gallery subtitle (`presets_dialog.py:58` uses it as a caption fallback), not a recipe key. Replaced with the contract that actually exists, and found two real defects while doing it. (1) The recipe list lives in a *second* hardcoded dict, `recipes_dialog.RECIPES`; `MainWindow._RECIPE_ACTIONS` is a third source of the same five names. They agree today by coincidence, and adding a recipe to one would silently produce a dialog entry with no action, so their key sets are now compared. (2) The flagship recipe description told users to "Open Cylinder Wake" while the action loads `High-Re Wake` — the same preset-identity confusion as C16, one layer over. Corrected. Also proved the resolver tests bite: replaying the old substring lookup shows `Show vortex shedding` resolving to None while the other three happened to agree, which is precisely why only the flagship was broken and nobody noticed. Full suite: 408 passed, 28 skipped. |
| C18 | 09-07 | **Done.** Deleted `workbench/app.py:_COLORMAPS`, a module-level list of all nine field names in the project. The Field combo and `_set_colormap` now read `SimEngine.get_field_names()`, so the UI can only offer fields the active engine produces. Measured the old behaviour against a plain `LBM2D`: the combo advertised five fields the engine could not produce at all (`color`, `component1`, `component2`, `phase`, `temperature`). Two distinct failure modes, both now impossible from the combo. Four raised `ValueError`, which `viewport.py:368-370` swallowed into a smoke fallback — that is C19. But `temperature` was the dangerous one: with thermal disabled, `LBM2D.get_field("temperature")` returned a real, correctly-shaped float32 array of the constant **0.5**, not an error. Picking Temperature on a non-thermal run showed a plausible flat field rather than anything obviously broken — this is the "flat 0.5" hazard P0-6 predicted, now pinned by a test. Removed two hardcoded engine-name sniffs as consequences: `_apply_and_refresh` no longer special-cases `LBM2DLiquid`/`LBM2DMultiComponent` to swap smoke for density/component1 (both engines declare those fields themselves), and `_auto_detect_view` now honours the scene's `recommended_colormap` when the engine offers it. `_set_colormap` refuses an unavailable field and **logs a warning** instead of dropping the request silently. New `tests/test_field_catalog.py` (17 tests): every declared field is actually producible, has a label and a LUT, plain LBM2D offers no multiphase fields, thermal toggling moves `temperature` in and out of the catalog, and an AST guard forbids a module-level field list in `app.py` from coming back. Full suite: 425 passed, 28 skipped. |
| C19 | 09-08 | **Done.** Removed the last two silent field fallbacks, both of which were `except ValueError: field = get_field("smoke")`. `viewport._upload_smoke` had the try/except around the per-frame field fetch; `export/image.py:export_image` had an identical one. Replaced with an explicit check in both: the viewport validates the name against `get_field_names()` in `set_colormap` and logs an **error** listing the fields that do exist, and `_upload_smoke` now lets a genuine mismatch propagate instead of masking it. Export raises a `ValueError` naming the engine and its fields, and writes no file. The export case is the one worth being strict about: a figure captioned "Temperature" that actually shows smoke is worse than a failed export, so it now fails. Grep confirms no `except ValueError`-to-smoke pattern survives anywhere in `workbench/`, `export/`, or `analysis/`. Four new tests in `tests/test_field_catalog.py`; verified against the pre-C19 code they fail (viewport accepted the bogus name silently, export wrote a smoke PNG under the wrong label). Full suite: 429 passed, 28 skipped. |
| C20 | 09-08 | **Done.** Removed the flat-0.5 temperature fallback. The old branch returned `np.full(self.grid_shape, 0.5, dtype=np.float32)` whenever `thermal_enabled` was false: a real, correctly-shaped float32 array containing a constant. On screen that is indistinguishable from a perfectly isothermal simulation, so selecting Temperature on a non-thermal run produced a plausible-looking answer to a question the model never simulated. Both `LBM2D` and `LBM3D` now raise a `ValueError` naming `thermal_enabled` and listing the fields that do exist. **Scope extended beyond `engines/lbm2d.py` as the plan specified:** `lbm3d.py:378` carried a byte-identical copy of the same branch, so fixing only 2D would have left a known wrong-render path in the 3D engine. Worth noting the inheritance worked in our favour here — `LBM2DLiquid` and `LBM2DMultiComponent` subclass `LBM2D` and inherit the corrected branch, and the tests assert the guarantee across all five engines rather than assuming it. Verified against the pre-C20 engines, five tests fail because the field is still served. Full suite: 440 passed, 28 skipped. |
| C21 | 09-09 | **Done.** `Probe.record` matched each requested field against a hardcoded `u`/`v`/`speed`/`pressure` chain and `continue`d past anything else, so an unrecognised field recorded nothing at all. Confirmed against the shipped library: `presets/scenes/oil_water_separation.json` ships a probe requesting `color`, `component1`, `component2` — **all three were dead**, flatlining at zero samples with no error anywhere. Reproduced directly against the pre-C21 probe: four requested channels, four empty deques. Any engine field is now read from `get_field()` at the probe cell, and `recordable_fields(sim)` derives the offerable set from the engine the same way C18 did for the field combo. Two details worth recording. **`speed` is deliberately kept local**, not redirected to the engine's `"speed"` field: the engine normalises that one against a running max for display (0..1), whereas the probe series has always been the physical magnitude. Swapping one for the other would have silently rescaled every saved speed trace, so `LOCAL_FIELDS` documents the distinction and a test pins it. **A per-step field cache is now shared across probes** (`AnalysisPanel.tick` builds one dict): `get_field()` returns a whole-domain array normalised against a running max, so N probes on one field meant N recomputations *and* each one perturbed the value the next probe saw. A genuinely unavailable field — reachable if thermal is toggled off mid-run — is logged once at ERROR and then skipped, and `unavailable_fields` exposes it, instead of being dropped silently every step. New `tests/test_probe_fields.py` (22 tests) covers all three engines. Full suite: 462 passed, 28 skipped. |
| C22 | 09-09 | **Done.** Added `recordable_fields`/`default_probe_fields`/`validate_probe_fields`/`validate_scene_probes` to `scene/probe.py`, and wired validation into `_apply_and_refresh` — the single path every scene change goes through, so preset load, saved-file open and engine swap are all covered by one call. Problems are logged at WARNING and surfaced in the status bar ("N probe field(s) unavailable on LBM2D — see log") instead of leaving a flat trace to puzzle over. Both placement sites (viewport click-to-probe and the scene-panel add button) now derive their default field list from the engine instead of relying on the `ProbeSpec` default. **Gap found and recorded rather than papered over: no preset declares which engine it expects.** `product.engine` is absent from all ten preset files, and the engine is fixed at startup by a CLI flag, so a preset genuinely cannot be validated against a declared engine at load time. That is the strongest argument for validating at apply time, and it is the path that now reports it: open `oil_water_separation.json` on a plain `LBM2D` and you are told its three component channels will not record, instead of discovering it from an empty plot. The tests pin the pairing explicitly (multiphase fields ⇒ `LBM2DMultiComponent`) and include `test_no_preset_declares_an_engine_yet`, which documents the gap and will fail loudly once presets gain `product.engine`, at which point it should be replaced by a per-preset check. 16 new tests. Full suite: 478 passed, 28 skipped. |
| C23 | 09-09 | **Done.** C22 proved `validate_scene_probes` returns clean, which is a statement about what the engine *declares* — not about whether anything is recorded. The C21 bug was silent and would have passed a declaration-only check. This commit verifies recording end to end: load each shipped preset, resolve the engine it needs, build it at the preset's **own declared grid** (512x256 for `karman_street`, so `apply_to_sim` is exercised against real dimensions), and assert every requested channel fills up with a finite sample and reports no unavailable fields. Two extra invariants fell out of doing it for real. Engine-returned samples are display-normalised to 0..1, so a sample outside that range means the probe read the wrong array — checked for every engine-sourced field on every preset. And a channel that records once then sticks is a distinct failure from one that never records, so `cylinder` is recorded four times to confirm the series grows. Confirmed this catches the C21 regression: emulating the old whitelist across the preset library reproduces three dead channels in `oil_water_separation.json`, which fails the new assertion. 4 new tests, 482 total. Full suite: 482 passed, 28 skipped. |
| C24 | 09-11 | **Done.** `apply_to_sim` calls `reset_for_scene`, which seeds the initial condition but never resizes — an engine built at 128x128 stays 128x128 when handed a 512x256 scene. Nothing raised and nothing logged. Why that is dangerous rather than merely untidy: obstacles land wherever they fall on the smaller domain, and **probes past the edge are clamped into the last column**, so Karman Street's probes at x=180, x=300 and x=180 all read the same cell. Three probes, one measurement. The field arrays are allocated in the engine constructor, so rebuilding is the only correct fix: `_ensure_engine_matches_scene()` rebuilds at the scene's dimensions and is called from `_apply_and_refresh` *before* `apply_to_sim`, which is the single path every preset load, file open and parameter change takes. While there, extracted `_adopt_engine`: the four-object re-pointing sequence (`self.sim`, viewport, scene panel, analysis panel, outcome panel) was already duplicated verbatim in `_set_physics_mode` and `_toggle_gpu`, and missing one object would leave a panel reporting from a discarded engine. All three engine-recreation paths now share it. The rebuild is idempotent and preserves the engine class, so it works for `LBM2DLiquid` and `LBM2DMultiComponent` too and a no-op apply stays a no-op. New `tests/test_scene_apply.py` (14 tests): pins the underlying non-resizing behaviour, the clamp-collapse it causes, and the rebuild across all three engines. Verified against the pre-C24 app, 11 of the 14 fail. Full suite: 496 passed, 28 skipped. |
| C24 (cont.) | 09-11 | **Done — second pass over the remaining `apply_to_sim` call sites.** Having fixed `_apply_and_refresh`, audited all five other callers. `scripts/render_presets.py` and `analysis/sweep.py` both already construct `LBM2D(width=scene.width, height=scene.height)`, so they were never affected — now pinned by tests so a future edit cannot quietly reintroduce the mismatch. `MainWindow.reset()` applies directly instead of routing through `_apply_and_refresh`, and since `scene.width`/`height` are editable in the properties editor, hitting Reset after a resize would have reintroduced exactly the bug just fixed; it now calls the same idempotent guard. `scene_panel._reapply()` cannot rebuild (the panel holds no engine factory) but is only reached from `_apply_and_refresh` *after* the rebuild, so its ordering is already correct. Full suite: 499 passed, 28 skipped. |
| C25 | 09-11 | **Done.** 14 new tests in `tests/test_scene_apply.py` pinning the invariant `(engine.height, engine.width) == (scene.height, scene.width)` as a standing property rather than a one-off fix. Covers all three engines, six scene/engine size combinations including deliberately awkward ones (65x33 onto 64x32, to catch silent rounding), odd aspect ratios, and confirmation that a follow-up `apply_to_sim` does not break the invariant. Also asserts ordering — the guard must run *before* `apply_to_sim`, since rebuilding afterwards would seed the engine that is about to be discarded and then resize, losing the seed — and that every shipped preset declares a positive grid, since a zero dimension would make the rebuild itself fail. **Measured the blast radius rather than trusting the plan's estimate: with the guard stubbed to a no-op, 8 of the 10 shipped presets violate the invariant, not the 7 the plan predicted.** Only `lid_driven_cavity` and `oil_water_separation` happen to be 128x128 and so agree with a default engine; every other preset, including `karman_street` at 512x256, silently landed on the wrong domain. Full suite: 513 passed, 28 skipped. |
| C26 | 09-12 | **Done.** Audited what `reset()` and `_apply_and_refresh()` each cleared and found they cleared **different subsets** of the same derived state. `reset()` touched only `step_count`, `_state_history` and the timeline widgets; everything else leaked. Five residue cases, all user-visible: probe plots kept carrying the previous run's samples; the analysis panel's stability trace kept drawing the old convergence curve beside the new one; `_tick_counter` stayed offset, so probe sampling no longer lined up with the step count it is plotted against; an active video recorder kept appending the new run's frames to the old video; and — the worst — a reset fired mid-drag left `_scrubbing` True, after which `tick()` skips `sim.step()` **forever**: the app looks hung with nothing wrong and no error to find. Added `_reset_runtime_state()` clearing all of it, and pointed both `reset()` and `_apply_and_refresh()` at it, so a new piece of derived state is cleared by construction instead of by remembering to add it to four call sites. The per-object clearing moved to where the state lives: `AnalysisPanel.reset_runtime()` (stability deque, tick phase, probe histories, curve) and `Viewport.invalidate_caches()` (GL textures are sized to the old grid, so a resize must invalidate them). Ordering checked rather than assumed: `_run_guided_demo` calls `reset(silent=True)` and *then* sets `_demo_running = True`, which is the only reason the guided demo still starts now that reset clears the flag — pinned by a test so a future reorder cannot silently disable every preset demo. `_apply_and_refresh` rebuilds probes before calling the shared reset, so it clears panel-side state rather than the fresh probe objects. New `tests/test_lifecycle_reset.py` (25 tests), including a guard that `reset()` no longer hand-rolls its own clearing, since two paths touching the same fields directly is how the drift started. Full suite: 538 passed, 28 skipped. |
| C26 (cont.) | 09-12 | **C27 and C28 are folded into C26 — the plan's two rows were the same defect.** C26 introduced `_reset_runtime_state()` as the single clearing path precisely so that "clear the probe and stability histories" and "clear the demo, recorder and scrub state" would be handled by construction rather than by remembering them at four call sites. Both landed in `AnalysisPanel.reset_runtime()` (stability deque, tick phase, probe histories, curve) and in `_reset_runtime_state()` (`_demo_running`, `_scrubbing`, `_state_history`, the timeline widgets, and a live recorder). Rows marked `[x]` with a note rather than renumbered, so the commit ids still match the history. `scene/probe.py` needed no change: a fresh `Probe` object is built per apply, so its deques start empty by construction. |
| C29 | 09-13 | **Done.** `Viewport.invalidate_caches()` only cleared `_tex_init`. Four further caches hold geometry derived from the domain and were invisible to it: `_streamline_cache` and `_contour_cache` store pixel coordinates built from `_canvas_rect()` and the grid shape, `_particle_cache` the same, and `_colorbar_pixmap` is sized from the canvas height while being keyed on the colormap name *alone*. Each is only recomputed when `_overlay_frame % 6` (or `% 4`) is zero, so a scene load, an engine rebuild at a new grid (the C24 path) or a window resize left the viewport drawing the previous domain's overlay — contours stretched away from the field underneath them — five frames out of six. Worse, the colorbar bug survived even a synchronous repaint, because the key that would have invalidated it does not change on a resize. There was also no `resizeEvent` override at all. `invalidate_caches()` now clears all five and `set_scene()` delegates to it instead of poking `_tex_init` directly, and a new `resizeEvent` calls the narrower `_invalidate_overlay_caches()` — deliberately *not* the full invalidation, because the field texture is sized to the grid rather than the widget, so a window resize must not force a reallocation on every resize frame. The invalidation happens before `super().resizeEvent()`, so the paint it schedules rebuilds at the new rectangle; afterwards, on a platform that paints synchronously, it would discard a cache the paint had just built correctly. New `tests/test_viewport_caches.py` (18 tests). Two test-harness facts are worth recording because they shaped the tests: the offscreen platform does not route a resize to a `QOpenGLWidget` at all, so the event is delivered explicitly after `resize()`, and delivering it paints synchronously and legitimately refills the caches — so the "caches are empty" assertions suppress the repaint, while the geometry assertions let it happen and check the result. **15 of the 18 fail against the pre-C29 viewport.** Also pinned the inverse: an overlay toggle keeps its cache, since a toggle does not change the domain and dropping it would recompute on every click. |
| C30 | 09-13 | **Done.** Five objects hold a `Scene`: the window plus the viewport, scene panel, analysis panel and outcome panel. `_undo`/`_redo` rebound `self.scene = dict_to_scene(...)` **on the panel only**, and `_on_scene_changed` then read the window's object rather than adopting the panel's. Nothing raised: after one undo the panel was editing a scene the viewport, probes, save and export could not see, and the next edit was applied to a document that was silently on its way out. Two fixes, deliberately separate. **(1) One entry point.** `MainWindow._adopt_scene()` is now the only way a Scene enters the app; `_share_scene()` re-points all four holders and replaces the hand-rolled sequence that `_apply_and_refresh` and `__init__` each maintained separately. `_file_new`, `_file_open`, `_load_preset_file` and `_on_wizard_template` all route through it. **(2) `ScenePanel.set_scene()`** replaces the raw `scene_panel.scene = ...` assignment, and clears the property editor rather than repainting it — `_PropEditor` holds a direct reference to the dataclass it edits, so leaving it alive across an adoption meant the next keystroke wrote into a discarded obstacle. `_on_scene_changed` keeps an explicit divergence guard: if a holder ever points elsewhere it is corrected there, rather than trusting that no path can cause it. That guard is deliberately not deleted in C31 even though in-place undo makes it inert, because it is the enforcement point for the invariant. New `tests/test_scene_ownership.py` (17 tests), 5 of which fail against the pre-C30 window. Three of those test the *mechanism* rather than the outcome, because an outcome test cannot tell a real fix from "the window copies the panel's reference and lives with it": an AST check asserts `self.scene` is written in exactly three places (`__init__`, `_adopt_scene`, and the guard), a source check asserts no load path assigns it directly, and another asserts the panel never calls `default_scene()`. Also found and killed a **second, independent leak of the same field**: `_PropEditor._set` did `setattr(self._obj, ...)` on an object reached through the tree, so an editor that outlived an adoption was not merely stale but writable. Full suite: 573 passed, 28 skipped. |
| C31 | 09-14 | **Done.** Three defects, all about the snapshot being taken in the wrong place or the wrong object. **(1) The first user edit was unreachable.** `_PropEditor._set` did `setattr(self._obj, ...)` and *then* emitted, so `_on_prop_changed` snapshotted the already-edited scene: a snapshot whose contents are the new value, making undo of that edit a no-op. The signal is now `about_to_change`, emitted before the assignment — the only place that knows a mutation is one keystroke away. The add/remove/import paths were already correct, which is why this survived: three-quarters of the panel behaved properly and the one broken path was the most-used one. **(2) Restore replaced the object** (`self.scene = dict_to_scene(...)`), which is what split the session in two — see C30. Added `scene.serializer.update_scene_in_place()`, which assigns every field of the parsed snapshot onto the existing object, so identity is preserved by construction rather than by every caller remembering to re-adopt. `_restore()` also clears the property editor: it holds a reference into the container list that the restore replaces, so leaving it open would have it edit a detached object. **(3) The four simulation-parameter spinboxes — viscosity, inflow, smoke diffusion, smoke decay — wrote to `Scene` as well as to the engine and pushed no snapshot at all**, so they were the only scene edits in the panel undo could not reach. Fixing that introduced a new failure, which is the part worth recording: `sync_params_from_scene()` drives those *same* widgets with `setValue`, which emits `valueChanged`. After an undo the values genuinely differ, so an unguarded sync would push an entry for a change nobody made **and clear the redo stack** — destroying the redo of the very edit just undone. `_syncing_params` guards it and is released in a `finally`. **(4) A fourth gap found on the way:** `MainWindow._on_viewport_probe` appended the probe itself, bypassing the panel, so clicking the canvas was not undoable even though adding a probe from the panel menu was. Added `ScenePanel.add_probe_from_viewport()` and routed the canvas through it, which also removed a duplicated rebuild/sync/refresh sequence. New `tests/test_undo_redo.py` (26 tests), **16 of which fail against the pre-C31 panel**. One pre-existing pass is worth naming: removal was *already* correctly snapshotted before mutating, so `test_a_removal_is_reachable` passes on both sides — the tests pin the whole surface, not just the holes. Full suite: 599 passed, 28 skipped. |
| C32 | 09-14 | **Done.** The undo stacks hold JSON snapshots of the document, and nothing discarded them on a load — so after opening a file, a preset or a wizard template, the Undo button was still live and pressing it replaced the newly opened scene with the contents of whatever was open before. No warning, no prompt, and nothing in the title to indicate the document had changed underneath: from the user's side the file they opened simply vanished. There is no meaningful undo that crosses a document boundary, so `ScenePanel.clear_history()` empties both stacks and disables both buttons, called from `MainWindow._adopt_scene()` — the single entry point C30 introduced, so the four load paths are covered by one call rather than four. Two placements are worth arguing for. It is **not** called from `_reset_runtime_state()`: a reset re-seeds the same document, and clearing its history would make the Reset button silently discard the user's edits — trading a recoverable bug for a destructive one. Nor is it called on an engine switch, where the document is unchanged and the history is still perfectly meaningful. The boundary that matters is *which document is open*, not *which solver is running*. Tests in `tests/test_undo_redo.py` (7 new, 4 fail against the pre-C32 window) pin both directions: every load path clears the stacks, and reset plus engine switch preserve them. Full suite: 606 passed, 28 skipped. |
| C33 | 09-15 | **Done.** Completes C26, and the gap between the two is instructive: `test_lifecycle_reset.py` pins what `_reset_runtime_state()` clears using a stub window holding only the fields it knows about, and **a stub cannot expose a field nobody remembered** — which is exactly how the residue got there in the first place. New `tests/test_state_reset.py` (18 tests) drives the real widgets offscreen. It found two live defects, and both share one cause: **the reset clears the input to a consumer, not what the consumer already drew.** **(1) The readouts.** `_update_physics` and `_update_field_stats` run every fifth tick, so zeroing `_tick_counter` left Re, Cd, St, min, max and mean describing a run that had just been discarded — and on a *paused* window nothing ever ticks, so the stale numbers sat there indefinitely rather than expiring. `reset_runtime()` now blanks all six. **(2) The probe y-ranges.** `_ProbePlot` deliberately never shrinks its range, which is right within a run (it stops the axis jumping between samples) and wrong across one: a probe whose signal lived in 0–0.05 after previously spanning 0–1.0 rendered as a flat line along the bottom of the old axis, reading as "nothing is happening" rather than as a bug. Ranges are now cleared on reset, with a test that asserts the new run can rescale rather than only that the field is `None`. A third, milder one: `update_outcome` refreshes every fifteenth step, and `reset()` forced it while `_apply_and_refresh()` did not — so after opening a file, preset or wizard template the regime card, confidence badge, drag and scorecard kept describing the discarded scene. `_apply_and_refresh()` now forces it, matching `reset()`. 7 of the 18 fail against the pre-C33 panel. The C26 stub tests were **updated rather than left passing**: registering six new labels and a widget list on a hand-built stub is precisely the weakness C33 exists to remove, so each stub now carries a note pointing at the real-widget tests, and C33 adds the mechanism guard C26 could not have — a list of cadence-driven labels asserted against the live widget, so adding a readout without registering it fails there instead of silently going stale. The recorder test asserts on the **file**, not the flag: `imageio.imread` of the closed MP4 must return its frame, because a dropped-but-unclosed writer leaves a file with no moov atom that looks fine until someone tries to open it. Full suite: 626 passed, 28 skipped. |
| C35 | 09-16 | **Done.** P1-2 counted five incompatible pressure definitions, but the factor-of-3 divergence had a **single root cause**: `get_pressure_gauge()` existed only on `LBM2D`, so an analysis routine could not call it on any other backend without `AttributeError`, and each engine compensated by inventing its own display normalization — `LBM2D` used the gauge `(rho-1)/3`, `LBM3D` and the GPU backends used `rho-1`. Same field, three times larger, depending on which backend produced it. The fix is structural rather than enumerative: `get_pressure_gauge()` is now a **concrete method on `SimEngine`**, defined once as `p - c_s^2 * rho0` from new class constants `SOUND_SPEED_SQUARED` and `REFERENCE_DENSITY`, so every backend inherits the identical definition and a hypothetical one with a different equation of state changes the constants rather than the formula. The now-redundant `LBM2D` override is deleted with a comment naming the failure it enables. `get_pressure()` stays absolute and is documented as *not* the display quantity — it is what the outlet BCs (`rho = 3p`) are stated in, so it is not redundant. **`plot_pressure()`'s docstring said `rho - 1` while its body plotted `rho/3`**: a difference of a factor of 3 plus an offset, invisible in the plot's shape and therefore invisible in review, but it is the ambiguity being chased. It now plots the gauge with a symmetric range, a `p' (lattice)` colorbar label and the formula in the title. Second half is the metadata: `FieldInfo` gains `quantity` (the defining formula) and `unit`, filled in for all ten fields, plus `field_quantity()` / `field_unit()` accessors that return `""` for unknown names rather than raising — the field catalog is engine-dependent, so they are called with names that may not exist. The pressure tooltip's false "Pressure field (rho - 1)" is corrected, and it now explains *why* the gauge form is used. The point of putting the definition in metadata is that a definition nobody can find is the original defect: it can drift from the code silently, which is exactly what a tooltip did. New `tests/test_pressure_canonical.py` (31 tests, 23 fail against the pre-C35 base — the module does not even import without the new accessors). Two tests are deliberately written as *mechanism* rather than outcome: `get_pressure_gauge` must be defined on `SimEngine` and must not be overridden by any engine, because an outcome test cannot distinguish "one shared definition" from "six copies that currently agree". One test is scoped to single-phase engines and one to multiphase, because the liquid and multi-component backends seed a `rho=2` droplet (C10) — asserting zero gauge pressure there would be asserting the absence of physics. Full suite: 656 passed, 29 skipped (the extra skip is `plot_pressure`, which needs matplotlib and is not installed). |
