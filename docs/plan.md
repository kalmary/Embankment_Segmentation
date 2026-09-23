# Embankment Segmentation Rebuild Plan

**Goal:** Incrementally divide the existing ground-profile segmentation code into tested configuration, geometry, classification, persistence, and optional visualization responsibilities without changing numerical output.

**Root-facing contract:** `GroundSegmenter.from_config(...).segment(points, labels)` returns a point-aligned copy with only the documented terrain labels changed.

**Design:** `../../../docs/rebuild.md`

**Branch requirement:** Perform all rebuild work on `development`. Verify the active branch first and request explicit approval before creating or switching it.

## Task 1: Establish the uv project

- [x] Create `.python-version`, `pyproject.toml`, and `uv.lock` for Python 3.12.
- [x] Replace the full environment freeze with direct dependencies discovered from imports.
- [x] Define a headless `basic` group and a `test` group including `basic`, `pytest`, `matplotlib`, and `pyvista`.
- [x] Keep plotting dependencies out of the basic import path; retain Open3D where computational code requires it.
- [x] Verify clean basic/test syncs and import the public segmenters.

## Task 2: Characterize public contracts

- [ ] Test every accepted config key, type, default, range, and unknown/missing-key behavior.
- [ ] Test input shape, dtype, copy-versus-mutation behavior, point ordering, preserved labels, and output labels.
- [ ] Add fixtures for straight, curved, sparse, degenerate, missing-rail, and missing-ground profiles.
- [ ] Pin current database-query behavior with a fake adapter and no live PostgreSQL dependency.
- [ ] Test standalone, parent-repository, direct-script, and module imports.

## Task 3: Stabilize imports and resources

- [ ] Replace conditional generic `utils` imports with package-relative imports.
- [ ] Keep `SegmentGround.py`, `Segment_embankment.py`, and `SegmentDitches.py` as compatibility modules while internals move.
- [ ] Resolve config and database parameter files from explicit paths or stable project-relative defaults.
- [ ] Move plotting imports behind plotting calls and test headless imports.
- [ ] Verify direct and module execution after each change.

## Task 4: Split GroundSegmenter by responsibility

- [ ] Extract configuration parsing/validation first.
- [ ] Extract database/rail-geometry retrieval behind a small injected interface.
- [ ] Extract centerline/section construction, profile smoothing, embankment detection, ditch detection, and label application one at a time.
- [ ] Retain existing methods or forwarding calls when external callers may use them.
- [ ] Compare arrays against characterization fixtures after every extraction.

## Task 5: Consolidate only proven duplication

- [ ] Compare legacy embankment/ditch classes with the active `GroundSegmenter` path.
- [ ] Share helpers only where behavior and units are identical under tests.
- [ ] Leave legacy behavior in place when equivalence is not demonstrated.
- [ ] Do not rename public files or config keys merely for style.

## Task 6: Verify

- [ ] Run all deterministic unit and integration fixtures without database or display access.
- [ ] Run a separate PostgreSQL integration smoke test with explicit credentials/configuration.
- [ ] Run plotting smoke tests in the test environment.
- [ ] Compare output labels on a representative railway tile before and after the rebuild.
- [ ] Run the root pipeline's ground-stage and border-tree integration tests.

## Completion Gate

Both uv groups reproduce; headless imports do not require visualization; the public segmenter and configs remain compatible; fixture and representative-tile labels match; database and visualization integrations are isolated.
