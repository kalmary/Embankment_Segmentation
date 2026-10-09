# Ground Segmentation Import Repair Plan

**Goal:** Keep the working ground-profile segmentation unchanged while making
its public class and standalone tools import reliably from every supported
execution context.

**Root-facing contract:**
`GroundSegmenter.from_config(...).segment(points, labels)` returns the same
point-aligned labels as before this import repair.

**Parent plan:** `../../../docs/plan.md`

## Frozen behavior

- Do not split or rewrite centerline construction, section construction,
  smoothing, embankment detection, ditch detection, label application, or
  database behavior.
- Preserve configuration keys, defaults, labels, array ordering, and numerical
  output.
- Keep the completed immutable `GroundConfig`; no further configuration
  redesign is planned.
- Keep every current file with `__main__` executable independently from this
  project, as both a module and a direct script where applicable.

## Completed foundation

- [x] Establish the Python 3.12 uv project and dependency groups.
- [x] Characterize configuration, arrays, labels, database queries, sparse
  inputs, and supported import contexts.
- [x] Extract and test `GroundConfig` without changing `GroundSegmenter`'s
  public attributes.
- [x] Keep Open3D out of the voxel-sampling import path.

## Task 1: Inventory executable and import contracts

- [x] List every current `__main__` guard and classify it as an operational,
  diagnostic, or developer command without removing it.
- [ ] Add or complete paired direct/module `--help` tests for every command.
- [ ] Assert that help does not load data, connect to PostgreSQL, create output,
  or initialize plotting.
- [ ] Verify `GroundSegmenter` import from this project root and the parent BRIK
  repository.

## Task 2: Repair imports and resource ownership

- [ ] Replace generic and wildcard utility imports with explicit imports from
  the defining module.
- [ ] Use package-relative imports in reusable modules without broad
  `ImportError` fallbacks.
- [ ] Keep direct-script compatibility in thin entry-point handling rather than
  in reusable algorithm modules.
- [ ] Resolve config and database-parameter defaults relative to this project or
  explicit user paths, never the caller's incidental working directory.
- [ ] Load Laspy, Open3D, PyVista, and plotting helpers only inside workflows
  that actually use them.

## Task 3: Preserve standalone tools

- [ ] Give operational, diagnostic, and developer scripts `main(argv=None)` and a
  non-destructive `--help` path.
- [ ] Replace developer-specific absolute-path examples with explicit CLI input.
- [ ] Verify `segment_ground.py`, `segment_embankment.py`, and
  `segment_ditches.py` in every declared direct/module form.

## Task 4: Verify

- [ ] Run import-boundary and invocation tests.
- [ ] Run the complete deterministic suite and compare `GroundSegmenter` arrays
  with the existing characterization fixtures.
- [ ] Verify the root ground stage without visualization dependencies.
- [ ] Run PostgreSQL and plotting smoke tests separately under `dev` when their
  external requirements are available.

## Completion gate

The public segmenter imports without visualization packages; standalone tools
work from their owning project; all algorithm fixtures remain unchanged; and
no computational segmentation method was refactored.
