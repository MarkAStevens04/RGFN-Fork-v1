# Refactor log — 2026-06-24 repo restructure

A running record of the restructure that introduced the `glue/` package and the
upstream/ours boundary. **If something is broken, start here**: this documents
what changed, why, what was verified, and what could not be verified (Balam was
down; work was done on a Mac laptop, so no GPU/docking/training runs were possible).

Owner: Mark + Claude. Goal context: `Logs/RESEARCH_CONTEXT.md`. Layout: `docs/ARCHITECTURE.md`.

---

## Why

The repo had fused upstream RGFN with our additions at the top level with no clear
boundary. We separated them: `rgfn/` + `configs/` stay pristine/mergeable; all new
work lives in `glue/` and clearly-owned top-level dirs. See `CLAUDE.md`.

Key finding during planning: the upstream package was *already* ~pristine. The
proxies/docking code/configs that looked like ours are actually upstream (2024
commits). Our only edits to upstream files were **three operational overrides**,
now documented in `docs/PATCHES.md`. So this was mostly **additive scaffolding +
moving our own files**, not extracting code out of `rgfn/`.

---

## What changed

### Added (new, zero risk to upstream)
- `glue/` package: `oracles/`, `rewards/`, `samplers/`, `proxies/`, `datasets/`,
  plus `registry.py` (import hub for gin discovery) and `__init__.py`.
  - `glue/proxies/example_glue_proxy.py` — working template adapter (returns QED)
    showing the `CachedProxyBase` + `@gin.configurable` pattern.
- `configs/glue/README.md` — overlay pattern for new configs.
- `scripts/train.py`, `scripts/infer.py` — wrappers that `import glue` (so gin
  sees our components) then delegate to the root entry points via `runpy`.
- `scripts/README.md`, `scripts/hpc/` (holds `submit.sh`).
- `benchmarks/README.md`, `models/README.md` + `.gitkeep`,
  `data/synthetic/README.md` + `.gitkeep`.
- `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/PATCHES.md`, this file.

### Moved
- `pre-processing/`  →  `research/preprocessing/` (whole tree via `git mv`, so
  internal `__file__`-relative and `../test-data` paths are preserved).
- `submit.sh`  →  `scripts/hpc/submit.sh` (and its train line now calls
  `scripts/train.py`).

### Edited (path fixups caused by the moves)
- `research/preprocessing/docking_gnina/submit_dock_crbn.sh` — `$REPO/pre-processing/...`
  → `$REPO/research/preprocessing/...` (2 refs).
- `research/preprocessing/docking_6td3/submit_dock_6td3.sh` — same (2 refs).
- `research/preprocessing/docking_gnina/analysis/plot_discrimination_curves.py` —
  `ROOT` now `parents[4]`; `PRE = ROOT / "research" / "preprocessing"`.
- `Logs/RESEARCH_CONTEXT.md` — "Where things live" updated + restructure note.
- `Logs/000_TEMPLATE.md` — `pre-processing/` → `research/preprocessing/` prefixes.
- `scripts/hpc/submit.sh` — train line → `python scripts/train.py ...`.

### Upstream files (deliberately left edited — see docs/PATCHES.md)
- `rgfn/trainer/logger/wandb_logger.py`, `configs/loggers/wandb.gin`,
  `configs/rgfn_seh_docking.gin`. Not reverted; documented as patches.

### Not changed
- `rgfn/` (apart from the documented wandb one-liner), `gin_config/`, upstream
  `configs/`, `train.py`, `grid_search.py`, upstream `data/` inputs, `tests/`.
- Historical experiment logs `Logs/001`–`005` keep their original
  `pre-processing/` paths as point-in-time records (noted in RESEARCH_CONTEXT).

---

## Validation

Environment: macOS laptop, **Balam down**. Could NOT run training, docking, or
anything needing the GPU/heavy deps. Verified what is checkable statically.

### Verified (static, on the laptop — 2026-06-24)
- **`py_compile`** of every `.py` under `glue/`, `scripts/`, `research/` → all compile.
- **No dangling `pre-processing` references** in any `*.py` / `*.sh` / `*.gin` /
  `*.toml` (only historical `Logs/001`–`005` still mention the old path, by design).
- **`bash -n`** on `scripts/hpc/submit.sh` and both moved `submit_dock_*.sh` → OK.
- **gin `include` integrity**: every `include '...'` in `configs/**/*.gin` resolves
  to an existing file.
- **glue import chain**: `glue/__init__` → `registry` → all 5 subpackages →
  `example_glue_proxy` all exist and reference real upstream symbols
  (`CachedProxyBase`, `ReactionStateEarlyTerminal`).
- **Upstream untouched this session**: `git status` shows no modifications to
  `rgfn/` or upstream `configs/` (only `configs/glue/` added). The 3 documented
  patches in `docs/PATCHES.md` are prior commits, not new edits.

### Could NOT verify locally (heavy deps not installed → do on Balam)
- `import glue` fails with `ModuleNotFoundError: No module named 'gin'` — this is
  an **environment** issue (gin-config/torch-geometric/etc. aren't installed on the
  laptop), **not a code issue**. Re-run `python -c "import glue"` in the `rgfn`
  conda env to confirm registration works.

### NOT verified (do on Balam when it's back)
- `python scripts/train.py --cfg configs/glue/<cfg>.gin` actually trains.
- `import glue` succeeds in the real env (depends on torch-geometric etc., which
  may not be installed on the laptop — a local import failure here is likely an
  environment issue, not a code issue; confirm on Balam).
- The moved docking scripts (`research/preprocessing/...`) run end-to-end with
  gnina on a full node.
- The example proxy produces a sensible reward in a real run.

---

## How to continue / repair

- **gin "No configurable matching @X":** the module defining `X` isn't imported on
  startup. Ensure it's imported by `glue/registry.py` (or its subpackage
  `__init__`), and that you launched via `scripts/train.py` (not root `train.py`).
- **A moved docking script can't find a file:** check it's reading relative to its
  own dir (`HERE`) or `../test-data`; repo-root-relative refs were updated to
  `research/preprocessing/` — grep for any stragglers: `grep -rn pre-processing .`
- **An upstream patch vanished after a merge:** re-apply from `docs/PATCHES.md`.
- **Adding new components:** follow the steps in `CLAUDE.md` ("How to extend").
