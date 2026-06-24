# CLAUDE.md — repository guide for AI agents and contributors

This is a research fork of **RGFN** (Reaction-GFlowNet) for generating novel
**molecular glue degraders**. We extend upstream RGFN with new oracles, reward
shaping, batch-selection strategies, benchmarks, and dataset tooling.

**Read this file and `docs/ARCHITECTURE.md` before making structural changes.**
For the science/goals, read `Logs/RESEARCH_CONTEXT.md`.

---

## The one rule that governs everything: old vs. new

| Path | Owner | Rule |
|---|---|---|
| `rgfn/` | **Upstream** | Pristine RGFN. **Do not edit.** Extend it from `glue/`. |
| `configs/` (except `configs/glue/`) | **Upstream** | Treat as pristine. New configs go in `configs/glue/`. |
| `gin_config/`, `train.py`, `grid_search.py`, `tests/` (upstream parts), `data/chemistry.xlsx`, `data/targets/`, `external/setup_{gnina,gneprop,reinvent,shared}.sh` | **Upstream** | Leave as-is. |
| `glue/` | **Ours** | All new Python: oracles, rewards, samplers, proxies, datasets. |
| `configs/glue/` | **Ours** | All new gin configs (overlay upstream via `include`). |
| `scripts/` | **Ours** | Entry points (`train.py`/`infer.py` wrappers) + `hpc/`. |
| `benchmarks/`, `models/`, `data/synthetic/` | **Ours** | Benchmark harness, input structures/checkpoints, generated datasets. |
| `research/preprocessing/` | **Ours** | Docking oracle validation pipeline (was `pre-processing/`). |
| `Logs/`, `docs/` | **Ours** | Experiment logs + project documentation. |
| `Logs/references/` | **Ours** | Canonical bibliography for papers we build on. Cite by key (`[koziarski2024rgfn]`); annotated index in its `README.md`. PDFs in `pdfs/` are git-ignored. |

We keep `rgfn/` and `configs/` mergeable with upstream. The only deliberate edits
to upstream files are three small operational overrides documented in
**`docs/PATCHES.md`** — read that before "fixing" anything odd in those files.

---

## How to extend the system (don't edit `rgfn/`)

1. **New oracle** → implement the science in `glue/oracles/`.
2. **Expose it to training** → write a `@gin.configurable` adapter in
   `glue/proxies/` subclassing upstream `CachedProxyBase`/`ProxyBase`. See
   `glue/proxies/example_glue_proxy.py` for the working template.
3. **New reward / sampler** → `glue/rewards/` or `glue/samplers/`, subclassing the
   upstream base classes in `rgfn/api/`.
4. **Register it** → make sure the module is imported by `glue/registry.py`
   (directly or via its subpackage `__init__`). Gin finds classes by name only
   after they are imported.
5. **Configure it** → add a gin config in `configs/glue/` that `include`s the
   upstream base and references your class as `@YourClass`.
6. **Run it** → `python scripts/train.py --cfg configs/glue/<cfg>.gin`. The
   wrapper imports `glue` first so gin can resolve your component; the root
   `train.py` (upstream) only knows about `rgfn`.

If gin says *"No configurable matching @X"*, the defining module wasn't imported
on the startup path — fix `glue/registry.py`, not the config.

---

## Project goals (what we're building toward)

- New **oracles** (ternary docking / neosubstrate differential; possibly MD).
- New **rewards** (differential reward isolating glue cooperativity).
- Possibly expanded **batch selection**.
- **Benchmarks** comparing RGFN vs. baselines and across protein systems.
- **Models & datasets** as input; **synthetic datasets** as output.

---

## Working notes for agents

- **Compute:** Heavy stack (torch-geometric, openbabel, meeko, gnina) + GPU
  docking runs on **Balam** (SciNet). Balam may be down; when it is, work on a
  Mac laptop and **validate what you can locally** (imports, `py_compile`, gin
  config-include integrity, `bash -n`) — full train/dock validation happens on
  Balam. State clearly in your summary what you did vs. couldn't verify.
- **Document as you go:** record structural changes and anything left unverified
  in `docs/REFACTOR_LOG.md` so the next agent can continue or repair the work.
- **Experiment logs:** use the `experiment-log` skill for any real computational
  experiment.
- **Historical logs** (`Logs/001`–`005`) predate the 2026-06-24 restructure and
  reference `pre-processing/`; read those paths as `research/preprocessing/`.
