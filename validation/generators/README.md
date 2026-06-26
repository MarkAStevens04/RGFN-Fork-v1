# validation/generators/

Baseline molecule generators we compare our RGFN pipeline against. Each is a
**thin adapter** exposing one common interface; the heavy third-party code is
installed separately (not vendored into this repo).

These are *entrants* in a benchmark, not part of the production pipeline. Nothing
in `glue/` imports from here (see the boundary rule in `../README.md`).

## Planned interface (not yet implemented)

A common `Generator` base (intended home: `base.py`) so the harness
(`validation/harness/`) can drive every entrant identically — roughly:

- `setup(config)` — load weights / build the model.
- `generate(oracle, budget) -> molecules` — produce candidates under a fixed
  oracle-call budget (so RGFN and the baselines are compared on equal compute).
- `name`, `is_synthesizable` — metadata the harness records per run.

The point of the equal-budget interface is the headline comparison reviewers will
ask for (`docs/RESEARCH_CONTEXT.md`, Objective 4): does our **synthesizable**
generator beat non-synthesizable baselines on the *same* oracle and budget?

## Entrants

| Dir | Generator | Synthesizable? | Install |
|---|---|---|---|
| `rgfn/` | **Our pipeline** (thin adapter → `glue/` + `rgfn/`) | ✅ by construction | in-repo |
| `synflownet/` | SynFlowNet (reaction-based GFlowNet baseline) | ✅ | `external/setup_synflownet.sh` |
| `fraggfn/` | Fragment-based GFlowNet | ❌ (fragment assembly, no route) | `external/setup_fraggfn.sh` |
| `vae_bo/` | VAE + Bayesian optimization | ❌ | `external/setup_vae_bo.sh` |

The `rgfn/` adapter is deliberately thin — it calls the real production pipeline
in `glue/` rather than copying it, so a benchmark entrant can never drift from
what we actually ship.

> Scaffolding only — directories are placeholders. See `external/` for the
> install-script stubs that will pull in each baseline's upstream code.
