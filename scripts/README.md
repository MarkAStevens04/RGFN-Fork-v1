# scripts/

Our entry points and operational scripts. These are **ours** (not upstream).

| Script | Purpose |
|---|---|
| `train.py` | Training wrapper — imports `glue` (so gin sees our components), then runs the upstream root `train.py`. Use this for any config under `configs/glue/`. |
| `infer.py` | Inference wrapper — same idea, delegates to the root `infer.py`. |
| `hpc/` | SLURM / cluster build & submit scripts for Balam. |

## Why wrappers exist

Gin discovers components by class name, but only after the defining module is
imported. Upstream `train.py` imports only `rgfn`, so it cannot see anything in
`glue/`. These wrappers `import glue` first (registering our oracles, rewards,
samplers, proxies) and then delegate to the unmodified upstream entry point.

Plain upstream RGFN configs can still be run with the root `train.py`/`infer.py`
directly; use these wrappers whenever the config references a `glue` component.
