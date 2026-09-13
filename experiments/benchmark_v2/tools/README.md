# campaign-specific drivers and verifiers

`manifest.py` resolves a cell from `grid.csv` + `matrix16/targets.py` + live filesystem status, the
way `matrix16/manifest.py` does for one pipeline. (This file used to say it was "deliberately not
written yet"; it has existed since the tree was set up.)

## ⛔ TWO INTERPRETERS, AND THE SPLIT IS LOAD-BEARING

**`manifest.py` is STDLIB ONLY on purpose.** Its own docstring says why: a bare SLURM batch shell has
no python on PATH until a conda env is activated, yet this module is what tells the launcher *which*
env the cell needs. So the launcher bootstraps with **conda base**, asks manifest for the spec, then
activates the cell's env. Importing anything heavier breaks that ordering — it already did once in
v1.

**Everything else here needs the `rgfn` env.** `measure_repeat_rate.py` (and therefore
`verify_cell.py`'s budget basis, which imports `_verdict` from it) reads `run_config.yaml` and so
needs **PyYAML**, which conda base does not have.

    manifest.py --emit ...            base is fine, and is the point
    everything else                   /home/markymoo/miniconda3/envs/rgfn/bin/python

A bare `python` on this cluster resolves to conda base. Four agents each reached for one; one of them
ran `measure_repeat_rate.py` from base, got 108 rows of "provider unresolved", and concluded the
landed cells could not be measured. They were measurable; the interpreter could not read their
configs. `measure_repeat_rate.py` now refuses up front with **exit 2** rather than producing that
report — exit 2 meaning *nothing was measured*, kept distinct from exit 1, *measured and some fell
short*, because a caller gating on non-zero would otherwise read a missing library as a finding.

**PyYAML is declared in no manifest in this repo.** `pyproject.toml`'s `[tool.poetry.dependencies]`
carries upstream RGFN's set, and adding a validation-tooling dependency there crosses the one-way
`validation/` → `glue/`/`rgfn/` arrow, so it is not an obvious edit. It works today because the
`rgfn` env happens to have it. Stated here so the next tool that needs it does not rediscover this
the same way.

## Exit codes, where they carry meaning

    verify_cell.py            0 accepted          1 rejected
    verify_backup.py          0 content verified  1 failure     2 structural pass only
    measure_repeat_rate.py    0 all measured, ok  1 some SHORT  2 REFUSED, nothing measured

`2` never means "worse than 1". It means the question was not answered, which a caller must not treat
as an answer.
