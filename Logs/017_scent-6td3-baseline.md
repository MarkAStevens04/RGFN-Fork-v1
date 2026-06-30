# SCENT / 6TD3 — cost-aware baseline head-to-head with RGFN
**Date:** 2026-06-30, ~5pm

## Question

When a competing generator actively tries to keep its molecules **cheap to
synthesize**, does our RGFN glue generator still produce comparably good 6TD3 glue
candidates — and what does that cost-awareness buy in diversity and synthesis cost —
given the same scoring oracle and the same budget?

## Context & Summary

**Context.** Entries `015` (FragGFN, the non-synthesizable foil) and `016` (RxnFlow,
a synthesizable peer) built out the baseline comparison reviewers will demand. SCENT
(`[gainski2025scent]`, arXiv:2506.19865) is the most pointed comparison of the three:
it is a **fork of RGFN from the same lab** that keeps RGFN's synthesizable
reaction-template machinery and adds explicit **cost-awareness** — it learns to steer
generation toward molecules whose synthesis routes use **cheap building blocks** and
**high-yield reactions**, plus two supporting tricks (a penalty that stops it from
over-exploiting the cheapest routes, and a "dynamic library" that promotes useful
intermediates to building blocks). RGFN is literally one of SCENT's own published
baselines, so this is the closest thing to a controlled test of "RGFN vs. RGFN + cost
guidance" on our glue task.

**Summary.** We run SCENT through the **same** active-learning loop, the **same** 6TD3
docking oracle, the **same** seed set of 408 labelled molecules, the **same** budget
(3 rounds × 32 molecules/round, 300 generator steps/round), the **same**
inverse-temperature β=8, and the **same** learned proxy reward as the RGFN run — so the
*only* thing that changes is the generator's internal machinery. SCENT runs its **full
cost-aware method** on its **SMALL building-block library**, which ships the
building-block prices and reaction yields its cost model needs (so cost guidance is
genuinely active, not a no-op). Because SCENT's python package is *named* `rgfn` (it is
an RGFN fork), it runs in its own software environment and reaches our docking oracle
through the shared scoring bridge — the same cross-environment setup FragGFN/RxnFlow
use, here recording each molecule's synthesis route so SCENT's output sits directly
alongside RGFN's.

## Answer

[TODO — after the Balam run: state whether RGFN matches/beats SCENT on glue score
(docking differential), how their diversity compares, and — the point of SCENT — how the
average synthesis **cost** of the good glues differs, and what that says about whether
cost-awareness helps or hurts glue quality at equal oracle budget.]

## Relevance to our Publication

This is the **cost-aware** arm of the baseline comparison (`docs/RESEARCH_CONTEXT.md`,
Objectives 4–5). FragGFN (`015`) controls for synthesizability, RxnFlow (`016`) controls
for a different synthesis-aware action space, and SCENT controls for **explicit cost
optimization** — letting us answer the reviewer question "is RGFN leaving easy
synthesis-cost wins on the table?" Because SCENT is RGFN's own published successor,
showing RGFN remains competitive (or showing exactly where cost-awareness changes the
candidate set) on the same oracle and budget is a strong, directly-citable result for a
methods venue (Digital Discovery / JCIM). SCENT also reports the AiZynth success rate we
adopt as our synthesizability metric (its Table 1, up to ≈0.75), tying the comparison to
`validation/harness/synthesizability.py`.

## Next Experiments

**Refining for publication.**
- Run ≥3 seeds for RGFN and SCENT (and FragGFN/RxnFlow) so the head-to-head carries error
  bars.
- A matched-oracle RGFN **GPU** rerun on a healthy node (entry `014`'s GPU run died on a
  wedged OpenCL node), so all entrants share the identical GPU oracle.
- Report **average synthesis cost** of the top-k glues (SCENT's headline axis) alongside
  glue score + diversity — the dimension SCENT is built to win.
- In-codebase ablation: rerun SCENT with cost guidance / exploitation penalty / dynamic
  library toggled off (the `include`s in SCENT's `configs/scent_base.gin`) to attribute
  any difference to the specific mechanism.

**Next steps in project.**
- Fold SCENT into the validation harness's top-k-vs-oracle-calls curve (the
  `[bengio2021gflownet]` Fig. 7 analogue) alongside RGFN, RxnFlow, FragGFN, and a
  random-acquisition baseline.
- Extend the comparison to a second protein system once a second oracle is validated
  (Objective 0 / 4).

# Re-creation

> **STATUS: STUB (START mode).** Code is implemented and locally validated by
> `py_compile` + `bash -n` + a static check that every gin `include` resolves into the
> SCENT clone, every symbol the adapter imports exists in SCENT's source, and the SMALL
> library's cost+yield data is present. The `scent` conda env build and GPU docking run
> on Balam/Trillium. Fill in Results + Answer after the run. Open items in
> `docs/REFACTOR_LOG.md`.

## Relevant Files

Root: repository root unless noted.

**Scripts / adapter** (`./validation/generators/scent/`):
- `run_scent_al.py` — entry point (runs in the `scent` env): absolutises our paths,
  `chdir`s into the SCENT clone, registers the gin classes, parses the gin config, runs
  the loop. Deliberately keeps the repo root OFF `sys.path` so SCENT's installed `rgfn`
  isn't shadowed by our repo-local `rgfn/` (uses sibling imports instead).
- `al_loop.py` — `ScentActiveLearningLoop`: the `[bengio2021gflownet]` Alg. 1 loop (fit
  proxy → train SCENT → sample batch + routes → label via bridge → accumulate → Top-K),
  with the all-NaN oracle-failure abort guard; `LabelStore` is the accumulating `D`.
- `proxy.py` — `LearnedDockingProxy`: the in-loop reward `M`. The same Bengio-2021
  atom-graph MPNN as RGFN's `glue.proxies.LearnedGlueProxy`, here subclassing **SCENT's**
  `CachedProxyBase` and importing **SCENT's** bundled `MPNNet`/`mol2graph` (identical
  symbols from its own `rgfn` fork). Fresh init, refit each round.
- `route.py` — self-contained copy of `glue.active_learning.route` (`extract_route`) for
  the `scent` env; reconstructs each molecule's synthesis route from its trajectory.

**Shared oracle bridge:**
- `./scripts/score_batch.py` — scores a batch of SMILES with a named glue oracle and
  writes the standard candidate-dataset format; route-aware (`--routes` per-round JSONL →
  joined into `routes.jsonl` on `--finalize`, `has_route=1`). Runs in the `rgfn` env.
  (Shared with the RxnFlow entrant; not modified here.)

**Install / run scaffolding:**
- `./external/setup_scent.sh` — clones `koziarskilab/SCENT` (pinned `af1fee5`) under
  `external/scent`, creates the `scent` conda env (py3.11.8, torch 2.3.0+cu118, dgl
  2.2.1+cu118), `pip install -e .`, and import-smoke-tests that SCENT's `rgfn` resolves.
- `./experiments/active_learning/scent_6td3/submit_scent_6td3.sh` — Balam SLURM submit
  (OpenCL health gate, `--exclude=balam008`, `$SCRATCH` outputs); activates `scent`, the
  bridge subprocess re-enters `rgfn`.

**Config / seed:**
- `./validation/configs/scent_6td3.gin` — gin AL config: `include`s SCENT's
  `scent_base.gin` (full cost-aware method) + `small.gin` (SMALL library + shipped
  cost/yield data), swaps the sEH proxy for `@LearnedDockingProxy`, and matches the RGFN
  run's budget/seed/β. `./validation/configs/scent_smoke.gin` — tiny mock-oracle dry run.
- `./external/scent/data/small/{fragments.txt,templates.txt,fragment_to_real_cost.json,templates_yields.csv}`
  — SCENT's SMALL building blocks (418), reaction templates (112), building-block
  **prices**, and reaction **yields** — the inputs that make Recursive Cost Guidance
  active. Shipped in the SCENT repo (git-ignored clone).
- `./experiments/active_learning/6td3/seed_6td3.csv` — the shared seed `D_0` (408
  validated docking labels), reused identically by RGFN, FragGFN, RxnFlow, and SCENT.

**Results** (git-ignored, on `$SCRATCH`):
- `/scratch/markymoo/rgfn_runs/experiments/active_learning/scent_6td3/<timestamp>/` —
  per-round `dataset_round_NNN.csv`, `top_k.csv`, and `suggestions/` (standard
  `candidates.csv` + `manifest.json` + `routes.jsonl`, `batch_metrics.csv`). [TODO — add
  the concrete run dir + SLURM job id after the run.]

## Relevant Versions

Relevant files are not yet committed.

[TODO — add commit hash after pushing.] Files to commit:
`validation/generators/scent/*`, `validation/configs/scent_{6td3,smoke}.gin`,
`experiments/active_learning/scent_6td3/*`, `external/setup_scent.sh`, and the reference
+ doc updates (`Logs/references/{README.md,references.bib}`,
`validation/generators/README.md`, `docs/REFACTOR_LOG.md`).

## Relevant Resources

**Sources.**
- `[gainski2025scent]` — SCENT: *Scalable and Cost-Efficient de Novo Template-Based
  Molecular Generation*, arXiv:2506.19865. Code: https://github.com/koziarskilab/SCENT.
- `[bengio2021gflownet]` — the active-learning loop (Alg. 1) every entrant follows.
- `[koziarski2024rgfn]` — RGFN, the generator SCENT forks and is compared against.
- `[slabicki2020cr8]` — the 6TD3 system (DDB1·CDK12–cyclinK·CR8) the oracle scores.

**Packages.**
- SCENT (its own `rgfn` fork) — the cost-aware generator; used by
  `validation/generators/scent/{proxy,route,al_loop}.py`.
- gnina + QuickVina2-GPU — the docking oracle backend, reached via
  `scripts/score_batch.py` (`Docking6TD3GpuOracle`).
- RDKit — SMILES canonicalization + descriptors, throughout.

## Method

> [TODO — confirm exact commands after the Balam run.]

1. `bash external/setup_scent.sh` — clone SCENT, build the `scent` env, install it.
2. CPU smoke: `conda run -n scent python validation/generators/scent/run_scent_al.py
   --cfg validation/configs/scent_smoke.gin --seed-csv
   experiments/active_learning/6td3/seed_6td3.csv --root-dir /tmp/scent_smoke`
   (mock oracle; validates the full loop + route logging; needs `module load cuda/11.8.0`
   for the bridge's `rgfn` env).
3. Real run: `sbatch experiments/active_learning/scent_6td3/submit_scent_6td3.sh` —
   3 rounds, GPU docking oracle, on a healthy node.

## Results

[TODO — after the run: per-round oracle mean / best / median `dvina`, fraction ≤ −2.0,
internal diversity, mean synthesis length (`num_reactions`), **mean synthesis cost** of
the top-k routes, and the side-by-side vs. the RGFN (entry `014`), FragGFN (`015`), and
RxnFlow (`016`) runs.]

### Local validation done this session (no heavy stack)
- `py_compile` of all new SCENT Python — passed.
- `bash -n` of `setup_scent.sh` + `submit_scent_6td3.sh` — passed.
- Static gin-integrity check: every `include` in `scent_{6td3,smoke}.gin` resolves into
  the SCENT clone; every symbol `proxy.py`/`route.py` import (`MPNNet`,
  `NUM_ATOMIC_NUMBERS`, `mol2graph`, `mols2batch`, `_chunks`, `CachedProxyBase`,
  `ReactionState{,EarlyTerminal,Terminal}`, `ReactionAction0`, `ReactionActionC`) is
  present in SCENT's source; `SCENT/data/small/{fragments,templates,
  fragment_to_real_cost,templates_yields}` all present (418 / 112 / 420 / 123 lines).
- `CachedProxyBase.compute_proxy_output` wraps a `List[float]` from
  `_compute_proxy_output` into `ProxyOutput(value=…)` — matches `LearnedDockingProxy`'s
  return type (same contract as SCENT's own `SehMoleculeProxy` and our `LearnedGlueProxy`).
