# RxnFlow / 6TD3 — synthesizable baseline head-to-head with RGFN
**Date:** 2026-06-30, ~4pm

## Question

When the generated molecules are *all* guaranteed to be synthesizable, does our RGFN
glue generator still match or beat another synthesis-aware generator (RxnFlow) at
producing good 6TD3 glue candidates, given the same scoring oracle and the same budget?

## Context & Summary

**Context.** Entry `015` ran our first baseline comparison: FragGFN, a generator that
builds molecules by gluing fragments together with **no synthesis route** — the
"non-synthesizable foil." It scored competitively with RGFN on glue quality (best
docking differential −4.86, median −2.06), which sharpens the story: if a
non-synthesizable model scores about as well, then RGFN's real selling point is that
its molecules come with a synthesis recipe. But reviewers will immediately ask the
harder question — how does RGFN compare against a generator that is *also*
synthesizable? RxnFlow (`[seo2024rxnflow]`, ICLR 2025) is exactly that peer: like RGFN
it assembles molecules from real building blocks and reaction templates, so every
molecule it proposes has a route too.

**Summary.** We run RxnFlow through the **same** active-learning loop, the **same**
6TD3 docking oracle, the **same** seed set of 408 labelled molecules, the **same**
budget (3 rounds × 32 molecules/round), the **same** inverse-temperature β=8, and the
**same** learned proxy reward as the RGFN run — so the *only* thing that changes is the
generator's internal machinery. RxnFlow runs in its own software environment and reaches
our docking oracle through the shared scoring bridge (the same cross-environment setup
FragGFN uses), now extended to record each molecule's synthesis route so RxnFlow's
output sits directly alongside RGFN's for comparison.

## Answer

[TODO — after the Balam run: state whether RGFN matches/beats RxnFlow on glue score
(docking differential), diversity, and route quality at equal oracle budget, and what
that says about RGFN's reaction-DAG action space vs. RxnFlow's template+block one.]

## Relevance to our Publication

This is the **synthesizable-vs-synthesizable** arm of the baseline comparison reviewers
will demand (`docs/RESEARCH_CONTEXT.md`, Objectives 4–5). FragGFN (entry `015`) shows
RGFN beats a non-synthesizable generator on the synthesizability axis; RxnFlow controls
for synthesizability so we can isolate what RGFN's specific action space buys. A clean
result here — RGFN competitive with a state-of-the-art synthesis-aware generator on the
same oracle and budget — is the kind of strong baseline a methods venue (Digital
Discovery / JCIM) expects before believing the headline claim.

## Next Experiments

**Refining for publication.**
- Run ≥3 seeds for RGFN, RxnFlow, and FragGFN so the head-to-head carries error bars.
- A matched-oracle RGFN **GPU** rerun on a healthy node (entry `014`'s GPU run died on a
  wedged OpenCL node), so all three entrants share the identical GPU oracle.
- Compare *route quality* (synthesis length, route diversity), not just glue score —
  the dimension where two synthesizable generators most plausibly differ.

**Next steps in project.**
- Fold RxnFlow into the validation harness's top-k-vs-oracle-calls curve (the
  `[bengio2021gflownet]` Fig. 7 analogue) alongside RGFN and a random-acquisition
  baseline.
- Extend the three-way comparison to a second protein system once a second oracle is
  validated (Objective 0 / 4).

# Re-creation

> **STATUS: STUB (START mode).** Code is implemented and locally validated by
> `py_compile` + a bridge round-trip; the RxnFlow heavy stack and GPU docking run on
> Balam/Trillium. Fill in Results + Answer after the run. RxnFlow upstream-API
> uncertainties are tracked in `docs/REFACTOR_LOG.md`.

## Relevant Files

Root: repository root unless noted.

**Scripts / adapter** (`./validation/generators/rxnflow/`):
- `run_rxnflow_al.py` — entry point: loads the YAML, builds the proxy + RxnFlow trainer
  + loop, runs it (runs in the `rxnflow` conda env).
- `al_loop.py` — `RxnFlowActiveLearningLoop`: the `[bengio2021gflownet]` Alg.1 loop
  (fit proxy → train RxnFlow → sample batch + routes → label via bridge → accumulate →
  Top-K), with the all-NaN oracle-failure abort guard; `extract_route` reconstructs each
  molecule's synthesis route from its trajectory.
- `task.py` — `RxnFlowTask` (reward = our proxy `M`) + `RxnFlowGlueTrainer` (RxnFlow's
  synthesis GFlowNet with our proxy, `num_workers=0`, constant β).
- `proxy.py` — re-exports FragGFN's `AtomMPNNProxy` so the proxy `M` is identical across
  all entrants.

**Shared oracle bridge:**
- `./scripts/score_batch.py` — scores a batch of SMILES with a named glue oracle and
  writes the standard candidate-dataset format; extended here to be **route-aware**
  (`--routes` per-round JSONL → joined into `routes.jsonl` on `--finalize`,
  `has_route=1`). Runs in the `rgfn` env.

**Install / run scaffolding:**
- `./external/setup_rxnflow.sh` — builds the `rxnflow` conda env (py3.12, torch
  2.5.1+cu121), installs RxnFlow + its bundled gflownet, prepares the env directory
  (building blocks + reaction templates).
- `./experiments/active_learning/rxnflow_6td3/submit_rxnflow_6td3.sh` — Balam SLURM
  submit (OpenCL health gate, `--exclude=balam008`, `$SCRATCH` outputs).
- `./experiments/active_learning/rxnflow_6td3/README.md` — run card + the RGFN/RxnFlow
  comparison table.

**Config / seed:**
- `./validation/configs/rxnflow_6td3.yaml` — budget/oracle/proxy matched field-for-field
  to `configs/glue/active_learning_6td3_gpu.gin`; `rxnflow:` block carries the generator
  knobs. `./validation/configs/rxnflow_smoke.yaml` — tiny CPU mock-oracle dry run.
- `./experiments/active_learning/6td3/seed_6td3.csv` — the shared seed `D_0` (408
  validated docking labels), reused identically by RGFN, FragGFN, and RxnFlow.

**Results** (git-ignored, on `$SCRATCH`):
- `/scratch/markymoo/rgfn_runs/experiments/active_learning/rxnflow_6td3/<timestamp>/` —
  per-round `dataset_round_NNN.csv`, `top_k.csv`, and `suggestions/` (standard
  `candidates.csv` + `manifest.json` + `routes.jsonl`, `batch_metrics.csv`). [TODO — add
  the concrete run dir + SLURM job id after the run.]

## Relevant Versions

Relevant files are not yet committed.

[TODO — add commit hash after pushing.] Files to commit:
`validation/generators/rxnflow/*`, `validation/configs/rxnflow_{6td3,smoke}.yaml`,
`experiments/active_learning/rxnflow_6td3/*`, `external/setup_rxnflow.sh`, the
route-aware `scripts/score_batch.py`, the reference + doc updates.

## Relevant Resources

**Sources.**
- `[seo2024rxnflow]` — RxnFlow: *Generative Flows on Synthetic Pathway for Drug Design*,
  arXiv:2410.04542 (ICLR 2025). Code: https://github.com/SeonghwanSeo/RxnFlow.
- `[bengio2021gflownet]` — the active-learning loop (Alg. 1) every entrant follows.
- `[koziarski2024rgfn]` — RGFN, the generator RxnFlow is being compared against.
- `[slabicki2020cr8]` — the 6TD3 system (DDB1·CDK12–cyclinK·CR8) the oracle scores.

**Packages.**
- RxnFlow (+ bundled Recursion `gflownet` v0.2.0) — the generator; used by
  `validation/generators/rxnflow/task.py`, `al_loop.py`.
- gnina + QuickVina2-GPU — the docking oracle backend, reached via
  `scripts/score_batch.py` (`Docking6TD3GpuOracle`).
- RDKit — SMILES canonicalization + descriptors, throughout.

## Method

> [TODO — confirm exact commands after the Balam run.]

1. `bash external/setup_rxnflow.sh` — build the `rxnflow` env, install RxnFlow, prepare
   the env directory (building blocks + 71 reaction templates).
2. CPU smoke: `conda run -n rxnflow python validation/generators/rxnflow/run_rxnflow_al.py
   --cfg validation/configs/rxnflow_smoke.yaml --seed-csv
   experiments/active_learning/6td3/seed_6td3.csv --device cpu --root-dir /tmp/rxnflow_smoke`
   (mock oracle; validates the full loop + route logging).
3. Real run: `sbatch experiments/active_learning/rxnflow_6td3/submit_rxnflow_6td3.sh` —
   3 rounds, GPU docking oracle, on a healthy node.

## Results

[TODO — after the run: per-round oracle mean / best / median `dvina`, fraction ≤ −2.0,
internal diversity, mean synthesis length (`num_reactions`), and the side-by-side vs.
the RGFN (entry `014`) and FragGFN (entry `015`) runs.]

### Local validation done this session (no heavy stack)
- `py_compile` of all new RxnFlow Python + the edited `scripts/score_batch.py` — passed.
- `bash -n` of `setup_rxnflow.sh` + `submit_rxnflow_6td3.sh` — passed.
- `rxnflow_6td3.yaml` `loop`/`oracle`/`proxy` blocks match `fraggfn_6td3.yaml` (budget +
  oracle identical) — confirmed by a parity check.
- Route-aware bridge round-trip (Trillium login, mock oracle): with `--routes`, the
  standard dataset emits `has_route=1`, `num_reactions` populated, `routes.jsonl` with one
  entry per candidate, `manifest.has_routes=True`, and `validate_candidate_dataset` →
  conformant; **without** `--routes` (FragGFN path) → `has_route=0`, no `routes.jsonl`,
  also conformant (no regression).
