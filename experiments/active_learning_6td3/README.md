# Active learning with the 6TD3 oracle — worked example

This directory is the runnable **example of the active-learning loop** with our
6TD3 docking oracle. It implements Algorithm 1 of `[bengio2021gflownet]` (see
`Logs/RESEARCH_CONTEXT.md` → "How the model learns") on the validated CDK12-DDB1
system.

## The loop (one round)

1. **Fit `M`** — train the proxy on everything labelled so far, `D_{i-1}`.
2. **Train RGFN** — `Trainer.train()` against the reward `M(x)^β`.
3. **Sample `B`** — draw a query batch from the trained policy.
4. **Score `B` with `O`** — the expensive 6TD3 docking differential.
5. **Accumulate** — `D_i = D̂_i ∪ D_{i-1}`; repeat for `N` rounds.

The expensive oracle enters training **only** by retraining the proxy — never as
a direct RGFN reward.

## Components (all under `glue/`)

| Role | Paper | Class |
|---|---|---|
| Oracle `O` | expensive scorer | `glue.oracles.Docking6TD3Oracle` (DDB1 differential) |
| Proxy `M` | MPNN over atom graph (A.4) | `glue.proxies.LearnedGlueProxy` — reuses RGFN's `MPNNet` |
| Dataset `D` | `D_i = D̂_i ∪ D_{i-1}` | `glue.datasets.OracleLabeledDataset` |
| Loop | Alg. 1 | `glue.active_learning.ActiveLearningLoop` |

## Seed datasets (`D_0`)

- **`seed_6td3.csv`** — 408 real validated-docking labels (160 known glues + 248
  decoys), `label` = the **`ddb1_dvina`** differential (`Vina(Tier2) −
  Vina(Tier1)`, more-negative = better) from
  `research/preprocessing/docking_6td3/{known,decoy_cdk}_results.csv`. This is the
  signal the evidence selected — **not** the CNNaffinity differential:
    - Log 002: +78pt discrimination (known median −2.20 vs decoy −0.60);
    - Log 006: best of six candidate signals, **AUROC 0.946** (CNN ΔT2−T1 only
      0.850; absolute Vina Tier 1 just 0.69);
    - Log 007: that edge is **molecular-weight-robust** (0.946 → 0.866 after
      size-matching, while absolute scores collapse — the differential cancels
      ligand size and isolates DDB1 recruitment).

  Smaller than the paper's `|D_0| = 2000` (documented divergence).
- **`seed_mock.csv`** — 14 molecules scored by `MockGlueOracle`, for the local
  smoke test only.

Rebuild the seeds with `python build_seeds.py` (regenerates both CSVs).

## Running

Local smoke test (CPU, no gnina — uses the mock oracle):
```bash
python scripts/active_learning.py --cfg configs/glue/active_learning_mock.gin
```

Real run (Balam — needs gnina, a GPU, and the prepared 6TD3 receptors):
```bash
python scripts/active_learning.py --cfg configs/glue/active_learning_6td3.gin --seed 42
```

Per-round `dataset_round_NNN.csv` and the final `top_k.csv` are written under the
run's `active_learning/` directory.

## Tests

`glue/tests/test_oracle_discrimination.py` is the **science-validated** guard
(complementing the wiring smoke test): it asserts the oracle metric separates
known glues from decoys, reproducing Log 002's +78pt gap, and would fail if the
seed were built from the wrong differential. Runs on a laptop (no gnina):

```bash
python -m pytest glue/tests/test_oracle_discrimination.py -v -s
# or, without pytest:
python glue/tests/test_oracle_discrimination.py
```

## Status / what still needs Balam

The wiring is validated locally with the mock oracle. The **`Docking6TD3Oracle`
itself is unvalidated locally** (no gnina/GPU/receptors on a laptop) — see
`docs/REFACTOR_LOG.md` for the full list of Balam validation items, including
reconciling this oracle's docking calls with the source-of-truth
`research/preprocessing/docking_6td3/dock_cluster.py`.
