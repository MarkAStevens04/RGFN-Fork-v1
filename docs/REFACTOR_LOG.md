# Refactor log — 2026-06-24 repo restructure

A running record of the restructure that introduced the `glue/` package and the
upstream/ours boundary. **If something is broken, start here**: this documents
what changed, why, what was verified, and what could not be verified (Balam was
down; work was done on a Mac laptop, so no GPU/docking/training runs were possible).

Owner: Mark + Claude. Goal context: `docs/RESEARCH_CONTEXT.md`. Layout: `docs/ARCHITECTURE.md`.

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
- `docs/RESEARCH_CONTEXT.md` — "Where things live" updated + restructure note.
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

---

# 2026-06-24 — Active-learning loop (branch `active-learning-loop`)

First implementation of the multi-round active-learning loop from
`[bengio2021gflownet]` Alg. 1 (transcribed in `docs/RESEARCH_CONTEXT.md`), wired
to our 6TD3 docking oracle and kept oracle-agnostic for future oracles (MD, etc.).

## Grounding decision (important — re-checked against the papers)

An earlier draft proposed a hand-rolled **MLP** proxy. That was wrong: the proxy
in `[bengio2021gflownet]` A.4 is an **MPNN over the RDKit atom graph** (NNConv +
GRU ×12 → Set2Set → 3-layer MLP, dim 64, LeakyReLU). RGFN already ships *exactly*
that network as `MPNNet` (the class behind `SehMoleculeProxy`, which loads
Bengio's pretrained sEH weights). Key point that resolves the "RGFN isn't
atom-by-atom" worry: RGFN's **reaction/fragment graph drives only the policy/flow
predictor**; the **proxy scores a finished molecule from its atom graph**
(`mol2graph(SMILES)`), identically in both `[bengio2021gflownet]` and RGFN's own
sEH proxy. So we **reuse `MPNNet` + `mol2graph`/`mols2batch` verbatim** rather
than reinventing an architecture.

**We reuse the architecture, NOT the weights.** `LearnedGlueProxy` never calls
`seh_proxy.load_original_model` / `SEHProxyWrapper` — it constructs a fresh
random-init `MPNNet` and trains it from scratch on `D_0` (and refits each round),
exactly as `[bengio2021gflownet]` does ("initializes an MPNN proxy"). Loading the
pretrained sEH weights would be wrong anyway: that proxy predicts sEH affinity,
not our DDB1 differential. (Verified at runtime: our init weights ≠ the pretrained
weights, and differ across seeds.)

## Added (all under `glue/`, zero edits to `rgfn/`)

- `glue/oracles/base.py` — `GlueOracle` ABC (`score(smiles) -> list[float]`): the
  modular seam so the loop is oracle-agnostic.
- `glue/oracles/mock_oracle.py` — `MockGlueOracle`: cheap CPU oracle (QED × MW
  band) so the whole loop runs on a laptop without gnina/GPU. Test fixture, not
  science.
- `glue/oracles/docking_6td3_oracle.py` — `Docking6TD3Oracle`: real two-tier
  gnina docking returning the DDB1 neosubstrate differential (`ddb1_dcnnaff`).
  **Mirrors `research/preprocessing/docking_6td3/dock_cluster.py`** (identical
  gnina flags, best-pose-by-CNNscore, Tier-1 `--score_only`, same differential);
  differs only in orchestration (one in-process batch vs sharded multi-GPU job).
- `glue/proxies/learned_proxy.py` — `LearnedGlueProxy`: trainable wrapper around
  RGFN's `MPNNet` with a `.fit()` (the in-loop reward `M`).
- `glue/datasets/oracle_labeled.py` — `OracleLabeledDataset`: the accumulating
  `D_i = D̂_i ∪ D_{i-1}` store (canonical-SMILES keyed, seed-CSV load, Top-K).
- `glue/active_learning/{__init__,loop}.py` — `ActiveLearningLoop`: the outer
  Alg. 1 orchestrator. New subpackage (registered in `glue/registry.py`).
- `scripts/active_learning.py` — entry point (imports `glue` **and** `Trainer`,
  parses an `configs/glue/` config, builds the loop, runs it).
- `configs/glue/active_learning_mock.gin` (local smoke) and
  `configs/glue/active_learning_6td3.gin` (Balam real run).
- `experiments/active_learning_6td3/` — worked example: `README.md`,
  `build_seeds.py`, `seed_6td3.csv` (408 real validated-docking labels = 160
  known + 248 decoys), `seed_mock.csv` (14 mols for the smoke test).
- Registrations: `glue/registry.py` (+`active_learning`), and the `oracles` /
  `proxies` / `datasets` subpackage `__init__.py` exports.

## How the loop works (and the one invariant)

Each round: **fit `M` on `D_{i-1}`** → `clear_cache()` → **`Trainer.train()`**
against `M(x)^β` → **sample query batch** from the trained forward policy →
**score with `O`** → **`D_i = D̂_i ∪ D_{i-1}`**. The trainer's reward and the loop
share **one** `LearnedGlueProxy` singleton, so refitting updates the reward RGFN
trains on. Invariant preserved: oracle labels enter training **only** by
retraining `M`, never as a direct RGFN reward. Critically, the in-loop reward is
the *cheap MPNN proxy*, so GFN training uses the `reaction` env (not
`reaction_docking`) — **docking never runs in the inner loop**, only on the
per-round query batch.

## Correction (2026-06-25): oracle uses the VINA differential, lower-is-better

The first draft of `Docking6TD3Oracle` computed the **CNNaffinity** differential
(`ddb1_dcnnaff`) with `higher_is_better=True`. That was wrong on both counts.
Checking log 002 + `compare_systems.py` + the result CSVs: the validated +78pt
discrimination is on **`ddb1_dvina` = `Vina(Tier2) − Vina(Tier1)`**, where *more
negative = better glue* (known median −2.20 vs decoy −0.60; 85.6% vs 7.3% below
−1.5). The CNNaffinity differential does **not** discriminate (known +0.23 vs
decoy +0.04, decoy range up to +1.80). CNNscore is still used — but only to pick
the pose (`max cnnsc`), matching `dock_cluster.py`. Fixed (user chose: store raw
`ddb1_dvina`, lower-is-better):
- `docking_6td3_oracle.py` now returns `vina_t2 − vina_t1`; `higher_is_better=False`.
- `LearnedGlueProxy.higher_is_better` is now a constructor arg (set `False` in
  `active_learning_6td3.gin`); dropped the positive output clip (the base config's
  exponential boosting makes the reward positive for any prediction) in favour of
  a symmetric `±clip` sanity bound; invalid molecules get a sign-aware worst-case.
- `ActiveLearningLoop.run()` now asserts `proxy.higher_is_better ==
  oracle.higher_is_better` to catch this class of bug.
- `seed_6td3.csv` / `build_seeds.py` rebuilt from the `ddb1_dvina` column.
- **Regression guard added:** `glue/tests/test_oracle_discrimination.py` — the
  "science validated" counterpart to the wiring smoke test. It checks the metric
  feeding the loop actually separates known glues from decoys (reproduces Log
  002: known 85.6% vs decoy 7.3% strong-bonus, +78pts), and a contrast check
  shows the Vina differential discriminates (+78pts) while CNNaffinity does not
  (+0pts). Verified non-vacuous: it FAILS if the seed is built from
  `ddb1_dcnnaff`. Runs on a laptop from committed data (no gnina); also runnable
  standalone (`python glue/tests/test_oracle_discrimination.py`).

**Process lesson (root cause of both this and the MLP/MPNN error):** each
consequential scientific choice was filled by inference instead of reading the
file that recorded the decision. The fix is to read all three where they exist —
the paper section, the upstream `rgfn/` code, AND our own `Logs/` experiment log
+ analysis script (`compare_systems.py`) — and to treat "smoke test passed" as
"wiring works", never "science is correct". The discrimination test above
operationalises the latter.

## Update (2026-06-25, cont.): Logs 006 + 007 confirm the Vina ΔT2−T1 choice

After the metric correction above, experiment logs **006** (six-way signal
ablation) and **007** (molecular-weight control) were written and independently
confirm `ddb1_dvina` (Vina Tier2−Tier1, lower-is-better) is the right oracle
signal — so **no reward-signal code change was needed**; the implementation was
already aligned. What changed is documentation + the guard:
- 006: ranked all six candidate signals on the entry-002 poses; **Vina ΔT2−T1
  wins, AUROC 0.946** (Cohen's d 2.38); its Youden cut −1.58 matches our −1.5.
  CNNaffinity differential is 0.850; absolute Vina Tier 1 is 0.69.
- 007: after matching glues/decoys on molecular weight, Vina ΔT2−T1 stays top
  (0.946 → 0.866) while absolute Vina Tier 1 falls below chance (0.38) — the
  differential isn't a size artifact.
- Citations to 006/007 added to: `docking_6td3_oracle.py` docstring,
  `test_oracle_discrimination.py`, the example README, and the gin config.
- Fixed two stale comments that still said `ddb1_dcnnaff` (config + build_seeds).
- Strengthened the discrimination test with an **AUROC check** (asserts > 0.90;
  reproduces 006's 0.946) and an AUROC-based contrast (Vina 0.946 vs CNN 0.850).
  Caught a sign slip while doing so: CNNaffinity is higher-is-better (pK), Vina is
  lower-is-better (energy) — using one orientation for both flips AUROC to
  1−AUROC. Now oriented per metric; both reproduce 006 exactly.

## Deliberate divergences from the publications (validate/revisit on Balam)

1. **Proxy target** — predicts our docking neosubstrate *differential*
   (`ddb1_dvina`, lower-is-better), not AutoDock sEH affinity. The project's
   novel oracle. Signal choice justified by Logs 002/006/007.
2. **Trainable proxy** — `SehMoleculeProxy` is inference-only with frozen
   weights; we add `.fit()`. This is what Alg. 1 requires; the shipped proxy just
   doesn't expose it.
3. **Random init each round** — we rebuild `MPNNet` weights every `fit()` rather
   than annealing from the previous round (avoids compounding drift; revisit).
4. **Label scaling** — we standardise labels at `fit` time and clip predictions
   to `[min_value, max_value]` at inference (mirrors the paper's renormalisation
   to a positive reward and `SehMoleculeProxy`'s `clip(1e-4, 100)`).
5. **Scale** — paper uses `|D_0|=2000` and 200 mols/round; our `D_0` is 408
   (all the validated labels we have). `query_batch_size=200` matches the paper.
6. **Validation split** — paper uses a fixed 3000-mol val set for early stopping;
   we hold out a fraction (`val_fraction`) — matters for small `D`.
7. **Replay buffer reset** — we clear it each round (its priorities go stale once
   `M` is refit). Best-effort, guarded against upstream layout changes.

## Verified locally (Mac, `rgfn` conda env — has torch/rdkit/gin/torch_geometric)

- `py_compile` of all new modules; `import glue` registers every new component.
- Both AL configs parse (after the `Trainer`-import fix below).
- `LearnedGlueProxy` fits on toy data, predicts positive rewards, handles invalid
  SMILES and early-terminal states.
- **Full mock loop end-to-end** via `scripts/active_learning.py` (2 rounds, CPU,
  `WANDB_MODE=offline`): seed `|D_0|=13` → fit → `Trainer.train()` → sample 16 →
  score → accumulate → `|D|=29` → round 2 → `|D|=45` → `top_k.csv` written. EXIT 0.
  RGFN-generated (reaction-assembled) molecules out-scored seeds under the mock
  oracle and rose to the top of the Top-K. (Smoke artifacts deleted, not committed.)
- `Docking6TD3Oracle` imports without gnina and raises a clear `FileNotFoundError`
  when receptors are absent (laptop) — Balam-only by design.

### Two bugs found and fixed during local validation
- **`scripts/active_learning.py` didn't register `Trainer`** → gin "No
  configurable matching 'Trainer'". Upstream `train.py` imports it explicitly;
  added `from rgfn.trainer.trainer import Trainer`.
- **`LearnedGlueProxy._compute_proxy_output` assumed `.molecule`** → crashed on
  `ReactionStateEarlyTerminal` after the loop's post-`fit` `clear_cache()` wiped
  the pre-seeded entry. Now guards early-terminal states → `min_value`.

## NOT verified — needs Balam (gnina + GPU + prepared receptors)

- `Docking6TD3Oracle.score()` against real `6TD3_tier{1,2}.pdbqt` + `crystal_RC8.pdb`
  (git-ignored, absent on laptop). **Reconcile/cross-validate it against
  `dock_cluster.py`** — the two duplicate the docking logic; `dock_cluster.py` is
  the source of truth. Consider unifying them into one shared `dock_batch_6td3()`.
- `scripts/active_learning.py --cfg configs/glue/active_learning_6td3.gin` full run.
- Hyperparameters: `β`, per-round `Trainer.n_iterations`, proxy `max_epochs` /
  `patience`, `query_batch_size` — all first-guess defaults, untuned.
- Reward-scale sanity: with `β=8` and standardised proxy outputs, confirm the
  exponential reward boosting doesn't explode/vanish on the real differential.
- The headline plots reviewers will want: Top-K-vs-oracle-calls curve and a
  random-acquisition baseline (`[bengio2021gflownet]` Fig. 7 analog).

---

# 2026-06-25 (cont.): documentation consolidation

Reduced duplication across the project docs (goal: one home per fact, fewer
chances for versions to drift):
- **Moved `Logs/RESEARCH_CONTEXT.md` → `docs/RESEARCH_CONTEXT.md`** (`git mv`);
  updated all ~14 referencing files (code docstrings, READMEs, CLAUDE.md, configs).
- **Slimmed `Logs/README.md`** to a short intro + pointers. Its duplicated
  "The systems" and "Common methodology" sections were deleted (already covered by
  RESEARCH_CONTEXT's systems table + terminology; 5HXB anchoring specifics remain
  in log 001). Its experiment **Index** moved into `RESEARCH_CONTEXT.md` (links now
  `../Logs/`). Its "Where results live" + "Datasets" moved into `ARCHITECTURE.md`.
- **`ARCHITECTURE.md`** is now the single source for repo layout + data/result
  locations (new "Data, datasets & results" section); RESEARCH_CONTEXT's
  "Where things live" trimmed to a pointer to it.
- Net: systems/terminology live once (RESEARCH_CONTEXT); locations live once
  (ARCHITECTURE); the experiment index lives once (RESEARCH_CONTEXT).
- **Path-corrected the historical logs** (001–005): their `pre-processing/` path
  references are now `research/preprocessing/` (real, clickable paths), so the
  "read these as research/preprocessing/" translation note was **removed** from
  both `RESEARCH_CONTEXT.md` and `CLAUDE.md`. Only verbatim *commit messages* in
  the logs' "Relevant Versions" blocks still say "pre-processing" — left intact,
  since those quote real historical commits.

---

# 2026-06-26: validation/benchmarking scaffold (second axis)

Introduced a **second organizing axis** for our own code — **production pipeline
vs. validation** — on top of the existing upstream/ours boundary. Goal: get the
structure ready for the comparative studies (multiple generators, PMO, our own
benchmarks, VAE-BO, Boltz-2) **before** Balam is back, keeping the validation
layer as separate as possible from the training pipeline. **Scaffolding only — no
implementation code imported yet** (per the user's request).

### Decisions (chosen by the user)
- **One umbrella folder, not two.** Initially scaffolded as two top-level dirs
  (`validation/` for entrants/scorers + `benchmarks/` for suites/harness/results),
  but the split was thin and tightly coupled (the harness only exists to run the
  generators) and produced two `configs/` dirs. Consolidated into a single
  `validation/` holding everything; the "benchmark" idea survives as
  `validation/suites/`. (`benchmarks/` was created and then removed in the same
  session — it was never committed.)
- **Validation oracles separate from in-loop oracles.** Boltz-2 / co-folding /
  high-fidelity checks go in `validation/oracles/` (never in-loop). In-loop
  oracles stay in `glue/oracles/`. Short-MD as a *complementary in-loop* oracle
  (CRBN ceiling, Objective 3) would still go in `glue/`, not here.
- **Baselines = thin adapters + external installs:** each baseline is a thin
  adapter in `validation/generators/<name>/`; heavy upstream code installs via
  `external/setup_<name>.sh` (not vendored).

### The rule that enforces the split
> `validation/` may import from `glue/` and `rgfn/`; the production pipeline
> (`glue/`, `scripts/train.py`, `configs/glue/`) must **never** import from
> `validation/`. The dependency arrow points one way.

### Created (directories + READMEs / `.gitkeep` only)
- `validation/` + `README.md` (boundary rule, full single-folder layout,
  what-goes-where guide, and the "why reviewers care" rationale).
  - `validation/generators/{rgfn,synflownet,fraggfn,vae_bo}/` + `generators/README.md`
    (entrant table, planned common `Generator` interface).
  - `validation/oracles/boltz2/` + `oracles/README.md`
    (in-loop vs. validation-oracle contrast table).
  - `validation/suites/{pmo,glue_suite}/` + `suites/README.md`.
  - `validation/harness/README.md`, `validation/configs/README.md`,
    `validation/results/` (`.gitkeep`).
- `external/setup_{synflownet,fraggfn,vae_bo}.sh` — **placeholder stubs** (valid
  bash, `exit 1` with a TODO; mirror `setup_reinvent.sh` when implemented).

### Docs updated
- `CLAUDE.md`: ownership table now tags `glue/` as *production pipeline* and adds
  a single `validation/` row as *validation*; new "second axis" subsection with
  the dependency rule.
- `docs/ARCHITECTURE.md`: layout tree expanded with the `validation/` subtree;
  new "Two axes" section + "Component flow (validation)" diagram.

### Verified
- `bash -n` on all three new `external/setup_*.sh` stubs (pass).
- Directory tree + placeholder files created as listed; `benchmarks/` removed.

### NOT done / deferred (by design — structure only)
- No `Generator` base class, adapters, harness, suite, or oracle code written.
- `__init__.py` files intentionally omitted until real code lands (each Python
  subtree gets one then, so the harness can import it).
- Config format for `validation/configs/` (gin vs. YAML) not decided yet.
- The setup stubs do not actually install anything.

---

## 2026-06-26 — sEH GPU-docking oracle for the active-learning loop

Added a fast docking oracle so the multi-round AL loop can actually finish (the
CPU-bound 6TD3 gnina oracle was killed by the login-node CPU cap; exp `009`/`010`).

### Created (ours)
- `glue/oracles/docking_seh_oracle.py` — `DockingSEHOracle(GlueOracle)`. Real sEH
  docking via QuickVina2-GPU-2.1. **Composes** upstream `DockingMoleculeProxy` and
  calls its `dock_batch_qv2gpu` directly on SMILES — the docking is byte-for-byte
  the upstream path (no duplication, unlike the 6TD3 oracle). `higher_is_better=
  False` (Vina energy); returns `nan` on failure per the `GlueOracle` contract;
  the heavy proxy is built lazily so the module imports on a laptop.
- `configs/glue/active_learning_seh.gin` — AL loop wiring (mirrors
  `active_learning_6td3.gin`): `LearnedGlueProxy` (MPNN) as `M`, `DockingSEHOracle`
  as `O`, `reaction` env (docking stays out of the inner loop).
- `research/active_learning_seh/` — sEH-specific run tooling (sibling of
  `research/preprocessing/`; bundles scripts + submit + README like the
  `pose_selection_ablation/` precedent): `make_seh_seed.py` (builds `D_0` by
  sampling the untrained RGFN policy and docking → `seed_seh.csv`; reports
  s/mol), `validate_seh_oracle.py` (quick live smoke), `submit_al_seh.sh` (Balam
  compute-node run; cuda module + boost libs; regenerates a 300-mol seed on
  `$SCRATCH`), `README.md`. The generic `scripts/active_learning.py` entry point
  stays in `scripts/` (shared with 6TD3).
- `experiments/active_learning_seh/seed_seh.csv` — committed starter seed.

### Edited (ours)
- `glue/oracles/__init__.py` — export/register `DockingSEHOracle`.
- `docs/RESEARCH_CONTEXT.md` — index row 010 + Objective 1 note.

### Verified (on balam-login01 A100, `rgfn` env)
- `py_compile`; import via `glue.registry`; gin parse of the config + class
  resolves by name.
- **Live GPU dock**: aspirin −6.30 (matches the recorded sEH validation),
  ibuprofen −6.90, caffeine −6.50; invalid SMILES → `nan`; ~3.7 s/mol.
- **Seed gen**: 36/40 RGFN-sampled molecules docked; dataset loads |D_0|=36;
  proxy/oracle sign check passes (both `higher_is_better=False`).
- `bash -n` on `submit_al_seh.sh`.

### NOT done / deferred
- The **full multi-round loop** has not been run yet (needs a compute node;
  `submit_al_seh.sh` is ready). No top-k-vs-oracle-calls curve / random baseline.
- Oracle outputs not yet cross-checked against an actual
  `configs/rgfn_seh_docking.gin` run on the same molecules (numbers should match).

---

## 2026-06-26 — per-phase timing for the active-learning loop

Experiment 009 could not say where the loop spent its time (only eyeballed
per-step rates; the run died at the oracle step with no number for it). Added
intrinsic, always-on phase timing.

### Added (ours)
- `glue/active_learning/timing.py` — `PhaseTimer`: a context-manager that times
  the four per-round phases (`fit_proxy`, `train_gfn`, `sample_batch`,
  `oracle_score`). On each phase exit it prints (`[AL]`), logs to the trainer
  logger under the `timing` prefix (→ wandb), and **appends** to
  `<run_dir>/active_learning/phase_timings.csv`. The append-before-next-phase
  ordering means a mid-run crash (the SIGXCPU that ended exp 009) still leaves a
  record of every phase that finished. `report_total()` ranks phases by share of
  total wall-clock — the "where to save time" answer.

### Edited (ours)
- `glue/active_learning/loop.py` — wrapped the four phases in `timer.phase(...)`,
  added `report_round`/`report_total`. No behavioural change to the algorithm;
  recorded via `finally`, so timing is robust to a raising phase.

### Verified (laptop, no heavy deps)
- `py_compile` on both files. Standalone smoke test of `PhaseTimer`: `_fmt`
  formatting, CSV append, crash-phase recorded via `finally`, share-ranked
  summary.

### NOT done / deferred
- Not yet run inside a real loop on Balam (needs the `rgfn` env / GPU). The four
  phase labels and their wiring are unverified against a live `Trainer.train()`.
- Overhead is negligible (one `perf_counter` + a CSV append per phase, 4/round),
  but unconfirmed under the real loop.

---

## 2026-06-26 — directory reorg: experiments/ (per-run), data/ (inputs), flat scripts/

Reorganized the top-level layout for clarity (one home per concept). No `glue/`
or `rgfn/` logic changed; this is moves + path-rewiring + docs.

### Moved / merged
- **`research/` dissolved into `experiments/`**, now **grouped by type, one dir
  per run**: `experiments/{active_learning,oracle_validation,ablations}/<run>/`.
  - `research/active_learning_{seh,6td3}/` → `experiments/active_learning/{seh,6td3}/`
    (merged with the matching `experiments/active_learning_*` seeds/outputs).
  - `research/preprocessing/{docking_6td3, docking_gnina→docking_crbn, clean*.py,
    compare_systems.py}` → `experiments/oracle_validation/…`.
  - `research/preprocessing/{pose_selection_ablation→pose_selection,
    full_comparison→sixway, full_comparison_mw→mw}` → `experiments/ablations/…`.
  - Old flat AL output dirs (`active_learning_mock`, `al_probe→probe`,
    `active_learning_6td3_inner→6td3_inner`) → under `experiments/active_learning/`.
  - `experiments/oracle_validation/` chosen over `validation/` to avoid clashing
    with the top-level `validation/` benchmarking layer.
- **`models/` + `research/preprocessing/test-data/` folded into `data/`** (the
  single inputs dir): `data/models/`, `data/validation-molecules/` (renamed from
  "test-data" — they're validation *inputs*, not fixtures). `data/` keeps its name
  because upstream hardcodes `data/chemistry.xlsx` + `data/targets/`.
- **`scripts/hpc/` removed**; the one generic `submit.sh` → `scripts/submit.sh`.
  Run-specific submit scripts already live with their experiment.

### Rewired
- All `git mv`/`mv` done so git detected pure **renames** (history preserved).
- Code paths: `glue/oracles/docking_6td3_oracle.py` `_DOCK_DIR`,
  `glue/tests/test_oracle_discrimination.py` (`SEED_CSV`/`DOCK_DIR`), the moved
  docking/analysis scripts (`../test-data`→`data/validation-molecules`,
  `docking_gnina`→`docking_crbn`, `models/`→`data/models/`), `clean*.py` in/out dirs.
- Configs: 5 AL `seed_csv` paths repointed to `experiments/active_learning/<run>/`.
- `scripts/active_learning.py`: run_name now groups outputs under
  `experiments/active_learning/<run>/<ts>/` so outputs co-locate with each run's
  committed material.
- `.gitignore`: replaced the blanket `experiments/*` ignore with a **pattern-based**
  scheme — ignore timestamped run-output dirs (`experiments/**/YYYY-MM-DD_*/`),
  force-track `seed_*.csv` / `*_results.csv`; updated `models/`→`data/models/`,
  `test-data`→`data/validation-molecules`, cluster_out/violins paths.

### Verified (balam-login01, `rgfn` env)
- `git status` shows clean renames; no committed file under a timestamped run dir;
  `git check-ignore` confirms run outputs ignored / seeds+results+code tracked.
- `py_compile` of all moved scripts; `import glue`; all 5 AL configs parse **and
  their seed paths resolve**; `pytest glue/tests/test_oracle_discrimination.py` (2
  passed); `bash -n` on all 6 submit scripts.
- READMEs updated/added: `experiments/README.md` (structure guide), per-group/run
  READMEs, `data/README.md` + `data/{models,validation-molecules}/README.md`,
  `scripts/README.md`; plus `CLAUDE.md`, `docs/ARCHITECTURE.md`, README/context.

### NOT done / deferred
- Balam-side `$SCRATCH` paths (e.g. where the user keeps `.cif`/receptors) may need
  the same `models/`→`data/models/` move on Balam; the repo-relative refs are fixed.
- The 6TD3 docking-oracle pdbqt receptors live (git-ignored) at the new
  `experiments/oracle_validation/docking_6td3/` path on this login node; confirm
  they're present at that path on Balam compute nodes before the next 6TD3 run.

---

## 2026-06-27 — `top_k` deliverable sort fix (`glue/datasets/oracle_labeled.py`)

`OracleLabeledDataset.top_k()` sorted `reverse=True` (largest label first), but
oracle labels are *more-negative = better* (`GlueOracle.higher_is_better = False`,
for both the 6TD3 `ddb1_dvina` differential and the sEH Vina energy), so the
"Top-K deliverable" returned the **worst** molecules. Found while reviewing the
first multi-round 6TD3 run (Logs/011, job 69445), whose `top_k.csv` listed +0.68
at rank 1. Fixed to sort ascending. Scope: deliverable file only — training's
reward/proxy path is separate and was correctly oriented (the run's generated
molecules dock *well*). Regenerated `top_k.csv` for that run from
`dataset_round_003.csv` (best now −4.92).

**Verified:** `py_compile` of the module. **Not verified here:** no unit test
covers `top_k`, and gin/rdkit/conda aren't on this node, so the loop wasn't
re-run; the regenerated `top_k.csv` was produced by replicating the (trivial)
ascending-sort logic inline. A `test_top_k_orientation` unit test would be cheap
insurance.

---

## 2026-06-28 — docking oracle sub-step timing (`OracleStepTimer`)

Added per-substep wall-clock timing inside the expensive docking oracle so the
`oracle_score` phase (35% of loop wall-clock, Logs/011) is no longer a black box.

- **New** `glue/oracles/step_timing.py::OracleStepTimer` — disabled-by-default
  (silent no-op) companion to `glue/active_learning/timing.PhaseTimer`. Appends
  `(round, step, seconds, n_molecules)` rows on each sub-step's completion
  (crash-safe), prints a `[dock]` line live.
- `glue/oracles/docking_6td3_oracle.py` — new `enable_step_timing(csv_path)`;
  `score()`'s four sub-steps wrapped: `embed` / `tier2_dock` / `pose_select` /
  `tier1_rescore`. **Docking logic unchanged** (like-for-like with Logs/011).
- `glue/active_learning/loop.py` — opt-in hook: if the oracle has
  `enable_step_timing`, the loop points it at
  `<run>/active_learning/docking_timings.csv`. Loop stays oracle-agnostic; mock /
  sEH oracles (no hook) are unaffected (`getattr` + `callable` guard).

**Verified:** `py_compile` (3 files); in the `rgfn` env `import glue` resolves the
oracle, disabled timer is a confirmed no-op, enabled timer writes the expected
4-column CSV with recoverable s/mol; `bash -n` on the new submit script.
**Not verified here:** a real multi-round run with the timer live — that is
**Logs/012** (job 69450, queued this session). The sEH oracle does **not** yet
implement `enable_step_timing`; add it there when that loop is next run.
