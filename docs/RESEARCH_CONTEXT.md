# RGFN Molecular Glue Research — Project Context

This file exists so that AI agents working on this project can orient themselves quickly. Read it before writing any experiment log. Update it as the project evolves.

> **Ground your claims in the real work.** The method (what RGFN/GFlowNets are) and the
> biology (why these systems and metrics) come from a short shelf of papers in
> `Logs/references/` — read its `README.md` before asserting how the model or the oracle
> works. Cite by key (e.g. `[koziarski2024rgfn]`); don't reconstruct a citation from memory.

---

## What this project is

We're building an end-to-end computational pipeline for generating novel **molecular glue degraders** — small molecules that force two proteins to interact, triggering degradation of a disease-causing protein. The generative model is **RGFN** (`[koziarski2024rgfn]`), a **GFlowNet** (`[bengio2021gflownet]`). A GFlowNet is *not* a reward-maximizing RL agent: it learns to **sample** candidates with probability **proportional to their reward**, so it returns many diverse high-scoring molecules instead of collapsing onto a single optimum. RGFN specifically builds each molecule by composing **chemical reactions over a building-block library**, so everything it proposes is synthesizable by construction. RGFN needs a *reward signal* to know whether a molecule is a good glue candidate — we're building and validating that signal (the **oracle**).

The oracle uses **molecular docking**: simulate how well a candidate molecule sits in the complex and — crucially — whether its *arm* helps recruit the second protein the glue must bring in. A good glue should score well not just because its warhead is anchored in a pocket, but because its arm earns an **extra** binding bonus once the recruited partner is present. We capture that bonus as a two-tier differential on the **same pose**: Tier 1 scores the molecule against only the protein its warhead anchors into; Tier 2 adds the recruited partner; the **differential (Tier 2 − Tier 1)** isolates the **arm's contribution once the recruited partner is present** — our working proxy for glue-specific cooperativity. (This is the project's own methodological contribution, not something inherited from a paper.)

What the differential *does* capture is the ligand arm's direct contacts with the recruited partner in the docked pose. What it *cannot* capture is protein–protein interface stabilization that doesn't run through the ligand — which is the most likely explanation for the CRBN ceiling below, where neosubstrate recognition is heavily PPI-driven.

Which protein is *anchored into* versus *recruited* **flips between our two systems** — they are mirror images, which is exactly why the same differential works for both:

- **CRBN / 5HXB** (`[matyskiela2016cc885]`): warhead (glutarimide) anchors in the **E3** (CRBN); the arm recruits the **neosubstrate** (GSPT1). Differential = GSPT1 bonus.
- **CDK12–DDB1 / 6TD3** (`[slabicki2020cr8]`): warhead (purine) anchors in the **target kinase** (CDK12); the arm recruits the **E3 adaptor** (DDB1). Differential = DDB1 bonus.

We call this metric the **neosubstrate differential** for historical reasons — it was first defined on CRBN, where the recruited partner *is* the neosubstrate. For 6TD3 the recruited partner is the E3, so there the same metric is the DDB1 bonus. (For in-silico glue-design context and why 6TD3 is a sound testbed, see `[bengeoffrey2025molde]`.)

### How the model learns: the active-learning loop

RGFN never reads the expensive scoring function directly. We train it the way `[bengio2021gflownet]` runs its multi-round molecule experiments (§4.3; **Algorithm 1** in A.5): a fast **proxy** `M` is the in-loop reward, and the expensive **oracle** `O` is queried only on small batches to keep that proxy honest.

**Before the loop (initialization).** Build a seed dataset `D_0` of `(x, O(x))` pairs labeled by the *true* oracle, then **warm-start** the proxy `M` by fitting it on `D_0`. Skip this and round 1 trains RGFN against an untrained proxy (noise). Also fix `N` (rounds), `β` (inverse temperature), and `K` (how many top molecules you want out at the end). One more constraint: a GFlowNet needs a **positive** reward, so `O(x)` — and hence `M(x)` — must be scaled/shifted to be `> 0` before use (docking is negative-is-better, so this matters).

**One round** (higher β peaks the sampling distribution harder toward high reward):

1. Fit the proxy `M` on everything labeled so far, `D_{i-1}`.
2. Train RGFN to sample proportional to `M(x)^β`.
3. Sample a query batch `B = {x_1,…,x_b}`, each `x_j ~ π_θ`, from the trained policy.
4. Score `B` with the expensive oracle `O`, giving `D̂_i = {(x_j, O(x_j)) : x_j ∈ B}`.
5. Accumulate the **full history**: `D_i = D̂_i ∪ D_{i-1}` (the proxy is refit on everything, not just the latest batch).
6. Repeat for `N` rounds; the deliverable is the top-`K` molecules in `D_N`.

Verbatim algorithm (`[bengio2021gflownet]`, Alg. 1, A.5; their generic oracle `O` = our docking/MD scorer, their proxy `M` = our fast learned reward):

```
Input:  D_0 = {(x_i, y_i)}   (y_i = true oracle reward);  K;  N;  β
Result: TopK(D_N)
Init:   proxy M;  policy π_θ;  oracle O;  i ← 1
while i ≤ N:
    fit M on dataset D_{i-1}
    train π_θ with unnormalized target reward  r(x) = M(x)^β
    sample query batch  B = {x_1,…,x_b},  x_j ~ π_θ
    evaluate B with O:  D̂_i = {(x_1, O(x_1)), …, (x_b, O(x_b))}
    update dataset  D_i = D̂_i ∪ D_{i-1}
    i ← i + 1
return TopK(D_N)
```

**The one gotcha to respect:** oracle scores enter the loop *only* by retraining `M`, never as a direct RGFN reward — the chain is `oracle → improves proxy → proxy scores everything → RGFN learns`, not `oracle → RGFN`.

Why this needs a *diverse* sampler — i.e. why a GFlowNet rather than reward-maximizing RL — is the paper's own argument: the proxy is only trained on what the generator proposes, so a mode-collapsed generator gives the proxy no signal outside the modes it already found (`[bengio2021gflownet]`, A.5). RGFN's diversity keeps the proxy's training distribution broad, which is what makes the loop work at all.

**Mapping the paper's vocabulary onto our systems.** The expensive oracle `O` is our **docking** score today (and **MD stability** later — see open experiments); the **proxy** `M` is a fast learned model fit on those scores. In their molecule experiment they instantiate exactly this — an MPNN proxy trained to predict AutoDock scores, refit on ~200 freshly docked molecules per round (`[bengio2021gflownet]`, A.5.2). Where docking is cheap enough to call in-loop, it can stand in for `M` directly in early single-system runs; the proxy/refresh machinery is what keeps the full run tractable once the per-molecule score (MD, co-folding) is far too slow to call millions of times.

The core argument of the paper: RGFN + a properly designed docking oracle, trained through this active-learning loop, can generate novel, drug-like, **synthesizable** molecular glue candidates. Each experiment is a piece of evidence for this argument.

---

## Paper target

Realistic tiering (revised down from an earlier "NeurIPS primary / Nature stretch" framing to match what an in-silico-only result can actually clear; see the project's venue-strategy notes):

- **Fast / first** — an ML workshop: **NeurIPS AI4Science**, **ICLR MLDD**, or **GEM**. Low-risk, quick, citable; most are non-archival, so a fuller journal version can follow. Best way to leave the lab with a result locked in.
- **Primary journal** — **Digital Discovery (RSC)**: the best fit for a rigorous in-silico methods+application paper without wet validation. **Journal of Cheminformatics** is a strong alternative if we lean on the reusable-method/codebase framing.
- **Stretch** — **JCIM**: achievable *with* ≥2 validated systems, clean/transparent data splits, recovery of known glues, and strong baselines including a non-synthesizable generator.
- **Out of scope without wet-lab** — NeurIPS/ICLR **main track** and **Nature-family**. The competitive ML tracks and the Nature tier effectively require a synthesized, tested compound.

**What reviewers will ask:**
- Does the oracle generalize beyond one system? (Need 2+ validated systems.)
- Does RGFN actually generate better molecules than random sampling / a non-synthesizable baseline? (cf. `[bengio2021gflownet]` Fig. 7 — GFlowNet beats random acquisition in the multi-round setting; we need the analogous plot.)
- Can you show the neosubstrate differential specifically matters (ablation vs. Tier 2 absolute score only)?
- How do you position against existing **conditioned** glue generators, and is "synthesizable" doing real work here? (Our answer: reaction-grounded synthesis routes, not merely valid/drug-like molecules.)
- (Top-tier only) Did you synthesize and test any generated molecules?

---

## Current project status

A list of objectives for our project. Tiers: **MVP** (minimum publishable result), **Target** (the full glue story), **Stretch** (upside). Check items `[X]` as they land; blocked items carry a ⚠️.


### Objective 0 — Oracle validation *(Foundation)*
- [X] Validate the docking oracle on **6TD3**: 78-pp separation between known glues and warhead-matched decoys on the DDB1 neosubstrate differential — the validated testbed for RGFN. (exp `002`)
- [X] Confirm the **neosubstrate differential** (Tier2 − Tier1, same pose) as the discrimination metric. **The specific signal is the *Vina* Tier2 − Tier1 differential.** Did six-way ablation and validated again with MW matching. (exp `006`, `007`)
- [X] Stand up the batched **gnina** pipeline on Balam (16 workers, 4× A100; ~14 min / 400 molecules). (exp `004`)
- [ ] ⚠️ **CRBN / 5HXB docking oracle** — blocked at a ceiling (−3 pp; docking can't see CRBN's PPI-driven recognition). Not usable as-is; revisit via MD (Objective 3) rather than sinking more time into box-docking tweaks. (exp `001`, `003`)
- [ ] Validate the oracle on **≥1 additional system** from MolGlueDB (generalization evidence the journals require).
### Objective 1 — First end-to-end RGFN run *(MVP)*
- [ ] Wire the glue oracle into RGFN's reward interface (`glue/` package → `[koziarski2024rgfn]` proxy/reward API).
- [ ] Build the seed dataset `D_0` and **warm-start the proxy** (`[bengio2021gflownet]` Alg. 1 init).
- [ ] Run RGFN through the **active-learning loop** on the validated 6TD3 oracle.
- [ ] **Instrument oracle-call counting from the very first run** (cannot be reconstructed later).
- [ ] Produce the **top-k-vs-oracle-calls** curve with a **random-acquisition baseline**.
### Objective 2 — Reward design & ablations *(Target)*
- [ ] Assemble the multi-objective reward: differential + QED (+ synthesizability comes free from RGFN).
- [ ] Ablation: **differential vs. Tier-2-absolute-only** — does the cooperativity term actually matter?
- [ ] Ablation: contribution of each reward component / building-block set.
### Objective 3 — Beat the CRBN ceiling with MD *(Target)*
- [ ] Test **MD stability** as oracle `O` for CRBN-type systems where docking plateaus.
- [ ] If MD discriminates, fold it in as the expensive `O` with a learned proxy `M` (multi-fidelity loop).
### Objective 4 — Generalization & baselines *(Target / journal bar)*
- [ ] Drive **≥2 systems** end-to-end through generation (not just oracle validation).
- [ ] Add a **non-synthesizable baseline generator** (e.g., graph GFlowNet / REINVENT) on the *same* oracle and budget.
- [ ] **≥3 seeds** per headline result, reported with error bars.
### Objective 5 — Evaluation suite *(Target)*
- [ ] **Synthesizability**: SA distribution + by-construction route advantage vs. the non-synthesizable baseline (the headline differentiator).
- [ ] **Diversity** (internal + scaffold) under a fixed oracle budget.
- [ ] **Recovery** of known glues / glutarimide chemistry (retrospective enrichment).
- [ ] **Anti-gaming**: in-loop-proxy vs. high-fidelity-oracle correlation on top-k; medchem sanity (PAINS, MW, logP).
- [ ] **High-fidelity validation** of top-k (co-folding / Boltz-2 / short MD).
### Objective 6 — Positioning & release *(always-on)*
- [ ] Write the **novelty paragraph** vs. the conditioned JT-VAE glue generator (reaction-grounded synthesizability + goal-directed sampling, not a conditioned VAE).
- [ ] Keep **code/data release-ready**: pinned env, fixed seeds, dataset provenance; archive every run's full output.


---

## Experiment log index

Chronological record; the objectives above cite these by number. Full entries in `../Logs/`.

| # | Date | Title | Verdict |
|---|------|-------|---------|
| [001](../Logs/001_5hxb-crbn-anchored-docking.md) | 2026-06-11 → 06-18 | 5HXB / CRBN — warhead-anchored docking oracle + decoy control | Works as a docker; **discrimination ceiling** — GSPT1 bonus is a geometric artifact, not a glue signal |
| [002](../Logs/002_6td3-cr8-validation-and-discrimination.md) | 2026-06-18 | 6TD3 / CDK12-DDB1 — oracle validation + discrimination run | **Decisive discrimination** — 85.6% vs 7.3% on DDB1 differential (+78 pts); validated oracle |
| [003](../Logs/003_crbn-vs-6td3-cross-system.md) | 2026-06-18 | 5HXB vs 6TD3 — head-to-head neosubstrate differential comparison | **6TD3 discriminates (+78 pts); 5HXB does not (−3 pts)** — failure is structural, not methodological |
| [004](../Logs/004_compute-benchmark.md) | 2026-06-18 | Compute benchmark — login node vs Balam debug_full_node (4× A100) | **CRBN 21× / 6TD3 6.6× faster; batching is the dominant lever** — conformer embedding is the new bottleneck |
| [005](../Logs/005_tier2-vina-roc-pr-curves.md) | 2026-06-23 | Tier 2 Vina — ROC and PR curves for 6TD3 and 5HXB | **6TD3 AUC=0.890 / AP=0.872; CRBN AUC=0.627** — absolute Tier 2 is a strong 6TD3 oracle; CRBN ceiling confirmed structural |
| [006](../Logs/006_6td3-violin-distributions.md) | 2026-06-25 | 6TD3 — which metric discriminates best (Tier 1 vs Tier 2 vs Δ, Vina vs CNN) | **Vina ΔT2−T1 wins (AUROC 0.946)**; Vina Tier 1 worst (0.691) — the signal lives in what DDB1 adds |
| [007](../Logs/007_6td3-molecular-weight-control.md) | 2026-06-25 | 6TD3 — controlling glue-vs-decoy discrimination for molecular weight | **Differential survives MW-matching (0.95→0.87); absolute scores collapse (Vina Tier 1 → 0.38)** — it isn't reading ligand size |

---

## The systems

| Name | PDB | E3 | Neosubstrate | Glue | Warhead | Oracle status |
|---|---|---|---|---|---|---|
| 5HXB | 5HXB | CRBN | GSPT1 | CC-885 | glutarimide | ⚠️ Ceiling — even with warhead-anchored docking, the neosubstrate differential doesn't discriminate real glues from random ones |
| 6TD3 | 6TD3 | DDB1 (adaptor) | cyclin K (CDK12-bound) | CR8 | purine (binds CDK12, *not* the E3) | ✅ Validated oracle |

> Note on 6TD3: the protein *recruited* in the differential is the **E3 adaptor DDB1**, not the degraded neosubstrate (cyclin K). This is the mirror-image flip described above and the reason "neosubstrate differential" is a slight misnomer for this system.

---

## Key terminology

- **RGFN**: Our generative model — a GFlowNet that builds molecules from reactions and **samples proportional to reward** (not reward-maximizing). Needs an oracle/proxy reward to learn what "good" looks like. See `[koziarski2024rgfn]`, `[bengio2021gflownet]`.
- **Oracle (`O`)**: The expensive, trusted scoring function. Currently docking on the ternary complex; MD stability is a candidate complementary/replacement oracle. In the active-learning loop it is queried only on the per-round query batch.
- **Proxy (`M`)**: The fast, learned in-loop reward, fit on oracle-labeled data and refit each round. RGFN trains against `M(x)^β`, never against `O` directly. See `[bengio2021gflownet]` §4.3 / Alg. 1.
- **Active-learning loop / multi-round**: The training procedure — fit proxy, train RGFN against it, sample a batch, label the batch with the oracle, refit the proxy, repeat for `N` rounds. `[bengio2021gflownet]` Alg. 1 (A.5).
- **β (inverse temperature)**: Controls how peaked the target reward is. Higher β concentrates sampling on high-reward modes.
- **Molecular glue**: Small molecule that induces proximity of two proteins (E3 ligase + neosubstrate), leading to ubiquitination and degradation of the neosubstrate.
- **E3 ligase**: The degradation machinery. We study CRBN (part of the CRL4 complex) and DDB1 (adaptor for CDK12).
- **Neosubstrate**: The protein being degraded. GSPT1 (CRBN system); cyclin K, presented by CDK12 (6TD3 system).
- **Warhead**: The part of the glue that anchors into a fixed pocket. That pocket is on the **E3** for CRBN (glutarimide → CRBN tri-Trp cage) but on the **target kinase** for 6TD3 (purine → CDK12 ATP pocket) — *not* the E3. This flip is the mirror-image point above.
- **Neosubstrate differential**: Tier2 − Tier1 score for the same docked pose. Isolates the arm's contribution to recruiting the second protein (the neosubstrate GSPT1 for CRBN; the E3 adaptor DDB1 for 6TD3). Our primary discrimination metric and the project's novel contribution. The validated signal uses the **Vina** score (lower = better binding, so the differential is lower-is-better; entries 006/007); gnina's CNN score is used only to pick the docked pose, not as the differential.
- **Tier 1 / Tier 2**: Tier 1 = the **warhead-anchoring protein only**. Tier 2 = that protein **plus the recruited partner**. Same pose scored against both. (CRBN: Tier 1 = CRBN, Tier 2 = CRBN+GSPT1. 6TD3: Tier 1 = CDK12, Tier 2 = CDK12+DDB1.)
- **Decoys**: Realistic fake glues — correct warhead, random drug-like arm. If decoys score like known glues, the oracle only reads warhead binding and is useless for RGFN.
- **gnina**: Our docking engine (v1.3.2, CNN-rescored). Launched from `/scratch/markymoo/gnina/run_gnina.sh`.
- **Balam**: SciNet GPU cluster (4× A100 per debug_full_node, 64 cores, 1 h max). Outputs go to `$SCRATCH` (`/scratch/markymoo/`), not `$HOME`.

---

## Where things live

- **Experiment logs**: `Logs/` (indexed above).
- **Key publications** (method + biology): `Logs/references/` — orientation sheet + PDFs; ground method/biology claims here.
- **Repo layout, code, datasets, and result/scratch locations**: see `docs/ARCHITECTURE.md` (the single source for where things live in the repo) — kept there so locations aren't duplicated across docs.
