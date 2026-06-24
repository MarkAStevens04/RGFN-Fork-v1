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

### Validated ✅
- **CDK12-DDB1 system (6TD3)** works as an oracle: straight box docking produces 78-percentage-point separation between known glues and random warhead-bearing decoys on the DDB1 neosubstrate differential. This is the right testbed for RGFN.
- The neosubstrate differential (Tier2 − Tier1, same pose) is the right metric — it isolates the arm's contribution to recruiting the second protein.
- Compute pipeline on Balam: batched gnina docking (16 workers, 4× A100) reduces wall time from ~90 min to ~14 min for 400 molecules.

### Does NOT work / ceiling hit ⚠️
- **CRBN system (5HXB)** shows near-zero neosubstrate differential discrimination (−3 percentage points) even with warhead-anchored docking. The oracle successfully docks into the CRBN binding pocket, but can't tell real glues from random warhead-bearing molecules. Most likely cause: CRBN neosubstrate recognition is dominated by protein–protein interface complementarity that a same-pose ligand-contact differential can't see (see the caveat above). This system is **not suitable as the RGFN oracle** in its current form.

### Open / next experiments 🔲
- Test whether **Molecular Dynamics stability** can serve as a better or complementary oracle `O` (especially for CRBN-type systems where docking has a ceiling). In active-learning terms, MD becomes the expensive `O` and a learned model becomes the proxy `M`.
- Validate oracle on additional systems (MolGlueDB has other known-glue systems to test).
- **Run RGFN with the validated 6TD3 oracle through the active-learning loop** (`[bengio2021gflownet]` Alg. 1) and evaluate generated molecules; produce the top-k-vs-oracle-calls curve and a random-acquisition baseline.
- Ablation: does Tier 2 absolute score alone work, or is the differential necessary?

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
- **Neosubstrate differential**: Tier2 − Tier1 score for the same docked pose. Isolates the arm's contribution to recruiting the second protein (the neosubstrate GSPT1 for CRBN; the E3 adaptor DDB1 for 6TD3). Our primary discrimination metric and the project's novel contribution.
- **Tier 1 / Tier 2**: Tier 1 = the **warhead-anchoring protein only**. Tier 2 = that protein **plus the recruited partner**. Same pose scored against both. (CRBN: Tier 1 = CRBN, Tier 2 = CRBN+GSPT1. 6TD3: Tier 1 = CDK12, Tier 2 = CDK12+DDB1.)
- **Decoys**: Realistic fake glues — correct warhead, random drug-like arm. If decoys score like known glues, the oracle only reads warhead binding and is useless for RGFN.
- **gnina**: Our docking engine (v1.3.2, CNN-rescored). Launched from `/scratch/markymoo/gnina/run_gnina.sh`.
- **Balam**: SciNet GPU cluster (4× A100 per debug_full_node, 64 cores, 1 h max). Outputs go to `$SCRATCH` (`/scratch/markymoo/`), not `$HOME`.

---

## Where things live

- **Experiment logs**: `Logs/` (this directory)
- **Key publications** (method + biology): `Logs/references/` — orientation sheet + PDFs; ground method/biology claims here
- **Pre-processing scripts**: `research/preprocessing/`
- **Docking scripts (CRBN)**: `research/preprocessing/docking_gnina/`
- **Docking scripts (6TD3)**: `research/preprocessing/docking_6td3/`
- **Protein models**: `models/`
- **Test datasets**: `research/preprocessing/test-data/`
- **Scratch (Balam)**: `/scratch/markymoo/rgfn_runs/`

> **Note (2026-06-24 refactor):** the docking/validation work moved from
> `pre-processing/` to `research/preprocessing/` (internals unchanged). New code
> for oracles/rewards/samplers lives in the `glue/` package, and new configs in
> `configs/glue/`. See `CLAUDE.md` and `docs/ARCHITECTURE.md` for the full layout.
> Experiment logs 001–005 still reference the old `pre-processing/` paths as a
> point-in-time record; translate `pre-processing/` → `research/preprocessing/`.
