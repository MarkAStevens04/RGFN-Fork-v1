# RGFN Molecular Glue Research — Project Context

This file exists so that AI agents working on this project can orient themselves quickly. Read it before writing any experiment log. Update it as the project evolves.

---

## What this project is

We're building an end-to-end computational pipeline for generating novel **molecular glue degraders** — small molecules that force two proteins to interact, triggering degradation of a disease-causing protein. The generative model is **RGFN** (a reinforcement-learning GFlowNet). RGFN needs a *reward signal* to know whether a molecule it generates is a good glue candidate. We're building and validating that reward signal (the **oracle**).

The oracle uses **molecular docking**: simulate how well a candidate molecule fits into the ternary complex (E3 ligase + neosubstrate + glue). A good glue should dock well in the full ternary complex but NOT just because of where its warhead sits — it should show extra benefit from the neosubstrate being present. We call this the **neosubstrate differential**: score with neosubstrate (Tier 2) minus score without (Tier 1), measured on the same pose.

The core argument of the paper: RGFN + a properly designed docking oracle can generate novel, drug-like molecular glue candidates. Each experiment is a piece of evidence for this argument.

---

## Paper target

- **Primary**: NeurIPS (ML methodology venue — focus on novelty of the approach, generalization across systems, clean ablations)
- **Stretch**: Nature (requires wet-lab validation of at least one generated compound, multiple protein systems)

**What reviewers will ask:**
- Does the oracle generalize beyond one system? (Need 2+ validated systems)
- Does RGFN actually generate better molecules than random sampling?
- Can you show the neosubstrate differential specifically matters (ablation vs. Tier 2 absolute score only)?
- (Nature) Did you synthesize and test any generated molecules?

---

## Current project status

### Validated ✅
- **CDK12-DDB1 system (6TD3)** works as an oracle: straight box docking produces 78-percentage-point separation between known glues and random warhead-bearing decoys on the DDB1 neosubstrate differential. This is the right testbed for RGFN.
- The neosubstrate differential (Tier2 − Tier1, same pose) is the right metric — it isolates glue-specific cooperativity.
- Compute pipeline on Balam: batched gnina docking (16 workers, 4× A100) reduces wall time from ~90 min to ~14 min for 400 molecules.

### Does NOT work / ceiling hit ⚠️
- **CRBN system (5HXB)** shows near-zero neosubstrate differential discrimination (−3 percentage points) even with warhead-anchored docking. The oracle successfully docks into the CRBN binding pocket, but can't tell real glues from random warhead-bearing molecules. This system is **not suitable as the RGFN oracle** in its current form.

### Open / next experiments 🔲
- Test whether Molecular Dynamics stability can serve as a better or complementary oracle (especially for CRBN-type systems where docking has a ceiling)
- Validate oracle on additional systems (MolGlueDB has other known-glue systems to test)
- Run RGFN with the validated 6TD3 oracle and evaluate generated molecules
- Ablation: does Tier 2 absolute score alone work, or is the differential necessary?

---

## The systems

| Name | PDB | E3 | Neosubstrate | Glue | Warhead | Oracle status |
|---|---|---|---|---|---|---|
| 5HXB | 5HXB | CRBN | GSPT1 | CC-885 | glutarimide | ⚠️ Ceiling — even with warhead-anchored docking, the neosubstrate differential doesn't discriminate real glues from random ones |
| 6TD3 | 6TD3 | DDB1 | CDK12–cyclinK | CR8 | purine | ✅ Validated oracle |

---

## Key terminology

- **RGFN**: Our generative model. Produces molecular glue candidates. Needs an oracle reward to learn what "good" looks like.
- **Oracle**: A scoring function that evaluates candidate molecules. Here: a docking-based score on the ternary complex.
- **Molecular glue**: Small molecule that induces proximity of two proteins (E3 ligase + neosubstrate), leading to ubiquitination and degradation of the neosubstrate.
- **E3 ligase**: The degradation machinery. We study CRBN (part of the CRL4 complex) and DDB1 (adaptor for CDK12).
- **Neosubstrate**: The protein being degraded. GSPT1 (for CRBN), CDK12/cyclinK (for 6TD3).
- **Warhead**: The part of the glue that anchors to the E3 ligase binding pocket (glutarimide for CRBN, purine for CDK12).
- **Neosubstrate differential**: Tier2_score − Tier1_score for the same docked pose. Isolates the arm's contribution to neosubstrate recruitment. Our primary discrimination metric.
- **Tier 1 / Tier 2**: Tier 1 = E3 pocket only (receptor = E3 subunit). Tier 2 = E3 + neosubstrate (receptor = full ternary complex minus glue). Same pose scored against both.
- **Decoys**: Realistic fake glues — correct warhead, random drug-like arm. If decoys score like known glues, the oracle only reads warhead binding and is useless for RGFN.
- **gnina**: Our docking engine (v1.3.2, CNN-rescored). Launched from `/scratch/markymoo/gnina/run_gnina.sh`.
- **Balam**: SciNet GPU cluster (4× A100 per debug_full_node, 64 cores, 1 h max). Outputs go to `$SCRATCH` (`/scratch/markymoo/`), not `$HOME`.

---

## Where things live

- **Experiment logs**: `Logs/` (this directory)
- **Pre-processing scripts**: `pre-processing/`
- **Docking scripts (CRBN)**: `docking_gnina/`
- **Docking scripts (6TD3)**: `pre-processing/docking_6td3/`
- **Protein models**: `models/`
- **Test datasets**: `pre-processing/test-data/`
- **Scratch (Balam)**: `/scratch/markymoo/rgfn_runs/`
