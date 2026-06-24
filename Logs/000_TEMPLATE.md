# 6TD3 / CR8 (CDK12-cyclinK) validation + discrimination run
**Date**: 2026-06-18 11:00am

## Question:
> This section should offer a succinct summary of the goal of this experiment. What question are we trying to answer? What problem are we trying to solve? When writing this question, really try to have the broader publication arc in mind, and think how this fits into the publication we're working towards. Focus on not using Jargon, and making it as simple to understand as possible.


Can our docking oracle accurately discriminate between random molecules and known binders for the CDK12-DDB1 system?


## Context & Summary:
> If this experiment journal is a story, this section is our introduction. We're creating the stakes, and defining a clear objective. I define a context & summary sub-section, but this is pretty flexible, so feel free to change it to better fit the structure. We'll start with some context, and explain WHY we need to perform this experiment. What other experiments have we run and why is this experiment necessary? How does this experiment fit into our investigation so far?

We previously found that known glues for the CRBN system did NOT have a significantly better docking score than randomly generated compounds, even when these random compounds had a warhead, and were constrained into the correct binding pocket (see `Logs/000_...`). We're curious if this is just an artifact of the CRBN system itself, or if there's a deeper methodological problem with our docking approach, so we're testing on the CDK12-DDB1 system.

> We'll next give a summary of what we're going to do. We just explained the context of WHY we're doing this, and what problems we need to solve. Now let's explain WHAT we're doing. Again, this is supposed to be very simple and easy to understand, keeping in mind how this experiment fits into the broader publication narrative.

We're checking the docking scores of known glues against randomly generated molecules in the CDK12-DDB1 system, and comparing those scores. If there's a strong difference in these two scores, then we know our system can work as an oracle. If the difference is weak, our current protocol cannot discriminate between randomly generated molecules and good glues, meaning our RGFN won't have good signal for generating molecules.



## Answer:
> This is another short section, briefly summarizing our key findings. Focus less on quantitative results, and more on what the results mean.

Yes, known glues have a much better docking score than randomly generated molecules in the DDB1-CDK12 system! We're confident we can use docking as an oracle for RGFN.


## Relevance to our Publication:
>  This section will explain how our experiment(s) fit into the publication. When we're looking through our logs at the end of this project, we'll reference this section heavily. Example included below:

We've shown our docking methodology can serve as a reliable oracle for generating molecular glues in our RGFN system. We've also shown that this methodology is dependent on the system being used (the DDB1-CDK12 system works well, but we previously found the CRBN system does not work well).

## Next experiments:
> This section should focus on what's next, and what needs to be done for our project to be published. What could be improved in this section, and what will journals likely want relevant to this experiment? What parts come next in the project?

**Refining for Publication**
- Publishers will likely want multiple trials of this experiment, so we may need to re-run with different seeds.
- Our realistic venue targets (see `RESEARCH_CONTEXT.md` → *Paper target*) are an ML workshop first (NeurIPS AI4Science / ICLR MLDD / GEM), then **Digital Discovery** or **J. Cheminformatics** as the primary journal. These want ≥2 validated systems, clean data splits, and strong baselines (including a non-synthesizable generator) — *not* wet-lab data. Synthesized/tested compounds are only required for the out-of-scope top tier (NeurIPS/ICLR main track, Nature-family).

**Next steps in project**
- Check if Molecular Dynamics can serve as a more accurate oracle. Is the *stability* of the ternary complex correlated strongly with molecular glues vs random molecules?
- Check additional systems beyond CDK12-DDB1. The Molecular Glue database has some good references of other molecular glues with known binders for us to investigate.


# Re-creation
> The following sections are a description of what was done, and how to recreate these results in the future! It also serves as an audit trail for us to figure out exactly *what* went wrong if things go wrong.
## Relevant Files:
> Looking at what files were used can help explain what was done in this experiment. We should help future users understand what files helped us answer our question, and where to look for each step of our experiment. The files should be roughly categorized by their type, and ordered by when they were used. Again, this is just a rough categorization, and the exact order and how the files fit together will be explained more clearly by our Method section. Focus on not using jargon and keeping this section understandable and tight. This section also helps us figure out if our results don't make sense, where exactly we should check for errors. If after we've done like 30 experiments we go back through our files and find that one python script had an error, this section will help us figure out exactly what experiments also had that error. I don't love how we note the "root" and stuff in this section, so we may need to rework the exact conventions we use.

Root: `research/preprocessing/docking_6td3/`

**Scripts**
- `research/preprocessing/clean_6td3.py`
    - Generates Tier 1, Tier 2, Tier 3 structure files
    - Cleans PDB structure, removes everything except proteins & ligands relevant to given Tier
- `research/preprocessing/docking_6td3/redock_cr8.py`
    - Performs docking on CR8 ligand to see if we can recover native known structure
- `make_decoys_cdk.py`
    - Generate purine-armed decoy molecules
- `research/preprocessing/docking_6td3/dock_cluster.py`
    - Run with `debug_full_node`.
    - Ran with both real glues and fake glues.
- `research/preprocessing/docking_6td3/submit_dock_6td3.sh`
    - Actual job submit script for Balam.
- `research/preprocessing/compare_systems.py`
    - Evaluation script for comparing our results.


**Models**:
> Non-Script files should include a rough explanation of what we BELIEVE each file has, so that we can come back later and verify that our beliefs about those files are accurate.
- `models/6TD3.pdb`
    - Base crystal structure of our CDK12 + DDB1 + RC8 system. Has 2 copies of ternary complex.
- `models/6TD3_tier1_CDK12.pdb`
    - Pose of CDK12 from the 6TD3 system. All other structures removed.
- `models/6TD3_tier2_CDK12_DDB1.pdb`
    - Pose of CDK12 + DDB1 complex from 6TD3 crystal. Only 1 copy of the pair, no RC8 or other atoms.
- `models/crystal_RC8.pdb`
    - Correct position of RC8 ligand in DDB1-CDK12 system.

**Datasets**
- `test-data/DDB1_CDK12_Glues.csv`
    - Reference set of our known DDB1-CDK12 molecular glues.
- Datasets generated from `make_decoys_cdk.py` (should put actual path here)
    - Set of randomly generated molecules with purine ring.

**Results**
- `docking_6td3/known_results.csv`
    - Docking scores of our known molecular glues on both the Tier 1 and Tier 2 systems
- `decoy_cdk_results.csv`
    - Docking scores of our randomly generated molecules on both the Tier 1 and Tier 2 systems.

**Job Logs**
- `/scratch/markymoo/rgfn_runs/dock_6td3_69271/`
    - Data produced from our run
- `/scratch/markymoo/rgfn_runs/dock6td3-69271.out`
    - Logs from SLURM

**Memory**
- `6td3-cr8-cyclink-glue-system`
- `balam-slurm-submission`
    - How to submit to balam

## Relevant Versions
> Insert link to Git PR or Git Commit here


## Relevant Resources
> Again, this is mostly for auditing. Write down what sources we referenced when generating our datasets, when creating scripts, etc.

**Sources**
- 6TD3 system was chosen in reference to a Molecular Glue Evaluator published at ACS HERE. https://pubs.acs.org/doi/10.1021/acsomega.4c08049

- Found known molecular glue binders HERE. https://www.molgluedb.com/browseDB

**Packages**
- Vina [Link here]
- `research/preprocessing/clean.py` helpers used heavily in `research/preprocessing/clean_6td3.py`

## Method
> This sections helps us both audit and understand how the experiment worked. Write down what commands were run and why. Focus on the big commands run, not on small fixes or tiny experiments to validate packages.

1. **Structure prep** — `research/preprocessing/clean_6td3.py` carves 6TD3 copy 1: Tier 1 = CDK12
   (`models/6TD3_tier1_CDK12.pdb`), Tier 2 = CDK12+DDB1 (`6TD3_tier2_CDK12_DDB1.pdb`), Tier 3 =
   +cyclinK; native CR8 → `crystal_RC8.pdb`. Receptors → pdbqt (obabel). No anchoring — straight box
   docking (`--autobox_ligand crystal_RC8.pdb --autobox_add 4`).
2. **Validation** — `docking_6td3/redock_cr8.py`: blind-redock CR8 + native in-place min + Tier1/Tier2.
3. **Discrimination run** — `docking_6td3/dock_cluster.py` on `debug_full_node` (job 69271):
   160 real glues (`test-data/DDB1_CDK12_Glues.csv`) vs 248 purine-armed decoys
   (`make_decoys_cdk.py`). Dock Tier2, take best-CNN pose, score that pose vs Tier1 →
   **DDB1 differential** = Tier2 − Tier1.


## Results
> In the `Answer` section, we wrote down broadly what we can conclude. Here we should focus on clearly stating the results that support those claims, and a brief explanation of how our results support our claims. I would rather have had TWO separate experiment log files for “DDB1-CDK12 discrimination ability” and “Tier 2 vs Tier 1”, even though both results came from the same experiment. In the first experiment log, we would JUST have the Tier 2 experiment results. In the second experiment log, we would have both Tier 1 and Tier 2 results, and compare.

**Validation:**
- Blind redock **recovers the native pose at 1.23 Å, ranked #1** (Vina −10.68, CNN 0.99). Native
  in-place min −10.56 / CNN 0.985. (Contrast CRBN: never sampled native.)
- **DDB1 cooperativity is captured:** CR8 Tier1 (CDK12) −8.39 vs Tier2 (CDK12+DDB1) −10.68 →
  **DDB1 bonus −2.16 to −2.29 kcal/mol** on the same pose.

**Discrimination run (job 69271, 13 min, full node):**
| metric | known (n=160) | decoy (n=248) | gap |
|---|---|---|---|
| **frac DDB1 dVina < −1.5** | **85.6%** | **7.3%** | **78 pts** |
| median DDB1 dVina | −2.20 | −0.60 | −1.60 |
| median Tier2 Vina | −10.15 | −7.96 | −2.19 |
| median Tier2 CNNaff | 7.82 | 6.70 | 1.13 |
| frac Tier2 Vina < −10 | 55.6% | 3.6% | 52 pts |

**Decisive discrimination.** 85.6% of real glues get a strong DDB1 bonus vs 7.3% of decoys — the
separation CRBN never produced. The proxy rewards a *productive DDB1-contacting arm*, not just
warhead presence. → This system is the right RGFN testbed. Reward = Tier2 Vina + DDB1-differential
gate. Warhead for warhead-constrained generation = the purine ATP-hinge binder.
