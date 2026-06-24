# models/

Protein structures and model checkpoints used as **inputs** to the pipeline.

Examples of what belongs here:
- Prepared receptor structures (Tier 1 / Tier 2 `.pdb` / `.pdbqt`) for docking.
- Trained RGFN checkpoints and oracle model weights (e.g. gneprop/seno `.ckpt`).

> Referenced historically as `models/` in `Logs/` and `research/preprocessing/`
> (e.g. `models/6TD3_tier2_CDK12_DDB1.pdb`). On Balam these often live in
> `$SCRATCH`; keep large/regenerable artifacts out of git.

**Weights and large structures are gitignored** (see `.gitignore`). Commit only
small, essential reference files; document where the rest come from / how to
regenerate them.
