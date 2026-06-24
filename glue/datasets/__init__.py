"""Datasets — loaders for input data and generators for synthetic datasets.

Two responsibilities:
    1. Load curated inputs (known glues, decoys, building blocks) for benchmarks
       and oracle validation.
    2. Generate synthetic datasets from trained RGFN runs (sample molecules,
       score them, emit a dataset artifact).

Conventions:
    - Raw / curated inputs live under `data/` and `research/preprocessing/test-data/`.
    - Generated synthetic datasets are written under `data/synthetic/` (gitignored).
    - Keep file formats simple (csv / smiles / parquet) and documented.

Import new dataset modules below so `glue.registry` registers any gin-configurable
loaders or generators.
"""

# from glue.datasets.synthetic import SyntheticDatasetGenerator  # noqa: F401
