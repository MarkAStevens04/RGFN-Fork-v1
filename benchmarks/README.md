# benchmarks/

Benchmark harness and committed results for evaluating our RGFN runs and oracles.

This is a first-class goal of the project (see `Logs/RESEARCH_CONTEXT.md`):
reviewers will ask whether RGFN beats random sampling **and a non-synthesizable
generator baseline** (does "synthesizable" do real work?), whether the oracle
generalizes across systems (≥2 validated), and whether the neosubstrate
differential specifically matters (ablation vs. Tier 2 absolute score alone).
A key target plot is the analogue of `[bengio2021gflownet]` Fig. 7 — top-k
quality vs. oracle calls, GFlowNet vs. random acquisition. This directory is
where those comparisons live.

Suggested layout (create as needed):

```
benchmarks/
├── harness/        # reusable benchmark code (imports glue + rgfn)
├── configs/        # benchmark specs (which runs/oracles/baselines to compare)
└── results/        # committed result tables + plots (small artifacts only)
```

Conventions:
- Benchmark *code* lives here or in `glue/`; keep heavy raw outputs in gitignored
  scratch and commit only summarized tables/figures.
- Each benchmark should be reproducible from a config + a documented command, and
  significant runs should get an entry in `Logs/`.
