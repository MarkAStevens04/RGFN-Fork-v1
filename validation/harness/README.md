# validation/harness/

The reusable **runner** that drives entrants through a suite, plus the
**evaluation metrics** used only for benchmarking.

Responsibilities (when implemented):

- read a spec from `validation/configs/`,
- load the named entrants from `validation/generators/` and run each through a
  `validation/suites/` suite at an **equal oracle-call budget** (so RGFN and the
  baselines are compared on equal compute),
- optionally score the resulting top-k with a `validation/oracles/` high-fidelity
  scorer (e.g. Boltz-2) for the anti-gaming check,
- write summarized tables/plots to `validation/results/`.

Metric placement: metrics used *only* for evaluation (diversity, SA distribution,
recovery/enrichment, anti-gaming correlations) live here. A metric the
**pipeline** also needs (e.g. scaffold counts during training) belongs in
`glue/metrics/` and is imported from there — never the reverse.

> Scaffolding only — no runner code yet.
