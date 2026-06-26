# validation/configs/

Run specs for comparative studies: *which generators*, on *which oracle /
benchmark*, at *what oracle-call budget and how many seeds*.

These are the validation-side analogue of `configs/glue/` (which configures the
production training pipeline). A spec here typically names:

- one or more entrants from `validation/generators/`,
- the oracle/benchmark to score against (a `validation/suites/` suite, and/or a
  `validation/oracles/` high-fidelity check),
- budget + seed settings so comparisons are apples-to-apples
  (`docs/RESEARCH_CONTEXT.md` Objective 4 wants ≥3 seeds with error bars).

Format (gin vs. YAML) is TBD and will be decided when the harness lands; keep
specs declarative so a run is reproducible from `config + documented command`.

> Scaffolding only.
