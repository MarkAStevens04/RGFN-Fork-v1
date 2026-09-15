# Publication-repo data handoff — RESOLVED 2026-09-12

This was a proposal asking who should own "which measurement generation is each committed
publication artifact from?". It is answered, so it has been cut to the answer rather than left
as a hundred lines of superseded reasoning.

**The answer: the publication repo owns it; the re-run owns the trigger.**

`RETRAIN_RUNBOOK.md` **§10.1** now carries the step — on completing a phase, re-run that
group's harvest script against the new tree, flip the generation marker, rebuild the manifest.
No `reproduce/` script changes; the harvest layer is the seam and is already parameterised for
retargeting.

## Why not in this tree

`benchmark_v2/tools/manifest.py` computes live cell status at load time *because the research
tree changes constantly*. Published artifacts change once per generation. Different problems —
and teaching `manifest.py` which of its cells someone downstream happened to publish would
couple producer to consumer backwards. (Checked, not assumed: `manifest.py` has no notion of
exhibits, artifacts or generations, and nothing else under `benchmark_v2/tools/` does either.)

## What lives in RGFN-LSD

- `artifacts/MANIFEST.csv` — generated, one row per committed file, with `generation`.
- `tools/build_manifest.py` — builds it; `--check` fails when stale; flags any artifact group
  no published exhibit reads.

Three generation states, and the last distinction is the one that matters:

| state | meaning |
|---|---|
| `v1` | measured under v1 standards; a newer measurement will exist once the re-run reaches it |
| `v1-stale` | its source has **already** been regenerated — re-harvest available now |
| `invalidated` | the quantity was **redefined**. No newer measurement of the same thing exists or will. Re-harvesting cannot fix it |

Collapsing `invalidated` into "superseded" is the dangerous direction: it implies a newer number
exists somewhere and sends someone looking for it.

Currently flagged: 27 files `invalidated` (every 6TD3 cell — 6TD3-B replaced that reward, so it
is a re-dock rather than a re-gate), 2 files `v1-stale` (the external head-to-head, whose
S3-GFN seed-43 pool was retrained 2026-09-12).

## The one claim to keep straight

The route dataset: **the machinery is what is proven today, not the dataset.** v1 supports route
artifacts for ~5 cell-seeds; matrix-wide exists only as a by-product of the re-run. Do not
describe the dataset as existing until the phase producing it passes verification — that is a
claim a chemist would act on.
