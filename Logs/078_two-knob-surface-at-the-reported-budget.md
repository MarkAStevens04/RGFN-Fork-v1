# Four generators / sEH — does the advantage survive both quality knobs at the budget we actually report?

**Date:** 2026-09-06, ~5pm

## Question

If a chemist has 100 reactions to spend, does hub-batching still deliver more distinct high-scoring
molecules than making the best molecules one at a time — no matter how strictly we define
"high-scoring" and "distinct", and no matter which generator produced the molecules?

## Context & Summary

**Context.** Every headline number in the workshop paper is quoted at one setting of two thresholds
a reviewer can argue with: the **reward bar** a molecule must clear to count at all, and the
**Tanimoto cutoff** below which two molecules count as the same thing. Set them loosely and
everything looks like a distinct hit; set them strictly and nothing does. Entry [052] answered this
for SCENT alone and entry [055] extended it to all four generators — both found no crossover
anywhere — but both measured at a budget of **300 reactions**. The project's headline readout became
**100 reactions** on 2026-08-17, so an appendix figure defending the thresholds at a budget the paper
never quotes invites the obvious question. The measurement is cheap: it re-scores an enumeration that
already exists, with no model, no oracle and no GPU.

**Summary.** We re-ran the whole grid at 100 reactions for all four generators on sEH, then added the
project's actual operating bar — **5.68**, the 5%-FPR point on sEH's own actives-versus-decoys set —
as an extra row so the figure marks where our numbers are really quoted rather than a nearby
gridline. Cells where either strategy ran out of library before spending the budget, or where nothing
cleared the bar at all, are classified separately and never counted as wins. The result was harvested
into the publication repo as a committed artifact plus a reproduce script, so the appendix figure
needs no cluster access.

## Answer

**Hub-batching is cheaper in 449 of 452 comparable cells, with three exact ties and no losses**, so
the advantage is not an artifact of where either threshold sits — and this now holds at the budget we
actually report, not only at the 300-reaction budget the earlier entries used.

The more useful result is that the surface **settles the 5.0-versus-7.0 question the paper has been
carrying**. At the 5.68 operating point all four generators are comparable. At 7.0, RGFN's entire row
is pool-limited — its baseline arm runs out of library at 9 molecules having spent 36 of its 100
reactions — so there is no valid comparison to quote there at all. That reproduces, on a fourth
generator and at a different budget, the starvation `targets.py` already recorded for both RxnFlow
cells on all three seeds. A ratio measured against a starved baseline understates hub-batching, so
7.0 is not the conservative choice it appears to be.

## Relevance to our Publication

The single largest threat to acceptance for a ratio metric is the suspicion that the authors chose
the thresholds they win on. GEM and ICLR reviewers will ask it directly. A surface answers it in a
way a sentence cannot: there is no corner of either knob where the comparison reverses, on any of the
four generators. It also lets the paper state one rule instead of four numbers — every gate is the
5%-FPR point of its own oracle — which is less to defend than any per-target justification, and it
removes the awkwardness of Figure 1 and Figure 3 currently gating sEH differently.

## Next Experiments

**Refining for publication.** The surface sweeps thresholds, not seeds: one cached enumeration per
generator, sEH only. Reviewers who accept the threshold argument will ask about seed variance next,
which is Figure 3's question and is currently n=1 for the external table. Extending the surface to a
second target would test whether the shape is a property of the method or of sEH's proxy, and costs
only CPU wherever an enumeration already exists.

**Next steps in project.** Decide whether Figure 3 and Table 1 move from 7.0 to the 5.68 operating
point. Figure 1 and this appendix are already there at no cost, but those two exhibits were solved
with SPARROW at 7.0 and moving them means re-solving the MILP, which is the expensive part and has a
history of not converging on sEH.

# Re-creation

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**
- `experiments/lsd_hubs/campaign/tau_similarity_surface.py` — the surface itself (from [052]): loads
  the candidate pool and cost table once, then runs both selection strategies to a fixed reaction
  budget in-process for every (bar, cutoff) cell. Unmodified.
- `experiments/lsd_hubs/matrix16/surface_all_generators.sh` — manifest-driven driver that runs the
  above for each generator with the correct per-generator child policy. **`OUTROOT` made overridable
  in this entry**; it was hard-coded, so a re-run at a different budget wrote into the same directory
  as the 300-reaction surfaces, which are the only copy of that measurement.
- `experiments/lsd_hubs/matrix16/manifest.py` — resolves each cell's sample dir, enumeration and
  checkpoint.

**Models**
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh_5k/seed42/additional_fragments/fragments_4000.json`
  — SCENT's promoted-fragment snapshot, required for the nested cost model. The same `scent_seh_5k`
  series Figure 1 draws on, which is why the SCENT cell agrees with Figure 1 (a pipeline consistency
  check, not independent corroboration).

**Datasets**
- `/scratch/markymoo/rgfn_runs/lsdflow/matrix16/<gen>_seh/enum/enum_children.json` — the cached
  one-step enumerations (200 hubs each) the surface re-scores. No new generation.
- `/scratch/markymoo/rgfn_runs/lsdflow/matrix16/<gen>_seh/sample/` — the candidate pools.

**Results**
- `experiments/lsd_hubs/matrix16/results/surface_r100/<cell>/surface.{csv,json,png,pdf}` — the four
  surfaces at budget 100. Note `*.csv` is git-ignored repo-wide, as it is for the 300-reaction
  surfaces; the durable copy of the numbers is the publication-repo artifact below.
- `experiments/lsd_hubs/matrix16/results/surface/<cell>/` — the earlier 300-reaction surfaces from
  [055], deliberately left untouched.

**Publication repo** (`/home/markymoo/projects/RGFN_Fork/RGFN-LSD`, `AC-MedChem-SDL/RGFN-LSD`)
- `tools/harvest/two_knob_surface.py` — copies the four surfaces into one tidy artifact, classifying
  every cell as comparable / pool-limited / no-modes and anything within 5e-4 of parity as a tie.
  Refuses to emit if the cells were run at different budgets.
- `artifacts/two_knob/{surface.csv,provenance.json}` — 520 rows, 44 KB, committed.
- `reproduce/figA2_two_knob_surface.py` — the appendix figure.
- `docs/CAPTIONS.md` — draft caption and the three things it must not overstate.

**Job Logs**
- Foreground login-node runs; combined stdout at
  `/home/markymoo/.claude/jobs/f8cc75e3/tmp/surf568.log`. No SLURM job — CPU only.

## Relevant Versions

Research repo, branch `worktree-tau-surface-appendix` (worktree at
`.claude/worktrees/tau-surface-appendix`, branched from `Hub-Analysis` at `6b9b92b`):

- `6463768` — "Two-knob surface at the budget the paper actually reports" (the `OUTROOT` override and
  the four budget-100 surfaces).

**This branch is not merged and not pushed** — the remote had no usable credential helper at the time
and `Hub-Analysis` was in active use by another session. **[TODO — merge `worktree-tau-surface-appendix`
into `Hub-Analysis` and push.]**

Publication repo, `main`, pushed:
- `88e5bfb` — "Appendix A2: sweep both thresholds instead of defending one point"
- `f92858e` — "Put the real operating point on the surface, and let it answer 5.0 vs 7.0"
- `129c645` — "Caption A2: the counts and the marker legend follow the 5.68 grid"

## Relevant Resources

**Sources**
- Entry [052] — the original two-knob surface, SCENT only, 300 reactions.
- Entry [055] — the same grid extended to four generators, still 300 reactions; already noted that
  RGFN's whole region above bar ≈6.3 is pool-limited *including* the (7.0, 0.5) operating point.
- Entry [051] — the pool-side control: intrinsic diversity is flat in the reward bar below ~7 and
  collapses past it, so the bar is a legitimate control rather than a hidden diversity dial.
- Entry [034] — the sEH proxy enriches known actives only weakly (AUROC 0.76/0.68) and 0 of 2,315
  real actives reach 8.0, which is why bar 7.0 is in the proxy's optimised tail.
- `experiments/lsd_hubs/matrix16/targets.py` — carries `mode_reward_threshold=5.68` for sEH with the
  reasoning inline; 7.0 retained only in `threshold_variants` as the paper-comparable number.

**Packages**
- `rdkit` (Morgan r=3 / 2048) — the fingerprint behind the diversity cutoff.
- `matplotlib` — `reproduce/figA2_two_knob_surface.py`.

## Method

1. Confirmed the four cached enumerations still exist and hold 200 hubs each, via
   `manifest.py --emit <gen> seh`. No generation, sampling or training was re-run at any point.
2. Made `OUTROOT` overridable in `surface_all_generators.sh` so the budget-100 run could not
   overwrite the committed 300-reaction surfaces.
3. Ran the grid at the reported budget, nine bars:
   `BUDGET=100 OUTROOT=.../results/surface_r100 bash experiments/lsd_hubs/matrix16/surface_all_generators.sh seh`
   — 9 bars × 13 cutoffs × 2 strategies × 4 generators, 2–5 min per generator.
4. Re-ran with the operating point on the grid (`--taus` accepts a comma list):
   `TAUS=4:8:0.5,5.68 BUDGET=100 OUTROOT=... bash .../surface_all_generators.sh seh` — 10 × 13.
5. Harvested into the publication repo, then rendered the figure:
   `python3 tools/harvest/two_knob_surface.py --surface-root .../results/surface_r100`
   `python3 reproduce/figA2_two_knob_surface.py`

## Results

**Grid:** 10 reward bars (4.0–8.0 by 0.5, plus 5.68) × 13 diversity cutoffs (0.3–0.9 by 0.05) = 130
cells per generator, 520 total, each run for both strategies at a 100-reaction budget.

| Generator | Comparable | Cheaper | Tie | Loss | Pool-limited | No modes | Median advantage | Range |
|---|---|---|---|---|---|---|---|---|
| RGFN | 91 | 91 | 0 | 0 | 13 | 26 | 3.06× | 1.01–3.84× |
| SCENT | 129 | 129 | 0 | 0 | 1 | 0 | 2.71× | 1.15–2.98× |
| RxnFlow | 102 | 101 | 1 | 0 | 15 | 13 | 2.61× | 1.00–2.80× |
| FragGFN | 130 | 128 | 2 | 0 | 0 | 0 | 4.40× | 1.00–4.80× |
| **Total** | **452** | **449** | **3** | **0** | **29** | **39** | | |

*Advantage = (reactions/mode, best-candidate) ÷ (reactions/mode, hub-batching). Cells are classified
comparable only if both strategies spent the budget; anything within 5e-4 of parity is a tie.*

**At cutoff 0.5, the three bars in play** (modes delivered, hub-batching vs best-candidate):

| Generator | bar 5.0 | bar 5.68 (operating point) | bar 7.0 |
|---|---|---|---|
| RGFN | 61 vs 25 | 57 vs 25 — **2.26×** | 35 vs 9 — **baseline pool-exhausted, not comparable** |
| SCENT | 77 vs 28 | 76 vs 28 — **2.71×** | 71 vs 28 |
| RxnFlow | 79 vs 33 | 73 vs 33 — **2.19×** | 44 vs 33 |
| FragGFN | 83 vs 20 | 80 vs 20 — **4.00×** | 76 vs 20 |

RGFN's best-candidate arm at bar 7.0 stops at 9 molecules having spent 36 of its 100 reactions; its
whole 7.0 row is pool-limited. This is the same failure `targets.py` records for both RxnFlow cells
on all three seeds, and that [055] saw on one seed at the 300-reaction budget.

**The three ties are exact**, not roundings: FragGFN at bar 8.0 with cutoffs 0.3 and 0.35 (20 modes
each arm) and RxnFlow at bar 6.5 / cutoff 0.3 (33 each). Both arms land on the same small set and
price it identically.

**The 39 no-modes cells are whole rows** — RGFN at bars 7.5 and 8.0, RxnFlow at 8.0, every cutoff.
Nothing in those pools clears the bar for either arm, so it is a property of the pool rather than of
any selection strategy.

**Cross-check against the 300-reaction surfaces** ([055], same grid minus the 5.68 row): 381/381
comparable cells cheaper, median advantage 2.60–4.47×. Same verdict, different budget.

**Consistency with Figure 1:** SCENT at bar 5.0 / cutoff 0.5 reads 77 vs 28 here and in
`fig1_seh_gate_variants.csv`. These share a checkpoint and enumeration, so this checks the pipeline,
not the result.

**Cost:** 2–5 min of login-node CPU per generator per grid; no GPU, no oracle calls, no SLURM job.
