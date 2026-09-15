# The competitor matrix — where the numbers are, and how to quote them

**State as of 2026-09-08: 106 of 108 cells complete.** Merged to `Hub-Analysis` at `8ca3cd9`
(originated on `worktree-fraggfn-stage2`). Full narrative in [`Logs/079`](../../../Logs/079_reprice-capped-rows-and-coarse-ladder.md);
this file is the operational summary for anyone comparing against these numbers.

This is the **competitor** half of the benchmark: 6 generators × 3 targets × 3 seeds × 2 pools.
Our own arm (hub-batching / LSD-Flow) is NOT here — it lives under
`$SCRATCH/rgfn_runs/lsdflow_sparrow/bc_sb*` and was not audited in this pass, so do not assume the
two halves share conventions until you have checked.

| | |
|---|---|
| generators | s3gfn, reinvent, saturn, tango, fraggfn, synformer |
| targets | seh, drd2, clpp — **6TD3 is deliberately paused** (`docs/RESEARCH_CONTEXT.md:669`) |
| seeds | 42, 43, 44 |
| pools | naive (top-500 by reward) and pruned (500 mutually distinct) |
| arms | greedy (prices a fixed list) and SPARROW-Batching (chooses the list) |

---

## Re-audit before trusting anything

```bash
python experiments/lsd_hubs/campaign/audit_matrix_coverage.py
```

Prints completeness per stage (pool / routes / greedy@R=100 / SB@R=100). It requires a **readable
R=100 row**, not a file that exists — a coarse ladder can write a CSV containing only its header, and
an "exists and non-empty" check passes that.

---

## Quote rules — each of these has already produced a wrong number

1. **`n_targets_priced`, not `n_modes`.** `n_modes` is the REQUEST, `n_targets_priced` the DELIVERY.
   They are equal on all 88 multiaiz cells and diverge on **all 19 SynFormer cells**, where delivery
   is 73-93% of request. Quoting the request overstates SynFormer by up to 27%.
2. **`cost_kept_rxns`, not `used_rxns`, outside the budget-binding regime.** For a `pool-exhausted`
   cell the unspent budget is not a choice, and the SB arm's `used_rxns` inflates. See CLAUDE.md.
3. **Never quote a `time_capped` row as an optimum.** It is a LOWER bound, so it flatters us. Four
   cells cannot be certified at any gap tried (1e-7, 1e-2, 1e-6); their best incumbents are in
   `$SCRATCH/rgfn_runs/lsdflow_sparrow/results/UNCERTIFIED_R100_BOUNDS.json`. Two of those main CSVs
   UNDERSTATE the competitor — `tango_clpp_44_pruned` holds 8 where 10 is measured,
   `saturn_seh_44` holds 4 where 5 is.
4. **Check the `gap_rel` column.** Re-solved R=100 rows carry `gap_rel = 0.01`; every other row is
   SPARROW's `1e-07`. On 70-92-mode rows a 1% gap is ±1 and immaterial; on a 3-14-mode cell it is
   about a quarter of the answer, and there a relaxed-gap `Optimal` can be BELOW a capped incumbent.
5. **Never mix the two axes** (modes-at-100-reactions vs reactions-for-100-modes) or two ladder
   resolutions in one table. A co-agent already lost a result to the second.

---

## What budget are these cells at? NOT 10,000 oracle calls

Stage 2 equalises **pools** (500 modes), not oracle calls, so the oracle-call axis is uncontrolled —
and uncontrolled in a generator-correlated way. Measured from each cell's
`$SCRATCH/rgfn_runs/stage2/<cell>/upsample_log.json` (`newly_sampled`, on top of Stage 1's 10,048):

| generator | total oracle calls |
|---|---|
| reinvent / saturn / tango | 12,018 - 12,048 |
| fraggfn | 12,047 - 14,046 |
| s3gfn | 10,048 - 18,048 |

Spread **1.80x** (10,048 - 18,048); total Stage-2 surcharge across 45 cells **122,411 calls**. The
largest surcharges are all S3-GFN's — `seh 43` +8,000, `drd2 42` +7,308, and the three ClpP cells
+6,300-6,500 each, all of which STALLED and still did not reach 500 modes.

**So these are not "10,000 oracle call" results and must not be placed in a column that claims to
be.** The honest label is *"arm-A training plus a measured Stage-2 surcharge of 2,000-8,000"*, with
the 1.80x spread and its S3-GFN concentration stated rather than averaged away.

**This is a LABELLING obligation, not a fairness hole — an earlier draft of this file had the
direction backwards.** Within the competitor field the surcharge does tilt things: S3-GFN's edge over
Saturn and TANGO is partly bought with up to 1.8x their oracle calls, and that is worth stating. But
the inference-time oracle axis is uncontrolled for *everyone*, and **we are by far its heavier
user** — our own campaign's per-cell `cum_reward_gen_calls` runs to **447,000-791,503** in the
budget-scale runs (`results/budget_scale_ours_*/curve_hub_batching.csv`), i.e. 50-100x the
competitors' entire Stage-2 surcharge. Equalising this axis is not the fix; the project deliberately
demoted the fixed-MODE readout that equalising would force. Measure it and report it per arm.

**Do not compute the surcharge from `rounds[].asked`.** That field is the CUMULATIVE escalating
request per round (4000, then 8000, then 12000), so summing it double-counts — it reports 24,000 for
a cell whose distinct set grew by 1,216. Use the top-level `newly_sampled`; `free_from_training` and
`distinct_scored` are the other honest fields.

**ONE CELL SPENT ONLY 6,950 OF ITS 10,000 CALLS: `synformer_drd2_seed43`.** Its siblings' traces
hold 10,000 rows; this one holds 6,950. It was NOT killed — the run ended normally after 29.72 h
(`[SF-FR] optimization done in 106973.4s (6950 scored, 6955 routed)`) with the last two generations
adding zero new scored molecules (`gen 80: 6950/10000`, `gen 81: 6950/10000`). The search exhausted
before the budget did, which is the same shape as Saturn's mode collapse and is a RESULT, not a
defect — but it means that cell sits at a **1.44x smaller oracle budget** than the rest of its row
and must be labelled, not averaged in. It still emitted a full 2,000-row `candidates.csv`, so every
downstream check passed it; the only trace is the row count.

*Beware the log.* `ch_sf_drd2_43-74719.out` contains an `oom-kill` block naming `job_74716` — that is
`ch_sf_seh_43`, a DIFFERENT cell which TIMEOUT'd at 3 days; node-level dmesg spilled into a
neighbouring job's log. Reading that OOM as this cell's cause is wrong, and it was my first
hypothesis. The superseding sEH cells (jobs 75753-55) are clean at 10,000 trace rows each.

**A second budget is UNAVAILABLE on the competitor side, on a measured basis.** SynFormer's three
sEH cells trained at 10,048 oracle calls in **19.30 h / 20.70 h / 20.17 h** wall-clock
(`timing.json`: top-level **`total_s`** = 69,489 / 74,502 / 72,605; jobs 75753-55, 10,000 trace rows
each). **Do NOT read these cells' `unaccounted_s: 0.0` as a passing audit** — an earlier draft did.
`phases` holds exactly ONE key (`total_run_s`) on all nine SynFormer cells, so `unaccounted_s` is
0.0 by construction and carries no information: a complete-looking breakdown of a single bucket.
What makes the wall-clock trustworthy here is different and weaker — `total_s` is a directly measured
job runtime, and the **trace row count is an independent witness** that the work happened (10,000
rows on eight cells, 6,950 on `drd2/43`, see below). `unaccounted_s` is only a detector where
`phases` has more than one entry to sum. NOTE the field name —
`total_run_s` is a PHASE key *inside* `phases`, not a top-level field, and an earlier draft of
this file cited it as though it were. Timing schemas are not uniform across this project: a
sweep of the v2 cells found `train_s` recorded as 0.0 on 9 of them (indistinguishable from
"training was skipped"), absent entirely on 24 more, and one cell with 99.5% of its runtime in
`unaccounted_s`. **No figure in this document is sourced from `train_s`**; the oracle counts
come from traces and `upsample_log.json`, and the wall-clock from `total_s` above, all of which
have `unaccounted_s` 0.0. `unaccounted_s` is the cheap detector if you extend this table.
Linear in calls, 320,000 would be ~615 h per cell — **~5,530 GPU-h for its nine cells alone**.
Saturn and TANGO are the same shape. So an iteration-matched external comparison at a second budget
is not merely expensive, it is out of reach; a budget-dependence check has to run on generators we
can actually train twice.

**THE PER-PHASE BREAKDOWN EXISTS — IT IS IN THE LOGS, NOT THE FILE.** `timing.json` holds one
bucket, which led to a conclusion (mine and a peer's) that any cross-generator per-phase cost table
must have a SynFormer-shaped hole. That is wrong. `run_synformer_fixed.py` computes
`ga_s`/`sanitize_s`/`project_s`/`score_s`/`gen_wall_s` per generation with an explicit residual and
PRINTS them as `[SF-TIME]` lines; the `timing.json` write records only `{"total_run_s": ...}`. The
hole is in the file, not the data — recover it with
`grep '\[SF-TIME\]' <job>.out` (117 / 110 / 112 generations on the three sEH cells).

Summed over those generations, **SynFormer is projector-bound, not oracle-bound**:

| phase | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| `project_s` | 68,174 s (**99.0%**) | 72,878 s (**99.1%**) | 71,144 s (**99.1%**) |
| `score_s` (the oracle) | 477 s (0.7%) | 504 s (0.6%) | 454 s (0.6%) |
| `ga_s` | 185 s (0.3%) | 150 s (0.2%) | 177 s (0.2%) |
| residual | 0 s | 0 s | 0 s |

So its 19-20 h per cell is ~99% mapping molecules into synthesizable space, and the 10,000 oracle
calls cost about **8 minutes**. That also says what scaling to a second budget would actually buy
time in — the projector, not the oracle.

*`total_s` is RUN time, not cell cost.* `run_t0` is set at `run_synformer_fixed.py:640`, after the
projector is constructed, so setup is outside it. The same pattern was found in
`run_s3gfn_fixed.py` and `run_reinvent_fixed.py`, and since `run_t0` is the only total in the file,
no field there answers "what did this cell cost to produce". The figures above are correct as RUN
time, which is how they are labelled.

*Write durations with explicit units.* On a cluster where jobs legitimately run for both twenty
minutes and twenty hours, `20:10` is ambiguous and the failure is silent — a reader can "correct" a
right number to a wrong one and be confident about it. This nearly happened on these very figures.

**Consequence for any budget-matched comparison:** the redundancy and route-sharing mechanisms below
are properties of the CONVERGED pool, so both are budget-dependent in principle. Nothing here has
been measured at a second budget. If you are comparing against a run at a different oracle budget,
measure the greedy→SB uplift on both rather than assuming it transfers — it is free, since both arms
already run for every cell.

---

## Reading the results tree

Root: `$SCRATCH/rgfn_runs/lsdflow_sparrow/`

| path | holds |
|---|---|
| `multiaiz_pools/<tag>_N*/` | `pool.smi`, `pool_scores.csv`, `multiaiz_routes.json`, `discovery_timing.json` |
| `results/<tag>_saturation/summary.json` | the pre-flight: `n_distinct_above_gate`, `modes_in_largest_prefix` (top-500 prefix, capped at 500 — NOT a headroom measure), `modes_available_whole_set` (what the pruned pool draws from) |
| `results/<tag>_greedy_N*/greedy_frontier.csv` | greedy arm, priced on the MODE axis |
| `results/<tag>_select_N*/select_frontier.csv` | SPARROW arm, priced on the REACTION axis |
| `results/UNCERTIFIED_R100_BOUNDS.json` | the four cells that cannot be certified |
| `greedy_csv_backup_20260906.tar.gz` | 330 greedy CSVs as they were before the re-pricing |

**`<tag>` = `<gen>_<target>_seed<N><suffix>[_pruned]`.** The suffix is `_stage2` for most cells, `""`
for SynFormer (no Stage 2), and `_stage2fix` for s3gfn drd2 seeds 43/44 — which marks **which cells
stalled in Stage 2**, not a second pipeline. Seed 42 reads `target-reached` and needed no re-run.

**LEGACY PRE-STANDARD POOLS SHARE THIS DIRECTORY — filter on the tag, not just the target.**
`multiaiz_pools/` also holds pools built before Stage 2 and before the ClpP gate was recalibrated,
and they are tagged WITHOUT `_stage2`. Their ClpP scores top out at exactly **-8.0**, the
pre-standard bar, so only **13-37%** of each clears the current **-9.1** gate. A glob like
`*clpp*` picks them up beside the real cells and they will look like badly-performing matrix cells
rather than correctly-performing old ones. Restricting to `_stage2` (plus bare `synformer_`) gives
109 pools, all of which are varied, unclamped, and on the correct side of their gate.

**Glob the size, never assume `_N500`.** A pool-limited cell writes `_N<actual>`
(`synformer_drd2_seed44_pruned` is `_N495`), and a hard-coded `_N500` reports a complete cell as
missing.

---

## The two cells that are not gaps

`s3gfn:drd2:42/naive` and `saturn:seh:42/naive` have 2-byte route files and empty result directories.
That is a **measured zero, not a crash**: `discovery_timing.json` records 500 targets over 4.9 h and
10.3 h at 35 and 74 s/target with `returncode 0`. Their reward-ranked top-500 is entirely
unsynthesizable while the same cell's diverse 500 prices 47 modes. Report them; do not re-run them.

---

## What the matrix says, in one table

Modes at R=100, seeds 42/43/44, delivery-quoted. DRD2 omitted — S3-GFN routes almost nothing there
([075] traced it to an amine absent from our ZINC stock).

| target | GREEDY naive | GREEDY pruned | SB naive | SB pruned |
|---|---|---|---|---|
| **sEH** S3-GFN | 45/50/75 | 40/60/70 | 56/38/49 | **98/100/100** |
| **sEH** REINVENT | 30/30/35 | 30/30/35 | **61/47/83** | 80/70/92 |
| **ClpP** S3-GFN | 40/40/40 | 40/40/45 | 45/43/42 | 64/57/62 |
| **ClpP** REINVENT | 30/35/30 | 30/35/30 | **81\*/69/71\*** | **86/74\*/72\*** |

`*` = still capped, i.e. a lower bound, so certifying can only raise REINVENT.

**The SELECTOR decides the winner, not the pool.** Greedy favours S3-GFN in 4 of 4 pool×target
combinations with disjoint three-seed bands; SPARROW favours REINVENT in 3 of 4 — all but sEH/pruned,
which is exactly where S3-GFN sits at the 1.0 rxn/mode arithmetic ceiling and cannot be beaten. So
the arm CLAUDE.md designates to carry the headline is the arm the competitor usually wins on. Say it
plainly rather than quietly preferring greedy.

Two measured mechanisms sit under that:
* **pool axis — redundancy.** Share of a reward-ranked top-500 that is mutually dissimilar:
  Saturn/TANGO 2-4%, S3-GFN 39-58%, REINVENT 59-87%, FragGFN 87-100%. The pruning gain is close to
  the ratio of distinct molecules in the two pools (median deviation 5%), so it is largely
  arithmetic, not a subtle interaction.
* **selector axis — route sharing.** Mean greedy→SB uplift: REINVENT 1.86×/2.57× (naive/pruned),
  S3-GFN 1.01×/1.65×, FragGFN ~1.2×, Saturn 0.17×/0.36×, TANGO 0.27×/0.22×. Below 1.0× is not a
  defect: on a collapsed pool SPARROW pursues its own objective and does not chase mode count.

---

## Traps that already cost time here

| trap | looked like | detector |
|---|---|---|
| coarse mode ladder | a low number, or a **header-only CSV** when the pool cannot reach the first rung | first rung in the CSV; require a readable R=100 row |
| two-arm launcher | an SB-only backfill silently re-priced greedy back to coarse **hours later**; both jobs logged success | audit first rungs tree-wide, not job logs. `RUN_GREEDY=0` now prevents it |
| `_longsolve` split | a certified row existed but the main CSV still held the capped one — my own audit requoted the stale value | merged via `merge_longsolve_rows.py`; prefer the merged file |
| pool rebuild mid-flight | the chain and the native launcher REBUILD pools, and `build_s3gfn_pools.py` has since gained the NaN gate | re-price with `submit_reprice_cached.sh`, which reads cached artifacts and refuses if they are absent |

---

## Tools added in this pass

| script | does |
|---|---|
| `audit_matrix_coverage.py` | completeness per stage across all 108 cells |
| `submit_reprice_cached.sh` | re-price from CACHED routes, changing one solver knob. `ARM=sb` re-solves a capped budget row, `ARM=greedy` re-prices on a ladder; `ROUTE_SOURCE=external` for SynFormer. Refuses rather than rediscovering |
| `merge_longsolve_rows.py` | merges certified R=100 rows into the main CSV and appends `gap_rel`. Refuses a still-capped row, and a changed mode count without `--allow-change` |
| `write_uncertified_bounds.py` | regenerates `UNCERTIFIED_R100_BOUNDS.json` |
| `_resolve_gate.py` | prints `<target> <gate> higher\|lower` from `matrix16/targets.py` so no launcher defaults the 5%-FPR bar |

Both launchers now default to the fine ladder
(`2,5,…,90,100,125,150`). The ladder is still 25-spaced **above 100**; no cell lands there today, but
a cell delivering ~110 modes would report 100.
