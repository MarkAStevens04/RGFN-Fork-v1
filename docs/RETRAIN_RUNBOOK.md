# `benchmark_v2` — specification for the clean re-run

**What this is.** The plan for regenerating the whole benchmark as one internally consistent set of
experiments, in one new tree (`experiments/benchmark_v2/`), under standards that did not exist when
the current results were produced. It is **not** a patch list. Where it is unclear whether a result
can be carried forward, the default is to re-run it.

**Why it exists.** The current results are correct but were assembled over three months during which
the taxonomy, the training budget, the hit gates, the pipeline shape and the route contract all
changed. Nothing is *wrong*; several things are no longer *commensurable*, and a journal reviewer
reads a benchmark's internal consistency as a proxy for its trustworthiness.

**When it runs.** After the workshop submission. It is deliberately not on a deadline: the point is a
clean foundation, not speed. Large GPU-hour totals are acceptable; avoidable ones are not.

**What it is for.** A backup body of evidence if the workshop/ICLR route does not land, and the
substrate for the journal (NCS) submission — including a chemist-facing route dataset that the
current results cannot support.

**Read first:** `docs/RESEARCH_CONTEXT.md` (definitions, the two pools, the logging spec) and
`experiments/benchmark_v2/README.md` (the operational twin of this document — what lives where).

---

## 0. What changed, and therefore what cannot be carried forward

Six decisions post-date most of the results on disk. Each one is a reason a cell has to be rebuilt,
and together they are why this is a re-run rather than a repair.

| # | Change | Date | Consequence |
|---|---|---|---|
| 1 | **Generator taxonomy.** FragGFN is a *non-reaction* GFlowNet and belongs with S3-GFN, not with our reaction-grounded generators | 2026-08-28 | The "16-cell matrix" is dissolved. Hub-batching is built on **three** generators (RGFN, RxnFlow, SCENT); FragGFN moves to the competitor block and needs the full route-less pipeline it has never had |
| 2 | **Training budget standardised** to the PMO convention (~10,000 oracle calls), with each generator at **its own paper's batch size** | 2026-08-21 → 08-28 | Our three generators ran at 320,000–640,000 calls, i.e. **32–64× the competitors**. Every one of our trained cells is superseded. FragGFN's old runs are ~30× over budget (`a327c3a`) |
| 3 | **Hit gates re-derived** on one rule: the score at which **5% of that target's property-matched decoys pass** | 2026-08-21 | sEH 7.0→**5.68**, DRD2 0.5→**0.345**, ClpP −8.0→**−9.10**, 6TD3-B **6.718**. Free for our side (gate is applied post-hoc, on CPU); **not** free for competitors — it changes pool composition, and therefore retrosynthesis and selection |
| 4 | **A fourth competitor stage: upsample-and-filter** (`upsample_to_modes.py`, `dd8f1a9`) | 2026-08-28 | A competitor "pool" is no longer a slice of a fixed 2,000-molecule sample. `pool-limited` used to conflate *the generator cannot* with *we did not ask for enough* |
| 5 | **Route contract enforced at write time**; RGFN/RxnFlow route emission implemented | 2026-08-24 | A fresh sample now produces `routes.json`. The route dataset can go from **5 cell-seeds to matrix-wide** as a by-product of re-running — see §6 |
| 6 | **6TD3-B** replaces the exploitable Tier2−Tier1 differential as the CDK12–DDB1 reward — reward *and* gate are `cnn_vs` (CNNscore × CNNaffinity) at **6.718** | 2026-08-21, settled 08-28 | Every 6TD3 cell of every generator is **INVALIDATED, not superseded** — the distinction is load-bearing. Row 2 supersedes: the same quantity, re-measured under a new budget, so a newer number will exist. Row 6 REDEFINES the quantity: 6TD3-B numbers are not a better measurement of old 6TD3, they measure something else, so **no newer measurement of the old quantity exists or ever will**. Calling it superseded implies a replacement number is out there and sends a reader looking for one. The old libraries do not survive re-gating. Oracle wiring in progress; see §7.1 for the threshold and the evidence |

### The one thing that survives unchanged

**Oracle calibration and hardware measurements are properties of oracles, molecule sets and GPUs, not
of any generator run.** They are copied, not re-run: the four oracle validations (Logs/034, 045, 066,
069), the pose-selection ablation (008), docking throughput and batch-size tuning (036, 057),
determinism findings (`PYTHONHASHSEED`, REINVENT nondeterminism, DRD2 env-invariance), the
baseline-config audit, TANGO's inventory and `num_top_results` measurements, and the AiZynth
stock-mismatch results (047, 048).

---

## 1. The grid

**Authoritative list: `experiments/benchmark_v2/grid.csv`** (108 training cells). This table is the
summary; the CSV is what drivers read.

|  | **GFlowNet** | **not a GFlowNet** |
|---|---|---|
| **reaction-grounded** | **RGFN, RxnFlow, SCENT** — hub-batching applies | SynFormer |
| **not reaction-grounded** | FragGFN, S3-GFN | REINVENT, Saturn, TANGO |

9 generators × 4 targets × 3 seeds (42/43/44) = **108 training cells**; 81 in phase 1, 27 in phase 2.

**Phase 1 — sEH, DRD2, ClpP.** Everything. **Phase 2 — 6TD3-B.** Reward and gate are settled
(§7.1); it starts once the oracle is wired, and is explicitly lower priority than the other three
targets, for every generator.

### Two arms, both defined on the oracle-call axis

Never on the step axis. Three generators have three different per-step call counts (RGFN 100,
SCENT 64, RxnFlow 64 at the authors' `num_from_policy`), and replay buffers make the arithmetic
unsettleable. **The trace counter decides, not multiplication.**

**And at arm B the counter must survive a PROCESS boundary.** `TraceWriter.__init__` sets
`n_scored = 0` and `n_train_scored = 0` fresh, then rotates any existing `trace.csv` to `trace.csv.N`
— **with no resume seed** (`_trace.py:78,83,95-99`). SCENT's arm B requeues two or three times on the
docking targets, so a stop reading the LIVE counter would restart from zero on each requeue and train
**640,000–960,000 calls against a declared 320,000**, with every artifact looking healthy. The arm-B
stop must therefore **COUNT `phase == "train"` rows across `trace.csv` AND its `.N` siblings** — which
the module's own comment already prescribes ("readers that want the FULL history across rounds should
concatenate trace.csv with its .N siblings"). This is the cumulative-counter trap in its third place,
after `asked` and `n_scored` (§8.2); here it crosses a process boundary rather than a phase or a round.

**THE READING SIDE HAS THE SAME TWIN, and the right answer is not "sum them".** `best_trace`
(`inventory_v1_cells.py:107`) takes the sibling with the MOST ROWS, and `write_arm_meta` /
`verify_cell` gate on it — so a requeued arm-B cell would record ONE round's budget and fail its own
budget gate while being genuinely complete. But blind summing is worse, because rotations come in
three shapes and only one of them sums. Surveyed across all 24 v1 cells that carry siblings:

| shape | what it is | cells | rule |
|---|---|---|---|
| **alternatives** | a re-invocation truncated `trace.csv` to a stub; the real history is in `.1` | 11 (s3gfn, fraggfn_drd2) | **max** |
| **duplicates** | a `.preunshape` rename backup, identical row counts | 9 (reinvent/saturn/tango clpp) | **max** |
| **superseded partials** | an earlier incomplete attempt beside a complete one (10,000 beside 5,384–6,867) | 4 (synformer) | **max** — summing gives 15,384–16,867 |
| **continuations** | disjoint contiguous `step` ranges, two halves of ONE run | **0 in v1**; arises only when v2 arm B requeues | **sum** |

So **max is correct for every v1 cell**, and spot-checking six copied cells confirms `copy_forward`
took the right file in each (10,000 / 10,048 / 10,024 as expected, never a sum). No existing cell is
short or inflated. The continuation rule is needed **prospectively only**, for v2 arm-B requeues.

**But the arm-B STOP does not need the discriminator at all** — `copy_forward` collapses v1's
alternatives on the way in (it resolves the rotation and lands the real history as `trace.csv`, which
is why no v2 cell has siblings). So a `trace.csv.N` inside a v2 TRAINING directory can only have come
from a v2 requeue, which is always a continuation. Inside a training run, SUM. Stage 2's re-invocation
does create an alternatives-shaped rotation, but that happens after the cell is frozen and writes to
its own directory, so the stop never sees it.

**Implemented 2026-09-12 as `best_trace(d, combine=...)`**, defaulting to `"max"` so every v1 path is
bit-identical, with `write_arm_meta` passing `"sum"` because it runs inside the training run. Three
details that are not obvious:

* **`n_distinct` is NOT summable** — it is a per-round dedup, so adding two rounds double-counts every
  molecule seen in both. It is recomputed as a true union over SMILES.
* **The caller declares the shape; the function does not guess.** An auto-detector that silently picks
  wrong is the failure this family keeps producing.
* **Asking to sum files whose `step` ranges overlap RAISES** rather than double-counting — the check
  that can fail (§6.10). Where steps are blank (SynFormer writes none) the overlap check cannot run,
  so the result carries `sum_verified: false` rather than choosing silently: legitimate for an arm-A
  requeue, wrong for a superseded partial, and indistinguishable from inside the function.

**Consequence for scheduling, since a call budget is not an iteration budget:** at a true 320,000
calls RGFN needs ~2,650 iterations, SCENT 5,000, RxnFlow ~10,300 — so RGFN roughly HALVES and RxnFlow
roughly DOUBLES against an iteration-matched run. Iteration-matching would instead give ~603k / 320k /
~155k, a **4× spread on the very axis this campaign claims to control**, which is why the arms are
defined on calls.

| arm | budget | who | purpose |
|---|---|---|---|
| **A** | **10,000 oracle calls** | all 9 generators | the PMO convention the competitors' own papers use |
| **B** | **320,000 oracle calls** | the 3 reaction-GFNs | **SCENT's own published protocol** — 64 forward trajectories × 5,000 iterations |

**Each exhibit uses the arm that makes it valid (decided 2026-09-08). Arm B's downstream is NOT
paused.**

| exhibit | arm | why |
|---|---|---|
| **external head-to-head** (us vs 6 competitors) | **A** | budget parity is what makes a cross-generator claim fair |
| **internal matrix** (hub-batching vs best-candidate) | **B** | within-generator on an identical pool, so parity was never needed — and at arm A one of our three generators is not itself (§7.1, Logs/078) |
| **compute↔reactions tradeoff** (new) | **A vs B**, our generators | prices GPU-hours against bench reactions: *"X more GPU-hours buys Y fewer reactions"*. Carries the composability control below |
| **route dataset** | **B** | SCENT promotes nothing at arm A, so its arm-A routes bottom out at stock and lose the nested promoted-fragment case the schema is built around |

**Why this is not "short vs long" but two literatures.** SCENT's paper contains the word "oracle"
**zero** times in 23 pages — it normalises on iterations, and explicitly matched RGFN ("we changed the
number of sampled forward trajectories to 64") and SynFlowNet ("64 forward trajectories and 5000
iterations") to its own 64 × 5,000 = 320,000. RGFN normalises on *training time*. S3-GFN is the PMO
paper ("oracle budget is limited to 10K"). So the reaction-GFN and PMO literatures normalise **32×
apart**, and this benchmark is the first thing to span both. Report under both and say so.

**Arm B is not available to the competitors, on a measured basis.** SynFormer trains at 10,048 calls
in 19.30 / 20.70 / 20.17 h (`timing.json`, `total_run_s` 69,489 / 74,502 / 72,605), so 320,000 would
be ~615 h/cell and ~5,530 GPU-h for its nine cells. Saturn and TANGO are the same shape. An
iteration-matched external comparison is therefore *unavailable*, not merely expensive.

**One training run yields both arms.** Train to 320,000 calls and checkpoint on the way past 10,000.
That *is* the continuation, on one trajectory, at zero extra training compute — and it sidesteps
reproducibility entirely, which matters because the docking reward is genuinely stochastic, so two
runs at the same seed would **not** agree on ClpP or 6TD3-B.

### The composability control: run HB-Enum-SB at BOTH arms

**This is a control, not an exhibit.** It is not a study of whether SPARROW beats greedy; it exists
so the arm-A external result and the arm-B internal matrix can be *discussed together* without an
unstated assumption between them.

**The risk it closes.** The two exhibits are not composed arithmetically, but the paper narrates them
together, and a reader will infer that the internal advantage explains the external win. That
inference needs the selector effect to be budget-independent, and nobody has checked. There is reason
to doubt it: the greedy→SB uplift depends on how much route sharing a pool offers, and a model
trained 32× longer converges somewhere with different sharing. The competitor side already shows the
uplift is strongly generator-dependent (REINVENT 1.86–2.57×, S3-GFN 1.01–1.65×) and that it
**reverses the ranking** — greedy favours S3-GFN 4 of 4 target×pool combos, SB favours REINVENT 3 of 4
— while `CLAUDE.md` designates SB to carry the headline.

**It is nearly free, because the experiment already exists.** `submit_hb_enum_sb.sh` hands SPARROW
hub-batching's own enumerated candidates — same optimizer, same gate, same budget, only the chooser
differs. That IS the greedy-vs-SB ratio on our pools; Logs/062 measured it at **2.27× (n=3, per-seed
1.67 / 2.03 / 3.13)**. The marginal ask is to run it at both arms rather than one.

**RUN IT ON DRD2, NOT sEH — the MILP does not converge on our sEH pools.** On the corrected uncapped
enumerations (123k–150k above gate) a 12 h cap and `gapRel` 1e-3 both returned TimeLimit, and the
capped answers do not trend (12 / 27 / 37 distinct across three pool sizes, no ordering), so sEH is
reportable only as a lower bound. DRD2 converged on all six budget points (Logs/062, jobs
74727/74728) — but note it was capped at `--top-n 50000` to match its competitor rows, so the DRD2
MILP saw ~⅓ the variables. Expect arm A to converge more readily than arm B, since a
less-trained model yields a smaller eligible pool (B measured 7,111 hubs at arm A against v1's
20,874) — which is the wrong direction for a paired comparison, so **check `time_capped` on every row
and do not form a ratio across one converged and one truncated arm.**

**What it can and cannot say.** It measures budget-sensitivity of the selector effect on *our* pools.
It cannot test the cross-generator ranking's budget-stability — that would need the competitors at
arm B, which is unavailable (§1). Converting an unstated assumption into a measured one is the whole
of the win.

### Resume, per generator — and why SCENT's arm B must not requeue

Verified from the checkpoint code 2026-09-07 (agent A), not from comments.

| generator | what the checkpoint holds | resumable? |
|---|---|---|
| **RxnFlow** | model + **both** optimizers + **both** LR schedulers + step (`al_loop.save_checkpoint`) | **yes**, fully |
| **RGFN** | model + optimizer + lr_scheduler + metrics + replay buffer | **yes, but only after a cache strip** |
| **SCENT** | the same five — and **not the dynamic fragment library** | **partially, and it silently corrupts** |

**A second, independent reason SCENT's arm B must not requeue (added 2026-09-08).** The RNG is not
checkpointed, so a resumed segment samples differently and promotes different fragments. Since
`7134642` a requeued run is *correct* — the library, its costs, the path-cost cache and the recipe
routes all restore — but it is **not reproducible**: it will not promote what an uninterrupted run
would. Do not relax the one-walltime rule on the grounds that the library is now safe; that fixes
correctness, not reproducibility.

**RGFN needs a cache strip.** Its forward policy carries lazily-populated `*_cache` buffers that a
freshly constructed model does not have, so a strict `load_state_dict` against a raw checkpoint
*fails* (Logs/021). The production chain works around it at resume time
(`experiments/fixed_reward/scale5k/submit_rgfn.sh` strips every `model` key ending in `_cache`), which
means **a raw RGFN checkpoint is not by itself resumable**. The caches are 78% of the file — 404 MB →
90 MB on `rgfn_seh_5k/seed42`. The arm-A checkpoint is therefore written **already stripped**; it is
the one checkpoint whose entire purpose is to be resumed from.

**SCENT's dynamic library is not in the checkpoint, and the damage is worse than a smaller library.**
`external/scent/rgfn/trainer/trainer.py:281-287` saves `model / optimizer / lr_scheduler / metrics /
replay_buffer`; the promoted-fragment library is only ever serialized to
`additional_fragments/fragments_<N>.json` for analysis and is never restored. So a resumed SCENT run
restarts with an **empty** library. v1 shows this happened: `scent_6td3_5k/seed42` progressed
400/800/**400**/800 (two resets, final 800) and `scent_clpp_5k/seed43,44` 400/**400**/800/1200
(one reset, final 1200), against a clean 1,600 for the cells that ran straight through — so the final
library size in v1 is a function of **requeue timing**, not of the science.

The second half is the nastier one, and it is silent. `FragmentOneHotEmbedding.weights` is
pre-allocated to `418 + max_additional` rows and `_get_embeddings()` returns `weights[:
current_fragments]` — **indexed by position**, while `on_update_fragments_library` updates only the
*count*. After a reset, a re-promoted fragment therefore inherits the **trained embedding row of the
slot's previous occupant**. Nothing crashes; the content-based `all_fingerprints` rebuilds correctly,
so only the one-hot half carries stale identity.

**Three consequences for this campaign:**

1. **Train one process straight through to arm B, checkpointing at arm A on the way past.** Already
   the plan above — but it is now load-bearing for a second reason, not just continuation semantics.
2. **"Train the cheap 10,000-call arm everywhere, extend later" is available for RGFN and RxnFlow
   only.** For SCENT the extension would not continue the same library trajectory.
3. **A SCENT arm-B run must finish inside one walltime.** Arm B is 320,000 calls = 5,000 iterations,
   SLURM `compute` caps at 3 days, and v1 proves SCENT 5k does *not* fit on at least some targets. A
   requeued SCENT arm-B cell is confounded and must be discarded, not repaired.

**FIXED 2026-09-07 — the library now persists across a requeue** (`validation/generators/scent/
library_io.py`, a sidecar beside `last_gfn.pt`, on the `guidance_io.py` precedent; the clone stays
untouched). The July audit declined this because it changes training behaviour mid-campaign; on a
clean slate that objection expires. Four things worth carrying, because they change what the fix is:

* **The trained rows were never lost, only made invisible.** `weights` is an `nn.Parameter`, so
  `model.load_state_dict` restores all 4,418 rows; it is `current_fragments` resetting to 418 that
  hides everything past it. So this restores *bookkeeping*, not a second weight sidecar.
* **It restores SCENT's design rather than deviating from it.** The one-hot table is sized
  `418 + num_additions × n_new_fragments` = **4,418** — built for a complete, never-reset run. The
  reset is an artifact of our 3-day requeue.
* **Three pieces of state, not two** — the ordered fragment list, the embedding rows, *and* the
  path-cost cache entry per fragment. The third was found by crashing into it: `assign_costs` only
  fills `molecule_num_reaction_to_cost` for states reached through a reaction action, so once the
  restore puts promoted fragments back in the action space the policy can select one as a *starting*
  block and hit a `None` the clone does not guard (it guards the identical lookup two lines away
  with `or inf`, so this is an upstream gap we compensate for, not a choice). The restore re-seeds
  the true cost, never `inf` — that value is what the next promotion records for the fragment.
* **A resumed run with the library restored but the cache cold CRASHES.** v1 never hit this only
  because the library was empty on resume — *the reset was accidentally what kept the crash away.*
  Anyone who fixes the reset without the cache seeding gets a run that dies at the first promotion
  after a resume. It is also the sharpest argument for the verification bar below: a count check
  never even runs, because the process is already dead.

**Verification: `validation/generators/scent/verify_library_recovery.py`.** Uninterrupted vs
killed-and-resumed at the same iteration, compared on a sha1 of the *ordered* `chosen_smiles` **and**
a sha1 of `weights[:current_fragments]` — a count check passes on precisely the bug being fixed —
plus a blind attribute-by-attribute diff of the library, the cost proxy and the replay buffer, so the
restore inventory is shown complete rather than merely sufficient for the cases we hit.

⛔ **v2's SCENT is deliberately NOT bit-comparable to v1's.** This changes training behaviour. Do not
diff a v2 SCENT cell against its v1 counterpart and report a regression.

### What "arm A" precisely means

Two details that a later reader will otherwise simplify away, both in
`validation/generators/_trace.py`:

* **The checkpoint fires at the first iteration BOUNDARY at or after 10,000 calls**, not at the
  10,000th row. A GFlowNet scores a whole minibatch inside one iteration, so the 10,000th call lands
  mid-iteration where no coherent optimizer/replay state exists to check-point. Overshoot is bounded
  by one batch (64–100 molecules, ≤1%), and `arm_a.json` records **both** numbers:
  `crossed_at_n_scored` is the truth, `saved_at_iteration` is where the weights are.
* **`step` is stamped from `on_start_sampling`, the budget checked from `on_end_sampling`.** Reward
  evaluation happens *during* sampling, so reading the step at the end would label every row with the
  previous iteration — off by one for the entire file.
* **`phase` is load-bearing.** RGFN and SCENT score the final candidate batch through the *same*
  proxy the training loop uses, so those rows are flipped to `phase="eval"` before sampling. Without
  it a smoke measured 600 training calls followed by 185 scoring calls, indistinguishable — and every
  budget or modes-vs-calls reading would count the second set. **Filter to `phase == "train"` for any
  budget claim** — and count ROWS, never the cumulative `n_scored`, which absorbs eval calls as it
  goes (measured on `s3gfn_drd2/42`: the same file reads 12,048 / 11,048 / 10,048 depending on which
  you take).
* **SCENT's periodic validation was inflating its own training count 2.84×**, and this is a measured
  figure rather than a rounding concern. Its config sets `valid_sampler = RandomSampler` with
  `valid_n_trajectories = 1000`, so every validation pass scores a thousand molecules through the
  same proxy — and they land labelled `train`, because `TrainingHooksMixin` has no validation hook.
  Per-iteration train rows in a 6-iteration smoke read 63/94/96/95/96/**1088**. The arm-A budget now
  gates on training-phase rows only, and `attach_proxy_trace` wraps the trainer's own `valid_step` to
  flip the phase for its duration; after the fix that iteration reads **96**.

### Batch sizes: each paper's own

| generator | batch | source |
|---|---|---|
| RGFN | 100/step | `configs/rgfn_base.gin` `train_forward_n_trajectories` (upstream, pristine) |
| SCENT | 64/step | `configs/scent_base.gin` (clone default) |
| RxnFlow | 64/step | `external/RxnFlow/src/gflownet/algo/config.py:187` `num_from_policy: int = 64`. The runner never overrides it, so it already inherits the authors' value |

**CORRECTION (2026-09-07, agent A).** An earlier draft of this runbook said RxnFlow ran at 2× its
authors' batch and had to be fixed. **That was wrong, and no config change is needed.** The `128` is
`reward.batch_size` in `rxnflow_*_5k.yaml`, which is passed to `SEHFrozenReward(batch_size=...)` —
the sEH proxy's *scoring* batch, a pure throughput knob with no effect on the training budget. The
*policy* batch is `algo.num_from_policy`, which `run_rxnflow_fixed.py` never sets, so it inherits the
authors' 64. FragGFN's own configs already record the same conclusion for the shared Recursion
codebase ("The batch was already right"), and `validation/generators/fraggfn/task.py:104` pins it
explicitly. Two different knobs, one name — check which one a number refers to before calling it an
error. The step count still follows from the budget, not the other way round.

---

## 2. Standards — the things that make cells comparable

Every one of these is a property a cell must have, checkable, not a convention to remember.

### 2.1 Hit gates — 5% FPR, and do not round

| target | column | gate | TPR | FPR | enrichment | AUROC |
|---|---|---|---|---|---|---|
| sEH | proxy value | **5.680** | 13% | 5.0% | 2.5× | 0.676 |
| DRD2 | activity probability | **0.345** | 74% | 5.0% | 14.9× | 0.949 |
| ClpP | raw Vina | **−9.100** | 47% | 5.0% | 9.5× | 0.895 |
| 6TD3-B | **`cnn_vs`** (CNNscore × CNNaffinity) | **6.718** | 78% | 4.4% | 17.8× | 0.917 |

Source of truth `experiments/lsd_hubs/matrix16/targets.py`; re-derive with
`experiments/oracle_validation/calibrate_gates.py` (verified 2026-08-28 to reproduce all four
exactly). **These are empirical grid points — rounding breaks the exact-FPR property they are defined
by.** Resolve a gate by importing `targets.py`, the way `upsample_to_modes.gate_for()` does; never
hardcode one, and never give one a default (§7 lists the places that still do).

**Carry this caveat into the paper:** sEH cannot reach a good operating point at *any* threshold
(AUROC 0.676, enrichment 2.4–2.7× across the whole range). That does not invalidate the benchmark —
all arms share the gate — but "sEH modes" is a weaker claim than "DRD2 modes", and the two must never
be averaged into one number without saying so.

### 2.2 Hub-batching configuration — one algorithm across all three generators

**`--pool all --child-policy free_frag --prebuild-k 0`.**

#### `--pool all` — no reward pre-filter on the hub set (decided 2026-08-28)

v1 ranked flow only over the parents of the top-`TOPK` candidates by reward (`TOPK=1000` in
production). That is a **reward pre-filter**, and `pick_hubs.py`'s own docstring describes it as
legacy continuity with the original Logs/028 recipe, not as a justified choice — `--pool all` was
added specifically so "does the flow signal buy anything?" is answerable.

**It is free to remove.** The flow estimate is read straight off the record's log-terms
(`logR + logP_B − logP_F(move) − logP_F(stop)`) — no model, no scoring. Measured on `scent_seh`
seed 42: **0.17 s over 575 filtered hubs, 0.17 s over all 20,874.** Enumeration cost is set by
`--n-hubs`, identical either way.

**It changes 60% of the walked hub set and ~1 mode of outcome.** Overlap between the two
flow-descending orders is **40% across the first 5/10/20/40 hubs** (46% at 200); three of the
unfiltered top-10 hubs are absent from the filtered list entirely. Yet Logs/053 measured, on
identical everything else:

| arm | pool | modes @ R=100 | reactions | rxn/mode | reward-gen calls | compute |
|---|---|---|---|---|---|---|
| `incumbent` | reward-filtered | 72 | 365 | 1.217 | 241,158 | 5,685 s |
| **`flow_top`** | **all** | **73** | **357** | **1.190** | 244,685 | 5,648 s |

Swap 60% of the hubs, move one mode. That is "flow finds the neighbourhood, not the rank" arriving
from a third direction, and it is the reason the switch is safe.

**Why do it:** it removes a confound from our own ordering ablation — with the pre-filter, "flow
ordering wins" is measured on a pool that was *itself reward-selected*, so a reviewer can say the
reward filter did the selecting and flow only sorted the survivors. It also deletes an arbitrary knob
(`TOPK`) from the method description.

**⚠ The consequence that must be reported, not just accepted: depth-0 exposure rises 4×.**
`pick_hubs.py` applies **no depth filter**, so `--pool all` admits bought building blocks that the
reward pre-filter happened to exclude — measured on the same cell, depth distribution of the top-200:

| pool | depth 0 | 1 | 2 | 3 | **depth-0 in the first 40 (the part actually walked)** |
|---|---|---|---|---|---|
| reward-filtered | 2 | 104 | 84 | 10 | **2** |
| `all` | **11** | 150 | 37 | 2 | **8** |

A depth-0 hub is legitimately priced at **zero** reactions — a chemist buys it
(`shallow_couplings(depth=0, promoted=()) == 0`) — and since `cc25046` the competitor gets catalogue
starting materials too, so this is not an unfair advantage. But "a bought scaffold costs 0 reactions
and still carries thousands of children, so it ranks near the top by flow" (`cc25046`), and the paper
itself names depth-0 catalogue picking as the metric's **degenerate optimum**. Leaning harder on it
without saying so would be the single most attackable move in the benchmark. So:

1. **Every cell reports the depth distribution of the hubs it walked AND the share of delivered modes
   that came off depth-0 hubs.** This is a new required output — v1's committed curves record
   `source_hub` but the hub-ordering arms kept only plots, so the delivered library's depth mix
   **cannot be recovered for the v1 comparison** and is unmeasured today.
2. **Run `--min-hub-depth 1` as a labelled sensitivity arm.** It is exactly the "is the win late-stage
   diversification or catalogue picking?" question a reviewer asks, it is cheap (same enumeration
   cache, different `hubs.csv`), and it matches the AL acquisition path's own default — which uses
   `min_hub_depth=1` precisely because depth-0 is "huge fan-out, zero amortization — not the 'build
   once, diversify' signal" (Logs/025). See §7.5: the campaign's `pick_hubs.py` has no such knob while
   the acquisition path does, which is an inconsistency between two implementations of the same idea.

**`--n-hubs` becomes the method's single width knob.** Under the pre-filter, `TOPK=1000` capped
eligibility at 575, so `--n-hubs 200` was a soft cap. Under `--pool all` it is the only thing between
the walk and 20,874 hubs. Report it beside the competitor's Stage-2 cap (§2.3).

#### `--child-policy free_frag --prebuild-k 0`

These are two knobs, not one. `free_frag` is the *filter*: keep only children whose last step attaches
an already-available fragment, so each kept child costs exactly one marginal reaction. `prebuild-k` is
the *stock*: pre-pay to synthesize the top-K promoted fragments up front, charged before the first
mode (`glue/samplers/lsdflow/campaign.py:451`).

- `free_frag` is **inert** on RGFN and RxnFlow (no dynamic library ⇒ it keeps everything ⇒ identical
  to `reward`), so switching it on universally costs nothing and removes a per-generator special case.
- `K=0` because Logs/049 traced the entire 12.69% count-once-vs-SPARROW gap on DRD2 to pre-select-K's
  up-front reactions (~17 of 26 never used by a 100-mode selection). **At K=0 the gap is exactly
  0.00%.** The cost-model audit is the credibility anchor for every number in the paper; do not
  hand it a 12% discrepancy that has a known cause.
- `K=20` remains available as a **labelled compute-saving variation** (Logs/037: 2–6× fewer
  oracle/enumeration calls at roughly constant reactions/mode). Revisit if SCENT runs short of oracle
  calls. It is never the core method.

### 2.3 Budgets: there are two, and only one is equalised

| budget | what it covers | treatment |
|---|---|---|
| **training** | oracle calls consumed while learning | **equalised at 10,000** (arm A) |
| **inference / selection** | Stage-2 upsampling (competitors, up to 50k distinct); hub enumeration (ours, up to ~700k children) | **measured and reported per arm, never equalised** |

Equalising the second would force the fixed-*mode* readout this project demoted to secondary. The
asymmetry is not directional — the calls make the competitor look expensive while the resulting pool
makes it stronger — so report it and say so. Our side's width knobs (`n_hubs=200`,
`n_traj=30,000`) belong in the same table as the competitor's Stage-2 cap, so a reader can see both.

**The competitor matrix is NOT a clean 10,000, and must not be labelled one.** Stage 2 equalises
POOLS (500 modes), not oracle calls, so each cell carries a measured surcharge on top of arm-A
training — REINVENT/Saturn/TANGO 12,018–12,048, FragGFN 12,047–14,046, **S3-GFN 10,048–18,048**. A
1.80× spread, 122,411 calls over 45 cells, concentrated on S3-GFN. Label it *"arm-A training plus a
measured Stage-2 surcharge of 2,000–8,000"* and state the spread rather than averaging it away.

**But the asymmetry runs toward us, not away.** Our own inference-time spend is 33,099–69,896
reward-gen calls per v1 campaign cell and reaches **791,503** on a budget-scale cell — **50–100×** the
competitors' entire Stage-2 cost. The inference-time axis is uncontrolled for everyone and we are by
far its heavier user. Equalising it is not the fix: that would force the fixed-MODE readout this
project demoted to secondary. Report it; do not equalise it. One within-field reading does survive:
S3-GFN's edge over Saturn and TANGO is partly bought with up to 1.8× their oracle calls.

### 2.4 Reporting conventions (unchanged, restated so a cell can be checked against them)

- **Primary readout: modes at a fixed 100-reaction budget.** Emit 50/100/150/200/300 for the
  reaction-GFNs (free — same ordering, CPU re-read) and headline 100. Multi-budget is **not**
  available on the competitor side without re-running retrosynthesis and re-solving SPARROW; out of
  scope.
- **Every cell carries its stop reason:** `budget-binding` (the only like-for-like case),
  `pool-exhausted`, `mode-capped`, `solver-truncated` (SB arm only), and — new from Stage 2 —
  `target-reached` / `stalled` / `cap`. Outside the budget-binding regime quote `cost_kept_rxns`,
  not `used_rxns`; the gap between them *is* the exhaustion detector.
- **Seeds 42/43/44**, `PYTHONHASHSEED=0` exported on every sample (load-bearing: `--seed` alone gave
  377 vs 387 routes; with it, 730/730 byte-identical).
- **Two pools per competitor cell** (naive / pruned), each read two ways (reactions/candidate,
  reactions/mode). Run `mode_saturation.py` first — cells whose two pools coincide need only one.

### 2.5 Every budget error found so far points the same way — say so before a reviewer does

Five separate measurement errors on the budget and cost axes have surfaced in this project. **All
five flatter us.** No individual one is suspicious; the pattern is the kind of thing a reviewer
notices and asks about, and the answer is much better volunteered than extracted.

| # | the error | size | direction | status |
|---|---|---|---|---|
| 1 | our three reaction-GFNs trained to 320,000 oracle calls against the competitors' 10,000 | **32×** | favours us | **fixed** — the arm split (§1) |
| 2 | SCENT's periodic validation scored 1,000 molecules per pass, labelled `train`, inflating its own training count | **2.84×** | favours us | **fixed** — phase-gated budget + `valid_step` wrapper (§1) |
| 3 | placing the arm-A checkpoint by step count rather than by oracle calls (RGFN scores 100 forward **+ 20 replay** = 120.6/iter, so it crossed at iteration **83**, not 100) | **~20%** | favours us | **avoided** — never shipped; the phase counter was written first (§7.2) |
| 4 | a CBC `TimeLimit` row is a feasible solution, not an optimum, so it understates the competitor's selector | unbounded | favours us | **structural — flag, cannot fix** (§6.7) |
| 5 | trimming an already-solved SPARROW selection to a smaller budget is feasible but not re-optimised, so it understates the competitor | unbounded | favours us | **structural — flag, cannot fix** (CLAUDE.md) |

The 1–3 / 4–5 split is the honest part and must not be collapsed. **1–3 were our own errors and are
corrected.** **4–5 are properties of the comparison itself**: a truncated solve and a trimmed solve
are both lower bounds on the competitor, and no amount of care removes that — it can only be
disclosed per row. Report both classes; do not let the fixed ones imply the structural ones went
away.

**THE COUNTEREXAMPLE, so "does it flatter us?" does not become the only filter.** A reader who
absorbs the table above will start triaging errors by direction. Here is one that points the other
way, found 2026-09-12 before it ran: **five of the six competitor docking bridges hardcode
`max(-float(raw)/norm, 0.0)` with no orientation seam** (`s3gfn`, `reinvent`, `synformer`, `saturn`,
`fraggfn`; only RxnFlow has `self.sign`). `cnn_vs` is higher-is-better in ~[0, 9], so that expression
returns **exactly 0.0 for every molecule** — a flat reward, silently, for ~3.3 h of docking per cell
across **18 cells**. It would have made the COMPETITORS look terrible on 6TD3-B, which is the
direction a reviewer would never think to question and we would never think to check.

**A TRAP SITS ON THE REPAIR PATH.** `oracle_higher_is_better` looks like the fix and is not: it is
consumed at `al_loop.py:230` as `min(valid) if not ... else max(valid)`, i.e. candidate **SELECTION**,
and FragGFN passes it too, so it does not even distinguish the affected generators. Flip it and you
change which molecules are kept, leave the reward flat, and believe you are done. The reward seam is
`self.sign` in the `_value` mapping; `rxnflow/fixed_reward.py:160-172` documents the same hazard from
the other side and is worth reading before touching any of the five.

**THE FIVE SITES, verified on `Hub-Analysis` 2026-09-12** (`fraggfn:257`, `s3gfn:243`,
`reinvent:237`, `saturn:236`, `synformer:353`; TANGO shares Saturn's runner, so five files cover six
generators). **Cite line numbers from the MERGED branch, never from a worktree** — this project runs
seven of them and two agents reported stale numbers as repo facts on the same day. `benchmark-v2-infra`
holds a 375-line `synformer/fixed_reward.py` against the merged 477, so the same statement is at 263
there and 353 here, and both greps are honest. A file-level check in a worktree is a check on the
worktree.

**And a banner did not prevent it.** `rxnflow_6td3b_docking_fixed_5k.yaml` carries a
"⚠ LOAD-BEARING — DO NOT DELETE" warning describing precisely this catastrophe. The fix landed in the
one generator whose header carries the warning; the five that needed the identical fix never got it,
because they had no 6td3b cell at the time it was written. **A banner protects the file it is written
in and nothing else** — the same lesson as §8.7/§8.9 and as the four backup gaps: a rule stated in one
place is not a rule applied in every place that needs it.

**Why this is worth a section rather than a footnote.** Errors 1–3 were each found by looking for
them, and each was corrected *against our own result*: #1 cost us the 32× training advantage, #2
removed 2.84× of SCENT's apparent budget, #3 gave up ~20% of headroom on the arm that carries the
external head-to-head. That is the claim to make — not "we made no errors", which is false, but "we
went looking in the direction that costs us, and we published what we found." Anyone adding a sixth
row should record its direction even when it points the other way; a table in which every row
flatters us is only credible if the other direction was also searched.

---

## 3. The two stage graphs

They are different. The runbook holds both; do not let a driver assume symmetry.

```
REACTION-GFNs (RGFN, RxnFlow, SCENT) — hub-batching
  train ──▶ sample ──▶ pick_hubs ──▶ enumerate ──▶ campaign
    │         │                        │             │
    │  ckpt@10k calls          children +        hub_batching vs
    │  + trace.csv             reactions         best_candidate
    │  + recipes (SCENT)       + rewards         at R = 50…300
    └─ ① routes.json prefix   └─ ② children[].reaction    ③ recipes expand promoted fragments

COMPETITORS (FragGFN, S3-GFN, SynFormer, REINVENT, Saturn, TANGO)
  train ──▶ upsample&filter ──▶ retrosynthesis ──▶ selection
    │            │                    │                │
    │      Stage 2: sample until      MultiAiZ, or      SPARROW MILP
    │      500 diverse modes;         native routes     + diversity-aware
    │      harvests trace FIRST       (SynFormer,       greedy
    │      as a free pool             TANGO ARM 2)
    └─ trace.csv is REQUIRED for the free pool
```

**Stage 2 does not apply to the reaction-GFNs.** Our pool is the enumerated hub neighbourhood, not a
sampled pool; there is nothing to upsample. **SynFormer is exempt from Stage 2 and that is a
finding** — its candidates are a slice of an accumulated GA population, so getting more molecules
means more *training*. Generate-then-sample methods can be upsampled; a genetic algorithm cannot.
Those cells stay pool-limited by construction.

**TANGO sits on two axes that do not coincide.** On *construction mechanism* it groups with Saturn and
REINVENT (Saturn's generator with a constrained-synthesizability reward; syntheseus decomposes
post-hoc, so synthesizability is reward-level, not construction-level). On *who ships routes* it groups
with SynFormer. **The 2×2 is on the first axis.** A table that implies the two coincide reads as an
error. TANGO uses **ARM 2** (re-run syntheseus over the emitted pool, ~3.5 h/cell), never ARM 1 —
ARM 1's coverage correlates with the generator's own collapse and covered 7 of 218 molecules on the
pruned sEH pool.

---

## 4. Generated vs copied — one tree, one ledger

**There is no `copied/` subdirectory.** `benchmark_v2` is a single source of truth with a uniform
layout; an agent reading it never has to ask which run a file came from. Provenance lives in **one
machine-readable ledger**, `experiments/benchmark_v2/PROVENANCE.csv`:

```
stage,generator,target,seed,arm,origin,source_path,source_md5,date,operator,note
```

`origin ∈ {generated, copied}` and nothing else.

### What is planned as copied (36 of 108 training cells)

| generator | targets | why it copies |
|---|---|---|
| S3-GFN | seh, drd2, clpp | corrected to ~10k calls 2026-08-21; `aux_coefficient` fixed to the authors' 0.001 |
| REINVENT | seh, drd2, clpp | corrected to batch 64 × 157 steps = 10,048 |
| Saturn | seh, drd2, clpp | config verbatim from the authors' `constrained_synthesizability/experiment.json` |
| TANGO | seh, drd2, clpp | same, plus the TANGO oracle |

Everything else generates: all three reaction-GFNs (budget change), FragGFN (~30× over budget),
SynFormer (1 of 9 exists, blocked), and every 6TD3-B cell (never run for any competitor).

### Copying is at the TRAINING stage only

**A copied training run does not carry its pools forward.** The gate change moves pool composition,
and therefore retrosynthesis and selection. Measured across all 37 trained competitor cells
(naive top-500):

- **27 of 37 unchanged** — pools *and* cached MultiAiZ routes stay valid; re-run selection only.
- **10 change**, and they cluster: `reinvent/clpp` all three seeds (587→122, 647→158, 511→92),
  `s3gfn/clpp` all three (263→41, 243→30, 302→47), `reinvent/seh/s44` (464→1202),
  `s3gfn/seh/s43` (65→476, verified independently), `s3gfn/seh/s44` (168→807),
  `s3gfn/drd2/s44` (97→186).
- Saturn and TANGO ClpP are untouched at 1800+, so **the ClpP damage is specific to REINVENT and
  S3-GFN.** Those six cells go pool-exhausted at R=100 and must be flagged, not quoted. Their
  survivors are a *subset* of already-routed molecules, so re-gating them is a filter, not a re-route.

⟦OPEN — not measured: whether **pruned** pools change more widely than naive ones. Plausible (the
sphere-exclusion scan reaches deeper below the gate) but unverified. Do not state it as established.⟧

### A copy MUST include `trace.csv`

Stage 2 harvests the training trace as a free pool of already-scored molecules at zero marginal
oracle cost — `saturn_clpp/s42` reaches 500 modes from its history alone, turning 28 GPU-h into 0. A
copy that brings the checkpoint but not the trace silently throws that away. `trace.csv` is a required
column in the ledger, not an optional extra.

---

### 4.1 Which source wins when the grid, the ledger and the disk disagree

They will disagree, routinely, and the answer is not "whichever was written last". Each answers a
different question and only one of them is evidence about the filesystem:

| the question | the authority |
|---|---|
| what EXISTS, and in what state | **the filesystem**, via `manifest.py --status` (computed at load time, cannot go stale). **The grid is not evidence.** |
| what PRODUCED an artifact | **`PROVENANCE.csv`**, append-only, one row per (stage, cell, arm) |
| which cells we INTEND to have | **`grid.csv`** — here the grid wins |
| the plan itself has changed | edit `build_grid.py`'s taxonomy and REGENERATE. **Never hand-edit the CSV.** |

**`train_plan` is a PLAN, not an OBSERVATION**, and `build_grid.py`'s own docstring says so — several
rows were known-optimistic at scoping and it names them, S3-GFN included. So **a driver must never
branch on `train_plan` to decide what is on disk**; read `manifest.py --status`, and use
`reconcile_plan_vs_disk.py` to report divergence (it writes nothing, because `grid.csv` is a build
product). Deriving `train_plan` from the filesystem would convert a plan column into a status column,
which is the thing `build_grid.py:13` exists to prevent — I proposed exactly that and was wrong.

---

## 5. Order of operations, per cell

Cells run concurrently and independently. Nothing waits for a phase to complete across all cells.

1. **Verify the plan.** Read the cell's row in `grid.csv`. If `train_plan=copy`, check the source
   still exists and its md5 matches; write the ledger row *before* copying.
2. **Train** to arm B's budget, with `_trace.py` writing continuously, checkpointing when the trace's
   `n_scored` crosses **10,000** (arm A) and at the end (arm B). SCENT trains with `--log-recipes`
   (default on since 2026-07-29 — confirm in the run's own config, do not assume).
3. **Freeze the training artifacts.** `chmod -R a-w` the cell's `train/` directory once step 4 passes.
   This is the structural fix for the hazard in §8.1 — every later stage writes to its own directory,
   so "do not re-invoke a runner against this cell" is enforced by the filesystem rather than
   remembered.
4. **Verify the training stage** (§6). A cell that fails here is re-trained now, at the cost of one
   run. A cell that fails here and is discovered in three weeks costs the same run plus everything
   built on it.
5. **Back up** to `/project/def-naeilum/.../RGFN_LSD_MarkStevens/backups` — per cell, per stage,
   as soon as that stage passes verification. Copy-verify-delete, never `mv`. Login node only
   (`/project` is not mounted on compute nodes).
6. **Downstream**, per the cell's pipeline in §3. Ours: sample → pick_hubs → enumerate → campaign.
   Competitors: Stage 2 → retrosynthesis → selection, both pools.
7. **Verify each stage** (§6) and write its ledger row before moving on.
8. **Record the delta.** A re-run cell's numbers *will* move. Diff against the corresponding v1 result
   and record the change rather than quietly replacing it.

---

## 6. Verification — every item is a failure that actually occurred

A cell is not accepted until all of these pass. Each corresponds to a silent failure that produced a
confident wrong answer rather than an error.

### 6.1 Routes were emitted (①)
```
python -c "import json;print(len(json.load(open('<cell>/sample/routes.json'))))"
```
Non-zero for rgfn / rxnflow / scent. `_routes.py` validates at write time and emits
`route_status.json` — check that file exists. *Why:* two workers hardcoded `json.dump({})`; a run
wrote a well-formed empty file, exited 0, and was promoted. 36 of 40 cell-seeds were in that state,
and downstream SPARROW priced the **empty library** as trivially `Optimal` at zero cost.

### 6.2 Every enumerated child carries its reaction (②)
Fraction of `children[].reaction` non-empty must be **1.00**. Anything strictly between 0 and 1 is
*more* dangerous than 0: a partial artifact prices a mixture — some children at their true molecule,
the rest at their hub — and still returns `Optimal`. Measured on merged artifacts: 56.6%, 24.6%,
**99.1%**. A 99.1% file is exactly the one whose warning gets scrolled past.

### 6.3 Recipes exist AND belong to this run (③)
Two parts. Part 1: does the snapshot have `smiles_to_route` at all? Part 2: **does it cover the
fragments THIS RUN used?** Part 2 is the one that was missing. Coverage of the snapshot's own
`chosen_smiles` is the wrong question — a snapshot from a *different* model is internally complete, so
it reads **100%** while only **53%** of the run's fragments are expandable (549 of 1,169 silently
*bought* rather than *built*).

**IN v2 THAT FAILURE RECURS ACROSS ARMS, and it is created by our own design.** `_recipe_health`
resolves the snapshot from the checkpoint's **run_id** (`check_route_readiness.py:170-176`) and takes
`snaps[-1]`, the LATEST. Because one training run now yields both arms in ONE run directory (§1), the
two arms **share a run_id** — so an arm-A cell's recipe check reads arm B's final snapshot and reports
arm B's coverage. Arm A promoted nothing; the number it prints is about a different arm, and the
verdict stays right only by luck. **The arm must disambiguate the snapshot, not just the run.**

**And an ABSENT library is `n/a`, not a failure** — `if not snaps: return None  # no dynamic library
-> nothing to expand`. An arm-A SCENT cell therefore passes on the same path RGFN and RxnFlow take,
and that is correct rather than lucky: with zero promotions its routes bottom out on the 418 base
blocks, so there is nothing to expand and a chemist can act on every molecule. Entry 070's
distinction is the operative one — **no recipes because logging was OFF while fragments were promoted
is broken; no recipes because NOTHING was promoted is complete.**

### 6.4 One command answers 6.1–6.3
```
python experiments/lsd_hubs/matrix16/check_route_readiness.py
```
**In `benchmark_v2` this is a hard gate, not a report.** A cell ships only if a chemist could act on
every molecule in it. That is what turns the route dataset from 5 cell-seeds into matrix-wide
coverage — the emitters exist now, so the coverage is a by-product of re-running, provided nothing
accepts a cell that fails.

### 6.5 The trace is present, continuous, and the checkpoint matches it
`trace.csv` exists, its `n_scored` reaches the arm's budget, `phase == "train"` rows are separable
(some generators score outside the training loop — S3-GFN's `evaluate()` scores 1,000 molecules, and
counting those inflates the budget), and the arm-A checkpoint sits at the row where `n_scored` first
crosses 10,000.

### 6.6 Determinism
`PYTHONHASHSEED=0` exported. Without it a sample is a one-of-a-kind artifact recoverable only from
backup.

### 6.7 Do not trust a solver's status
CBC reports `Optimal` when it has merely run out of time (PuLP's `sol_status` does not distinguish).
Detect by wall-clock, not status. A capped row is a **lower bound on the competitor**, i.e. it
flatters us, and cannot carry a ratio.

### 6.8 Freeze inputs for any comparative arm
Any experiment claiming two arms saw "the same candidates" reads from a frozen snapshot with the
source md5 recorded — and the snapshot must include `meta.json` and `compositions.json`, because the
provenance and coverage guards resolve them *relative to their inputs*. A snapshot omitting them turns
both checks into silent no-ops, which is what happened on the first attempt.

### 6.9 The one-line rule
Every failure above was an **existence check where the real question was a match or a content check**.
The file was present, well-formed, and wrong. When adding a guard, ask what it would take for the
check to pass on bad data — and check *that* instead.

---


### 6.10 A check that cannot fail is not evidence — ask what it would look like if the thing were broken

Three assertions were made and withdrawn on 2026-09-12, all by careful people, all on checks that
**had no failing branch**:

| the check | why it could not have failed |
|---|---|
| "`unaccounted_s` is 0.0 and `phases` sums exactly to `total_s`, so these cells are trustworthy" | `phases` held exactly ONE key. A single-phase file's residual is 0.0 **by construction** and its sum matches **by construction**. Both statements true, neither evidence |
| "no `population_checkpoints/` directory exists" | the glob looked one level too deep. A glob under the wrong parent cannot find the directory whether or not it is there |
| "the Stage-2 surcharge is 3.4x" | `asked` is cumulative, so summing it as if incremental **cannot come out small** |

The common shape is not carelessness about the measurement. It is that nobody asked *what this check
would print if the thing it tests were broken.* When the answer is "the same thing", the check is
decoration.

**A check that cries wolf is discarded, which costs the same in the end.** An orphan-detector that fires on a `README.md` teaches its reader to skip the output, and a check nobody reads has the same value as one that cannot fail. Exclude the meta-cases explicitly, with a comment saying why.

**Before recording any verification as passed, state its failing case.** If you cannot name an input
that makes it fail, it is not a check — find one that distinguishes, or say plainly that the property
is unverified. A negative result is worth stating only when a positive one was possible.

Corollary, learned the same day: **`unaccounted_s` is a detector only where `phases` has more than one
entry to sum.** Same field, same value, opposite meanings depending on how many phases exist
(SynFormer 0.0 = nothing separated; FragGFN 19.1 = an honest 10% residual).

## 7. Build items — before any cell launches

### 7.1 6TD3-B: reward and gate are both `cnn_vs` at 6.718 (settled 2026-08-28)

**Reward and gate: `cnn_vs` — gnina's virtual-screening score, CNNscore × CNNaffinity — higher is
better, gate 6.718.** Another agent is wiring the oracle; this section records the threshold and why
it is the right one.

The problem it solves. Three candidate signals were compared on one docking pass —
`dock_6td3_matched_74500` + `dock_6td3_scentsample_2452248`: 160 real glues, 160 property-matched
decoys, and 400 of our **known reward-exploiting** candidates from the old differential. Each at its
own 5%-FPR gate (recomputed independently 2026-08-28):

| signal | gate | AUROC glues vs decoys | **AUROC glues vs OUR exploiters** | glues | decoys | **ours** |
|---|---|---|---|---|---|---|
| `ddb1_dvina` *(old reward)* | −2.0 | 0.688 | 0.327 | 66.9% | 31.2% | **75.5%** |
| `cnnaff_t2` | 7.970 | 0.804 | **0.521 — chance** | 37.5% | 5.0% | **35.8%** |
| `cnnsc_t2` | 0.9066 | 0.923 | 0.965 | 78.8% | 5.0% | 1.0% |
| **`cnn_vs`** | **6.718** | **0.917** | **0.946** | **78.1%** | **4.4%** | **3.5%** |

Each single column failed one of the two jobs. `cnnaff_t2` is an unbounded pK estimate, so it makes a
good *reward* — but it scores our known exploiters at **chance** (0.521; they clear its gate at 35.8%
against real glues' 37.5%), so it cannot serve as the *gate*. `cnnsc_t2` is the sharp discriminator
but a bounded [0,1] pose-quality probability whose real-glue p90 is 0.987, leaving little headroom
once a pose is good.

**`cnn_vs` does both, and it forecloses the exploit structurally rather than merely failing to
observe it.** Because CNNscore enters multiplicatively, a molecule with poor pose confidence cannot
buy its way past the gate with affinity: at our exploiters' median CNNscore of 0.334, reaching 6.718
would need CNNaffinity ≈ **20**, against a real-glue range of roughly 5.6–9.4. The old reward's
failure mode — make Tier 1 worse rather than Tier 2 better, with no requirement that the pose be
physical — has no analogue here.

It also has the reward shape the training run needs. Our candidates span 0.22–8.37 against the glues'
0.72–8.71, with our median at 2.63 against the glues' 7.59 — ample gradient, no saturation. And
climbing it drags pose confidence up with it, which is exactly what `cnnaff_t2` failed to do:

| our candidates, ranked by | median `cnn_vs` | median `cnnsc_t2` | median `cnnaff_t2` |
|---|---|---|---|
| top 10 by `cnn_vs` | 7.190 | **0.834** | 8.757 |
| top 50 by `cnn_vs` | 6.166 | 0.764 | 8.308 |
| all 400 | 2.634 | 0.334 | 7.767 |
| *(top 10 by `cnnaff_t2`, for contrast)* | — | *0.627* | *9.032* |
| **real glues** | **7.592** | **0.975** | **7.824** |

**One consequence to keep.** Reward and gate are now the same column again, so the standing
exploitation check must score against something that is **not a component of the reward**. Use
`vina_t2` — a different scoring function entirely — alongside the two CNN components, which remain
separately emitted so a passing molecule can be decomposed into *why* it passed. Note our old
candidates already dock *better* than real glues on raw `vina_t2` (median −11.13 vs −10.15), so "beats
real glues on Vina but not on CNN VS" is itself the diagnostic signature.

**Verify the column at wiring time.** The current result CSVs carry `cnnsc_t2`, `cnnaff_t2`,
`cnnaff_t1` and `ddb1_dcnnaff` — there is **no `cnn_vs` column yet**. gnina emits `CNN_VS` natively as
exactly this product, and reconstructing it as `cnnsc_t2 × cnnaff_t2` reproduces the 5%-FPR point to
6.7134 against the adopted 6.718, so the two agree. Confirm which one the oracle captures, and keep
both components in the output either way — not recording a cheap column is what made Logs/069
expensive.

**Generalisation still open (§11).** sEH, DRD2 and ClpP all use the same column for reward and gate,
so none has an independent exploitation check. 6TD3 is the only target where gaming was caught — not
because it is the only one being gamed, but because it is the only one where a second signal happened
to be recorded.

### 7.2 Wire `_trace.py` into the three reaction-GFNs

`validation/generators/_trace.py` exists and is wired into the four competitor runners. It is **not**
wired into `run_scent_fixed.py`, `run_rxnflow_fixed.py`, `run_fraggfn_fixed.py`, or the RGFN path —
they emit no `trace.csv` and no `timing.json`. Without it: no modes-vs-oracle-calls curve on our side,
no per-phase wall-clock, and **no way to place the arm-A checkpoint exactly**. This blocks §1.

Free side effect: the trace's first rows *are* the untrained-policy samples, so the t=0 baseline for a
learning curve costs nothing extra.

### 7.3 Close the stale-gate and unknown-target defaults

- ~~`submit_competitor_routes.sh` still carries pre-standard gates as shell defaults.~~
  **STRUCK 2026-09-08 — this was already done and the item was written from a stale read.** Line 88
  of that script reads `from targets import get_target` and line 90 calls it, on both this branch
  and the competitor branch. Flagged by the Standard Pipeline session and verified. The remaining
  bullets below were NOT re-checked at the same time and should be treated as still open.
- ~~`upsample_to_modes.py`'s `--target` choices are `sorted(DEFAULT_CAP)` = clpp/drd2/seh, and
  `DEFAULT_CAP` has no `6td3b` entry.~~ **STRUCK 2026-09-12 — correctly NOT done.** That script is
  the Stage-2 **competitor** pool builder; the three reaction-GFNs never call it. 6TD3-B is scoped to
  our three models only, so adding a `6td3b` entry there would assert that the competitors run 6TD3-B.
  The argparse rejection is the right behaviour, not a gap. (Raised by agent C against this bullet.)
- ~~About eight hardcoded `("6td3","clpp")` membership tests and `_HIGHER_IS_BETTER_BY_REWARD`
  dicts~~ **DONE 2026-09-12, `69c1251` / `947493f`** — and the fix was not the one this bullet
  described. **The same literal `("6td3","clpp")` means two different things** in the four places it
  appears, and 6TD3-B belongs in one of them and not the other:

  | the test really asks | 6TD3-B? | why |
  |---|---|---|
  | is the reward produced by the docking bridge? | **yes — added** (`rxnflow_worker`, `scent_worker`) | it is a docking reward |
  | does the gate read a DIFFERENT column than the reward? | **no — deliberately absent** | 6TD3/ClpP reward is `clip(-vina)` while the bar is raw Vina; 6TD3-B's reward *is* `cnn_vs` and it gates on `cnn_vs`, so reward and gate are the same column |

  With one literal serving both questions, 6TD3-B was landing on the second answer **by coincidence**,
  exactly as the `.get(name, True)` default was landing on the first. Both
  `_HIGHER_IS_BETTER_BY_REWARD` dicts now carry an explicit `"6td3b": True`, and `rgfn_adapter`
  records that the *absence* from the second test is load-bearing so nobody "completes" it later.
  Verified end to end: `scent/6td3b/42` resolves to `REWARD_TYPE=docking`, `HIGHER_IS_BETTER=true`,
  gate `6.718`, `ORACLE=docking_6td3b_gpu`, `PHASE=2`, with the oracle name read from the config
  rather than a literal.
- **STILL OPEN, and it is what actually blocks phase 2: there are no `6td3b` TRAINING CONFIGS.**
  Plumbing alone cannot launch a cell. The three generators have `fixed_reward_6td3_5k.gin`,
  `scent_6td3_fixed_5k.gin` and `rxnflow_6td3_docking_fixed_5k.yaml`; there is **no `6td3b`
  equivalent of any of them** (a repo-wide search for `*6td3b*` returns one log, one analysis script
  and its `.pyc`). The oracle side is ready — `docking_6td3b_gpu` → `Docking6TD3BGpuOracle` is
  registered in `glue/oracles/__init__.py` and dispatched in `docking_server.py:268` — so this is
  three configs cloned from the `6td3` ones and repointed at that oracle, not new science. Until they
  exist, all 27 phase-2 cells are unlaunchable.

### 7.4 ~~Correct RxnFlow's batch size~~ — WITHDRAWN, no change needed

Verified 2026-09-07 (agent A): RxnFlow already runs at the authors' `num_from_policy = 64`. The
`128` that prompted this item is `reward.batch_size`, the sEH proxy's scoring batch. See §1.

### 7.5 `pick_hubs.py` — pass `--pool all`, and give it the depth knob it lacks

- **The v2 driver must pass `--pool all`** (§2.2). v1's `submit_cell.sh` does not, and **must not be
  edited** — jobs are in the queue, bash resumes a script at a byte offset, and SLURM runs a submit
  script as of its start. Leave v1 alone; put the flag in the v2 driver.
- **Add a depth band to `pick_hubs.py`.** It has no depth filter at all, while the AL acquisition path
  defaults to `min_hub_depth=1 / max_hub_depth=3` for reasons that apply equally here (depth 0 = huge
  fan-out, zero amortization; depth 4 = at the reaction cap, `P_B` unrecoverable). Two
  implementations of "pick hubs" should not disagree about what a hub is. The default stays
  unfiltered — the sensitivity arm needs the knob, not a new default.
- **Emit the depth mix.** The per-cell campaign output must carry the walked hubs' depth distribution
  and the share of delivered modes sitting on depth-0 hubs. `source_hub` is already recorded; joining
  it to depth is all that is missing.
- **Two incidental bugs, both of which vanish under `--pool all`:** `pick_hubs.py`'s own
  `--top-k-candidates` default is **100** (yielding 64 hubs, never reaching `--n-hubs 200`) while
  `submit_cell.sh` passes 1000 — so invoking the script directly runs a silently different method.
  And `submit_cell.sh`'s Knobs comment says `TOPK (100)` on line 18 while line 35 sets 1000.

### 7.6 A `manifest.py` for `benchmark_v2`

Spanning both pipelines, resolving a cell from `grid.csv` + `targets.py` + live filesystem status, the
way `matrix16/manifest.py` does for one. Deliberately not written yet — a bad duplicate is worse than
none.

---

### 7.7 There is no v2 TRAINING driver — this is what actually gates the launch

`submit_cell_v2.sh` begins by *verifying* that a cell's `train/` already exists; it runs
sample → enumerate → campaign. **Nothing in `benchmark_v2/` trains a production cell.** Cells still
train in the v1 layout and reach v2 through `copy_forward`. Verified 2026-09-12:

| generator | launcher | can it write into a v2 tree? |
|---|---|---|
| fraggfn, rxnflow, scent, reinvent, saturn, tango, synformer, s3gfn (**8 of 9**) | `scale5k/submit_baseline.sh` | **yes** — line 101 is `FR_ROOT_DIR=${OUT_ROOT:-$SCRATCH/rgfn_runs/experiments}` |
| **RGFN** | `scale5k/submit_rgfn.sh` | **no** — line 38 hardcodes `FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments` and line 41 derives `RUN_NAME` from (system, seed) with no override. Pointed at seh/42 it resolves to the **live v1 cell** and resumes from its 5,000-iteration checkpoint |

**The pilot trainer is not the answer and says so itself.** `benchmark_v2/pilot/submit_pilot_train.sh`
covers `scent|rgfn` only, `seh|drd2` only for RGFN, and requires a hand-computed `N_ITERS`; its own
header states that approximate placement "is NOT adequate for the production campaign, which must
checkpoint on the trace counter." That prerequisite is now met — `_trace.py` is wired into all three
reaction-GFNs (§7.2) — so the trainer can be promoted rather than replaced.

**Three things to build, in order:**
1. Give `submit_rgfn.sh` the `OUT_ROOT` + run-name override the other eight already have, plus the
   pilot's refuse-if-inside-v1 guard. One generator, and it is the one whose v1 cells a mistake
   would overwrite.
2. Promote the pilot trainer to production: all nine generators, all four targets, budget placed by
   the trace counter rather than by a hand-set `N_ITERS`.
3. A driver that walks `grid.csv` and submits cells — none exists; `manifest.py --list` already emits
   the cell list it should consume.

Until (1) and (2) exist, **no phase-1 cell can be trained into `benchmark_v2` at all.** This is a
larger gap than the 6TD3-B configs, which only block phase 2.

## 8. Hazards

### 8.1 Re-invoking a generator runner OVERWRITES that cell's outputs

`trace.csv`, `candidates.csv`, `pairs.csv`, `timing.json`, `run_config.yaml`. This cost
`s3gfn_seh/seed43`'s entire training history (unrecoverable) and `s3gfn_seh/seed42`'s budget-faithful
2,000-molecule `candidates.csv` (overwritten at 20,000 rows). Mitigated in `3281bce` (TraceWriter
rotates) and `dd8f1a9` (Stage 2 snapshots `fixed_reward/` → `fixed_reward.budget_faithful/`).

**This threatens copy-forward directly:** a copied training run destroyed by a later runner
invocation destroys exactly what was copied. The structural fix is §5 step 3 — freeze `train/`
read-only after verification. Do not rely on remembering.

### 8.2 Fields whose names lie — check the writer before you quote the reader

Two near-misses on 2026-09-08 were the **measurement being right and the reading being wrong**, and
neither would have been caught by re-running anything: a cumulative field summed as if incremental,
and `19:18` read as minutes when it was hours. Both were caught only because the wrong number looked
surprising. The unsurprising instances of this class are still out there.

The defence is not vigilance, it is a lookup. Every field below has produced a wrong number in this
project at least once:

| field | reads as | actually is |
|---|---|---|
| `asked` (`upsample_log.json`) | this round's request | **cumulative** across rounds — summing it double-counts (24,000 reported for a cell whose distinct set grew by 1,216) |
| `used_rxns` | what the selection cost | **inflates** outside the budget-binding regime; quote `cost_kept_rxns` (65 molecules priced at 247 read as 300→387) |
| `n_modes` | modes delivered | modes **requested**; `n_targets_priced` is delivered (89 vs 100 on native routes) |
| `n_modes_kept` | same as `n_modes` | post-filter count — a co-agent lost a result to the difference |
| `n_scored` (`trace.csv`) | training oracle calls | **all phases**, eval interleaved; count `phase == "train"` ROWS, and note that `max(n_scored)` over filtered rows is still contaminated |
| `total_modes` | a total | capped at the 500-molecule prefix |
| `sample_s` / any timing component | 0.0 when absent | **absent ≠ zero** — a missing component must be omitted from the total, never written as 0.0, or an untimed stage becomes a free one |
| durations in prose | `20:10` = 20 min 10 s | on this cluster it is as likely **20 h 10 min**. Always write `19.30 h` |
| a complete-looking `candidates.csv` | the budget was spent | the file is written at full size regardless. `synformer_drd2/43` scored **6,950 of 10,000** — its search exhausted before its budget did — and passed every downstream check as a complete cell. The **trace row count is the only witness**; label the cell, do not average it in |
| a `oom-kill` block in a job's `.out` | this job was OOM-killed | node-level dmesg spills into NEIGHBOURING jobs' logs. `ch_sf_drd2_43-74719.out` carries one naming `oom_memcg=.../job_74716`, a different cell. **Check the job id inside the block before believing it** — this produced a confident wrong diagnosis once already |
| `train_s: 0.0` (`timing.json`) | training was free / instant | **the resume path ran and no training happened.** `run_s3gfn_fixed.py:344` writes `train_s = 0.0` when it finds a checkpoint and skips straight to sampling. **9 of the 54 copied v2 cells record it** — 8 of the 9 S3-GFN cells plus `saturn_clpp_s43`. Only `s3gfn_seh_s44` carries a real S3-GFN training cost (598.959 s) |
| `phases` keys across generators | one schema | **four, and only 21 of the 54 copied cells carry a directly readable training cost.** Counted 2026-09-12 — **21** `train_s` real (reinvent/s3gfn/saturn/tango); **9** `train`/`sample`, no suffix (FragGFN) — recoverable by a mapper; **9** whose only phase key is `total_run_s` (SynFormer) — no split **in the file**, but the breakdown EXISTS: the runner computes `ga_s`/`sanitize_s`/`project_s`/`score_s` per generation and prints them as `[SF-TIME]`, then serialises only the total. Recoverable by `grep '\[SF-TIME\]'` on the SLURM log, not by a file read (and note the two names: the top-level field is `total_s`, `total_run_s` is a key *inside* `phases`; citing the wrong one has already propagated two hops); **9** `train_s: 0.0` (see the row above); **6** with **no `timing.json` at all** (Saturn seh + drd2 — its clpp cells have one, unexplained). Keep MISSING and MISMATCHED apart: one is a parsing problem, the other a measurement never taken, and no re-run recovers the second |
| SynFormer's `unaccounted_s: 0.0` | the breakdown is complete | everything is in **one bucket**. `phases` holds only `total_run_s`, so nothing is unaccounted because nothing was separated |
| `total_s` | what the cell cost to produce | **setup is outside the bracket.** `run_s3gfn_fixed.py:330-331` constructs `SynthSmilesTrainer(...)` and only THEN sets `run_t0`, so loading the 200k-block `zincfrag_hb105` env is untimed — job 76229 spent **6+ minutes** in setup against a recorded `total_s` of 618 s for the whole cell. `run_reinvent_fixed.py:316` has the same shape. This applies to **all 21 otherwise-good cells**, and because `run_t0` is the TOTAL timer it is not only `train_s` that excludes it. "S3-GFN trains in ten minutes" and "an S3-GFN cell takes ten minutes" are different claims and the file supports only the first. Not uniform across runners — check where `run_t0` sits before quoting any cell cost |
| `total_s` vs its own `phases` | the phases sum to the total | not necessarily. `saturn_clpp_s43` reads `total_s` 10,264 s against phases summing to **47 s** — `unaccounted_s` 10,217, i.e. **99.5% of the run is outside the accounting**. Check `unaccounted_s` before quoting any per-phase cost |

**The rule:** read the field's writer before quoting its reader. The names lie by omission, and a
plausible reading of a plausible number is exactly what no re-run will catch.

### 8.3 Never edit a running bash script

bash resumes at a **byte offset**; an edit mid-job runs garbage hours later (this killed job 74318
five hours in, *after* its work had succeeded). The chain scripts snapshot their callee to `/tmp` for
this reason. Copy that pattern; do not reinvent it.

**The same hazard exists in PYTHON, via re-import rather than byte offset.** A multi-process harness
that launches runs sequentially re-imports every module per process — so editing a module between
run A and run B means the two runs execute different code, and if the edit touched a written schema
(a CSV column, a sidecar key) the comparison is between incomparable artifacts while both report
success. Caught 2026-09-12 mid-flight on `_trace.py` during the RNG verification, and reverted for
exactly this reason. **Freeze every module a comparison spans for the duration of that comparison**,
the same way §6.8 freezes inputs for a comparative arm.

**And save state at the point it will be RESTORED, not at a convenient earlier one.** The RNG sidecar
was captured inside the `make_checkpoint` wrapper, which fires part-way through an iteration's
epilogue, while a resume re-enters at the TOP of the next iteration — so the restore faithfully
reinstated a position the uninterrupted run never occupied there. The restore mechanism was correct
throughout; the capture point was not, and no amount of verifying the restore would have found it.
The fix is a pending flag at the checkpoint and the write at the next iteration's top, which is
correct **whatever** consumes randomness in between — so it does not depend on identifying the
consumer, and the two candidate consumers considered were both wrong.

### 8.4 Shared scratch is rewritten by other agents

On 2026-08-19 all three `scent_seh` enumerations were re-run mid-experiment (a correct fix) *after* a
comparison arm had read the old ones — silently turning a same-chemistry comparison into a
cross-chemistry one, with both runs reporting success. The pools differed by 6× and nothing in the
outputs showed it. See §6.8.

### 8.5 Environment traps that a login smoke cannot catch

`$HOME` is read-only on compute nodes — export `TRITON_CACHE_DIR`, `MPLCONFIGDIR`, `XDG_CACHE_HOME`,
`HF_HOME`, `TORCH_HOME`, `SYNTHESEUS_CACHE_DIR` to `$SCRATCH` in every submit script. Any job that
docks must `source ~/bin/rgfn-smoke-env.sh` — omitting it leaves QuickVina2-GPU with three unresolved
boost libraries, which surfaces as all-`nan` and reads exactly like a degraded GPU.

### 8.5b The login node is saturated — a smoke can fail for reasons that are not your code

Two traps that each cost an agent a run on 2026-09-12, both reading exactly like code bugs:

* **`RLIMIT_NPROC` is 1024 and counts THREADS, not processes.** Our own concurrent Claude sessions
  hold ~490 of them, so a smoke that lets OpenBLAS spawn 64 threads per process dies with
  `blas_thread_init: pthread_create failed for thread 17 of 64` at **59 processes**. Prefix every
  login smoke with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
* **Without `source ~/bin/rgfn-smoke-env.sh` the ingest subprocess dies on `libnvrtc.so.11.2`** — the
  same failure that destroyed a 214-row history once. It is survivable now only because the trace,
  timing and library saves were moved BEFORE ingest; the run still exits non-zero.

Same shape as §8.5: a login smoke cannot catch a compute-only bug, and now a saturated login node
cannot reliably run the smoke either.

### 8.6 Do not submit a cell another chain has already claimed

Chains claim cells before any file appears. Grep every queued job's `CELLS=` line first, and prefer
`scontrol hold` over cancel.

### 8.9 A backup scope silently NARROWS as the data grows around it

Four backup gaps surfaced on 2026-09-12, and none was a mistake at the time it was written. Each was
a scope decision that stayed still while the data moved:

| gap | size | why the rule stopped covering it |
|---|---|---|
| `lsdflow_sparrow/` — the **whole 106-cell competitor matrix** | 2.4 G | the script copies `experiments/` and `lsdflow/`; the matrix lives in a THIRD tree that postdates the `lsdflow/` line. The header says "the LSD-Flow artifacts derived from them", which reads as though it covers it |
| `routes.jsonl` (SynFormer's native routes) | — | the tier-1 include list predates it; without it SynFormer's Stage 3 cannot be reproduced at all |
| `*.out` / `*.err` SLURM logs | 139 M | excluded **deliberately** — correct while logs were transcripts, wrong the moment instrumentation printed without serialising (§8.8) |
| the tier-1 name allowlist | 0.74% of the tree | written from our generators' conventions; `last_gfn.pt` matched 9 files, `checkpoint*.pt` matched none |

**The lesson is not "be careful".** It is that an include-list or a named-subtree scope is a claim
about data that existed when it was written, and it degrades silently and successfully — a backup
protecting 0.7% logs exactly what one protecting 100% logs.

**How to check one:** compare DESTINATION against SOURCE, per irreplaceable artifact type, and
prefer no filter where the data is small. 139 MB against 949 GB free does not justify a clever rule.
And when comparing, **match on CONTENT, not on filename** — S3-GFN writes three targets' models to the
same path tail (§8.7), so a basename comparison reports three distinct models as one and passes.

### 8.8 Instrumentation that PRINTS but does not SERIALISE — and the logs are not backed up

`run_synformer_fixed.py` measures four phases per generation and prints them as `[SF-TIME]` lines,
then writes `timing.json` as `{"total_run_s": ...}` and discards every one. **The measurement was
taken; only the serialisation is single-bucket.** So "SynFormer has no per-phase timings" was wrong in
the most expensive direction — it would have designed a figure around a hole that is not there. The
data recovers from the SLURM log with a grep: 117/110/112 generations on the sEH cells,
130/143/129 on ClpP.

And it is a RESULT. Summed across generations, SynFormer is **projector-bound, not oracle-bound**:

| phase | sEH seeds 42/43/44 | share |
|---|---|---|
| `project_s` | 68,174 / 72,878 / 71,144 s | **99.0 / 99.1 / 99.1 %** |
| `score_s` | 477 / 504 / 454 s | 0.7 / 0.6 / 0.6 % |
| `ga_s` | 185 / 150 / 177 s | 0.3 / 0.2 / 0.2 % |

19–20 h per cell is ~99% mapping molecules into synthesizable space; **its 10,000 oracle calls cost
about eight minutes.** So the ~5,530 GPU-h arm-B estimate is a PROJECTION cost, not an oracle cost,
and must not be argued about as one.

**THE EXPOSURE THIS REVEALED.** That data lived only in `/scratch/markymoo/rgfn_runs/*.out`, which
is purge-eligible, and the backup held **zero** `.out`/`.err` files — it covered cell trees only.
A measurement whose sole home is a SLURM log is one purge from gone. Fixed 2026-09-12: all 2,633 log
files (139 MB) now mirror to the backup, verified 1320/1320 `.out`, 1313/1313 `.err`, 100% of bytes,
18/18 files carrying `[SF-TIME]`. **No name filter** — 139 MB against 949 GB free does not justify a
clever rule, and a clever rule is what made the checkpoint allowlist copy 0.74%.

**The rule:** before concluding an artifact lacks a measurement, grep the RUN LOG, not just the
summary file. And any instrumentation that prints without serialising needs its log treated as a
primary artifact.

### 8.8 Retired thresholds live on inside DATA, where fixing the script does not reach them

`multiaiz_pools/` holds PRE-Stage-2 pools, tagged without `_stage2`, built before ClpP was
recalibrated from **−8.0 to −9.1**. Their scores top out at exactly −8.0, so only **13–37%** of each
clears the current gate. A `*_N*` glob picks up twelve of them — `reinvent_clpp_seed42_N500`,
`s3gfn_clpp_seed42_N339` and siblings — and they read as badly-performing current cells rather than
correctly-performing retired ones. Restricting to `_stage2` (plus bare `synformer_`) gives the clean
109.

**Quiet in both directions:** a glob that picks them up understates the field, and a −8.0 ceiling
looks like a plausible ClpP score rather than an obvious sentinel. This is the same family as §7.3's
pre-standard shell defaults with one difference that matters — **the stale bar is baked into the
DATA, so correcting the script does not correct these files.** They have to be excluded by tag or
re-built.

### 8.7 Artifacts that are not where the naming convention says

A driver that locates a cell's artifact by **describing** it — a glob, a filename pattern, a
directory shape — has been wrong silently every time it has been tried here. Two live instances, both
verified 2026-09-12:

| generator | what a weight-glob finds | what the artifact actually is |
|---|---|---|
| **SynFormer** | **nothing** — zero `.pt` and zero `.ckpt` anywhere under `synformer/` | it is a genetic algorithm; its arm-A artifact is a **population**, `population_checkpoints/pop_10000.csv` (ten `pop_*.csv` per cell, 100 rows each). **The directory sits at the CELL ROOT** — `seed<N>/population_checkpoints/`, NOT under `fixed_reward/`; a check one level deeper reports absence, which has already happened once. A driver globbing for weights reads a healthy cell as a failed one. It is also exempt from Stage 2 — a population cannot be upsampled |
| **S3-GFN** | three targets' models at the **same path tail** | it hardcodes its run name, so every target writes `<cell>/seed<N>/s3gfn_seh/s3gfn_seh-seed<N>_model.pt`. Only the outermost cell directory carries the truth; keying on basename or immediate parent silently collides three different models, each of which loads fine and produces plausible numbers for the wrong target |

Related but distinct from §8.2: there the field name lies about its *contents*; here the convention
lies about the artifact's *location or form*. Both fail successfully.

**The rule:** consult the record the pipeline already produces — `.copy_manifest.json`,
`arm_meta.json`, the cell's own inventory — instead of restating the convention. Where a filter is
unavoidable, assert on the RESULT (destination bytes vs source bytes) rather than on the rule; a
transferred-file count reads ~0 on a correct incremental re-run and so cannot distinguish success
from a no-op.

---

## 9. What gets rebuilt downstream

Re-running a cell invalidates everything derived from it. This is the dependency map, so nothing is
quoted from a mixed vintage.

| exhibit | depends on | status in v2 |
|---|---|---|
| Reaction-budget readout (the headline) | every reaction-GFN cell's campaign curves | rebuild; **note the headline moves 2.93× → 2.73×** once FragGFN leaves (verified 2026-08-28: 3 reaction-GFNs, n=23 comparable at R=100, range 1.67–3.56) |
| External head-to-head | competitor pools + routes + selection | rebuild — and on the **primary** axis. The committed figure is still on the secondary axis (reactions-for-100-modes) |
| Diversity-aware SPARROW comparison | our enumerations + competitor selection | rebuild |
| Ordering floor + ceiling | per-cell enumerations | rebuild; robust to the taxonomy change (median recovery 94.6% → 94.0%). **The claim gets stronger under `--pool all`**: v1 measured flow ordering over a reward-pre-filtered pool, so "flow does the selecting" was confounded. Re-run the arms on the unfiltered pool |
| Filter ablation | one cell's enumeration | rebuild; currently `scent_seh` seed 42 only |
| Two-knob surfaces | per-cell enumerations | rebuild at **R=100**; currently budget 300, sEH only, seed 42 |
| Cost-model audit vs SPARROW MILP | selected libraries | rebuild — CPU-cheap, and it is the credibility anchor |
| Compute-time accounting | per-stage timings | rebuild; `rgfn_6td3`'s v1 attribution is broken (all 184,069 s in `unattributed_s`) |
| Route dataset | ① + ② + ③ on every cell | **new capability** — matrix-wide instead of 5 cell-seeds, arm A only |
| Oracle calibration, hardware benchmarks | oracles + molecule sets + GPUs | **copy** (§0) |

---

## 10. Harvest contract to the publication repo

`AC-MedChem-SDL/RGFN-LSD` reads this tree through `tools/harvest/*.py`, already parameterised by
`--matrix-root` / `--campaign-root` / `--research-root`, so retargeting is a flag change. Each exhibit
must record, in `benchmark_v2/results/<exhibit>/PROVENANCE.md`: the script that produced it, the
artifact path it lands in, and the `reproduce/` script that consumes it. Note that **every** 6TD3 artifact in the publication repo is invalidated by the 6TD3-B swap — not only
`artifacts/reaction_budget/seed42/*_6td3/`. Per 0 row 6 it is every 6TD3 cell of every generator, so seed
is irrelevant and `seed43/` and `mechanism/greedy_ceiling/` are included: 27 files across three locations.

Figures are build products and are never committed. Run `tools/check_identity.py --all` before
committing there — the default scans only *tracked* files, so a new file is invisible to it.

### 10.1 When a generation lands, re-harvest and flip the marker — this is OUR step, not theirs

**Every artifact in `RGFN-LSD/artifacts/` is v1 and is superseded by this re-run.** The publication
repo tracks which measurement generation each committed artifact belongs to
(`artifacts/MANIFEST.csv`, built by `tools/build_manifest.py`, with `--check` failing on a stale one).
That marker cannot flip itself, and the publication side cannot know when a generation lands. So:

> **On completing a phase, re-run the named harvest script in `tools/harvest/` against the new tree
> and flip `GENERATION` in the publication repo, then rebuild the manifest.** No `reproduce/` script
> changes — the harvest layer is the seam, and it is already parameterised by `--matrix-root` /
> `--campaign-root` / `--research-root`, so retargeting is a flag change.

This deliberately lives here and NOT as a generation column inside `benchmark_v2/`. The research
tree's tooling computes live status at load time *because the research tree changes constantly*;
published artifacts change once per generation. Teaching `manifest.py` which of its cells someone
downstream happened to publish would couple the producer to the consumer in the wrong direction, and
a fourth hand-maintained inventory beside `grid.csv`, `PROVENANCE.csv` and `manifest.py` is exactly
the document that goes stale and misleads.

**At the FIRST re-harvest, add the single-generation assertion — the trigger is ours.** Nothing today
prevents a figure drawing on artifacts from two generations. The publication repo's CI gates a stale
manifest, a broken figure script, the test suite, T0 end-to-end, and notebook/toy-dataset staleness —
but the `generation` column is **data, not an assertion**: nothing reads it and refuses. Re-harvest
three artifact groups, leave a fourth at v1, and every check stays green while the figure is silently
mixed. The protection is a human reading the column before writing a caption, which is a real
improvement on it being unrecorded and a lower bar than "fails loudly".

The fix is a reproduce-time check that every artifact a script reads shares one generation. It is
deliberately NOT written yet, and the reason is §6.10: with everything at v1 it could not fail, so it
would ship as an untested green light. **It becomes testable the moment the first phase lands and a
mixed state is reachable — add it then, and verify it by deliberately mixing two generations and
watching it go red.** This is the failure mode the project has already suffered once (a metric that
changed mid-flight, read as a seed effect), so it is not hypothetical.

**The route dataset is the sharpest case.** v1 can support route artifacts for only 5 cell-seeds; the
matrix-wide, chemist-facing dataset exists only as a by-product of this re-run (`benchmark_v2/README`,
route readiness as a hard gate). The exporter is being built now against the cells that exist, so
**the machinery is what is proven today, not the dataset.** Do not describe the dataset as existing
until the phase that produces it has passed §6.

---

## 11. Open questions

1. **A second untrained signal for sEH / DRD2 / ClpP** (§7.1). 6TD3-B is settled, but its resolution
   made reward and gate the same column again, so even there the standing exploitation check needs a
   signal that is not a component of `cnn_vs` — `vina_t2` is the candidate. The other three targets
   have no such signal at all. Free for the docking targets; ~100 molecules/cell for sEH.
2. **How much of the delivered library sits on depth-0 (bought) hubs?** (§2.2) — unmeasured, and
   `--pool all` raises depth-0 exposure 4× in the walked prefix. v1's hub-ordering arms kept only
   plots, not the per-step curves, so the v1 baseline for this cannot be recovered. Answer it in the
   arm-A pilot, alongside the `--min-hub-depth 1` sensitivity arm.
3. **Do pruned pools change more widely than naive ones under the new gates?** (§4) — unmeasured.
4. **Arm B's downstream** — paused by decision; revisit only if arm A trains badly.
5. **SynFormer** — 1 of 9 cells; blocked on a worker-pool memory leak. If it stays blocked, the
   reaction-grounded / not-a-GFlowNet quadrant has one entrant and thin coverage.
