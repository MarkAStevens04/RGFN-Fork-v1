"""Stop a training run when the ORACLE-CALL budget is spent — across requeues.

WHY THIS IS NOT ``BudgetCheckpointer``. That class SAVES at arm A and lets the run continue; nothing
in the trace machinery ever STOPPED a run. Without a stop, a cell runs to its config's iteration
count, and an iteration count is not a call budget for two of the three reaction-GFNs:

    RGFN     ~120.6 calls/iter -> 5,000 iters = ~603,000   (1.9x OVER a declared 320,000)
    SCENT      64.0 calls/iter -> 5,000 iters =  320,000   (exact -- it is SCENT's own arithmetic)
    RxnFlow    ~31   calls/iter -> 5,000 iters = ~155,000   (0.48x UNDER, and unreachable)
               -- and that ~31 is the VALID-molecule count, not a dedup effect. See below.

A 4x spread on the axis the campaign says it controls is not a budget. RGFN's figure is measured from
its own pilot (``arm_a.json``: 10,007 calls at iteration 83), not derived from a nominal batch size —
which is the point, since its nominal batch is 100 while RxnFlow's measured rate is half ITS nominal
64.

CORRECTION (2026-09-13): I first attributed RxnFlow's ~31/iter to a dedup cache absorbing
re-proposed molecules. That was WRONG, and the trace says so: of 187 training rows across six
iterations, 181 were distinct, only 2 molecules appeared in more than one iteration, and the
per-iteration repeat count was 2/0/0/0/0/2. A dedup would have produced that pattern; so does simply
not repeating. The real cause is `valid_smis` -- rxnflow/task.py:73 passes the batch with INVALID
molecules dropped, so ~31 of 64 sampled molecules reach the reward. RxnFlow's seh/drd2 path holds no
cache at all (SEHFrozenReward / DRD2FrozenReward have no `_cache`); only its docking bridge does.

CONSEQUENCE FOR THE STOP ON THOSE SIX CELLS (rxnflow x {seh, drd2} x 3 seeds): with no cache, every
presentation IS a real oracle invocation, so halting at 320,000 DISTINCT means the cell has already
made distinct/(1 - repeat_rate) real calls. At the measured 3.2% repeat rate that is ~330,600, a
~3.3% overshoot -- bounded and small, but real, and in the direction that flatters us. The clean fix
is to give those two reward classes the memoisation PMO's own harness has
(optimizer.py:150-173, a dict keyed by canonical SMILES), which makes distinct == invocations
everywhere instead of almost everywhere. Not done here: it changes competitor-shared reward classes
and is the researcher's call.

⚠ THE COUNT MUST SURVIVE A REQUEUE, AND THE LIVE COUNTER DOES NOT. ``TraceWriter.__init__`` sets
``n_scored`` and ``n_train_scored`` to 0 and THEN rotates the previous file to ``trace.csv.N``, with
no resume seed. So a stop that reads the live counter restarts from zero on every requeue. A SCENT
arm-B cell requeues two or three times on the docking targets, which at a declared 320,000 would
train 640,000-960,000 calls with every artifact looking healthy and only a concatenation of the trace
files telling the truth.

⚠ AND IT COUNTS DISTINCT TRAINING MOLECULES, NOT ROWS AND NOT A COUNTER (researcher's ruling,
2026-09-13). Three readings of one finished trace give 12,048 / 11,048 / 10,048 depending on whether
you take the last ``n_scored``, its max over train rows, or a COUNT of train rows — S3-GFN
interleaves a 2,000-molecule evaluation sample with training, so any counter absorbs evaluation. But
the row count is not the budget either: several reward paths CACHE, so a re-proposed molecule never
reaches the oracle and must not be charged. Upstream had already defined this —
``CachedProxyBase.n_proxy_calls`` returns ``len(self.cache)``, a proxy call IS a distinct state —
while our budget counted rows. Measured repeat rates: 0.0% for the four competitors whose surrogate
rewards hold no cache, 5.6% FragGFN, up to 65.6% S3-GFN, 28.4% RGFN.

So the gate is ``len(distinct training SMILES)``, taken as a UNION across rounds. Adding per-round
distinct counts would double-charge exactly the molecules a resumed run is most likely to revisit.

WHY SIBLINGS CAN BE SUMMED WITHOUT A SHAPE CHECK. Inside a v2 TRAINING directory a ``trace.csv.N``
can only have come from a v2 requeue, which is always a continuation: ``copy_forward`` resolves v1's
rotations on the way in (it picks the real history and lands it as ``trace.csv``), and Stage 2's
re-invocation — the one case that produces an alternatives-shaped rotation — happens after the cell
is frozen and writes to its own directory. So the prior rounds are disjoint earlier halves of this
same run and add. Outside that setting, use ``best_trace(..., combine=...)``, which discriminates.

COST. The siblings are immutable once rotated, so they are read ONCE at construction and their
molecules are seeded into the writer's own train-distinct set. Every subsequent check is then
``len(that set)`` — O(1), and correct as a union for free. Re-parsing a 320,000-row file, or
unioning two large sets, at every iteration boundary would not be.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


class BudgetReached(Exception):
    """Raised at an iteration boundary once the budget is spent.

    A control-flow signal, not an error: the caller catches it around ``trainer.train()`` and
    proceeds to its normal check-pointing and sampling. It is a dedicated type precisely so a caller
    cannot confuse it with a training failure and so a bare ``except Exception`` around the loop
    does not swallow it silently.
    """

    def __init__(self, n_train_scored: int, budget: int, iteration: int) -> None:
        super().__init__(
            f"oracle-call budget reached: {n_train_scored:,} train calls >= {budget:,} "
            f"at iteration {iteration}"
        )
        self.n_train_scored = n_train_scored
        self.budget = budget
        self.iteration = iteration


def prior_train_smiles(trace_path: Path | str) -> set:
    """DISTINCT training SMILES already recorded in ROTATED siblings of ``trace_path``.

    A SET, not a count, and that is load-bearing in two directions.

    THE BUDGET COUNTS MOLECULES THAT REACHED THE ORACLE (researcher's ruling, 2026-09-13), not
    molecules presented to the reward. Several reward paths cache -- the docking bridges all do, and
    RGFN/SCENT go through ``SehMoleculeProxy``, a ``CachedProxyBase`` -- so a re-proposed molecule
    costs nothing and must not be charged. Upstream had already settled this: ``n_proxy_calls``
    returns ``len(self.cache)``, i.e. a proxy call IS a distinct state, while our budget was counting
    rows. Measured gaps: RGFN 28.4% of presentations are repeats, s3gfn up to 65.6%, and the four
    competitors with no cache 0.0%.

    AND THE UNION MUST BE TAKEN ACROSS ROUNDS, NOT PER ROUND AND ADDED. A molecule sampled in round 1
    and again in round 2 is ONE oracle call: it is already in the reward's cache when round 2 asks.
    Adding per-round distinct counts double-charges exactly the molecules a resumed run is most
    likely to revisit. This is the same trap ``best_trace`` hit when summing ``n_distinct``.

    ``trace.csv`` itself is deliberately excluded -- that is the live file, and the live writer
    tracks it.
    """
    p = Path(trace_path)
    seen: set = set()
    for sib in sorted(p.parent.glob(p.name + ".*")):
        try:
            with open(sib, newline="") as fh:
                for r in csv.DictReader(fh):
                    if (r.get("phase") or "") == "train":
                        s = r.get("smiles")
                        if s:
                            seen.add(s)
        except Exception as exc:  # noqa: BLE001 - a damaged sibling must not kill a training run
            print(
                f"[budget-stop] WARNING could not read {sib.name} ({exc}); its molecules are NOT "
                f"counted, so this run may train PAST its budget rather than short of it",
                flush=True,
            )
    return seen


class BudgetStopper:
    """Raises :class:`BudgetReached` at the first iteration boundary at or after ``budget``.

    Checked at an iteration BOUNDARY rather than per molecule, for the same reason
    ``BudgetCheckpointer`` is: a run interrupted mid-iteration has incoherent optimizer and replay
    state. The overshoot is therefore under one batch, which is the same tolerance the arm-A
    checkpoint already carries and is recorded rather than assumed.
    """

    def __init__(
        self, trace, budget: int, trace_path: Path | str, tag: str = "budget-stop"
    ) -> None:
        self.trace = trace
        self.budget = int(budget)
        self.tag = tag
        self.fired = False

        # SEEDED INTO THE WRITER'S OWN SET rather than kept beside it. The writer already adds every
        # training molecule to `_train_seen`, so seeding it with the earlier rounds' molecules makes
        # `len(_train_seen)` the CELL-level union, maintained incrementally and read in O(1). The
        # alternative -- holding a second set here and unioning on every check -- is O(n) per
        # iteration, which at 320,000 molecules over 5,000 iterations is not affordable. It is also
        # the only way the union is correct: a molecule seen in an earlier round AND this one must
        # count once, and that falls out of a shared set for free.
        prior = prior_train_smiles(trace_path)
        self.n_prior = len(prior)
        if prior and hasattr(trace, "_train_seen"):
            trace._train_seen |= prior
        elif prior:
            print(
                f"[{self.tag}] WARNING trace has no _train_seen set; {len(prior):,} molecules from "
                f"earlier rounds cannot be credited and this run will train PAST its budget",
                flush=True,
            )
        if prior:
            print(
                f"[{self.tag}] resuming with {self.n_prior:,} DISTINCT training molecules already "
                f"scored in rotated trace siblings; budget {self.budget:,}",
                flush=True,
            )

    @property
    def total_train_scored(self) -> int:
        """Distinct training molecules this CELL has sent to the oracle, across all rounds."""
        return int(getattr(self.trace, "n_train_distinct", 0))

    def note_iteration(self, iteration_idx: int) -> None:
        if self.fired:
            return
        total = self.total_train_scored
        if total < self.budget:
            return
        self.fired = True
        print(
            f"[{self.tag}] budget reached at iteration {iteration_idx}: {total:,} DISTINCT training "
            f"molecules ({self.n_prior:,} of them from earlier rounds; "
            f"{int(getattr(self.trace, 'n_train_scored', 0)):,} presentations this process) "
            f">= {self.budget:,}",
            flush=True,
        )
        raise BudgetReached(total, self.budget, iteration_idx)

    def summary(self) -> dict:
        return {
            "budget_oracle_calls": self.budget,
            "fired": self.fired,
            # THE GATE: distinct molecules that reached the oracle, cell-wide.
            "n_train_distinct_total": self.total_train_scored,
            "n_train_distinct_prior_rounds": self.n_prior,
            # Diagnostics. The gap between presentations and distinct is the repeat rate, which is a
            # real per-generator property (0.0% for the uncached competitors, 28.4% for RGFN), so it
            # is reported rather than discarded.
            "n_train_presented_this_process": int(getattr(self.trace, "n_train_scored", 0)),
        }


def arm_b_budget(default: Optional[int] = None) -> Optional[int]:
    """Arm-B budget from ``BENCHMARK_V2_ARM_B_CALLS``; ``default`` (usually None) when unset.

    Unset means NO STOP, which is correct for arm-A-only cells and for every existing config. A
    caller that wants arm B must say so, so no run silently acquires a stop it was not launched with.
    """
    import os

    raw = os.environ.get("BENCHMARK_V2_ARM_B_CALLS")
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        print(
            f"[budget-stop] WARNING BENCHMARK_V2_ARM_B_CALLS={raw!r} is not an int; no stop applied",
            flush=True,
        )
        return default
    if val <= 0:
        return default
    return val
