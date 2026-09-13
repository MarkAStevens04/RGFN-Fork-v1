"""Stop a training run when the ORACLE-CALL budget is spent — across requeues.

WHY THIS IS NOT ``BudgetCheckpointer``. That class SAVES at arm A and lets the run continue; nothing
in the trace machinery ever STOPPED a run. Without a stop, a cell runs to its config's iteration
count, and an iteration count is not a call budget for two of the three reaction-GFNs:

    RGFN     ~120.6 calls/iter -> 5,000 iters = ~603,000   (1.9x OVER a declared 320,000)
    SCENT      64.0 calls/iter -> 5,000 iters =  320,000   (exact -- it is SCENT's own arithmetic)
    RxnFlow    ~31   calls/iter -> 5,000 iters = ~155,000   (0.48x UNDER, and unreachable)

A 4x spread on the axis the campaign says it controls is not a budget. RGFN's figure is measured from
its own pilot (``arm_a.json``: 10,007 calls at iteration 83), not derived from a nominal batch size —
which is the point, since its nominal batch is 100 and RxnFlow's measured rate is half ITS nominal 64
because a dedup cache absorbs re-proposed molecules.

⚠ THE COUNT MUST SURVIVE A REQUEUE, AND THE LIVE COUNTER DOES NOT. ``TraceWriter.__init__`` sets
``n_scored`` and ``n_train_scored`` to 0 and THEN rotates the previous file to ``trace.csv.N``, with
no resume seed. So a stop that reads the live counter restarts from zero on every requeue. A SCENT
arm-B cell requeues two or three times on the docking targets, which at a declared 320,000 would
train 640,000-960,000 calls with every artifact looking healthy and only a concatenation of the trace
files telling the truth.

⚠ AND IT MUST COUNT ROWS, NOT READ A COUNTER. One finished trace reads 12,048 / 11,048 / 10,048
depending on whether you take the last ``n_scored``, its max over train rows, or an actual COUNT of
train rows — S3-GFN interleaves a 2,000-molecule evaluation sample with training. Only the count is
immune, so only the count is used here.

WHY SIBLINGS CAN BE SUMMED WITHOUT A SHAPE CHECK. Inside a v2 TRAINING directory a ``trace.csv.N``
can only have come from a v2 requeue, which is always a continuation: ``copy_forward`` resolves v1's
rotations on the way in (it picks the real history and lands it as ``trace.csv``), and Stage 2's
re-invocation — the one case that produces an alternatives-shaped rotation — happens after the cell
is frozen and writes to its own directory. So the prior rounds are disjoint earlier halves of this
same run and add. Outside that setting, use ``best_trace(..., combine=...)``, which discriminates.

COST. The siblings are immutable once rotated, so they are counted ONCE at construction; every
subsequent check is ``prior + trace.n_train_scored``, i.e. O(1). Re-parsing a 320,000-row file at
every iteration boundary would not be.
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


def count_prior_train_rows(trace_path: Path | str) -> int:
    """Train rows already recorded in ROTATED siblings of ``trace_path``.

    Counted once: a rotated file is never written again. ``trace.csv`` itself is deliberately NOT
    counted -- that is the live file, and the live writer's own counter tracks it.
    """
    p = Path(trace_path)
    total = 0
    for sib in sorted(p.parent.glob(p.name + ".*")):
        try:
            with open(sib, newline="") as fh:
                total += sum(1 for r in csv.DictReader(fh) if (r.get("phase") or "") == "train")
        except Exception as exc:  # noqa: BLE001 - a damaged sibling must not kill a training run
            print(
                f"[budget-stop] WARNING could not count {sib.name} ({exc}); its rows are NOT "
                f"included, so this run may train PAST its budget rather than short of it",
                flush=True,
            )
    return total


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
        self.prior = count_prior_train_rows(trace_path)
        self.fired = False
        if self.prior:
            print(
                f"[{self.tag}] resuming with {self.prior:,} train calls already recorded in "
                f"rotated trace siblings; budget {self.budget:,}",
                flush=True,
            )

    @property
    def total_train_scored(self) -> int:
        """Calls spent by this CELL, not by this process."""
        return self.prior + int(getattr(self.trace, "n_train_scored", 0))

    def note_iteration(self, iteration_idx: int) -> None:
        if self.fired:
            return
        total = self.total_train_scored
        if total < self.budget:
            return
        self.fired = True
        print(
            f"[{self.tag}] budget reached at iteration {iteration_idx}: {total:,} train calls "
            f"(this process {int(getattr(self.trace, 'n_train_scored', 0)):,} + "
            f"{self.prior:,} from earlier rounds) >= {self.budget:,}",
            flush=True,
        )
        raise BudgetReached(total, self.budget, iteration_idx)

    def summary(self) -> dict:
        return {
            "budget_oracle_calls": self.budget,
            "fired": self.fired,
            "n_train_scored_total": self.total_train_scored,
            "n_train_scored_this_process": int(getattr(self.trace, "n_train_scored", 0)),
            "n_train_scored_prior_rounds": self.prior,
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
