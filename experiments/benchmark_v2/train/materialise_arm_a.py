#!/usr/bin/env python
"""Materialise a cell's ARM A from the arm-B run that produced it.

WHY ARM A IS EXTRACTED RATHER THAN TRAINED. A reaction-GFN cell trains ONE process straight through
to arm B and check-points on the way past 10,000 oracle calls. Training arm A separately would cost a
second run AND would not describe the same trajectory: the docking reward is genuinely stochastic, so
two runs at one seed do not agree on ClpP or 6TD3-B. Extraction gives both arms from one trajectory
at zero extra training compute -- which is also why arm A's directory is the one artifact whose name
and provenance can disagree, and why TRAIN_DONE.json states the arm explicitly.

WHAT IS COPIED. The checkpoint the BudgetCheckpointer took at the crossing, plus every sidecar that
checkpoint needs to be USABLE -- for SCENT that means the guidance models (the backward policy that
``last_gfn.pt`` silently drops, unrecoverable without it), the dynamic library, and the RNG state.
A checkpoint without its sidecars verifies fine and is worthless for flow analysis.

HOW THE TRACE IS SLICED, and why not with a filter. The prefix ends at the Nth row with
``phase == "train"``. Eval rows are carried along IN POSITION rather than dropped, so the result is a
genuine PREFIX of the run's history -- the file the run would have had if it had stopped there. A
filtered copy would renumber nothing and still be a different object: it would claim the cell never
evaluated, and any modes-vs-calls curve read off it would place molecules at the wrong call counts.
``n_scored`` / ``n_distinct`` are NOT recomputed for the same reason: they are the run's own
cumulative counters at those rows, and rewriting them would fabricate a history.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

# Checkpoint + sidecars each generator writes at the arm-A crossing. Written out per generator
# rather than globbed: a glob would silently carry an arm-B artifact into the arm-A directory the
# first time someone adds a file, and arm A is the half nobody re-reads until analysis.
ARM_A_ARTIFACTS = [
    "arm_a_10k.pt",
    "guidance_models_arm_a_10k.pt",
    "dynamic_library_arm_a_10k.json",
    "rng_state_arm_a_10k.pt",
]
# Where a generator keeps its checkpoints under the run dir. Checked in order.
CKPT_SUBDIRS = ["train/checkpoints", "checkpoints", "train", "."]


def _find_artifacts(src: Path) -> tuple[Path | None, list[Path]]:
    for sub in CKPT_SUBDIRS:
        d = src / sub
        if not d.is_dir():
            continue
        found = [d / n for n in ARM_A_ARTIFACTS if (d / n).is_file()]
        if any(p.name == "arm_a_10k.pt" for p in found):
            return d, found
    return None, []


def slice_trace(src_trace: Path, dst_trace: Path, budget: int) -> dict:
    """Copy the prefix of ``src_trace`` ending at the ``budget``-th DISTINCT train molecule.

    ⚠ DISTINCT, NOT ROWS, AND THAT IS THE WHOLE POINT. ``BudgetCheckpointer`` fires when
    ``n_train_distinct`` reaches the budget (_trace.py), because the researcher's ruling is that the
    budget counts molecules that reached the oracle. Slicing by ROWS then ends the arm-A trace at a
    DIFFERENT, EARLIER point than the checkpoint it accompanies: measured on the first real smoke
    (job 76243, rgfn/seh/42 at a 600-call arm A), 600 rows held only **500 distinct**, 16.7% under
    budget, while the checkpoint had correctly fired at 600 distinct. The checkpoint and its trace
    disagreed about what the budget was, and the direction UNDER-trains our own generator.

    Counting distinct makes the slice end on the same row the checkpoint fired on BY CONSTRUCTION
    rather than by coincidence -- they are now two readings of one rule instead of two rules.

    Eval rows are carried IN POSITION so the result is a genuine prefix of the run's history, and the
    cumulative counters are left exactly as the run wrote them. A filtered copy would claim the cell
    never evaluated and would place molecules at the wrong call counts on any modes-vs-calls curve.
    """
    if not src_trace.is_file():
        return {"ok": False, "reason": f"{src_trace} does not exist"}
    kept, train_rows, total = [], 0, 0
    train_seen: set = set()
    with open(src_trace, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        for row in reader:
            total += 1
            kept.append(row)
            if (row.get("phase") or "") == "train":
                train_rows += 1
                smi = row.get("smiles")
                if smi:
                    train_seen.add(smi)
                if len(train_seen) >= budget:
                    break
    if len(train_seen) < budget:
        return {
            "ok": False,
            "reason": (
                f"only {len(train_seen):,} DISTINCT training molecules in the source trace "
                f"({train_rows:,} rows), short of the {budget:,} arm-A budget -- the arm-A "
                f"checkpoint cannot have fired, so there is nothing to materialise"
            ),
            "n_train_rows": train_rows,
            "n_train_distinct": len(train_seen),
        }
    dst_trace.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_trace, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    return {
        "ok": True,
        "n_rows": len(kept),
        # THE GATE: matches the budget the checkpoint fired on.
        "n_train_distinct": len(train_seen),
        # Diagnostics. The gap between rows and distinct is this cell's repeat rate -- 16.9% for RGFN
        # on sEH in the first real smoke, so it is reported rather than thrown away.
        "n_train_rows": train_rows,
        "n_eval_rows_carried": len(kept) - train_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from", dest="src", required=True, type=Path, help="the arm-B run dir")
    ap.add_argument("--to", dest="dst", required=True, type=Path, help="the arm-A dir to create")
    ap.add_argument("--budget", type=int, default=10_000, help="arm-A oracle-call budget")
    a = ap.parse_args()

    src: Path = a.src
    dst: Path = a.dst
    if not src.is_dir():
        print(f"REFUSED: source {src} does not exist", file=sys.stderr)
        return 1
    # Never overwrite a materialised arm A that something may already have verified or frozen.
    if (dst / "trace.csv").is_file():
        print(f"REFUSED: {dst} already holds a trace; refusing to overwrite", file=sys.stderr)
        return 1

    ckpt_dir, artifacts = _find_artifacts(src)
    if ckpt_dir is None:
        print(
            "REFUSED: no arm_a_10k.pt under "
            f"{', '.join(str(src / s) for s in CKPT_SUBDIRS)} -- the arm-A checkpoint never fired, "
            "which for a completed arm-B run means the cell never reached 10,000 train calls",
            file=sys.stderr,
        )
        return 1

    sliced = slice_trace(src / "trace.csv", dst / "trace.csv", a.budget)
    if not sliced["ok"]:
        print(f"REFUSED: {sliced['reason']}", file=sys.stderr)
        return 1

    dst_ckpt = dst / "train" / "checkpoints"
    dst_ckpt.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in artifacts:
        # Land arm_a_10k.pt under the name the rest of the pipeline looks for, and keep each
        # sidecar's own role in its name rather than its arm.
        name = {
            "arm_a_10k.pt": "last_gfn.pt",
            "guidance_models_arm_a_10k.pt": "guidance_models.pt",
            "dynamic_library_arm_a_10k.json": "dynamic_library.json",
            "rng_state_arm_a_10k.pt": "rng_state.pt",
        }[p.name]
        shutil.copy2(p, dst_ckpt / name)
        copied.append(f"{p.name} -> {name}")

    for side in (
        "arm_a.json",
        "timing.json",
        "run_config.yaml",
        "config.gin",
        "operative_config.gin",
    ):
        s = src / side
        if s.is_file():
            shutil.copy2(s, dst / side)

    prov = {
        "materialised": True,
        "materialised_from": str(src),
        "arm": "a",
        "budget_oracle_calls": a.budget,
        "trace": sliced,
        "artifacts": copied,
        "note": (
            "Arm A was EXTRACTED from the arm-B run above, not trained separately: one trajectory, "
            "check-pointed on the way past the arm-A budget. The trace here is a genuine PREFIX of "
            "that run -- eval rows are carried in position, and the cumulative counters are the "
            "run's own, not recomputed."
        ),
    }
    (dst / "MATERIALISED.json").write_text(json.dumps(prov, indent=2))
    print(
        f"materialised arm A -> {dst}\n"
        f"  trace  : {sliced['n_rows']:,} rows -- {sliced['n_train_distinct']:,} DISTINCT training "
        f"molecules (the budget), from {sliced['n_train_rows']:,} train rows + "
        f"{sliced['n_eval_rows_carried']:,} eval carried in position\n"
        f"  ckpt   : {len(copied)} artifact(s) from {ckpt_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
