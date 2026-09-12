#!/usr/bin/env python
"""Write ``arm_meta.json`` for a GENERATED cell -- the half of a contract that had no implementer.

WHY THIS EXISTS. ``verify_cell.py`` requires every train/ directory to carry ``arm_meta.json``, and
says so in its own failure text: "missing -- the runner must record which call count this checkpoint
sits at". No runner did. ``copy_forward.py`` writes it for COPIED cells, so the 54 competitor cells
verify; nothing on the TRAINING path wrote it at all. The consequence was silent and total: every
generated cell -- all 27 reaction-GFN cells of arm A, not merely one retrain -- would fail
verification on a missing file before any substantive check ran, and a cell that cannot verify
cannot be frozen, and a cell that cannot be frozen cannot be accepted. The gate was not lenient,
it was unreachable.

⛔ RUN THIS INSIDE THE TRAINING JOB, NOT AFTERWARDS FROM A LOGIN NODE. ``--pythonhashseed`` must be
passed from the live environment of the run (``--pythonhashseed "${PYTHONHASHSEED-}"``). Reading it
in a later process would record THIS process's environment and attribute it to the run -- a
fabricated provenance that looks identical on disk to a real one. The whole point of the field is
to say what the run actually executed under, so it is an explicit argument and never a default.

THE MISSING-HASHSEED ASYMMETRY IS DELIBERATE, AND IT IS THE OPPOSITE OF THE COPIED CASE. For a
copied v1 cell the value is unrecoverable -- the v1 launchers never exported it, so nobody can ever
know it -- and ``verify_cell`` exempts that, but only when it is DECLARED (``pythonhashseed: null``
+ a note + ``origin: "copied"``). For a generated cell an absent value is not unrecoverable, it is
a launcher defect that a re-run repairs in minutes. So this tool records null + a note and lets
verification FAIL, because passing it would assert reproducibility the run does not have. A gate
you can satisfy by writing "0" is worse than one that fails honestly.

``n_scored_at_checkpoint`` COUNTS phase=="train" ROWS, and never ``max(n_scored)``. ``n_scored`` is
one cumulative counter shared across phases, and S3-GFN INTERLEAVES its evaluation sample with
training. Measured end-to-end on job 76229 (s3gfn/seh/43): the finished trace holds 12,048 rows --
10,048 train + 2,000 eval -- so the overall ``max(n_scored)`` reads 12,048, and even
``max(n_scored)`` restricted to train rows reads 11,048, because 1,000 eval rows land before the
last train row. The true training budget is the train ROW COUNT, 10,048. Reading either counter
would overstate every S3-GFN cell's budget and quietly break the arm-A comparison.

Usage (from inside the job, after the run completes):
    python experiments/benchmark_v2/tools/write_arm_meta.py \
        --run-dir "$RUN_DIR" --generator s3gfn --target seh --seed 43 \
        --pythonhashseed "${PYTHONHASHSEED-}" --job-id "$SLURM_JOB_ID" \
        --launcher experiments/benchmark_v2/pilot/submit_pilot_train.sh

Exit 0 = written. 1 = refused (and the reason is printed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from copy_forward import _final_checkpoint  # noqa: E402  the naming heuristic, not a second guess
from inventory_v1_cells import CKPT_GLOBS, best_trace  # noqa: E402


def build(run_dir: Path, gen: str, target: str, seed: int, arm: str, budget: int,
          hashseed: str | None, job_id: str | None, launcher: str | None) -> dict:
    t = best_trace(run_dir) or {}
    # DEDUPLICATED: the per-generator glob lists OVERLAP by design. S3-GFN's are
    # ["*/model_state*.pt", "*/*.pt", ...], so any file matching the first also matches the second
    # and a naive `+=` would report n_checkpoints twice its true value -- a count that later reads
    # as "this cell checkpointed more often than it did".
    rels = sorted({str(p.relative_to(run_dir))
                   for g in CKPT_GLOBS.get(gen, [])
                   for p in run_dir.glob(g) if p.is_file()})

    meta = {
        "arm": arm,
        "budget_calls": budget,
        # train ROWS -- see the module docstring. Never max(n_scored).
        "n_scored_at_checkpoint": t.get("train_rows") or t.get("n_scored"),
        "checkpoint": _final_checkpoint(rels),
        "n_checkpoints": len(rels),
        "generator": gen,
        "target": target,
        "seed": seed,
        "origin": "generated",
        "provenance": (
            f"trained in this campaign by {launcher or 'an unrecorded launcher'}"
            + (f" as SLURM job {job_id}" if job_id else "")
        ),
        "run_dir": str(run_dir),
        "trace_source": t.get("source"),
        "trace_rows": t.get("n_rows"),
    }
    if hashseed:
        meta["pythonhashseed"] = hashseed
    else:
        # Absent for a GENERATED cell is a defect, not a limitation -- and it stays failing.
        meta["pythonhashseed"] = None
        meta["pythonhashseed_note"] = (
            "the launcher did not export PYTHONHASHSEED. For a GENERATED cell this is repairable "
            "by re-running under a v2 driver, so it is NOT exempted the way a copied v1 cell is -- "
            "verification is expected to fail until the cell is re-trained with it set."
        )
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--arm", default="a")
    ap.add_argument("--budget", type=int, default=10_000)
    ap.add_argument("--pythonhashseed", default=None,
                    help='pass "${PYTHONHASHSEED-}" from the LIVE job env; empty is honest')
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--launcher", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing arm_meta.json that this tool did not write")
    a = ap.parse_args()

    run_dir: Path = a.run_dir
    if not run_dir.is_dir():
        print(f"REFUSED: {run_dir} does not exist", file=sys.stderr)
        return 1

    out = run_dir / "arm_meta.json"
    if out.is_file() and not a.force:
        try:
            prev = json.loads(out.read_text())
        except Exception:
            prev = {}
        # Never quietly relabel a COPIED cell as generated: that would erase the one field saying
        # the artifact was not produced here.
        if prev.get("origin") and prev["origin"] != "generated":
            print(f"REFUSED: {out} already records origin={prev['origin']!r}; "
                  f"re-run with --force only if you mean to relabel it", file=sys.stderr)
            return 1

    meta = build(run_dir, a.generator, a.target, a.seed, a.arm, a.budget,
                 a.pythonhashseed, a.job_id, a.launcher)
    if not meta["n_scored_at_checkpoint"]:
        print(f"REFUSED: no trace rows under {run_dir} -- refusing to declare a budget of 0",
              file=sys.stderr)
        return 1

    out.write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}")
    print(f"  n_scored_at_checkpoint = {meta['n_scored_at_checkpoint']} (train rows)")
    print(f"  checkpoint             = {meta['checkpoint']}  ({meta['n_checkpoints']} found)")
    print(f"  pythonhashseed         = {meta['pythonhashseed']!r}"
          + ("  <- VERIFICATION WILL FAIL until re-trained with it set"
             if meta["pythonhashseed"] is None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
