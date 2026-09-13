#!/usr/bin/env python
"""Migrate landed ``arm_meta.json`` files off the pre-2026-09-12 key name. Ships WITH the rename.

WHY A MIGRATION EXISTS AT ALL. ``n_scored_at_checkpoint`` was renamed to
``n_train_scored_at_checkpoint`` (the gate) plus ``n_total_scored_at_checkpoint`` (diagnostic),
because the old name did not say WHICH count it held and one trace yields three defensible readings
-- on s3gfn/seh/43 the cumulative counter ends at 12,048, ``max(n_scored)`` over train rows reads
11,048, and the real budget is 10,048 train rows. That ambiguity had already produced one bug in the
budget checkpointer.

WHAT MADE THE RENAME NON-TRIVIAL, checked before it was decided rather than after: 54 arm_meta.json
files had already landed and **52 of them were FROZEN** (``chmod -R a-w``). A bare rename would have
stranded 52 ACCEPTED cells behind a key the verifier no longer reads, in directories nothing can
write. What makes it tractable is that ``arm_meta.json`` is AUTHORED by ``copy_forward.py``, not
copied from v1 -- it is absent from ``.copy_manifest.json``, so rewriting it does not invalidate any
backup md5 and ``verify_backup.py`` stays green. Confirmed by reading a real manifest, not assumed.

IT RE-DERIVES FROM THE TRACE RATHER THAN RENAMING THE OLD VALUE IN PLACE. Carrying the recorded
number across under a new name would migrate a wrong value as faithfully as a right one. Both
figures are recomputed from the cell's own landed trace, and a disagreement with what was recorded
is REPORTED AND REFUSED, never silently corrected -- a migration that quietly changes a budget is
indistinguishable from one that quietly corrupts it.

FREEZE/UNFREEZE GOES THROUGH freeze_cell.sh, not through hand-rolled chmods. That script already
owns the invariant that a cell is verified before it is sealed, so re-freezing re-runs verification
and a cell that fails for any other reason stays unfrozen and visible instead of being resealed on
trust.

Usage:
    python experiments/benchmark_v2/tools/migrate_arm_meta.py --all            # dry run
    python experiments/benchmark_v2/tools/migrate_arm_meta.py --all --execute
    python experiments/benchmark_v2/tools/migrate_arm_meta.py --cell reinvent/clpp/42 --execute

Exit 0 = nothing left to migrate (or dry run clean). 1 = something needs a human.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manifest  # noqa: E402
from inventory_v1_cells import best_trace  # noqa: E402

LEGACY = "n_scored_at_checkpoint"
TRAIN_KEY = "n_train_scored_at_checkpoint"
TOTAL_KEY = "n_total_scored_at_checkpoint"
COMBINE_KEY = "n_train_scored_combine"

# THIS BRINGS arm_meta UP TO THE CURRENT SCHEMA; IT IS NOT A ONE-SHOT RENAME. The first version
# checked only for TRAIN_KEY and called anything holding it "already migrated" -- so when
# `n_train_scored_combine` was added to both writers hours later, all 54 landed cells reported done
# while carrying none of it. A migration that defines "done" as "has the field I added" stops being
# a migration the moment the schema moves again. It now reconciles against the full expected set, so
# the next field costs an entry here and a re-run.


def plan_one(cell, arm: str) -> tuple[str, str, dict | None]:
    """(state, detail, new_meta). States: done / migrate / skip / PROBLEM."""
    d = cell.train_dir(arm)
    p = d / "arm_meta.json"
    if not p.is_file():
        return "skip", "no arm_meta.json (cell not landed)", None
    try:
        meta = json.loads(p.read_text())
    except Exception as e:
        return "PROBLEM", f"unparseable arm_meta.json: {e}", None

    if TRAIN_KEY not in meta and LEGACY not in meta:
        return "PROBLEM", f"neither {LEGACY} nor {TRAIN_KEY} present", None

    # Re-derive rather than rename in place, and take the MODE FROM best_trace rather than asserting
    # it. These are copied v1 cells, so the default (max over a cell's alternative trace files) is
    # the right reading -- but reading it back out of the function is what keeps this honest if that
    # default ever changes underneath us.
    t = best_trace(d) or {}
    train_rows = t.get("train_rows") or 0
    total = t.get("n_scored") or 0
    combine = t.get("combined", "max")
    if not train_rows:
        return "PROBLEM", "no readable trace -- cannot re-derive the budget", None

    recorded = meta.get(TRAIN_KEY, meta.get(LEGACY))
    if int(recorded) != int(train_rows):
        # Do NOT migrate a value we cannot reproduce. Renaming it would carry a wrong number across
        # under a name that asserts more precision than the old one did.
        return (
            "PROBLEM",
            (
                f"recorded {int(recorded):,} but the landed trace has "
                f"{train_rows:,} train rows -- refusing to migrate a value that does "
                f"not reproduce"
            ),
            None,
        )

    new = dict(meta)
    new.pop(LEGACY, None)
    new[TRAIN_KEY] = int(train_rows)
    new[TOTAL_KEY] = int(total)
    # THE MODE TRAVELS WITH THE VALUE, on every cell and not just the ones written from now on.
    # write_arm_meta sums a requeue's rotations; copy_forward takes the max of a v1 cell's
    # alternatives. Both are right for their own shape, and they coincide on every landed cell today
    # only because none has two non-empty traces -- an accident, which is exactly why the value must
    # never appear without the mode beside it.
    new[COMBINE_KEY] = combine

    if new == meta:
        return "done", "already at the current schema", None

    added = sorted(set(new) - set(meta))
    detail = f"{train_rows:,} train rows, combine={combine}"
    if LEGACY in meta:
        detail += f"; renamed {LEGACY}"
    if added:
        detail += f"; +{','.join(added)}"
    return "migrate", detail, new


def run(cmd: list[str]) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def migrate_one(cell, arm: str, new: dict) -> tuple[bool, str]:
    d = cell.train_dir(arm)
    # target_name, NOT target: `Cell.target` resolves to a Target dataclass, so an f-string here
    # interpolates its whole repr and freeze_cell.sh is handed a "cell" that cannot exist.
    tag = f"{cell.generator}/{cell.target_name}/{cell.seed}"
    was_frozen = not (d.stat().st_mode & 0o200)

    if was_frozen:
        rc, out = run([str(HERE / "freeze_cell.sh"), tag, "--arm", arm, "--unfreeze"])
        if rc != 0:
            return False, f"unfreeze failed: {out}"

    try:
        (d / "arm_meta.json").write_text(json.dumps(new, indent=2))
    except Exception as e:
        return False, f"write failed: {e}"

    if was_frozen:
        # freeze_cell.sh VERIFIES before it seals, so a cell that fails for any other reason stays
        # unfrozen and loud rather than being resealed on trust.
        rc, out = run([str(HERE / "freeze_cell.sh"), tag, "--arm", arm])
        if rc != 0:
            return False, f"written, but RE-FREEZE FAILED (cell left unfrozen): {out}"
    return True, "migrated" + (" and re-frozen" if was_frozen else "")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cell", default=None, help="GEN/TARGET/SEED")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--arm", default="a", choices=("a", "b"))
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()

    if not a.all and not a.cell:
        ap.error("pass --cell GEN/TARGET/SEED or --all")

    if a.cell:
        gen, tgt, seed = a.cell.split("/")
        cells = [manifest.get_cell(gen, tgt, int(seed))]
    else:
        cells = [c for c in manifest.load_grid() if c.has_arm(a.arm)]

    todo, problems, done = [], [], 0
    for c in cells:
        state, detail, new = plan_one(c, a.arm)
        if state == "done":
            done += 1
        elif state == "migrate":
            todo.append((c, detail, new))
        elif state == "PROBLEM":
            problems.append((c, detail))

    print(
        f"arm {a.arm}: {done} already migrated, {len(todo)} to migrate, "
        f"{len(problems)} problem(s)"
    )
    for c, why in problems:
        print(f"  PROBLEM  {c.tag:<24} {why}")
    for c, detail, _ in todo:
        print(f"  migrate  {c.tag:<24} {detail}")

    if not a.execute:
        if todo:
            print("\nDRY RUN -- nothing written. Re-run with --execute.")
        return 1 if problems else 0

    failed = 0
    for c, _, new in todo:
        ok, msg = migrate_one(c, a.arm, new)
        print(f"  {'ok      ' if ok else 'FAILED  '} {c.tag:<24} {msg}")
        failed += 0 if ok else 1
    print(f"\n{len(todo) - failed}/{len(todo)} migrated")
    return 1 if (failed or problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
