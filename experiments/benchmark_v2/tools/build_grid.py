#!/usr/bin/env python
"""Regenerate ``benchmark_v2/grid.csv`` — the campaign's cell list — from the taxonomy.

WHY THIS IS GENERATED AND NOT HAND-MAINTAINED. ``grid.csv`` is what every driver reads to know
which cells exist, so a hand-edited copy is a single point of silent divergence: one stale row and a
launcher trains a cell nobody expects, or skips one everybody assumed was running. Generating it
means the taxonomy lives in ONE place (this file) and the CSV is a build product that can be
diffed and regenerated at will.

WHAT THE FILE DOES AND DOES NOT CONTAIN. It holds the PLAN — which cells exist, which class each
generator belongs to, which pipeline it runs through, which phase it lands in, and what we EXPECT
its training artifact's origin to be. It deliberately holds **no live status**: whether a cell is
trained, verified or frozen is a property of the filesystem, and a status column in a plan file goes
stale the moment anything runs. ``manifest.py`` computes that live, every load. This is the same
discipline ``matrix16/manifest.py`` already applies ("live filesystem status ... computed at load
time so the manifest never goes stale").

``train_plan`` IS A PLAN, NOT AN OBSERVATION. It records what we intended when the campaign was
scoped; it is not evidence about what is on disk. Re-derive it against reality before relying on it
-- ``manifest.py --status`` prints both side by side, and the copy-forward step owns reconciling
them. As of scoping, several rows are known to be optimistic: FragGFN and SynFormer were scoped as
``generate`` and have since been produced at the normalised budget, while S3-GFN was scoped as
``copy`` and is the one whose traces actually need regenerating.

THE TAXONOMY (researcher's decision, 2026-08-28) — four classes, and hub-batching applies to
exactly one of them:

                        | GFlowNet                      | not a GFlowNet
    reaction-grounded   | RGFN, RxnFlow, SCENT  <-ours  | SynFormer
    not reaction-ground | FragGFN, S3-GFN               | REINVENT, Saturn, TANGO

FragGFN sits with S3-GFN, NOT with our generators: its move is a fragment attachment, not a
reaction, so its "reactions" are not bench steps and it cannot carry a hub-batching arm. It is a
competitor and a cost-model control.

Run:  python experiments/benchmark_v2/tools/build_grid.py [--out grid.csv] [--check]

``--check`` regenerates in memory and diffs against the committed file, exiting non-zero on drift —
so CI (or a human) can prove the CSV still matches this definition.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_ROOT = HERE.parent
REPO_ROOT = V2_ROOT.parents[1]

# Resolve targets from the ONE source of truth so a gate or a new system is a one-line edit there,
# never a second copy here. Unknown targets raise (targets.get_target does), which is deliberate:
# a silently-defaulted target is how 6td3b nearly inherited the wrong direction.
sys.path.insert(0, str(REPO_ROOT / "experiments" / "lsd_hubs" / "matrix16"))
from targets import TARGETS  # noqa: E402

# ---------------------------------------------------------------------------------------------
# The taxonomy. `role` decides the stage graph: hub_batching runs sample -> pick_hubs -> enumerate
# -> campaign; competitor runs upsample -> retrosynthesis -> selection.
# ---------------------------------------------------------------------------------------------
GENERATORS = [
    # name,        gen_class,             role,           train_plan(phase 1), note
    (
        "rgfn",
        "reaction_gfn",
        "hub_batching",
        "generate",
        "batch 100/step = upstream configs/rgfn_base.gin train_forward_n_trajectories",
    ),
    (
        "rxnflow",
        "reaction_gfn",
        "hub_batching",
        "generate",
        "batch 64/step = authors' algo.num_from_policy, inherited (the runner never sets it). The 128 "
        "in rxnflow_*_5k.yaml is reward.batch_size -- the sEH proxy's SCORING batch, a throughput knob "
        "with no effect on the training budget. Two knobs, one name",
    ),
    (
        "scent",
        "reaction_gfn",
        "hub_batching",
        "generate",
        "batch 64 = clone default; only generator with a dynamic library, so only one needing recipes",
    ),
    (
        "fraggfn",
        "nonreaction_gfn",
        "competitor",
        "copy",
        "reclassified 2026-08-28 and RETRAINED at the normalised budget, so the ~30x-over-budget runs "
        "in fraggfn_*_5k are superseded rather than reused. Stage-2 capable (it has a sampler)",
    ),
    (
        "s3gfn",
        "nonreaction_gfn",
        "competitor",
        "copy",
        "corrected to ~10k calls 2026-08-21; aux_coefficient restored to the authors' 0.001",
    ),
    (
        "synformer",
        "reaction_nongfn",
        "competitor",
        "copy",
        "EXEMPT from Stage 2: its pool is a slice of an accumulated GA population, so more molecules "
        "means more TRAINING. Those cells stay pool-limited by construction, and that is a finding",
    ),
    (
        "reinvent",
        "nonreaction_nongfn",
        "competitor",
        "copy",
        "corrected to batch 64 x 157 steps = 10,048 scored",
    ),
    (
        "saturn",
        "nonreaction_nongfn",
        "competitor",
        "copy",
        "config verbatim from the authors' constrained_synthesizability/experiment.json",
    ),
    (
        "tango",
        "nonreaction_nongfn",
        "competitor",
        "copy",
        "Saturn's generator + a constrained-synthesizability reward; ships routes natively (ARM 2 only)",
    ),
]

# Phase 1 is everything except the CDK12-DDB1 system, which is explicitly lower priority for every
# generator. Both phases use the same seeds and the same standards.
PHASE = {"seh": 1, "drd2": 1, "clpp": 1, "6td3b": 2}
SEEDS = (42, 43, 44)

# PER-CELL OVERRIDES. Three cells differ from their generator's plan, and each difference is a fact
# about that one run rather than about the generator, so it cannot live in the table above.
# Reconciled against disk 2026-09-07 (agent D measured, verified independently here).
#
# `resample` is a FOURTH verdict, deliberately not folded into `generate`: drawing a fresh pool from
# a surviving frozen policy costs minutes, where re-training costs GPU-days. Collapsing them would
# make a cheap cell look expensive and invite someone to skip it.
#
# WHY s3gfn LOOKED LIKE THE PROBLEM CHILD AND IS NOT. Reading `trace.csv` without checking its
# rotations made nine cells across three generators appear historyless. A re-invoked runner truncates
# trace.csv to its 59-byte header while the full history survives as `trace.csv.1` -- the defect
# 3281bce fixed by rotating. Verified here: all three fraggfn_drd2 seeds show trace.csv at 0 rows and
# trace.csv.1 at 10,048. The copy step resolves the real trace, so those cells copy normally.
CELL_OVERRIDES = {
    # (generator, target, seed): (train_plan, note)
    #
    # THE SIX ClpP CELLS RE-SCOPED FROM copy TO generate (researcher's decision, 2026-09-24).
    # These were copied forward from v1 and accepted, then found to be 6-15% SHORT of arm A's
    # 10,000-DISTINCT budget: fraggfn 9,258-9,375, s3gfn 8,451-8,912. The cause is the same one
    # that put fraggfn's 6TD3-B cells at 2.50x -- these two entrants have no "stop at N molecules"
    # setting, so their round count IS the budget, and a round count is a PRESENTATION budget while
    # arm A is a DISTINCT one. They agree only at a ~100% unique rate, which neither entrant has.
    #
    # Re-scoped rather than re-copied because the correction is a RE-TRAIN: the v1 run stopped at
    # 157 rounds and the budget needs more. A cell we train in v2 is a `generate` cell, and saying
    # so here is what lets the v2 driver touch it -- submit_train_v2.sh refuses any other plan, by
    # design, so that copy_forward keeps sole ownership of the cells it produced.
    #
    # ⚠ THE PROVENANCE CHANGES WITH THE PLAN. arm_meta.json for these six currently records
    # origin="copied" and a v1 source_run_dir; once retrained they are v2 runs and that record must
    # be rewritten, not left to say they came from v1.
    ("fraggfn", "clpp", 42): (
        "generate",
        "re-scoped 2026-09-24: v1 run reached 9,371 of 10,000 distinct (6% short). fraggfn EXTENDS "
        "from its checkpoint (remaining = n_train_steps - loop._it), so 157 -> 169 rounds continues "
        "the same run rather than restarting it",
    ),
    ("fraggfn", "clpp", 43): (
        "generate",
        "re-scoped 2026-09-24: 9,375 of 10,000 distinct; extends from checkpoint to 169 rounds",
    ),
    ("fraggfn", "clpp", 44): (
        "generate",
        "re-scoped 2026-09-24: 9,258 of 10,000 distinct (the worst seed, which set the 169); "
        "extends from checkpoint",
    ),
    ("s3gfn", "clpp", 42): (
        "generate",
        "re-scoped 2026-09-24: v1 run reached 8,451 of 10,000 distinct (15% short). s3gfn does NOT "
        "extend -- its runner SKIPS training entirely when a model file exists ('already present -- "
        "skipping training and going straight to sampling'), so this is a clean re-run at 202 rounds",
    ),
    ("s3gfn", "clpp", 43): (
        "generate",
        "re-scoped 2026-09-24: 8,912 of 10,000 distinct; clean re-run at 202 rounds (no extend path)",
    ),
    ("s3gfn", "clpp", 44): (
        "generate",
        "re-scoped 2026-09-24: 8,735 of 10,000 distinct; clean re-run at 202 rounds (no extend path)",
    ),
    ("s3gfn", "seh", 42): (
        "resample",
        "candidates.csv lost to a re-invoked runner, but 10 checkpoints and a 10,048-call trace "
        "survive -> re-draw the pool from the frozen policy (minutes), NOT a re-train",
    ),
    ("s3gfn", "seh", 43): (
        "copy",
        "trace unrecoverable -- trace.csv and every rotation are header-only stubs. Copyable, but "
        "Stage 2 loses its free-pool harvest here and must sample the checkpoint instead",
    ),
    ("synformer", "drd2", 43): (
        "copy",
        "trace short at 6,950 of 10,000 training calls -- usable but under budget; do not quote "
        "this cell in a budget-matched comparison without saying so",
    ),
}

# S3-GFN's trace needs care and the reason is not obvious. `n_scored` is a SINGLE cumulative counter
# shared across phases, and S3-GFN INTERLEAVES its 2,000-molecule evaluation sample with training
# rather than appending it (measured on s3gfn_drd2/42: three phase switches, eval rows spanning
# n_scored 65..12,048). So the final counter reads 12,048 while the training budget is 10,048 -- the
# real figure is the COUNT of `phase == "train"` rows, which is what verify_cell gates on. For the
# five entrants with no eval phase the two agree; for S3-GFN's nine cells, reading the counter makes
# an on-budget cell look 20% over, and would let a genuinely short cell pass.

# Both arms are defined on the ORACLE-CALL axis, never the step axis: the three reaction-GFNs have
# three different per-step call counts and replay buffers make the arithmetic unsettleable, so the
# trace counter decides when a checkpoint is written, not multiplication.
ARM_A_CALLS = 10_000  # PMO convention; the only budget at which cross-generator comparison holds
ARM_B_CALLS = 320_000  # our previously-published budget; reaction-GFNs only, internal comparison

COLUMNS = [
    "generator",
    "gen_class",
    "role",
    "pipeline",
    "target",
    "seed",
    "phase",
    "arm_a_calls",
    "arm_b_calls",
    "train_plan",
    "note",
]


def build_rows():
    rows = []
    for gen, klass, role, plan1, note in GENERATORS:
        for target, phase in sorted(PHASE.items(), key=lambda kv: (kv[1], kv[0])):
            if target not in TARGETS:
                raise KeyError(f"target {target!r} is not in matrix16/targets.py")
            for seed in SEEDS:
                # A phase-2 cell has never been run for ANY generator, so it always generates.
                plan = plan1 if phase == 1 else "generate"
                cell_note = (
                    note if (phase == 1 or plan1 == "generate") else "never run on this target"
                )
                override = CELL_OVERRIDES.get((gen, target, seed))
                if override and phase == 1:
                    plan, cell_note = override
                rows.append(
                    {
                        "generator": gen,
                        "gen_class": klass,
                        "role": role,
                        "pipeline": "ours" if role == "hub_batching" else "competitor",
                        "target": target,
                        "seed": seed,
                        "phase": phase,
                        "arm_a_calls": ARM_A_CALLS,
                        # Arm B is the reaction-GFNs' continuation only. Giving it to a competitor would
                        # hand it a budget its own paper never uses, which is the error this campaign
                        # exists to correct.
                        "arm_b_calls": ARM_B_CALLS if role == "hub_batching" else "",
                        "train_plan": plan,
                        "note": cell_note,
                    }
                )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=V2_ROOT / "grid.csv")
    ap.add_argument(
        "--check",
        action="store_true",
        help="diff against the existing file instead of writing; exit 1 on drift",
    )
    a = ap.parse_args()

    rows = build_rows()
    lines = [",".join(COLUMNS)]
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    text = buf.getvalue()

    if a.check:
        if not a.out.exists():
            print(f"MISSING: {a.out}", file=sys.stderr)
            return 1
        if a.out.read_text() != text:
            print(f"DRIFT: {a.out} does not match build_grid.py", file=sys.stderr)
            return 1
        print(f"OK: {a.out} matches build_grid.py ({len(rows)} cells)")
        return 0

    a.out.write_text(text)
    n_phase = {}
    n_role = {}
    n_plan = {}
    for r in rows:
        n_phase[r["phase"]] = n_phase.get(r["phase"], 0) + 1
        n_role[r["role"]] = n_role.get(r["role"], 0) + 1
        n_plan[r["train_plan"]] = n_plan.get(r["train_plan"], 0) + 1
    print(f"wrote {a.out}  ({len(rows)} training cells)")
    print(f"  phase      : {dict(sorted(n_phase.items()))}")
    print(f"  role       : {dict(sorted(n_role.items()))}")
    print(f"  train_plan : {dict(sorted(n_plan.items()))}  <- a PLAN; reconcile against disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
