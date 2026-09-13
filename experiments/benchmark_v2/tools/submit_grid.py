#!/usr/bin/env python
"""Walk the grid and submit the cells that are missing. The thing that makes the tooling a pipeline.

WHY THIS EXISTS. Every piece of the campaign existed except the one that walks it: `manifest.py`
knows which cells there are and what state each is in, `verify_cell.py` grades one, `accept_cell.sh`
finishes one, and nothing decided WHICH to run next. That gap is not neutral -- it is filled by a
person typing 55 sbatch lines over a weekend, which is exactly the process that has already
double-submitted a cell, smoke-tested into a live run directory, and left a finished cell unbacked.

⛔ IT READS LIVE STATUS, NOT ``train_plan``. This is the single rule the driver exists to enforce.
``build_grid.py`` says it outright -- "train_plan IS A PLAN, NOT AN OBSERVATION ... it is not
evidence about what is on disk" -- and the column is already known-optimistic on the S3-GFN rows.
A driver that believed it would skip a cell that was never produced, or retrain one that exists.
So every "is this done" question goes to ``Cell.status(arm)``, which manifest computes from the
filesystem on every load. ``train_plan`` is consulted for exactly one thing: WHICH OPERATION a cell
wants (train vs re-sample), because that is intent, and intent is what a plan file is for.

WHAT EACH STATUS MEANS TO THIS DRIVER, and why three of them are refusals rather than submissions:

    frozen        done. Backed up, verified, sealed. Skip.
    verified      NOT done -- checks pass, awaiting a login-node backup sweep. Never resubmitted,
                  and never silently skipped either: /project is not mounted on compute, so this
                  driver structurally cannot finish it, and a cell left here forever is a cell whose
                  only copy is on a purge-eligible filesystem. Listed loudly with the command.
    unverified    artifacts exist and verification has not passed. Could be a real defect or just an
                  unrun check. Retraining would destroy evidence of which. REFUSED.
    no-trace      artifacts but no trace. The budget cannot be evidenced. Retraining overwrites the
                  checkpoints that are the only thing left. REFUSED -- this is how s3gfn_seh/43's
                  history was lost the first time.
    short-trace   training stopped early. Resubmitting resumes it. SUBMITTED.
    not-started   nothing on disk. SUBMITTED.

CLAIMS, AND THE HOLE SLURM LEAVES. "Chains claim cells before files appear", so checking for output
files finds nothing and the cell gets submitted twice. The existing trick -- grep each job's
``CELLS=`` echo -- only works once a job is RUNNING and has written its first stdout line. A PENDING
job's environment is not exposed: ``scontrol`` gives Command, SubmitLine and Comment, and a shell
prefix like ``CELLS=... sbatch`` never reaches argv, so it appears in none of them. Verified on a
live queue before relying on it.

So this driver stamps its own claim where a PENDING job still shows it: ``--comment`` carries
``v2cells:<tag>,<tag>``, which ``squeue -o %k`` prints in any state. Detection then reads both --
the comment for anything this driver submitted, and the ``CELLS=`` line for hand-submitted chains
already running. A cell claimed by either is refused.

DRY RUN IS THE DEFAULT. ``--execute`` submits. A driver whose first invocation queues 36 jobs at
320,000 oracle calls is not a tool, it is an incident.

Usage:
    python experiments/benchmark_v2/tools/submit_grid.py --arm a --phase 1
    python experiments/benchmark_v2/tools/submit_grid.py --arm a --phase 1 --execute
    python experiments/benchmark_v2/tools/submit_grid.py --arm b --generator rgfn --limit 3

Exit 0 = plan printed (or submitted). 1 = something was refused that needs a human.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # experiments/benchmark_v2/tools
REPO = HERE.parent.parent.parent  # the checkout root
sys.path.insert(0, str(HERE))

import manifest  # noqa: E402

CLAIM_PREFIX = "v2cells:"

# ---------------------------------------------------------------------------------------------
# THE LAUNCHER. ONE script for all 54, confirmed with its author 2026-09-12 rather than inferred:
#
#     OUT_ROOT=<v2 train root> sbatch .../submit_train_v2.sh <gen> <target> <seed>
#
# No ARM argument: a reaction-GFN cell is ONE job that produces BOTH arms (train to 320,000 into
# armb/, checkpoint at the 10,000-train-row crossing, then materialise arma/ by copying that
# checkpoint, slicing the trace to a genuine prefix and re-sampling). The trainer reads
# `manifest.py --emit` itself and takes CFG, CONDA_ENV, RUNNER, TRAIN_DIR, ARM_CALLS and the gate
# from there, so THIS DRIVER PASSES NO PATHS -- one table, and it is grid.csv. It detects the six
# arm-A-only competitors from gen_class itself, and asked not to be special-cased here.
#
# NO N_ITERS, EVER. The budget comes from ARM_CALLS and the checkpoint lands where the TRACE counter
# says, never where batch x steps predicts: B's pilot measured RGFN crossing 10,000 at iteration 83
# rather than 100, because it scores 100 forward plus 20 replay.
TRAINER = "experiments/benchmark_v2/train/submit_train_v2.sh"
REACTION_GFNS = ("rgfn", "rxnflow", "scent")


def claimed_cells() -> dict[str, str]:
    """Tags currently claimed by a queued or running job -> the job id that holds them.

    TWO SOURCES BECAUSE SLURM ONLY EXPOSES ONE OF THEM. `--comment` is set by this driver and is
    visible while a job is still PENDING; the `CELLS=` stdout line covers hand-submitted chains but
    only once they are RUNNING. Neither alone is sufficient and the union is still not a lock --
    it is a check, and two drivers racing inside the same second can both pass it.
    """
    out: dict[str, str] = {}
    try:
        q = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i|%T|%k"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except Exception as e:  # a queue we cannot read is not an
        print(f"WARNING: could not read the queue ({e}).", file=sys.stderr)
        print(
            "         Claim detection is DEGRADED -- submitting could double-run a cell.",
            file=sys.stderr,
        )
        return out

    for line in q.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jid, _state, comment = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if comment.startswith(CLAIM_PREFIX):
            for tag in comment[len(CLAIM_PREFIX) :].split(","):
                if tag.strip():
                    out.setdefault(tag.strip(), jid)

    # Hand-submitted chains: their first stdout line carries `CELLS=gen:target:seed ...`.
    for jid in re.findall(r"^(\d+)\|", q, re.M):
        if jid in out.values():
            continue
        try:
            info = subprocess.run(
                ["scontrol", "show", "job", jid], capture_output=True, text=True, timeout=30
            ).stdout
            m = re.search(r"StdOut=(\S+)", info)
            if not m:
                continue
            p = Path(m.group(1))
            if not p.is_file():
                continue  # PENDING: nothing written yet
            head = p.open(errors="replace").readline()
            cm = re.search(r"CELLS=(\S+(?:\s+\S+)*)", head)
            if cm:
                for triple in cm.group(1).split():
                    bits = triple.split(":")
                    if len(bits) == 3:
                        out.setdefault(f"{bits[0]}_{bits[1]}_s{bits[2]}", jid)
        except Exception:
            continue
    return out


# Where each generator's DOCKING reward decides its sign. A table of FILE LOCATIONS, deliberately
# not a table of verdicts: the verdict is read out of the file at call time, so it cannot disagree
# with the code. TANGO has no directory of its own -- submit_baseline.sh:71 points it at Saturn's
# runner -- so five files cover six generators. RGFN is not a validation bridge at all; its
# OracleRewardProxy takes the sign off the ORACLE object, which is why no config-shaped search finds
# it and why it was never at risk.
REWARD_SOURCE = {
    "fraggfn": "validation/generators/fraggfn/fixed_reward.py",
    "reinvent": "validation/generators/reinvent/fixed_reward.py",
    "saturn": "validation/generators/saturn/fixed_reward.py",
    "tango": "validation/generators/saturn/fixed_reward.py",
    "s3gfn": "validation/generators/s3gfn/fixed_reward.py",
    "synformer": "validation/generators/synformer/fixed_reward.py",
    "rxnflow": "validation/generators/rxnflow/fixed_reward.py",
    "scent": "validation/generators/scent/docking_bridge_proxy.py",
    "rgfn": "glue/proxies/oracle_reward_proxy.py",
}
_SEAM = re.compile(r"self\._?sign\b")


def reward_orientation_broken(cell) -> str | None:
    """Why this cell's reward cannot express its target's orientation, or None if it can.

    ⛔ A SILENT FAILURE THAT FAILS GREEN. Five generators' docking rewards hardcode the
    lower-is-better mapping ``max(-float(raw) / self.norm, 0.0)`` -- fraggfn:257, reinvent:237,
    s3gfn:243, saturn:236 (which TANGO shares), synformer:353. That mapping is CORRECT for ClpP and
    has been correct for every docking target this project has ever run. 6TD3-B is the first
    HIGHER-is-better DOCKING target: cnn_vs is roughly [0, 9], so the expression is non-positive for
    every molecule and the clamp makes it exactly 0.0. A flat reward raises nothing, logs nothing and
    produces no nan -- the cell burns its whole docking budget (~1.2 s/mol) training against a
    constant and finishes looking healthy. RxnFlow (:245), SCENT (:154) and RGFN's proxy already take
    the sign from a parameter and are unaffected.

    THE DEFECT IS A PROPERTY OF THE (GENERATOR, TARGET) PAIR, WHICH COST ME NINE FALSE REFUSALS.
    My first version keyed on the TARGET alone -- "higher-is-better docking" -- which is the half of
    the condition that identifies the new case but not the half that says who mishandles it. It
    refused all 27 phase-2 cells including the 9 reaction-GFN ones, which are correctly wired. That
    is the worse direction to be wrong in: a guard that fires on healthy cells gets switched off, and
    then it is not there for the 18 that need it.

    WHY THIS READS THE SOURCE RATHER THAN LISTING THE FIVE, and why it is NOT keyed on an empty CFG
    as both reviewers suggested. A hardcoded list of five goes stale in the dangerous direction the
    moment someone adds a seam (cries wolf) or adds a tenth generator without one (silent). An empty
    CFG is worse for THIS defect specifically: it tracks whether a config exists, and the config and
    the seam are exactly the two things that must land together. Write a competitor's 6td3b config
    without touching its sign and a CFG-keyed guard goes green while the flat reward ships -- the
    failure it was put there to stop. Reading the file keys on the defect itself, so it stops firing
    the moment the seam lands and never stops firing for any other reason.

    ⛔ DO NOT COLLAPSE THIS WITH THE TRAINER'S EMPTY-CFG REFUSAL. They agree on all 18 cells TODAY,
    purely because the three generators that have configs are also the three that have seams -- a
    coincidence of scheduling, not a relationship. Keeping one is the obvious tidy-up later and it
    silently removes half the cover:

        missing config, seam present  -> the trainer's CFG check fires. This one does not, correctly.
        config present, NO seam       -> ONLY this one fires. A CFG check is green, the run starts,
                                         and every molecule scores 0.0 for the full docking budget.

    The second row is the dangerous one, and it is UNREACHABLE TODAY: verified 2026-09-12 across all
    branches (`git log --all --diff-filter=A`) that no competitor 6td3b config has ever existed --
    only our three, for rgfn/rxnflow/scent, which are also the three with seams. Nobody is assigned
    to write the other six, and whether the competitors run 6TD3-B at all is an open question for the
    researcher. So the overlap is not merely an accident of scheduling, it is an accident of work
    that has not been scoped.

    That is precisely why the row has to be written down rather than waited for. The moment those six
    configs exist -- whenever that is, and by whoever -- a CFG-keyed guard goes green while the five
    sign seams may still be absent, which is when the flat-reward run first becomes possible. Two
    guards, two different failures, both required.

    (An earlier version of this comment said the six configs "are being written now". They are not,
    and never were; I had merged an open question from the trainer's author -- who owns the six? --
    with an unrelated assignment to write OUR three, and asserted the result as a schedule. The
    argument did not depend on it, so it is stated as a condition instead of a date.)
    """
    t = cell.target
    if not (getattr(t, "reward_type", "") == "docking" and getattr(t, "higher_is_better", False)):
        return None
    rel = REWARD_SOURCE.get(cell.generator)
    if rel is None:
        return (
            f"{cell.generator} has no known reward source to check for a sign seam; refusing "
            f"rather than guessing on a higher-is-better docking target"
        )
    p = REPO / rel
    try:
        src = p.read_text()
    except Exception as e:
        return f"cannot read {rel} to check the sign seam ({e}); refusing rather than assuming"
    if _SEAM.search(src):
        return None
    return (
        f"{cell.generator}'s docking reward ({rel}) hardcodes max(-raw/norm, 0) with no sign "
        f"seam, but {t.name} is HIGHER-is-better docking -- every molecule would score exactly "
        f"0.0 and the cell would train against a flat reward and look healthy"
    )


def decide(cell, arm: str) -> tuple[str, str]:
    """(action, reason). Actions: submit / done / refuse / accept-pending / resample."""
    st = cell.status(arm)
    if st == "n/a":
        return "skip", f"no arm {arm}"
    if st == "frozen":
        return "done", "accepted"
    if st == "verified":
        return "accept-pending", "checks pass; needs a LOGIN-NODE backup + freeze"
    if st == "unverified":
        return "refuse", (
            "artifacts exist but verification has not passed -- retraining would "
            "destroy the evidence of why. Run verify_cell.py first"
        )
    if st == "no-trace":
        return "refuse", (
            "artifacts but NO trace: the budget cannot be evidenced, and the "
            "checkpoints are all that survive. Retraining overwrites them"
        )
    if st.startswith("short-trace"):
        return "submit", f"incomplete ({st}) -- resubmit resumes"
    if st == "not-started":
        # train_plan read HERE, for intent only: a re-sample is minutes against a frozen policy and
        # is NOT a training. A driver that trained it would destroy a surviving trace for nothing.
        if getattr(cell, "train_plan", "") == "resample":
            return "resample", (
                "plan says re-sample: re-draw the pool from the frozen policy, " "NOT a re-train"
            )
        return "submit", "nothing on disk"
    return "refuse", f"unrecognised status {st!r}"


def submit_cmd(cell, out_root: str) -> list[str]:
    # target_name, NOT target: `Cell.target` resolves to a Target dataclass whose repr carries the
    # whole gate definition. Interpolating it would hand the launcher an argument that is not a
    # target name and would not fail early -- it would just build a path nothing lives at. Cost me
    # one failed migration run before I read the dataclass.
    return [
        "sbatch",
        f"--job-name=v2_{cell.tag}",
        f"--comment={CLAIM_PREFIX}{cell.tag}",
        f"--export=ALL,OUT_ROOT={out_root}",
        TRAINER,
        cell.generator,
        cell.target_name,
        str(cell.seed),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", default="a", choices=("a", "b"))
    ap.add_argument("--phase", type=int, default=None)
    ap.add_argument("--role", default=None, choices=("hub_batching", "competitor"))
    ap.add_argument("--generator", action="append", default=None)
    ap.add_argument("--target", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None, help="submit at most N cells")
    # THE ROOT COMES FROM manifest, NOT FROM A STRING HERE. I first spelled it
    # ".../rgfn_runs/benchmark_v2" from the directory's name and the real one is ".../rgfn_runs/v2";
    # I found it only by going to look for the landed files instead of trusting what I had typed.
    # Two places that spell a path are two places that drift, so there is now one.
    ap.add_argument("--out-root", default=str(manifest.SCRATCH_ROOT))
    ap.add_argument(
        "--execute",
        action="store_true",
        help="actually sbatch; without it this only prints the plan",
    )
    a = ap.parse_args()

    cells = manifest.select(phase=a.phase, role=a.role, generators=a.generator, targets=a.target)
    cells = [c for c in cells if c.has_arm(a.arm)]
    if not cells:
        print("no cells match those filters")
        return 0

    claims = claimed_cells()
    buckets: dict[str, list] = {
        k: []
        for k in (
            "submit",
            "done",
            "accept-pending",
            "refuse",
            "resample",
            "claimed",
            "skip",
            "no-launcher",
        )
    }

    # The trainer is agent A's and may not have landed yet. Refusing up front beats sbatch'ing a
    # path that does not exist 54 times and reading 54 identical errors.
    trainer_missing = not (REPO / TRAINER).is_file()

    for c in cells:
        if c.tag in claims:
            buckets["claimed"].append((c, f"already queued as job {claims[c.tag]}"))
            continue
        action, why = decide(c, a.arm)
        if action == "submit":
            broken = reward_orientation_broken(c)
            if broken:
                buckets["refuse"].append((c, broken))
                continue
            if trainer_missing:
                buckets["no-launcher"].append((c, f"{TRAINER} does not exist yet"))
                continue
        buckets[action].append((c, why))

    print(
        f"arm {a.arm}"
        + (f"  phase {a.phase}" if a.phase else "")
        + f"   {len(cells)} cells in scope\n"
    )
    for name, label in (
        ("done", "ACCEPTED (frozen)"),
        ("claimed", "already queued -- not resubmitted"),
        ("skip", "no such arm"),
    ):
        if buckets[name]:
            print(f"{label}: {len(buckets[name])}")

    if buckets["accept-pending"]:
        print(
            f"\nVERIFIED BUT NOT ACCEPTED -- {len(buckets['accept-pending'])} cell(s). These are "
            f"NOT done: their only copy is on scratch."
        )
        print("  This driver cannot finish them: /project is not mounted on compute nodes.")
        for c, _ in buckets["accept-pending"]:
            print(f"    {c.tag}")
        print(
            f"  Run on a LOGIN NODE: experiments/benchmark_v2/tools/accept_cell.sh "
            f"--all --arm {a.arm}"
        )

    for name, label in (
        ("refuse", "REFUSED -- needs a human, not a resubmit"),
        ("resample", "RE-SAMPLE, not a training"),
        ("no-launcher", "NO LAUNCHER WIRED"),
    ):
        if buckets[name]:
            print(f"\n{label}: {len(buckets[name])}")
            for c, why in buckets[name]:
                print(f"    {c.tag:<24} {why}")

    todo = buckets["submit"]
    if a.limit is not None:
        todo = todo[: a.limit]
    print(
        f"\nTO SUBMIT: {len(todo)}"
        + (
            f" (of {len(buckets['submit'])}, --limit {a.limit})"
            if a.limit is not None and len(buckets["submit"]) > len(todo)
            else ""
        )
    )
    for c, why in todo:
        cmd = submit_cmd(c, a.out_root)
        print(f"    {c.tag:<24} {why}")
        print(f"      {' '.join(cmd)}")

    if not a.execute:
        print("\nDRY RUN -- nothing was submitted. Re-run with --execute.")
        return 1 if buckets["refuse"] else 0

    rc = 0
    for c, _ in todo:
        cmd = submit_cmd(c, a.out_root)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"submitted {c.tag}: {r.stdout.strip()}")
        else:
            print(f"FAILED    {c.tag}: {r.stderr.strip()}", file=sys.stderr)
            rc = 1
    return 1 if (rc or buckets["refuse"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
