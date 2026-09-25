"""Pre-flight checks for cells that are QUEUED but have not started yet.

WHY THIS EXISTS. Every check below is a failure this campaign actually shipped to the cluster and
discovered only from the wreckage, hours or days later. None of them raised at submit time; several
produced runs that looked completely healthy and were wrong.

    RxnFlow import          nine cells died 13 s in on `No module named 'gflownet'`
    flat reward             cnn_vs is higher-is-better; the bridge negated it, so every molecule
                            scored exactly 0.0 for the whole docking budget, silently
    wrong metric            score_batch hardcoded `dvina`, so 6TD3-B never saw cnn_vs at all
    s3gfn skips training    its runner exits to sampling if a model file exists; three cells
                            "extended" for 25 minutes and trained zero steps
    fraggfn restarts        the same resubmit RESTARTS instead of extending when the checkpoint
                            is missing, which reads as a normal run
    budget not transferred  a round count copied between targets put one entrant at 2.50x budget
    walltime rejected       a site submit filter changed under us; nothing queued at all

Run before a launch, or against whatever is queued right now:

    python experiments/benchmark_v2/tools/preflight.py            # everything in the queue
    python experiments/benchmark_v2/tools/preflight.py --cell fraggfn/clpp/42

Exits non-zero if any check FAILS. WARN is advisory. No GPU, no cluster, seconds to run.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))

import manifest  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self, tag: str):
        self.tag = tag
        self.rows: list[tuple[str, str, str]] = []

    def add(self, verdict: str, name: str, detail: str = "") -> None:
        self.rows.append((verdict, name, detail))

    @property
    def failed(self) -> bool:
        return any(v == FAIL for v, _, _ in self.rows)

    def print(self) -> None:
        bad = self.failed
        print(f"\n{'=' * 78}\n{self.tag}{'   <-- FAILS' if bad else ''}\n{'=' * 78}")
        for v, name, detail in self.rows:
            print(f"  {v:<4} {name:<38} {detail}")


def _queued_tags() -> set[str]:
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%k"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:
        return set()
    return {l.replace("v2cells:", "").strip() for l in out.split("\n") if "v2cells:" in l}


def _yaml_or_gin_value(text: str, key: str):
    m = re.search(rf"^\s*{re.escape(key)}\s*[:=]\s*([^\s#]+)", text, re.M)
    return m.group(1).strip().strip("'\"") if m else None


def check_cell(cell, arm: str) -> Report:
    r = Report(f"{cell.tag}  arm{arm}")

    # ---- 1. the cell is allowed to be trained by this driver at all -------------------------
    plan = getattr(cell, "train_plan", "")
    if plan not in ("generate", ""):
        r.add(FAIL, "train_plan", f"{plan!r}: submit_train_v2.sh refuses anything but 'generate'")
    else:
        r.add(PASS, "train_plan", plan or "(unset)")

    if cell.frozen(arm):
        r.add(FAIL, "not frozen", "train dir is read-only; the runner cannot write")
    else:
        r.add(PASS, "not frozen", "")

    # ---- 2. the config resolves, exists and parses ------------------------------------------
    try:
        cfg = cell.cfg
    except KeyError as e:
        r.add(FAIL, "config registered", str(e)[:60])
        return r
    p = REPO / cfg
    if not p.is_file():
        r.add(FAIL, "config exists", cfg)
        return r
    text = p.read_text()
    r.add(PASS, "config exists", cfg.split("/")[-1])

    # ---- 3. THE BUDGET KNOB, and that it is the right one for THIS target --------------------
    # A round count copied between targets is what put fraggfn 6TD3-B at 2.50x its budget.
    steps = _yaml_or_gin_value(text, "n_train_steps") or _yaml_or_gin_value(
        text, "Trainer.n_iterations"
    )
    budget = _yaml_or_gin_value(text, "budget")
    if arm == "b":
        want = manifest.arm_b_iterations(cell.generator)
        if want is None:
            r.add(WARN, "arm-B iterations", f"no forward rate for {cell.generator}")
        elif steps is None:
            r.add(FAIL, "arm-B iterations", "no n_iterations/n_train_steps in the config")
        elif int(steps) != want:
            r.add(FAIL, "arm-B iterations", f"config {steps} != protocol {want}")
        else:
            r.add(
                PASS,
                "arm-B iterations",
                f"{steps} x {manifest.FORWARD_PER_ITER[cell.generator]}"
                f" = {manifest.ARM_B_FORWARD_TRAJECTORIES:,} trajectories",
            )
    else:
        if budget is not None:
            r.add(PASS, "arm-A budget", f"budget: {budget} (explicit stop)")
        elif steps is not None:
            r.add(
                WARN,
                "arm-A budget",
                f"no budget field; n_train_steps={steps} IS the budget "
                f"-- must be calibrated for {cell.target_name}",
            )
        else:
            r.add(FAIL, "arm-A budget", "neither a budget field nor a step count")

    # ---- 4. REWARD ORIENTATION. A wrong sign is a flat 0.0 reward, silently ------------------
    # WHO NEEDS THE LITERAL DEPENDS ON THE PATH, and getting that wrong makes this check cry wolf.
    # A CROSS-ENV BRIDGE (DockingBridgeProxy / DockingBridgeReward, i.e. `type: docking` shelling out
    # to score_batch.py) cannot see the oracle class, so the config is the ONLY place that can state
    # the orientation -- omit it on a higher-is-better target and every molecule scores exactly 0.0.
    # An IN-ENV proxy has the oracle object: OracleRewardProxy does
    # `self._sign = 1.0 if oracle.higher_is_better else -1.0` (oracle_reward_proxy.py:82), so the
    # literal is redundant there and demanding it flags healthy cells. Verified against the live
    # rgfn_6td3b traces, which carry 78,933 distinct reward values over [0, 9.977] with no literal
    # anywhere in their config -- the first version of this check called exactly those a FAIL.
    t = cell.target
    if t.is_docking:
        blob = text
        for inc in re.findall(r"^\s*include\s+['\"]([^'\"]+)['\"]", text, re.M):
            q = REPO / inc
            if q.is_file():
                blob += "\n" + q.read_text()
        uses_bridge = bool(re.search(r"DockingBridge(Proxy|Reward)|type:\s*docking", blob))
        in_env = "OracleRewardProxy" in blob
        hib = _yaml_or_gin_value(text, "higher_is_better")
        want_hib = bool(t.higher_is_better)
        if hib is None:
            if not want_hib:
                r.add(PASS, "higher_is_better", "absent, and this target is lower-is-better")
            elif uses_bridge and not in_env:
                r.add(
                    FAIL,
                    "higher_is_better",
                    f"{cell.target_name} is higher-is-better, this cell "
                    f"scores across the env boundary, and the config does "
                    f"not say so -> flat 0.0 reward",
                )
            elif in_env:
                r.add(
                    PASS,
                    "higher_is_better",
                    "not needed: in-env proxy takes the sign from oracle.higher_is_better",
                )
            else:
                r.add(
                    WARN,
                    "higher_is_better",
                    "higher-is-better target, no literal, and the reward path could not be "
                    "resolved from the config or its includes -- check before launching",
                )
        else:
            got = hib.lower() in ("true", "1", "yes")
            if got != want_hib:
                r.add(FAIL, "higher_is_better", f"config {got} != target {want_hib}")
            else:
                r.add(PASS, "higher_is_better", f"{got} (matches targets.py)")

        oracle = _yaml_or_gin_value(text, "oracle")
        if oracle and t.oracle and oracle != t.oracle:
            r.add(FAIL, "oracle", f"config {oracle!r} != targets.py {t.oracle!r}")
        elif oracle:
            r.add(PASS, "oracle", oracle)

    # ---- 5. WILL IT ACTUALLY TRAIN? per-generator resume semantics ---------------------------
    d = cell.train_dir(arm)
    gen = cell.generator
    if gen == "s3gfn":
        models = list(d.rglob("*_model.pt")) if d.is_dir() else []
        if models:
            r.add(
                FAIL,
                "s3gfn will train",
                f"{len(models)} model file(s) present -> runner SKIPS training and samples only",
            )
        else:
            r.add(PASS, "s3gfn will train", "no model file")
    elif gen == "fraggfn":
        ck = d / "checkpoints" / "last_gfn.pt"
        if d.is_dir() and any(d.iterdir()) and not ck.is_file():
            r.add(WARN, "fraggfn extends", "artifacts but no last_gfn.pt -> restarts from step 0")
        else:
            r.add(PASS, "fraggfn extends", "checkpoint present" if ck.is_file() else "fresh start")
    elif gen == "rgfn" and arm == "b":
        ck = d / "train" / "checkpoints" / "last_gfn.pt"
        if d.is_dir() and any(d.iterdir()) and not ck.is_file():
            r.add(WARN, "rgfn resumes", "artifacts but no last_gfn.pt -> restarts from iteration 0")
        else:
            r.add(PASS, "rgfn resumes", "checkpoint present" if ck.is_file() else "fresh start")

    # ---- 6. the runner exists and its env is real -------------------------------------------
    runner = REPO / cell.runner
    r.add(PASS if runner.is_file() else FAIL, "runner exists", cell.runner)
    env = cell.conda_env
    envs = subprocess.run(
        ["conda", "env", "list"], capture_output=True, text=True, timeout=60
    ).stdout
    r.add(PASS if re.search(rf"^{re.escape(env)}\s", envs, re.M) else FAIL, "conda env", env)
    return r


def check_walltime() -> Report:
    """The site submit filter, which changed under this campaign on 2026-09-24."""
    r = Report("submit filter / walltime")
    sh = (REPO / "experiments/benchmark_v2/train/submit_train_v2.sh").read_text()
    m = re.search(r"^#SBATCH --time=(\S+)", sh, re.M)
    if not m:
        r.add(FAIL, "trainer --time", "no #SBATCH --time line")
        return r
    req = m.group(1)
    out = subprocess.run(
        [
            "sbatch",
            "--test-only",
            "--partition=compute",
            f"--time={req}",
            "--gpus-per-node=1",
            "--wrap=echo x",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    blob = out.stdout + out.stderr
    if "ERROR" in blob:
        r.add(
            FAIL, "trainer --time accepted", f"{req}: " + blob.strip().split("\n")[1].strip()[:60]
        )
    else:
        r.add(PASS, "trainer --time accepted", req)
    part = subprocess.run(
        ["scontrol", "show", "partition", "compute"], capture_output=True, text=True, timeout=30
    ).stdout
    mt = re.search(r"MaxTime=(\S+)", part)
    if mt:
        r.add(
            WARN if mt.group(1) != req else PASS,
            "partition MaxTime",
            f"{mt.group(1)} (the SCHEDULER's limit; the submit filter may be stricter)",
        )
    return r


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cell", help="GEN/TARGET/SEED; default = everything queued")
    ap.add_argument("--arm", default=None, choices=("a", "b"))
    a = ap.parse_args()

    cells = manifest.select()
    if a.cell:
        g, t, s = a.cell.split("/")
        want = [c for c in cells if c.generator == g and c.target_name == t and c.seed == int(s)]
    else:
        q = _queued_tags()
        want = [c for c in cells if c.tag in q]
        if not want:
            print("nothing queued; pass --cell GEN/TARGET/SEED to check one directly")
            return 0

    reports = [check_walltime()]
    for c in want:
        arm = a.arm or ("b" if c.has_arm("b") else "a")
        reports.append(check_cell(c, arm))
    for rep in reports:
        rep.print()

    nfail = sum(1 for rep in reports if rep.failed)
    print(f"\n{'=' * 78}")
    print(f"{len(reports) - 1} cell(s) checked; {nfail} report(s) with at least one FAIL")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
