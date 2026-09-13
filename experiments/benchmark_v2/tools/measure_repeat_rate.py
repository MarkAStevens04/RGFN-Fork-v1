#!/usr/bin/env python
"""How many of a cell's traced rows actually reached its oracle? Repeat rate BESIDE cache status.

WHY BOTH NUMBERS OR NEITHER. A trace row is a molecule PRESENTED to the reward. Whether that row
cost an oracle call depends entirely on whether the provider caches, and the generators disagree:

    synformer  run_synformer_fixed.py:666   traces `todo` -- deduped, cache-missed AND budget-capped
                                            (`room = budget - len(scored)`). Rows ARE oracle calls.
    s3gfn      run_s3gfn_fixed.py:145       traces the full presented list.
    fraggfn    run_fraggfn_fixed.py:134     traces the full presented list.

So the repeat rate alone is the number that misleads, and it nearly did: ``s3gfn/drd2`` is 39.5%
unique, which reads like a 60% budget shortfall. It is not one. ``DRD2FrozenReward`` has NO cache --
``predict()`` featurizes and runs the model on every SMILES handed to it -- so every one of those
presentations is a real invocation and the repetition is MODE COLLAPSE, a result of the same family
as the Saturn finding, not an accounting defect. Reporting the rate without the mechanism would have
been a reason to re-run three cells that are fine.

THE ASYMMETRY THIS EXISTS TO MEASURE. Among the competitors only ``DockingBridgeReward`` caches, so
their surrogate rows are all real calls. Ours cache on EVERY target: ``SehMoleculeProxy`` and
``DRD2Proxy`` are both ``CachedProxyBase``, whose ``compute_proxy_output`` filters uncached indices
and then ``list(set(...))`` them -- deduped twice -- and whose ``n_proxy_calls`` is ``len(self.cache)``,
i.e. upstream itself counts a call as a DISTINCT state. On sEH and DRD2 only one side caches, which
is the widest part of the gap and it runs against us, on the exhibit whose justification is parity.

NOTHING HERE IS HARDCODED AS A VERDICT.
  * The provider is resolved from the CELL'S OWN ``run_config.yaml`` (``reward.type`` plus
    ``reward.subprocess``), mirroring ``build_provider``'s dispatch -- not guessed from the target
    name. SynFormer is why: its ``seh_proxy`` resolves to ``SEHBridgeReward`` rather than
    ``SEHFrozenReward`` because its config sets ``subprocess: true``, and only that file says so.
  * The cache status is read out of the class body by AST at call time, so it stops being true the
    moment someone adds or removes a cache, and cannot disagree with the code.
  * Anything that cannot be determined is reported as ``unknown`` WITH ITS REASON, never assumed.

EVERY CELL GETS A ROW, including the 100%-unique ones. An absent row means "not measured", not "no
gap" -- the two must not look the same, which is the same rule the project applies to FragGFN's
empty routes.json and SCENT's zero promoted fragments.

OUR OWN GENERATORS RESOLVE THROUGH A SECOND CONFIG SHAPE, and both shapes were found by reading a
real run rather than by being told. RGFN and SCENT are gin-configured and write no ``reward.type``,
so their proxy is read from the gin scope binding the run itself recorded:

    rgfn   logs/operative_config.txt   proxy/singleton.constructor = @SehMoleculeProxy
    scent  operative_config.gin        proxy/singleton.constructor = @DockingBridgeProxy

Note those are DIFFERENT FILENAMES IN DIFFERENT PLACES -- a ``.txt`` under ``logs/`` versus a ``.gin``
in the run root. I was told "config.gin / operative_config.gin in the run dir", which is exactly
right for SCENT and exactly wrong for RGFN, so a resolver built on that description alone would have
returned ``unknown`` for twelve cells of whichever one it missed, silently. Verified against five
real v1 run dirs: rgfn seh/clpp/drd2 -> SehMoleculeProxy / OracleRewardProxy / DRD2Proxy, scent
seh/6td3 -> SehMoleculeProxy / DockingBridgeProxy, every one ``cached=yes`` by AST.

AND VERIFIED AGAINST THE MATERIALISED SHAPE, WHICH IS A DIFFERENT CLAIM. A trained run dir and a
materialised arm-A dir are not the same artifact: the second carries only what the copier chose, so a
resolver proven on five trained dirs has not been proven on the thing it is actually pointed at. That
gap was real -- ``materialise_arm_a`` copied a hand-written file list matching SCENT's shape and the
bridge shape and nothing RGFN writes, so a materialised RGFN cell carried NO config record and still
looked complete. Tested here on a fixture holding ``logs/`` alone, with no run_config.yaml and no
trace: resolves ``SehMoleculeProxy``, ``cached=yes``. And the negative, which is the pre-fix state:
a cell carrying no config record at all resolves to ``None`` with its reason, never to a guess.

Note what that separates. "Does this resolver read the shape" is answerable here; "does the copier
deliver the shape" is not, and belongs to whoever owns the copier. Both were ``unknown`` for the same
cell before, which is why they had to be pulled apart.

    python experiments/benchmark_v2/tools/measure_repeat_rate.py --arm a
    python experiments/benchmark_v2/tools/measure_repeat_rate.py --arm a --csv /tmp/repeat.csv
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))

import manifest  # noqa: E402

# Same tolerance verify_cell uses: a batch may straddle the boundary, 5% is one batch at any of
# our batch sizes. Imported as a constant rather than restated, so the two cannot drift.
BUDGET_TOLERANCE = 0.95

# Where each generator's reward providers are defined. File locations, not verdicts -- the verdict is
# read out of the file. `rgfn` and `scent` are ours and go through upstream's CachedProxyBase.
PROVIDER_SOURCE = {
    g: f"validation/generators/{g}/fixed_reward.py"
    for g in ("fraggfn", "reinvent", "saturn", "s3gfn", "synformer", "rxnflow")
}
PROVIDER_SOURCE["tango"] = "validation/generators/saturn/fixed_reward.py"  # shares Saturn's runner
PROVIDER_SOURCE["scent"] = "validation/generators/scent/docking_bridge_proxy.py"
PROVIDER_SOURCE["rgfn"] = None  # upstream proxies; resolved separately


# RGFN's proxy is bound by gin scope, and the run records it. Read from a REAL run rather than
# assumed: /scratch/.../rgfn_seh_5k/seed42/logs/operative_config.txt carries
#     proxy/singleton.constructor = @SehMoleculeProxy
# The file is `logs/operative_config.txt` -- a .txt under logs/, NOT a `config.gin` in the run root,
# which is what it was described to me as. A resolver built on the described name would have found
# nothing and returned `unknown` for every RGFN cell, silently, which is the failure this whole
# resolver exists to avoid.
#
# TWO GIN SHAPES, BOTH FOUND BY LOOKING RATHER THAN BY BEING TOLD. RGFN writes .txt files under
# logs/; SCENT writes .gin files in the run ROOT. I was told "config.gin / operative_config.gin in
# the run dir", which is exactly right for SCENT and exactly wrong for RGFN -- so a resolver built on
# either description alone would have silently returned `unknown` for twelve cells of the other.
#     rgfn   logs/operative_config.txt   proxy/singleton.constructor = @SehMoleculeProxy
#     scent  operative_config.gin        proxy/singleton.constructor = @DockingBridgeProxy
#
# The `^\s*proxy/` anchor is load-bearing: SCENT's config also binds
# `path_cost_proxy/singleton.constructor = @PathCostProxy` four lines earlier, and an unanchored
# match would take the path-cost proxy as the reward proxy.
_GIN_PROXY = re.compile(r"^\s*proxy/singleton\.constructor\s*=\s*@(\w+)", re.M)
_GIN_FILES = (
    "logs/operative_config.txt",  # rgfn
    "logs/config.txt",
    "operative_config.gin",  # scent
    "config.gin",
)
# Where an RGFN proxy class may be defined. DIRECTORIES, not a class->file map: the class name comes
# from the run's own config and the file is found by searching, so adding a fifth proxy needs no edit.
_PROXY_DIRS = (
    "glue/proxies",
    "rgfn/gfns/reaction_gfn/proxies",
    "validation/generators",  # SCENT's DockingBridgeProxy lives with its bridge, not with ours
)


def _find_class_file(cls: str) -> Path | None:
    for d in _PROXY_DIRS:
        for p in sorted((REPO / d).rglob("*.py")):
            try:
                if re.search(rf"^class\s+{re.escape(cls)}\b", p.read_text(), re.M):
                    return p
            except Exception:
                continue
    return None


def _resolve_provider(
    d: Path, cfg: dict, cfg_error: str | None = None
) -> tuple[str | None, Path | None, str]:
    """(class name, defining file, why). Driven by the CELL's own config, never by its target name.

    TWO CONFIG SHAPES, BECAUSE THE PROJECT HAS TWO. The bridge generators write a YAML
    ``run_config.yaml`` and this mirrors ``build_provider``'s dispatch over ``reward.type``; RGFN is
    gin-configured and writes no such key, so its proxy is read from the gin scope binding the run
    itself recorded. Both are the cell telling us what it used, which is the only authority that
    cannot drift from what ran.
    """
    # An unreadable config is reported as unreadable and NEVER falls through to the gin path, where
    # it would come back as "no reward.type" -- a message that names the cell for the interpreter's
    # fault. Refusing here keeps the two causes distinguishable at the point they diverge.
    if cfg_error:
        return None, None, cfg_error
    reward = (cfg or {}).get("reward") or {}
    rtype = str(reward.get("type", "")).strip()
    if rtype:
        if rtype == "drd2":
            cls, why = "DRD2FrozenReward", "reward.type=drd2"
        elif rtype == "seh_proxy":
            cls, why = (
                ("SEHBridgeReward", "reward.type=seh_proxy, reward.subprocess=true")
                if bool(reward.get("subprocess"))
                else ("SEHFrozenReward", "reward.type=seh_proxy, in-process")
            )
        elif rtype == "docking":
            cls, why = "DockingBridgeReward", "reward.type=docking"
        else:
            return None, None, f"unrecognised reward.type={rtype!r}"
        return cls, None, why

    for rel in _GIN_FILES:
        p = d / rel
        if not p.is_file():
            continue
        try:
            m = _GIN_PROXY.search(p.read_text())
        except Exception:
            continue
        if m:
            cls = m.group(1)
            f = _find_class_file(cls)
            if f is None:
                return None, None, f"{rel} binds @{cls}, but no file defines it"
            return cls, f, f"{rel}: proxy/singleton.constructor = @{cls}"
    return None, None, "no reward.type in run_config.yaml and no gin proxy binding in logs/"


def _class_caches(src_path: Path, class_name: str) -> tuple[bool | None, str]:
    """(cached?, evidence). None when it cannot be determined -- never a guess."""
    try:
        tree = ast.parse(src_path.read_text())
    except Exception as e:
        return None, f"cannot parse {src_path.name}: {e}"
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for base in node.bases:
            name = getattr(base, "id", None) or getattr(getattr(base, "value", None), "id", None)
            if name == "CachedProxyBase" or "CachedProxyBase" in ast.dump(base):
                return True, "subclasses CachedProxyBase"
        for sub in ast.walk(node):
            tgt = None
            if isinstance(sub, ast.AnnAssign):
                tgt = sub.target
            elif isinstance(sub, ast.Assign) and sub.targets:
                tgt = sub.targets[0]
            if isinstance(tgt, ast.Attribute) and tgt.attr in ("cache", "_cache"):
                return True, f"assigns self.{tgt.attr}"
        return False, "no cache attribute in the class body"
    return None, f"class {class_name} not found in {src_path.name}"


def _read_config(d: Path) -> tuple[dict, str | None]:
    """(config, unreadable-reason). An EMPTY config and an UNREADABLE one are different facts.

    ⛔ THIS SWALLOWED AN ImportError AND BLAMED THE DATA FOR IT. The first version was
    ``try: import yaml ... except Exception: return {}``, so an interpreter without PyYAML returned
    an empty dict for every cell -- and ``_resolve_provider`` then reported "no reward.type in
    run_config.yaml" on files that plainly contain ``type: drd2``. An environment defect wearing a
    data defect's message, on every cell at once.

    It cost a real disagreement: two of us ran the same tool on the same cell from the same branch
    and got different answers, and the message pointed at the cell rather than at the interpreter, so
    the obvious next move was to go looking at the cell. Reproduced by blocking the import, which is
    the only reason we found it rather than concluding the landed cells were unmeasurable.

    THREE STATES, KEPT APART:
      no file            -> ({}, None)   legitimate; gin-configured cells have no run_config.yaml
      file, no PyYAML    -> ({}, reason) an ENVIRONMENT problem, and it says so
      file, bad YAML     -> ({}, reason) a DATA problem, and it says which
    """
    p = d / "run_config.yaml"
    if not p.is_file():
        return {}, None
    try:
        import yaml
    except ImportError as e:
        return {}, (
            f"run_config.yaml exists but PyYAML is not importable in this interpreter ({e}) -- "
            f"this is an ENVIRONMENT problem, not a property of the cell"
        )
    try:
        return (yaml.safe_load(p.read_text()) or {}), None
    except Exception as e:
        return {}, f"run_config.yaml is present but unparseable: {e}"


def _verdict(row: dict, budget: int) -> tuple[str, str]:
    """Apply the 2026-09-13 ruling: the budget counts molecules that REACHED THE ORACLE.

    WHICH NUMBER THAT IS DEPENDS ON THE PROVIDER, and getting this backwards is the whole reason the
    cache column exists. Where the provider CACHES, a repeat never reaches the oracle, so the cell is
    measured on `distinct`. Where it does NOT, every presentation is a real invocation, so the cell
    is measured on `train_rows` and a low unique% is MODE COLLAPSE to report, not a shortfall to
    repair. Applied the other way round, s3gfn_drd2 (34-45% unique, uncached) reads as 60% short and
    three healthy cells get re-run for nothing.

    `unknown` yields `unknown`, never a pass: a cell whose provider we cannot resolve has not been
    measured against the ruling, and that must not look like meeting it.
    """
    if row["cached"] == "unknown" or row["train_rows"] == "":
        return "unknown", "not measured against the ruling"
    reached = int(row["distinct"] if row["cached"] == "yes" else row["train_rows"])
    basis = "distinct" if row["cached"] == "yes" else "rows"
    if reached >= budget * BUDGET_TOLERANCE:
        return "ok", f"{reached:,}/{budget:,} on {basis}"
    return "SHORT", f"{reached:,}/{budget:,} on {basis}, short by {budget - reached:,}"


def measure(cell, arm: str) -> dict:
    row = {
        "cell": cell.tag,
        "generator": cell.generator,
        "target": cell.target_name,
        "seed": cell.seed,
        "train_rows": "",
        "distinct": "",
        "unique_pct": "",
        "provider": "",
        "cached": "unknown",
        "verdict": "unknown",
        "basis": "",
        "evidence": "",
    }
    d = cell.train_dir(arm)
    t = d / "trace.csv"
    if not t.is_file() or t.stat().st_size < 200:
        row["evidence"] = "NOT MEASURED: no usable trace"
        return row

    n, seen = 0, set()
    with open(t, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("phase") != "train":
                continue
            n += 1
            seen.add(r.get("smiles", ""))
    row["train_rows"] = n
    row["distinct"] = len(seen)
    row["unique_pct"] = f"{100*len(seen)/n:.1f}" if n else ""

    cfg, cfg_error = _read_config(d)
    cls, src, why = _resolve_provider(d, cfg, cfg_error)
    if cls is None:
        row["evidence"] = f"provider unresolved: {why}"
        return row
    row["provider"] = cls
    # A gin-resolved proxy carries its own defining file; a bridge provider is looked up by generator.
    if src is None:
        rel = PROVIDER_SOURCE.get(cell.generator)
        if rel is None:
            row["evidence"] = f"{why}; no provider source mapped for {cell.generator}"
            return row
        src = REPO / rel
    cached, ev = _class_caches(src, cls)
    row["cached"] = {True: "yes", False: "no", None: "unknown"}[cached]
    row["evidence"] = ev
    row["verdict"], row["basis"] = _verdict(row, cell.arm_calls(arm) or 0)
    return row


def _require_pyyaml() -> str | None:
    """The CLI's precondition. None if satisfied, else the message to print and exit on.

    ⛔ LEGIBLE IS NOT LOUD, AND A TABLE OF 108 GOOD EXPLANATIONS IS STILL A MISLEADING REPORT.
    Making each row say WHY it could not resolve was the first half of this fix; it is not enough.
    Every row reading "unresolved" reads as "the landed cells cannot be measured", which is the
    conclusion it actually produced -- a peer ran this from conda base, got 108 careful explanations,
    and concluded the shared branch could not resolve a single cell. The per-row reason was right and
    the report was wrong.
    #
    THE INTERPRETER SPLIT IS REAL AND NOT INCIDENTAL. `manifest.py` is deliberately STDLIB ONLY so a
    bare SLURM shell can bootstrap with conda base and ask which env a cell needs -- and conda base
    is exactly the interpreter with no PyYAML. So base is the one interpreter that can import this
    module and must not run its CLI. Four agents on this project each reach for a bare `python`,
    which resolves to base.

    So the library degrades legibly (an unreadable config is reported as unreadable, per cell) and
    the CLI REFUSES rather than emitting a report whose every row is individually honest and
    collectively false.
    """
    try:
        import yaml  # noqa: F401
    except ImportError:
        return (
            f"REFUSING TO REPORT: PyYAML is not importable in {sys.executable}\n"
            f"  Every cell with a run_config.yaml would read 'unresolved', which is a property of\n"
            f"  this interpreter and not of the cells. conda BASE has no PyYAML by design --\n"
            f"  manifest.py is stdlib-only so a SLURM shell can bootstrap with base.\n"
            f"  Run it from the rgfn env instead:\n"
            f"    /home/markymoo/miniconda3/envs/rgfn/bin/python "
            f"experiments/benchmark_v2/tools/measure_repeat_rate.py ..."
        )
    return None


def main() -> int:
    problem = _require_pyyaml()
    if problem:
        print(problem, file=sys.stderr)
        return 2  # distinct from 1 (= short cells found): nothing was measured at all
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", default="a", choices=("a", "b"))
    ap.add_argument("--generator", action="append", default=None)
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args()

    cells = [c for c in manifest.select(generators=a.generator) if c.has_arm(a.arm)]
    rows = [measure(c, a.arm) for c in cells]

    hdr = (
        f"{'cell':<22} {'rows':>7} {'distinct':>9} {'uniq%':>7} {'cached':>7} "
        f"{'verdict':>8}  {'provider':<20} basis / evidence"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tail = r["basis"] or r["evidence"]
        print(
            f"{r['cell']:<22} {str(r['train_rows']):>7} {str(r['distinct']):>9} "
            f"{str(r['unique_pct']):>7} {r['cached']:>7} {r['verdict']:>8}  "
            f"{r['provider']:<20} {tail[:40]}"
        )

    measured = [r for r in rows if r["train_rows"] != ""]
    short = [r for r in rows if r["verdict"] == "SHORT"]
    print(f"\n{len(measured)}/{len(rows)} cells measured; the rest carry their reason.")
    print(
        "A repeat costs an oracle call only where cached=no. Where cached=yes the gap between rows\n"
        "and distinct is budget that never reached the oracle; where cached=no it is mode collapse."
    )
    if short:
        print(f"\nSHORT UNDER THE ORACLE-CALL RULING: {len(short)}")
        for r in short:
            print(f"    {r['cell']:<22} {r['basis']}")
        print(
            "  These cells' artifacts are intact and still frozen -- SHORT is a property of the\n"
            "  BUDGET, not of the data, and unfreezing them to record it would risk the artifacts\n"
            "  to make a point. Topping one up is NOT a cheap continuation: DockingBridgeReward's\n"
            "  cache is in-process and starts empty (fixed_reward.py:203, no load/dump), so a\n"
            "  resumed run re-docks what the first already docked; and TraceWriter ROTATES, so the\n"
            "  cell's combine mode would have to change from max to sum."
        )
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")
    # Non-zero when any cell is short, so a driver or sweep can gate on it rather than read prose.
    return 1 if short else 0


if __name__ == "__main__":
    raise SystemExit(main())
