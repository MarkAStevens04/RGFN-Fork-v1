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

⚠ OUR OWN GENERATORS RESOLVE TO ``unknown`` TODAY, AND THAT IS DELIBERATE RATHER THAN AN OVERSIGHT.
RGFN and SCENT are gin-configured, so their run dirs do not carry the ``reward.type`` /
``reward.subprocess`` keys this dispatch reads, and no cell of either exists yet to read. Their
providers ARE cached -- ``SehMoleculeProxy`` and ``DRD2Proxy`` both subclass ``CachedProxyBase``, and
``OracleRewardProxy`` does too -- but wiring that resolution now would mean inventing the config
shape before a single cell has produced one, which is how a lookup ends up confidently pointing at
the wrong class. Wire it against the first landed RGFN/SCENT cell, from what that cell actually
writes. Until then the rows say ``unknown`` with the reason, which is the honest state.

    python experiments/benchmark_v2/tools/measure_repeat_rate.py --arm a
    python experiments/benchmark_v2/tools/measure_repeat_rate.py --arm a --csv /tmp/repeat.csv
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))

import manifest  # noqa: E402

# Where each generator's reward providers are defined. File locations, not verdicts -- the verdict is
# read out of the file. `rgfn` and `scent` are ours and go through upstream's CachedProxyBase.
PROVIDER_SOURCE = {
    g: f"validation/generators/{g}/fixed_reward.py"
    for g in ("fraggfn", "reinvent", "saturn", "s3gfn", "synformer", "rxnflow")
}
PROVIDER_SOURCE["tango"] = "validation/generators/saturn/fixed_reward.py"  # shares Saturn's runner
PROVIDER_SOURCE["scent"] = "validation/generators/scent/docking_bridge_proxy.py"
PROVIDER_SOURCE["rgfn"] = None  # upstream proxies; resolved separately


def _resolve_provider(cfg: dict) -> tuple[str | None, str]:
    """(class name, why). Mirrors build_provider's dispatch, driven by the cell's own config."""
    reward = (cfg or {}).get("reward") or {}
    rtype = str(reward.get("type", "")).strip()
    if not rtype:
        return None, "run_config.yaml has no reward.type"
    if rtype == "drd2":
        return "DRD2FrozenReward", "reward.type=drd2"
    if rtype == "seh_proxy":
        if bool(reward.get("subprocess")):
            return "SEHBridgeReward", "reward.type=seh_proxy, reward.subprocess=true"
        return "SEHFrozenReward", "reward.type=seh_proxy, in-process"
    if rtype == "docking":
        return "DockingBridgeReward", "reward.type=docking"
    return None, f"unrecognised reward.type={rtype!r}"


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


def _read_config(d: Path) -> dict:
    p = d / "run_config.yaml"
    if not p.is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


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

    cls, why = _resolve_provider(_read_config(d))
    rel = PROVIDER_SOURCE.get(cell.generator)
    if cls is None:
        row["evidence"] = f"provider unresolved: {why}"
        return row
    row["provider"] = cls
    if rel is None:
        row["evidence"] = f"{why}; ours, upstream proxy source not mapped here"
        return row
    cached, ev = _class_caches(REPO / rel, cls)
    row["cached"] = {True: "yes", False: "no", None: "unknown"}[cached]
    row["evidence"] = ev
    return row


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", default="a", choices=("a", "b"))
    ap.add_argument("--generator", action="append", default=None)
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args()

    cells = [c for c in manifest.select(generators=a.generator) if c.has_arm(a.arm)]
    rows = [measure(c, a.arm) for c in cells]

    hdr = f"{'cell':<22} {'rows':>7} {'distinct':>9} {'uniq%':>7} {'cached':>7}  {'provider':<20} evidence"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['cell']:<22} {str(r['train_rows']):>7} {str(r['distinct']):>9} "
            f"{str(r['unique_pct']):>7} {r['cached']:>7}  {r['provider']:<20} {r['evidence'][:44]}"
        )

    measured = [r for r in rows if r["train_rows"] != ""]
    print(f"\n{len(measured)}/{len(rows)} cells measured; the rest carry their reason.")
    print(
        "A repeat costs an oracle call only where cached=no. Where cached=yes the gap between rows\n"
        "and distinct is budget that never reached the oracle; where cached=no it is mode collapse."
    )
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
