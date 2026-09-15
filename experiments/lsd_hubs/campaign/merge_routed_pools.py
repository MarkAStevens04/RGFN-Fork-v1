#!/usr/bin/env python
"""OURS. Union every cached MultiAiZ route artifact for one (target, seed) into a single larger
candidate pool, so the competitor is not judged on whichever 500-molecule slice happened to be routed.

WHY A UNION IS VALID HERE. `multiaiz_worker.py` flattens each route and stitches every
discovered-intermediate leaf back down to REAL ZINC stock before writing, so an emitted recipe is
self-contained: it does not depend on which pool's set-based discovery found it. Merging therefore
changes which molecules SPARROW may choose from, not what any individual route costs. SPARROW then
re-derives sharing across the merged network itself.

WHAT IT IS NOT. This is a stopgap that reuses sunk routing, NOT a substitute for one properly sampled
and routed pool -- the constituent pools were built at DIFFERENT GATES (sEH 7.0 and 5.68) and with
different pruning, so the union is not a sample from any single well-defined population. Every
molecule is re-filtered to the CURRENT gate here, but the composition is still a union of
convenience. Report it as such, and prefer a clean re-run where the hours exist.

Writes a pool dir in the standard shape (pool.smi / pool_scores.csv / multiaiz_routes.json /
pool_meta.json) so every downstream script consumes it unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

POOLS = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/multiaiz_pools")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gate", type=float, required=True)
    ap.add_argument("--higher-is-better", type=lambda v: v.lower() != "false", default=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=float, default=0.5, help="tau for the sphere-exclusion walk")
    a = ap.parse_args()

    routes: dict = {}
    scores: dict = {}
    used = []
    for d in sorted(POOLS.glob(f"s3gfn_{a.target}_seed{a.seed}_*")):
        rf, sf = d / "multiaiz_routes.json", d / "pool_scores.csv"
        if not rf.exists():
            continue
        try:
            R = json.loads(rf.read_text())
        except Exception:  # noqa: BLE001 - a truncated artifact must not abort the merge
            continue
        n_new = 0
        for k, v in R.items():
            if v and k not in routes:
                routes[k] = v
                n_new += 1
            elif v:
                routes[k] = routes[k] + v  # more candidate recipes = more for SPARROW to pick
        if sf.exists():
            for r in csv.DictReader(sf.open()):
                smi = r.get("smiles")
                try:
                    val = float(r.get("score"))
                except (TypeError, ValueError):
                    continue
                if val != val or not smi:  # NaN never passes a gate; see build_s3gfn_pools
                    continue
                if smi not in scores or (
                    (val > scores[smi]) if a.higher_is_better else (val < scores[smi])
                ):
                    scores[smi] = val
        used.append((d.name, len(R), n_new))

    keep = {
        s: v
        for s, v in scores.items()
        if s in routes and ((v > a.gate) if a.higher_is_better else (v < a.gate))
    }
    if not keep:
        raise SystemExit("[merge] nothing survives the gate — wrong gate or wrong target?")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ranked = sorted(keep.items(), key=lambda t: (-t[1] if a.higher_is_better else t[1]))

    # PRUNE THE UNION. Merging PRUNED pools with NAIVE ones re-introduces the near-duplicates the
    # pruned pools existed to remove, and the damage is large: measured on the unpruned union, SPARROW
    # selected 100 molecules that collapsed to 65 distinct (35% redundant) and 500 that collapsed to
    # 251 (50%), against 1-2% on a properly pruned pool. That HANDICAPS the competitor at small
    # budgets while flattering it at large ones -- the opposite of what a bigger pool is for. So the
    # union gets the same sphere-exclusion walk every other pool gets.
    n_pre = len(ranked)
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from validation.lsdflow.metrics.diversity import (  # noqa: E402
        ecfp,
        mode_representatives,
    )

    smis = [s_ for s_, _ in ranked]
    rews = [v for _, v in ranked]
    idx = mode_representatives(
        smis,
        rews,
        higher_is_better=a.higher_is_better,
        reward_threshold=a.gate,
        similarity_threshold=a.cutoff,
        fps=[ecfp(s_) for s_ in smis],
    )
    ranked = [ranked[i] for i in idx]
    print(f"[merge]   pruned union at tau={a.cutoff}: {n_pre} -> {len(ranked)} mutually distinct")
    (out / "pool.smi").write_text("\n".join(s for s, _ in ranked) + "\n")
    with (out / "pool_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score"])
        w.writerows(ranked)
    (out / "multiaiz_routes.json").write_text(json.dumps({k: routes[k] for k, _ in ranked}))
    (out / "pool_meta.json").write_text(
        json.dumps(
            {
                "pool_variant": "union-of-cached-routes-PRUNED",
                "cutoff": a.cutoff,
                "provenance": "STOPGAP: reuses sunk MultiAiZ routing across pools built at DIFFERENT gates; "
                "re-filtered to this gate, but the composition is a union of convenience",
                "gate": a.gate,
                "higher_is_better": a.higher_is_better,
                "n_routed_union": len(routes),
                "n_scored_union": len(scores),
                "n_written": len(ranked),
                "source_pools": [{"pool": p, "routed": n, "new_to_union": k} for p, n, k in used],
            },
            indent=2,
        )
    )
    print(
        f"[merge] {a.target} s{a.seed}: {len(used)} pools -> {len(routes)} routed, "
        f"{len(ranked)} above gate {a.gate} -> {out}"
    )


if __name__ == "__main__":
    main()
