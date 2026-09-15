#!/usr/bin/env python
"""Template id -> named reaction: the campaign's view of the ONE naming table.

The table itself moved to :mod:`glue.export.naming` when the route exporter
(``docs/ROUTE_DATASET_SCHEMA.md``) was built: the exporter needs exactly these names in
``routes.json``/``steps.csv``/``protocol.md``, and ``glue/`` may not import from ``experiments/``.
One table, two importers -- a second copy here would let the same template acquire two names in two
artifacts, which is the one failure mode a reading aid must not have.

This module is unchanged for every existing caller (``parallel_groups.py``, ``yield_preference.py``,
...): ``parse_template`` / ``named_reaction`` / ``is_single_reactant`` / ``RULES`` / ``FGI`` are
re-exported verbatim. What still lives HERE is the audit CLI, :func:`emit_review_table`, which prints
every id with its count, coverage and assigned name so a chemist can review the whole mapping in one
pass -- an analysis over campaign artifacts, not a library function.
"""
from glue.export.naming import (  # noqa: F401  (re-exported for existing callers)
    FGI,
    RULES,
    is_single_reactant,
    named_reaction,
    parse_template,
)


def emit_review_table(routes_paths, out_csv=None) -> None:
    """Print every template with count, coverage and assigned name, for a one-pass chemist audit."""
    import collections
    import csv as _csv
    import json

    cnt, ex = collections.Counter(), {}
    for p in routes_paths:
        for rt in json.loads(open(p).read()).values():
            for s in rt.get("steps") or []:
                t = s.get("reaction") or ""
                tid, _ = parse_template(t)
                cnt[tid] += 1
                ex.setdefault(tid, t)
    tot = sum(cnt.values()) or 1
    rows, run = [], 0
    for tid, n in cnt.most_common():
        run += n
        nm = named_reaction(ex[tid])
        rows.append(
            dict(
                template_id=tid,
                steps=n,
                cum_coverage=round(run / tot, 4),
                named_reaction=nm,
                single_reactant=is_single_reactant(ex[tid]),
                smarts=parse_template(ex[tid])[1],
            )
        )
    unnamed = sum(r["steps"] for r in rows if r["named_reaction"] == "?")
    print(f"templates: {len(rows)}   steps: {tot:,}   UNNAMED: {unnamed:,} ({unnamed/tot:.2%})")
    print(f"{'id':>5}{'steps':>9}{'cum':>8}  {'1-react':>7}  named reaction")
    for r in rows[:34]:
        print(
            f"{r['template_id']:>5}{r['steps']:>9,}{r['cum_coverage']:>8.1%}"
            f"  {str(r['single_reactant']):>7}  {r['named_reaction']}"
        )
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nfull table -> {out_csv}")


if __name__ == "__main__":
    import sys

    SNAP = "/scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820"
    paths = sys.argv[1:] or [
        f"{SNAP}/{c}/routes.json"
        for c in ("seh_seed43", "seh_seed44", "drd2_seed43", "drd2_seed44")
    ]
    emit_review_table(paths, "experiments/lsd_hubs/campaign/results/template_map.csv")
