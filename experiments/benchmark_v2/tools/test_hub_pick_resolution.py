#!/usr/bin/env python
"""Regression test for the bug that made every depth field null on the pilot's first real run.

THE BUG. `run_campaign` resolved `pick_hubs_timing.json` TWICE: `load_hub_pick_s` honoured the
`--hub-pick-timing` override, while the depth-provenance block fifteen lines away re-derived the path
from `--enum-children` and ignored the flag. B's pilot then ran a campaign against an enumeration
produced elsewhere -- `hubs_head.csv` at the cell root, no sidecar beside `enum/enum_children.json` --
and `hub_pool`, `min_hub_depth`, `max_hub_depth` and `walked_depth_hist` all came back null while the
job reported success.

Two lessons, both already project memories, both committed by the person writing the guards:
locating an artifact by DESCRIBING where it should be rather than reading where it is, and a second
definition of one rule living feet from the first.

Runs without dgl: the functions are extracted by AST, because importing run_campaign pulls in
rgfn -> dgl -> graphbolt and cannot load on a login node.

    python experiments/benchmark_v2/tools/test_hub_pick_resolution.py
"""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = REPO_ROOT / "experiments" / "lsd_hubs" / "campaign" / "run_campaign.py"

ENUM = "/a/b/enum/enum_children.json"


def _extract(*names):
    tree = ast.parse(CAMPAIGN.read_text())
    ns = {"Path": Path, "json": json}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<extracted>", "exec"), ns)
    missing = [n for n in names if n not in ns]
    if missing:
        raise AssertionError(f"not found in run_campaign.py: {missing}")
    return [ns[n] for n in names]


def test_explicit_override_is_honoured():
    """The actual bug: the depth block ignored --hub-pick-timing and re-derived the path."""
    resolve, = _extract("resolve_hub_pick_timing")
    got = resolve(ENUM, "/elsewhere/pick_hubs_timing.json")
    assert str(got) == "/elsewhere/pick_hubs_timing.json", got
    print("  ok  explicit --hub-pick-timing is honoured")


def test_fallback_is_beside_enum_children():
    resolve, = _extract("resolve_hub_pick_timing")
    assert str(resolve(ENUM, None)) == "/a/b/enum/pick_hubs_timing.json"
    print("  ok  fallback resolves beside enum_children.json")


def test_both_callers_resolve_to_one_path():
    """compute-time and depth provenance must read the SAME file, or a run gets one from each."""
    resolve, load = _extract("resolve_hub_pick_timing", "load_hub_pick_s")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "custom_timing.json"
        p.write_text(json.dumps({"hub_pick_s": 12.5, "pool": "all",
                                 "walked_depth_hist": {"1": 158, "2": 39}}))
        assert load(ENUM, str(p)) == 12.5
        assert resolve(ENUM, str(p)) == p
    print("  ok  compute-time and depth provenance resolve to ONE path")


def test_absent_sidecar_is_nameable():
    """B's case. The point is not that it returns 0.0 -- it is that the path can be REPORTED.

    A null depth field with no path attached is indistinguishable from a legitimately empty answer;
    that is what let the pilot finish 'successfully' without a required deliverable.
    """
    resolve, load = _extract("resolve_hub_pick_timing", "load_hub_pick_s")
    enum = "/scratch/rgfn_runs/v2_pilot/campaign/rgfn_armA_all/enum/enum_children.json"
    p = resolve(enum, None)
    assert not p.is_file()
    assert load(enum, None) == 0.0
    assert p.name == "pick_hubs_timing.json" and "rgfn_armA_all" in str(p)
    print(f"  ok  absent sidecar is a nameable path, not a silent null:\n        {p}")


if __name__ == "__main__":
    print("hub-pick timing resolution:")
    test_explicit_override_is_honoured()
    test_fallback_is_beside_enum_children()
    test_both_callers_resolve_to_one_path()
    test_absent_sidecar_is_nameable()
    print("\nall passed")
