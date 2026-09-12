#!/usr/bin/env python
"""Export one cell's delivered library as the chemist-facing route dataset.

Implements ``docs/ROUTE_DATASET_SCHEMA.md`` for a cell = (generator, target, seed): runs both
selection strategies on IDENTICAL chemistry (same enumeration, same gate, same budget -- only the
chooser differs), resolves every selected molecule's logged assembly route down to purchasable
leaves, groups the molecules into the plates they are made in, and writes the tree of §3.

    <out-root>/<generator>_<target>_seed<N>/
      README.md                  gate + budget + every input by absolute path and md5 (§5, §7)
      hub_batching/
        molecules.csv  batches.csv  routes.json  steps.csv
        batch_01/{buy_list.csv, protocol.md, scheme.png}
        batch_02/ ...
      best_candidate/ ...

THIS SCRIPT IS THE ONLY PART THAT KNOWS WHERE ANYTHING LIVES. The exporter itself is
``glue/export/`` and takes objects, not paths, because ``docs/RETRAIN_RUNBOOK.md`` moves every input
under ``benchmark_v2/`` and re-derives the gates. A v2 driver is a new file beside this one; nothing
in ``glue/export/`` changes. Everything here is a flag with no default that encodes a run: the gate
is required and never guessed (§2.1 of the runbook: "These are empirical grid points -- rounding
breaks the exact-FPR property they are defined by").

WHAT IT REUSES, AND WHY NOT A NEW LOADER. The candidate pool and the enumerated hubs come from
``run_campaign._load_candidates`` and ``parallel_groups.load_hubs_with_reactions`` -- the same
loaders the committed campaign and Logs/073 ran, so the library exported here is the library those
analyses describe rather than one re-derived by a second implementation that could drift.
``load_hubs_with_reactions`` parity-checks itself against ``run_campaign._load_enumerated_hubs``
before anything is written; a mismatch aborts, because groups over a different hub list would
describe a library we never selected.

THE THREE THINGS THAT ARE VERIFIED, not asserted (each names its failing case):
  * **every leaf is purchasable** (§2) -- a route that says *buy X* for an unbuyable X is a broken
    cell, not a footnote. The failing case is a run whose fragment snapshot belongs to a different
    model: internally complete, so coverage of its own ``chosen_smiles`` reads 100%, while the
    run's own promoted fragments are silently bought instead of built.
  * **the tree and the flat table agree** (§4.4) -- built by different code paths (a recursive
    expansion over ``by_product`` vs a linear walk of the ordered steps), so the check can fail:
    an intermediate reachable twice, or a final step whose product is not the molecule, breaks it.
  * **the stereo-aware SMILES matches the sampler's own record** -- ``child_stereo_key`` in
    ``records.csv`` is an independent record of the same fact. The failing case is exactly the one
    ``render_route_scheme.py``'s docstring assumed: a generation whose ``routes.json`` products are
    stereo-STRIPPED, where the recovered ``smiles`` would silently be the flat form.

Pure CPU; no GPU, no oracle calls -- a re-selection over a cached enumeration plus file writing.
Reads a FROZEN enumeration (§7), never a live path.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/export_library.py \\
        --analysis-dir /scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed43/scent_seh/sample \\
        --enum-children /scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820/seed43/enum_children.json \\
        --snapshot /scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh_5k/seed43/additional_fragments/fragments_4000.json \\
        --reward-threshold 7.0 --seed 43 --budget-reactions 100
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from parallel_groups import load_hubs_with_reactions, parity_check
from run_campaign import _load_candidates, build_strategy

from glue.export import (
    CellProvenance,
    LoggedRouteSource,
    RouteAssembler,
    build_library,
    leaf_audit,
    write_cell_readme,
    write_dataset_readme,
    write_strategy,
)
from glue.samplers.lsdflow.campaign import rank_fragments
from glue.samplers.lsdflow.child_select import make_child_policy
from glue.samplers.lsdflow.mode_select import _FP_BITS, _FP_RADIUS
from validation.lsdflow.metrics.cost.dynamic_amortization import (
    load_cost_table_from_snapshot,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "lsdflow-routes"
SCHEMA_DOC = HERE.parents[2] / "docs" / "ROUTE_DATASET_SCHEMA.md"

STRATEGIES = ("hub_batching", "best_candidate")


# ----------------------------------------------------------------- inputs
def load_stereo_records(analysis_dir: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    """``flat smiles -> stereo-aware smiles`` from the sampler's own records, plus how often that
    map is AMBIGUOUS.

    The map is used only to check the stereo-aware SMILES the exporter recovered from the logged
    route: it is an independent record of the same fact, which is what makes the check a check.

    The ambiguity count is a measurement §6 asks for by name. Selection ran on stereo-stripped
    structures, so "two enantiomers counted as ONE molecule during selection" -- and a flat key that
    the sampler saw in two different configurations is exactly one of those merges, caught in the
    act. Reporting the rate turns a stated limitation into a number for this cell.
    """
    out: Dict[str, str] = {}
    forms: Dict[str, set] = {}
    p = analysis_dir / "records.csv"
    if not p.exists():
        return out, {"n_keys": 0, "n_multi_stereo_keys": 0}
    with open(p) as fh:
        rdr = csv.DictReader(fh)
        if "child_stereo_key" not in (rdr.fieldnames or []):
            return out, {"n_keys": 0, "n_multi_stereo_keys": 0}
        for r in rdr:
            for flat, stereo in (
                (r.get("child_key"), r.get("child_stereo_key")),
                (r.get("hub_key"), r.get("hub_stereo_key")),
            ):
                if flat and stereo:
                    out.setdefault(flat, stereo)
                    forms.setdefault(flat, set()).add(stereo)
    multi = sum(1 for v in forms.values() if len(v) > 1)
    return out, {"n_keys": len(forms), "n_multi_stereo_keys": multi}


def final_step_index(
    hubs,
) -> Tuple[Dict[Tuple[str, str], List[dict]], Dict[Tuple[str, str], tuple], int]:
    """``(hub_key, child_smiles) -> [the logged final step]`` for every enumerated child.

    The step is rebuilt from the fields ``RxChild`` carries, which is the LAST logged step. An
    enumerated child is one reaction off its hub, so that is the whole of it -- and the count of
    children whose enumeration recorded more than one step is returned so the caller can refuse to
    export a generation where that stopped being true instead of silently dropping steps.
    """
    finals: Dict[Tuple[str, str], List[dict]] = {}
    extra: Dict[Tuple[str, str], tuple] = {}
    n_missing = 0
    for h in hubs:
        for c in h.children:
            key = (h.hub_key, c.smiles)
            if key in finals:
                continue
            if not c.product:
                n_missing += 1
                continue
            finals[key] = [
                {
                    "reaction": c.template,
                    "reactants": list(c.reagents),
                    "input": c.parent_input or h.hub_key,
                    "product": c.product,
                }
            ]
            extra[key] = tuple(c.added_promoted)
    return finals, extra, n_missing


# ----------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--analysis-dir",
        required=True,
        help="sample dir: records.csv, compositions.json, routes.json, meta.json",
    )
    ap.add_argument("--enum-children", required=True, help="FROZEN enum_children.json (§7)")
    ap.add_argument("--snapshot", required=True, help="fragments_<N>.json (recipes + catalogue)")
    ap.add_argument(
        "--reward-threshold",
        type=float,
        required=True,
        help="the cell's hit gate. Required and never defaulted: gates are empirical 5%%-FPR grid "
        "points, so a guessed or rounded one silently redefines the library.",
    )
    ap.add_argument("--similarity", type=float, default=0.5, help="diversity cutoff tau")
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=None)
    ap.add_argument("--budget-reactions", type=int, default=100)
    ap.add_argument("--child-policy", default="free_frag", choices=["reward", "free_frag"])
    ap.add_argument("--prebuild-k", type=int, default=20)
    ap.add_argument("--rank-by", default="build_score")
    ap.add_argument("--generator", default="", help="default: meta.json 'model'")
    ap.add_argument("--target", default="", help="default: meta.json 'reward_name'")
    ap.add_argument("--seed", default="", help="seed label for the cell directory name")
    ap.add_argument("--cell", default="", help="override the whole cell directory name")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--no-schemes", action="store_true", help="skip the batch scheme PNGs")
    ap.add_argument(
        "--no-parity-check",
        dest="parity_check",
        action="store_false",
        default=True,
        help="skip re-parsing the enumeration to verify the hub loader against run_campaign's. "
        "On by default: it is the only thing that catches a loader drift, which would silently "
        "export a library we never selected.",
    )
    ap.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="only write the first N batch_NN/ dirs (smoke test; 0 = all). The CSVs are always "
        "complete -- this caps the per-batch directories, and the README says so.",
    )
    ap.add_argument(
        "--limitation",
        action="append",
        default=[],
        help="extra 'known limitations' line for this cell's README (repeatable).",
    )
    a = ap.parse_args()

    adir = Path(a.analysis_dir)
    meta = json.loads((adir / "meta.json").read_text()) if (adir / "meta.json").exists() else {}
    generator = a.generator or meta.get("model") or "unknown"
    target = a.target or meta.get("reward_name") or "unknown"
    higher = a.higher_is_better
    if higher is None:
        higher = bool(meta.get("higher_is_better", True))
    cell = a.cell or (f"{generator}_{target}_seed{a.seed}" if a.seed else f"{generator}_{target}")
    out_root = Path(a.out_root)
    cell_dir = out_root / cell

    # ---------------------------------------------------------------- load (one pass each)
    cands, comps = _load_candidates(adir, higher)
    hubs, cov = load_hubs_with_reactions(Path(a.enum_children), comps)
    print(
        f"[export] {cell}: {len(cands):,} candidates, {len(hubs)} hubs, "
        f"{cov['n_children']:,} children, reaction coverage {cov['rxn_coverage']:.4%}, "
        f"multi-step children {cov['n_multi_step_children']}"
    )
    if a.parity_check:
        # A second full parse of the enumeration, which is why it is a flag: the check is worth it
        # (a drifted loader would describe a library we never selected) but it is not free.
        if not parity_check(hubs, comps, a.enum_children):
            sys.exit(
                "[export] ABORT: re-derived hubs differ from run_campaign._load_enumerated_hubs"
            )
        print("[export] loader parity vs run_campaign._load_enumerated_hubs: OK")
    if cov["rxn_coverage"] is None or cov["rxn_coverage"] < 1.0:
        print(
            f"[export] WARNING: {1 - (cov['rxn_coverage'] or 0):.4%} of enumerated children carry "
            "no reaction. A partially-reacted enumeration prices a mixture (RETRAIN_RUNBOOK §6.2); "
            "those children cannot be routed and will be dropped if selected."
        )
    if cov["n_multi_step_children"]:
        sys.exit(
            f"[export] ABORT: {cov['n_multi_step_children']} enumerated children record more than "
            "one reaction step. The final-step index keeps only the last one, so exporting would "
            "drop real chemistry. Extend final_step_index() before re-running."
        )

    snapshot = json.loads(Path(a.snapshot).read_text())
    cost_table = load_cost_table_from_snapshot(snapshot)
    purchasable = set(snapshot.get("initial_smiles_set") or ())
    recipes = snapshot.get("smiles_to_route") or {}
    sampled_routes = (
        json.loads((adir / "routes.json").read_text()) if (adir / "routes.json").exists() else {}
    )
    # Recipes win on overlap: a promoted fragment's own logged build is the authority on how it is
    # made, and it is the sub-route §2 promises to expand inline.
    routes = {**sampled_routes, **recipes}
    print(
        f"[export] catalogue {len(purchasable)} blocks | {len(recipes)} fragment recipes | "
        f"{len(sampled_routes):,} sampled routes -> {len(routes):,} route entries"
    )

    prebuilt = None
    if a.prebuild_k > 0:
        ranked = rank_fragments(
            hubs, cost_table, a.reward_threshold, method=a.rank_by, higher_is_better=higher
        )
        prebuilt = {f for f, _ in ranked[: a.prebuild_k]}

    finals, extra_builds, n_missing_product = final_step_index(hubs)
    if n_missing_product:
        print(f"[export] {n_missing_product:,} children carry no product and cannot be routed")

    assembler = RouteAssembler(routes, purchasable)
    stereo_records, stereo_stats = load_stereo_records(adir)
    print(
        f"[export] {len(stereo_records):,} stereo records for the independent SMILES check; "
        f"{stereo_stats['n_multi_stereo_keys']:,} of {stereo_stats['n_keys']:,} flat keys were "
        "seen in more than one configuration (the §6 enantiomer merge, measured)"
    )

    common = dict(
        target=target,
        reward_threshold=a.reward_threshold,
        similarity=a.similarity,
        higher_is_better=higher,
    )
    budget = ("reactions", a.budget_reactions)

    # ---------------------------------------------------------------- export each strategy
    summaries, audits = [], {}
    for name in [s.strip() for s in a.strategies.split(",") if s.strip()]:
        if name == "hub_batching":
            result = build_strategy(
                "hub_batching",
                hubs,
                cost_table,
                comps,
                child_policy=make_child_policy(a.child_policy),
                prebuilt_fragments=prebuilt,
                **common,
            ).run(budget)
            source = LoggedRouteSource(assembler, final_steps=finals, extra_builds=extra_builds)
        else:
            result = build_strategy("best_candidate", cands, cost_table, comps, **common).run(
                budget
            )
            source = LoggedRouteSource(assembler)  # sampled terminals carry their own route

        lib = build_library(
            result,
            source,
            purchasable,
            reward_name=target,
            budget_reactions=a.budget_reactions,
            stereo_check=stereo_records or None,
        )
        audit = leaf_audit(lib.trees, purchasable)
        audits[name] = audit
        summary = write_strategy(
            cell_dir / name,
            lib,
            f"{cell} / {name}",
            schemes=not a.no_schemes,
            max_batch_dirs=a.max_batches,
        )
        summaries.append(summary)
        print(
            f"[export] {name}: {summary['n_molecules']} molecules in {summary['n_batches']} "
            f"batches, {summary['reactions_used']}/{a.budget_reactions} reactions "
            f"(stop={summary['stop_reason']}, largest batch {summary['largest_batch']}) | "
            f"leaves {audit.n_leaves}, not purchasable {audit.n_unpurchasable} | "
            f"stereo disagreements {summary['n_stereo_disagreements_vs_records']} | "
            f"dropped (no route) {summary['n_dropped_no_route']}"
        )
        if summary["scheme_errors"]:
            print(f"[export]   scheme errors: {summary['scheme_errors'][:3]}")

    # ---------------------------------------------------------------- provenance (§5, §7)
    enum_path = Path(a.enum_children)
    sidecar = enum_path.with_suffix(".md5")
    artifact_md5 = {}
    if sidecar.exists():
        artifact_md5["enumeration (frozen)"] = sidecar.read_text().strip().split()[0]
    prov = CellProvenance(
        cell=cell,
        generator=generator,
        target=target,
        reward_name=target,
        hit_threshold=a.reward_threshold,
        higher_is_better=higher,
        budget_reactions=a.budget_reactions,
        similarity_cutoff=a.similarity,
        similarity_metric=f"Tanimoto on ECFP Morgan r={_FP_RADIUS}, {_FP_BITS} bits, stereo-blind",
        artifacts={
            "model checkpoint": meta.get("checkpoint", "(not recorded in meta.json)"),
            "enumeration (frozen)": str(enum_path),
            "fragment snapshot": str(Path(a.snapshot)),
            "sampled routes": str(adir / "routes.json"),
            "sample records": str(adir / "records.csv"),
        },
        artifact_md5=artifact_md5,
        config={
            "child policy": a.child_policy,
            "pre-select K": a.prebuild_k,
            "fragment ranking": a.rank_by,
            "hubs enumerated": len(hubs),
            "enumerated children": cov["n_children"],
            "candidate pool": len(cands),
            "training config": meta.get("config", "(not recorded)"),
            "sample dir": str(adir),
        },
        limitations=_limitations(a, cov, summaries, audits, stereo_stats) + list(a.limitation),
    )
    write_dataset_readme(out_root, SCHEMA_DOC)
    write_cell_readme(cell_dir / "README.md", prov, summaries, audits)
    (cell_dir / "export_summary.json").write_text(
        json.dumps(
            {
                "cell": cell,
                "generator": generator,
                "target": target,
                "budget_reactions": a.budget_reactions,
                "reward_threshold": a.reward_threshold,
                "higher_is_better": higher,
                "similarity": a.similarity,
                "child_policy": a.child_policy,
                "prebuild_k": a.prebuild_k,
                "inputs": {
                    "analysis_dir": str(adir),
                    "enum_children": str(enum_path),
                    "snapshot": str(Path(a.snapshot)),
                    **cov,
                },
                "strategies": summaries,
                "leaf_audit": {
                    k: {
                        "n_routes": v.n_routes,
                        "n_leaves": v.n_leaves,
                        "n_unpurchasable": v.n_unpurchasable,
                        "examples": v.examples,
                    }
                    for k, v in audits.items()
                },
            },
            indent=2,
        )
    )
    print(f"[export] wrote {cell_dir}")


def _limitations(a, cov, summaries, audits, stereo_stats) -> List[str]:
    """The caveats §5 requires, stated from what this export actually measured."""
    lims = [
        "**Selection ran on stereo-stripped structures**, so two enantiomers counted as ONE "
        "molecule during selection and a library can contain a near-duplicate enantiomeric pair "
        "the diversity metric merged. The stereochemistry is known (it is in `smiles`); it just "
        "did not participate in the choosing. Measured on this cell: "
        f"**{stereo_stats['n_multi_stereo_keys']:,} of {stereo_stats['n_keys']:,} "
        "stereo-stripped keys the sampler recorded were reached in more than one configuration**, "
        "so `smiles` is the configuration of the route that is shipped, not the only one that "
        "flat key can stand for.",
        "**The budget counts reactions, and a reaction count is yield-blind** (Logs/071): a "
        "100-reaction library could be 100 reactions of the safest chemistry or 100 of the "
        "riskiest and the number would be the same. The template library ships per-reaction "
        "yields (0.65-0.95) which this dataset does not carry.",
        "**Failures inside a batch are correlated** (Logs/071): one bad step in a shared prefix "
        "loses the whole plate. `prefix_reactions` in `batches.csv` is how many steps that "
        "exposure runs through, and the largest batch is how many molecules ride on it.",
        "**A batch is not necessarily one plate** (Logs/073): members share a substrate, but not "
        "necessarily one set of conditions. Group `steps.csv` on (`product` of the prefix, "
        "`named_reaction`) over the `is_shared=false` rows to see the parallel arrays a batch "
        "actually decomposes into.",
        "**Molecule count overstates bench workload where a substrate is polyfunctional** "
        "(Logs/073): one reaction on a two-site substrate gives two separable products from one "
        "vessel. Distinct (substrate, reagent) pairs among the diverging steps is the well count.",
        "**`routes.json` is the logged assembly route, not a retrosynthesis.** Every step is a "
        "template that fired; none is a proposal.",
    ]
    for s in summaries:
        if s["stop_reason"] != "reactions":
            lims.append(
                f"`{s['strategy']}` stopped with `{s['stop_reason']}` after "
                f"{s['reactions_used']} of {s['budget_reactions']} reactions: the budget did NOT "
                "bind, so this arm's molecule count is pool-limited and is not a like-for-like "
                "reading against an arm that spent its budget."
            )
        if s.get("n_dropped_no_route"):
            lims.append(
                f"`{s['strategy']}`: {s['n_dropped_no_route']} selected molecule(s) had no "
                "recoverable route and were omitted from this dataset (they remain part of the "
                "selection the campaign reported)."
            )
        if s.get("n_stereo_disagreements_vs_records"):
            lims.append(
                f"`{s['strategy']}`: {s['n_stereo_disagreements_vs_records']} molecule(s) whose "
                "stereo-aware SMILES recovered from the route differs from the sampler's own "
                "record of the same flat key. These are the enantiomer merges above showing up in "
                "the library itself: the shipped `smiles` is the configuration the shipped ROUTE "
                "makes, which is the one a chemist following `protocol.md` would get."
            )
        if s.get("n_steps_creating_a_stereocentre"):
            lims.append(
                f"`{s['strategy']}`: **{s['n_steps_creating_a_stereocentre']} of {s['n_steps']} "
                "steps CREATE an undefined stereocentre** "
                f"({', '.join(s.get('stereocentre_creating_reactions') or [])}), so the usual "
                "chemist's intuition -- a new centre comes out racemic and separating enantiomers "
                "is painful -- does apply to those molecules. It does not apply to the rest: every "
                "other centre in this library is inherited from a catalogue block. "
                f"{s['n_molecules_with_unassigned_stereo']} molecule(s) carry an undefined centre "
                "and are flagged by `has_unassigned_stereo` in `molecules.csv`."
            )
        if (
            s.get("n_batches_with_assigned_hub")
            and s.get("n_batches_prefix_is_assigned_hub", 0) < s["n_batches_with_assigned_hub"]
        ):
            lims.append(
                f"`{s['strategy']}`: for "
                f"{s['n_batches_with_assigned_hub'] - s['n_batches_prefix_is_assigned_hub']} of "
                f"{s['n_batches_with_assigned_hub']} batches the cost model's assigned hub is not "
                "the actual shared prefix of the members' logged routes. `shared_intermediate` "
                "reports the real prefix (what you weigh into the well), not the accounting hub."
            )
    for name, audit in audits.items():
        if not audit.solved:
            lims.append(
                f"**`{name}` has {audit.n_unpurchasable} leaves that are NOT in the catalogue** "
                f"(e.g. {', '.join(audit.examples[:2])}). §2 promises every route bottoms out in "
                "purchasable blocks; this cell does not keep that promise and should not ship."
            )
    if (cov.get("rxn_coverage") or 1.0) < 1.0:
        lims.append(
            f"Only {cov['rxn_coverage']:.4%} of enumerated children carried a logged reaction."
        )
    return lims


if __name__ == "__main__":
    main()
