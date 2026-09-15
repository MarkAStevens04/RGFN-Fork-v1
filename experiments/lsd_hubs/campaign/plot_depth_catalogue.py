#!/usr/bin/env python
"""OURS. The three figures from the depth / catalogue / catalogue-distinct experiments.

Reads results off disk and draws whatever is present, so re-running as replicates land extends the
error bars rather than requiring a rewrite. Every panel carries the caveat that governs how it may be
read, IN the figure rather than only in the caption -- these plots have already been wrong once by
mixing two different x-axes, and a caption is easy to crop away.

FIG 1  catalogue ladder: route-solve rate vs purchasable catalogue. The random-draw rungs are the
       point -- a random slice of ZINC the same size as ZINCFrag routes nothing, so the effect is
       CURATION and not size.
FIG 2  depth-cost on ONE axis: both methods' depth counted from ZINC. This is the headline. Ours is
       drawn as a BAND because the two unroutable policies bracket the truth: `drop` removes the
       molecules furthest from the competitor's catalogue (understates us) and `deep` counts them as
       passing (overstates us).
FIG 3  catalogue-distinct sweep: modes at a 100-reaction budget vs the tau that governs BOTH mutual
       and block dissimilarity.

Usage:  python plot_depth_catalogue.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results")
LAD = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/ladder")
OURS, THEIRS, BC = "#2a6f97", "#c1462f", "#8a8a8a"


def _read_ladder():
    out = {}
    for f in sorted(LAD.glob("*_ladder.jsonl")):
        cell = f.name.replace("_stage2_pruned_N500_ladder.jsonl", "")
        per = defaultdict(lambda: [0, 0])
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            per[r["stock"]][1] += 1
            per[r["stock"]][0] += int(bool(r["solved"]))
        if per:
            out[cell] = {k: (v[0], v[1]) for k, v in per.items()}
    return out


def _read_zinc_axis():
    """{(target, seed, policy): {depth: (modes, unjudged)}} — the LAST round of a run is its result."""
    out = defaultdict(dict)
    conv = {}
    for d in sorted(RES.glob("zinc_axis_*_d*_*")):
        m = re.search(r"zinc_axis_(?:(\w+?)_)?seed(\d+)_d(\d)_(drop|deep)$", d.name)
        if not m:
            continue
        tgt, seed, depth, pol = (m.group(1) or "seh"), int(m.group(2)), int(m.group(3)), m.group(4)
        dr = d / "driver_rounds.json"
        if not dr.exists():
            continue
        j = json.loads(dr.read_text())
        rounds = [d / f"round{r['round']}" for r in j["rounds"]]
        sm = [p for p in rounds if (p / "summary.json").exists()]
        if not sm:
            continue
        s = json.loads((sm[-1] / "summary.json").read_text())
        out[(tgt, seed, pol)][depth] = s["hub_batching"]["case1_modes_at_100rxn"]
        conv[(tgt, seed, pol, depth)] = j["rounds"][-1]["n_unjudged"] == 0
    return out, conv


# GENERATIONS OF THE SAME CELL SIT SIDE BY SIDE ON DISK, and the newest is not the one a plain
# sorted() glob picks. `s3gfn_seh_seed42_{CLEAN,UNION,big_pruned_N239,stage2_pruned_N500}_depth1_select`
# all match one glob and all key on (seh, 42, 1); sorted() takes `stage2...` last because lowercase
# sorts after uppercase, so the figure silently drew the OLDEST pool. Precedence has to be stated,
# not inferred from a name, and the chosen generation has to be printed -- a wrong pick here is
# invisible in the output, which is exactly how the _big pools were published and then retracted.
_GEN_ORDER = ("CLEAN", "UNION", "stage2fix", "stage2", "big", "pruned")


def _gen_rank(tag: str) -> tuple[int, str]:
    """(rank, generation-name) for a cell's tag segment; higher rank wins."""
    for i, g in enumerate(reversed(_GEN_ORDER), start=1):
        if g in tag:
            return i, g
    return 0, tag or "?"


def _pick_generation(cands: dict, what: str):
    """cands: {key: {(rank, gen): value}} -> {key: value}, printing what was chosen and dropped."""
    out, chosen = {}, defaultdict(set)
    for key, by_gen in cands.items():
        rank, gen = max(by_gen)
        out[key] = by_gen[(rank, gen)]
        chosen[key[:-1] if len(key) > 1 else key].add(gen)
        dropped = sorted(g for r, g in by_gen if g != gen)
        if dropped:
            print(f"[plot] {what} {key}: using {gen}, superseding {', '.join(dropped)}")
    mixed = {k: sorted(v) for k, v in chosen.items() if len(v) > 1}
    for k, v in mixed.items():
        print(f"[plot] WARNING {what} {k} mixes generations {v} across depths")
    per_target = defaultdict(set)
    for k, v in chosen.items():
        per_target[k[0]].update(v)
    for t, v in sorted(per_target.items()):
        if len(v) > 1:
            print(
                f"[plot] WARNING {what}: target {t} mixes generations {sorted(v)} ACROSS SEEDS -- "
                f"seeds are not comparable to each other until the laggard re-runs"
            )
    return out


def _read_competitor_depth():
    """{(target, seed): {depth: (modes, capped, used, budget)}} from the depth-pruned cells."""
    cands = defaultdict(dict)
    for d in sorted(RES.glob("s3gfn_*_depth*_select")):
        m = re.search(r"s3gfn_(\w+?)_seed(\d+)_(.*)_depth(\d)_select$", d.name)
        f = d / "select_frontier.csv"
        if not m or not f.exists():
            continue
        for r in csv.DictReader(f.open()):
            if r["budget_rxns"] == "100":
                key = (m.group(1), int(m.group(2)), int(m.group(4)))
                # cost_kept_rxns, not used_rxns: outside the budget-binding regime used_rxns inflates
                # and the gap between the two IS the exhaustion detector (CLAUDE.md, reaction-budget rule).
                cands[key][_gen_rank(m.group(3))] = (
                    int(r["n_modes_kept"]),
                    r["time_capped"] == "True",
                    float(r["cost_kept_rxns"]),
                    float(r["budget_rxns"]),
                )
    flat = _pick_generation(cands, "fig2")
    out = defaultdict(dict)
    for (tgt, seed, dep), v in flat.items():
        out[(tgt, seed)][dep] = v
    return out


def _read_budget(target):
    """({seed: {R: modes}}, {seed: {R: (modes, used, budget, capped)}}) for one target.

    Our side is read off the campaign curve, and ONLY at budgets the curve actually reaches -- past
    its end the campaign stopped for its own mode budget, and reading that as a plateau would
    truncate US rather than them.
    """
    ours, theirs = {}, {}
    # sEH was measured in TWO passes: R=50-500 before the scripts were target-parameterised (dirs
    # without a target segment) and R=600-2000 after (dirs with one). Both are the same cells at the
    # same gate, so the globs are merged -- reading only the newer pattern silently drops the entire
    # low-budget half of the curve, which is exactly where the crossover lives.
    pats_ours = [f"budget_scale_ours_{target}_seed*"]
    # The clean-pool re-runs write `budget_scale_s3gfn_<target>CLEAN_seed<N>` with NO underscore
    # before the generation, so `..._{target}_seed*` does not match them at all -- the fixed pools
    # would have been omitted from fig4 in silence rather than merely mis-ranked.
    pats_th = [f"budget_scale_s3gfn_{target}_seed*", f"budget_scale_s3gfn_{target}[A-Z]*_seed*"]
    if target == "seh":
        pats_ours.append("budget_scale_ours_seed*")
        pats_th.append("budget_scale_s3gfn_seed*")
    for d in sorted(x for pat in pats_ours for x in RES.glob(pat)):
        m = re.search(r"seed(\d+)$", d.name)
        f = d / "curve_hub_batching.csv"
        if not m or not f.exists():
            continue
        rows = list(csv.DictReader(f.open()))
        if not rows:
            continue
        reach = int(rows[-1]["cum_reactions"])
        cur = {}
        for R in (50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000):
            if R > reach:
                continue
            n = 0
            for r in rows:
                if int(r["cum_reactions"]) <= R:
                    n = int(r["cum_modes"])
                else:
                    break
            cur[R] = n
        ours.setdefault(int(m.group(1)), {}).update(cur)
    cands = defaultdict(dict)
    for d in sorted(set(x for pat in pats_th for x in RES.glob(pat))):
        m = re.search(rf"budget_scale_s3gfn_(?:{target})?(.*?)_seed(\d+)$", d.name)
        f = d / "select_frontier.csv"
        if not m or not f.exists():
            continue
        cur = {}
        for r in csv.DictReader(f.open()):
            # used_rxns inflates once the pool is exhausted; cost_kept_rxns is the true price of the
            # selected set, and used-minus-cost is how the plot tells exhaustion from binding.
            cur[int(float(r["budget_rxns"]))] = (
                int(r["n_modes_kept"]),
                float(r["cost_kept_rxns"]),
                float(r["budget_rxns"]),
                r["time_capped"] == "True",
            )
        cands[(target, int(m.group(2)))][_gen_rank(m.group(1))] = cur
    for (_, seed), cur in _pick_generation(cands, f"fig4:{target}").items():
        theirs.setdefault(seed, {}).update(cur)
    return ours, theirs


def _read_known():
    """{set: (n_input, n_solved, n_one_step)} -- how deep are molecules people actually make."""
    out = {}
    for f in sorted(LAD.glob("known/*_zinc.jsonl")):
        rs = [json.loads(l) for l in f.open() if l.strip()]
        sol = [r for r in rs if r["solved"] and r.get("min_steps")]
        if not rs or not sol:
            continue
        out[f.name.replace("_zinc.jsonl", "")] = (
            len(rs),
            len(sol),
            sum(1 for r in sol if r["min_steps"] == 1),
        )
    return out


def _read_cmode():
    ours, theirs = defaultdict(dict), defaultdict(dict)
    for d in sorted(RES.glob("cmode_ours_seed*_t*")):
        m = re.search(r"cmode_ours_seed(\d+)_t([\d.]+)$", d.name)
        f = d / "summary.json"
        if m and f.exists():
            j = json.loads(f.read_text())
            ours[float(m.group(2))][int(m.group(1))] = (
                j["hub_batching"]["case1_modes_at_100rxn"],
                j["best_candidate"]["case1_modes_at_100rxn"],
                j["hub_batching"]["stop_reason"],
            )
    for d in sorted(RES.glob("s3gfn_seh_seed*_cmode*_select")):
        m = re.search(r"seed(\d+)_cmode(\d\d)_N(\d+)_select$", d.name)
        f = d / "select_frontier.csv"
        if not m or not f.exists():
            continue
        tau = float(f"0.{m.group(2)[1]}")
        for r in csv.DictReader(f.open()):
            if r["budget_rxns"] == "100":
                theirs[tau][int(m.group(1))] = (int(r["n_modes_kept"]), int(m.group(3)))
    return ours, theirs


def _band(ax, xs, lo, hi, color, label):
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(xs, hi, "-o", color=color, ms=5, lw=2, label=label)
    ax.plot(xs, lo, "--", color=color, lw=1.2, alpha=0.8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default="experiments/lsd_hubs/campaign/results/depth_catalogue")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- FIG 1: catalogue ladder -------------------------------------------------------------
    lad = _read_ladder()
    if lad:
        order = [
            ("zinc", "ZINC\n17.4M"),
            ("zincfrag", "ZINCFrag\n178,598"),
            ("zinc_rand178k", "random ZINC\n178,598"),
            ("rgfnlib", "our blocks\n456"),
            ("zinc_rand418", "random ZINC\n456"),
        ]
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        cells = sorted(lad)
        w = 0.8 / max(len(cells), 1)
        for i, cell in enumerate(cells):
            xs, ys = [], []
            for k, (key, _) in enumerate(order):
                if key in lad[cell]:
                    s, n = lad[cell][key]
                    xs.append(k + i * w - 0.4 + w / 2)
                    ys.append(100.0 * s / n)
            ax.bar(xs, ys, width=w * 0.9, label=cell.replace("s3gfn_", ""))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([lbl for _, lbl in order], fontsize=8)
        ax.set_ylabel("molecules with a route (%)")
        ax.set_title(
            "Catalogue ladder: it is CURATION, not size\n"
            "a random slice of ZINC the same size as ZINCFrag routes nothing",
            fontsize=10,
        )
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.99,
            0.97,
            "ZINC and ZINCFrag are NOT nested in this build\n"
            "0% from our blocks = USPTO templates cannot reverse our chemistry,\n"
            "NOT chemical unreachability (cf. Logs/048 Exp A)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#555",
        )
        fig.tight_layout()
        fig.savefig(out / "fig1_catalogue_ladder.png", dpi=160)
        fig.savefig(out / "fig1_catalogue_ladder.pdf")
        plt.close(fig)
        print(f"[plot] fig1: {len(cells)} cells -> {out}/fig1_catalogue_ladder.png")

    # ---- FIG 2: depth-cost on ONE axis, per target -------------------------------------------
    za, conv = _read_zinc_axis()
    comp = _read_competitor_depth()
    targets = sorted({t for t, _, _ in za})
    if targets:
        fig, axes = plt.subplots(1, len(targets), figsize=(5.0 * len(targets), 4.8), squeeze=False)
        for ax, tgt in zip(axes[0], targets):
            seeds = sorted({sd for t, sd, _ in za if t == tgt})
            depths = sorted({d for (t, sd, p), v in za.items() if t == tgt for d in v})
            for sd in seeds:
                # The `deep` edge is the CONVERGED one; `drop` rungs mostly hit the round cap and are
                # upper bounds on the conservative reading, so `deep` is the line and `drop` the floor.
                hi = [za.get((tgt, sd, "deep"), {}).get(d) for d in depths]
                lo = [za.get((tgt, sd, "drop"), {}).get(d) for d in depths]
                if all(x is not None for x in hi):
                    if all(x is not None for x in lo):
                        ax.fill_between(depths, lo, hi, color=OURS, alpha=0.15, lw=0)
                        ax.plot(depths, lo, "--", color=OURS, lw=1.0, alpha=0.7)
                    ax.plot(
                        depths,
                        hi,
                        "-o",
                        color=OURS,
                        ms=4,
                        lw=2,
                        label="hub-batching" if sd == seeds[0] else None,
                    )
            for (t, sd), v in sorted(comp.items()):
                if t != tgt or len(v) < 2:
                    continue
                xs = sorted(v)
                ax.plot(
                    xs,
                    [v[d][0] for d in xs],
                    "-s",
                    color=THEIRS,
                    ms=4,
                    lw=2,
                    label="S3-GFN+MultiAiZ+SPARROW"
                    if sd == min(s2 for t2, s2 in comp if t2 == tgt)
                    else None,
                )
                for d in xs:
                    modes, capped, used, bud = v[d]
                    # A cell that could not spend the budget is a CAPACITY limit, not a cost result.
                    if used < bud - 1:
                        ax.plot([d], [modes], "x", color="k", ms=9, mew=2)
            ax.set_title(f"{tgt}", fontsize=11)
            ax.set_xlabel("min synthetic depth\n(reactions from ZINC)")
            ax.set_xticks(depths)
            ax.grid(alpha=0.25)
            if ax is axes[0][0]:
                ax.set_ylabel("distinct molecules at 100 reactions")
                ax.legend(fontsize=7, loc="lower left")
        fig.suptitle(
            "Depth vs delivered library — BOTH sides measured from ZINC\n"
            "band = unroutable policy: solid `deep` (converged, OVERSTATES us) to dashed `drop`\n"
            "(understates us, and not converged so the true floor is lower still)"
            "  ·  x = pool-exhausted, a capacity limit not a cost",
            fontsize=9.5,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(out / "fig2_depth_zinc_axis.png", dpi=160)
        fig.savefig(out / "fig2_depth_zinc_axis.pdf")
        plt.close(fig)
        print(f"[plot] fig2: targets {targets} -> fig2_depth_zinc_axis.png")

    # ---- FIG 3: catalogue-distinct sweep -----------------------------------------------------
    co, ct = _read_cmode()
    if co:
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        taus = sorted(co, reverse=True)
        for arm, idx, col, lbl in ((0, 0, OURS, "hub-batching"), (1, 1, BC, "best-candidate")):
            ys = [sum(v[idx] for v in co[t].values()) / len(co[t]) for t in taus]
            er = [
                (max(v[idx] for v in co[t].values()) - min(v[idx] for v in co[t].values())) / 2
                for t in taus
            ]
            ax.errorbar(taus, ys, yerr=er, fmt="-o", color=col, ms=5, lw=2, capsize=3, label=lbl)
        if ct:
            tt = sorted(ct, reverse=True)
            ax.plot(
                tt,
                [sum(v[0] for v in ct[t].values()) / len(ct[t]) for t in tt],
                "-s",
                color=THEIRS,
                ms=5,
                lw=2,
                label="S3-GFN + MultiAiZ + SPARROW",
            )
            for t in tt:
                for _, (mo, pool) in ct[t].items():
                    if pool < 500:
                        ax.annotate(
                            f"pool {pool}\n(pool-limited)",
                            (t, mo),
                            fontsize=6,
                            color=THEIRS,
                            xytext=(4, 6),
                            textcoords="offset points",
                        )
        ax.invert_xaxis()
        ax.set_xlabel(
            "tau — governs dissimilarity from other modes AND from every building block\n"
            "(stricter to the right)"
        )
        ax.set_ylabel("distinct molecules delivered at 100 reactions")
        ax.set_title(
            "Catalogue-distinct modes: what if the library must differ from what you can buy?\n"
            "no retrosynthesis involved — a structural constraint",
            fontsize=10,
        )
        ax.set_xticks(taus)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.text(
            0.01,
            0.03,
            "tau=0.3 is BOTH pool-limited and draw-sensitive (yield swings ~2.7x between\n"
            "two draws from one frozen checkpoint) — it carries no conclusion alone.\n"
            "Block reference set is ZINCFrag + our 418, NOT the 17.4M ZINC stock\n"
            "(zinc_stock.hdf5 is InChIKeys only and cannot be fingerprinted).",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#555",
        )
        fig.tight_layout()
        fig.savefig(out / "fig3_catalogue_distinct.png", dpi=160)
        fig.savefig(out / "fig3_catalogue_distinct.pdf")
        plt.close(fig)
        print(
            f"[plot] fig3: taus {taus}, ours seeds "
            f"{sorted({s for t in co for s in co[t]})} -> fig3_catalogue_distinct.png"
        )

    # ---- FIG 4: budget scaling / capacity ceiling (the headline) ------------------------------
    for tgt in ("seh", "drd2", "clpp"):
        bo, bt = _read_budget(tgt)
        if not bo or not bt:
            continue
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.8))
        Rs = sorted({R for v in bo.values() for R in v} | {R for v in bt.values() for R in v})
        for sd, v in sorted(bo.items()):
            xs = sorted(v)
            axL.plot(
                xs,
                [v[x] for x in xs],
                "-o",
                color=OURS,
                ms=4,
                lw=2,
                label="hub-batching" if sd == min(bo) else None,
            )
        for sd, v in sorted(bt.items()):
            xs = sorted(v)
            axL.plot(
                xs,
                [v[x][0] for x in xs],
                "-s",
                color=THEIRS,
                ms=4,
                lw=2,
                label="S3-GFN+MultiAiZ+SPARROW" if sd == min(bt) else None,
            )
            # mark where the budget stops being spendable -- the ceiling IS the finding
            ex = [x for x in xs if v[x][1] < v[x][2] - 1]
            if ex:
                axL.plot(
                    ex,
                    [v[x][0] for x in ex],
                    "x",
                    color="k",
                    ms=9,
                    mew=2,
                    label="pool-exhausted (budget unspendable)" if sd == min(bt) else None,
                )
        axL.set_xscale("log")
        axL.set_xlabel("reaction budget (log)")
        axL.set_ylabel("distinct molecules delivered")
        axL.grid(alpha=0.25, which="both")
        axL.legend(fontsize=7, loc="upper left")
        axL.set_title("delivered library vs budget", fontsize=10)
        for sd, v in sorted(bo.items()):
            xs = [x for x in sorted(v) if v[x]]
            axR.plot(xs, [x / v[x] for x in xs], "-o", color=OURS, ms=4, lw=2)
        for sd, v in sorted(bt.items()):
            xs = [x for x in sorted(v) if v[x][0]]
            axR.plot(
                xs, [min(v[x][1], v[x][2]) / v[x][0] for x in xs], "-s", color=THEIRS, ms=4, lw=2
            )
        axR.set_xscale("log")
        axR.set_xlabel("reaction budget (log)")
        axR.set_ylabel("reactions per delivered molecule")
        axR.axhline(1.0, color="#999", ls=":", lw=1)
        axR.annotate(
            "metric floor: 1 reaction per molecule",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=6.5,
            color="#666",
        )
        axR.grid(alpha=0.25, which="both")
        axR.set_title("cost per molecule vs budget", fontsize=10)
        fig.suptitle(
            f"{tgt}: the competitor holds the metric's FLOOR only while cheap molecules last\n"
            "beyond that its pool runs out and the budget becomes unspendable "
            "(x) — a capacity ceiling, not a cost",
            fontsize=9.5,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        fig.savefig(out / f"fig4_budget_{tgt}.png", dpi=160)
        fig.savefig(out / f"fig4_budget_{tgt}.pdf")
        plt.close(fig)
        print(f"[plot] fig4 {tgt}: ours {sorted(bo)}, theirs {sorted(bt)} -> fig4_budget_{tgt}.png")

    # ---- FIG 5: are the molecules people actually make deep? ---------------------------------
    kn = _read_known()
    if kn:
        label = {
            "glues_crbn_gspt1": "CRBN/GSPT1\nglues",
            "glues_ddb1_cdk12": "DDB1/CDK12\nglues",
            "actives_clpp": "ClpP actives\n(ChEMBL)",
            "actives_drd2": "DRD2 actives\n(ChEMBL)",
            "actives_seh": "sEH actives\n(ChEMBL)",
        }
        order = [
            k
            for k in (
                "glues_crbn_gspt1",
                "glues_ddb1_cdk12",
                "actives_clpp",
                "actives_drd2",
                "actives_seh",
            )
            if k in kn
        ]
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        xs = range(len(order))
        share = [100.0 * kn[k][2] / kn[k][1] for k in order]
        ax.bar(list(xs), share, color=OURS, width=0.6, label="known molecules")
        ax.axhline(49, color=THEIRS, ls="--", lw=2, label="S3-GFN's delivered library (49%)")
        for i, k in enumerate(order):
            n, sol, one = kn[k]
            ax.annotate(
                f"{one}/{sol}\nof {n} routed",
                (i, share[i]),
                fontsize=6.5,
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
            )
        ax.set_xticks(list(xs))
        ax.set_xticklabels([label.get(k, k) for k in order], fontsize=8)
        ax.set_ylabel("% of routed molecules that are ONE step from ZINC")
        ax.set_ylim(0, 62)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(
            "Molecules people actually make are NOT one step from the catalogue\n"
            "so requiring depth is a calibration to known chemistry, not a handicap",
            fontsize=10,
        )
        ax.text(
            0.99,
            0.97,
            "CRBN/GSPT1 routes only 4/100 — read as 'not one-step', not as a rate.\n"
            "Same planner, budget and catalogue as every other panel.",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#555",
        )
        fig.tight_layout()
        fig.savefig(out / "fig5_known_depth.png", dpi=160)
        fig.savefig(out / "fig5_known_depth.pdf")
        plt.close(fig)
        print(f"[plot] fig5: {len(order)} sets -> fig5_known_depth.png")


if __name__ == "__main__":
    main()
