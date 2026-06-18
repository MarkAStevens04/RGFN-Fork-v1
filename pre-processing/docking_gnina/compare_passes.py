#!/usr/bin/env python3
"""Head-to-head: does Pass B (flexible anchor) sharpen the known-glue vs decoy separation?

Reads the pass-A and pass-B result CSVs for both the known-glue sample and the decoy set and
prints, per pass, the discrimination signal: fit-rate, median scores, and the strong-affinity
tail -- plus the known-minus-decoy gap (the number that actually matters for an RGFN reward).
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {
    "A": {"known": "batch_results.csv", "decoy": "decoy_results.csv"},
    "B": {"known": "batch_results_passB.csv", "decoy": "decoy_results_passB.csv"},
}


def load(path):
    rows = []
    with open(os.path.join(HERE, path)) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def fnum(rows, col, only_ok=True):
    out = []
    for r in rows:
        if only_ok and r["status"] != "ok":
            continue
        v = r.get(col, "")
        if v not in ("", None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def med(xs):
    return st.median(xs) if xs else float("nan")


def summarize(rows):
    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    nv = sum(1 for r in rows if r["status"] == "no_valid_pose")
    vina = fnum(rows, "vina_min")
    caff = fnum(rows, "cnn_affinity")
    cpose = fnum(rows, "cnn_score")
    strong_vina = sum(1 for v in vina if v < -10) / len(vina) if vina else float("nan")
    strong_caff = sum(1 for v in caff if v > 7) / len(caff) if caff else float("nan")
    return {
        "n": n,
        "ok": len(ok),
        "ok_rate": len(ok) / n if n else float("nan"),
        "no_valid_rate": nv / n if n else float("nan"),
        "vina_med": med(vina),
        "caff_med": med(caff),
        "cpose_med": med(cpose),
        "vina_min": min(vina) if vina else float("nan"),
        "frac_vina_lt_-10": strong_vina,
        "frac_caff_gt_7": strong_caff,
    }


def main():
    for p in ("A", "B"):
        print(f"\n{'='*78}\nPASS {p}")
        if not all(os.path.exists(os.path.join(HERE, SETS[p][s])) for s in ("known", "decoy")):
            print("  (results not present yet)")
            continue
        k = summarize(load(SETS[p]["known"]))
        d = summarize(load(SETS[p]["decoy"]))
        print(f"  {'metric':<22}{'known':>12}{'decoy':>12}{'gap(k-d)':>12}")

        def line(label, kk, dd, lower_is_better=False, pct=False):
            gap = kk - dd
            f = (lambda x: f"{x*100:5.1f}%") if pct else (lambda x: f"{x:7.2f}")
            arrow = ""
            if not pct:
                # which direction favors known being a 'better binder'
                arrow = ""
            print(
                f"  {label:<22}{f(kk):>12}{f(dd):>12}{gap*100 if pct else gap:>11.2f}{'%' if pct else ''}"
            )

        line("ok / fit-rate", k["ok_rate"], d["ok_rate"], pct=True)
        line("NO_VALID rate", k["no_valid_rate"], d["no_valid_rate"], pct=True)
        line("median Vina", k["vina_med"], d["vina_med"])
        line("median CNNaff", k["caff_med"], d["caff_med"])
        line("median CNN pose", k["cpose_med"], d["cpose_med"])
        line("frac Vina < -10", k["frac_vina_lt_-10"], d["frac_vina_lt_-10"], pct=True)
        line("frac CNNaff > 7", k["frac_caff_gt_7"], d["frac_caff_gt_7"], pct=True)
        print(
            f"  best Vina: known {k['vina_min']:.2f}  decoy {d['vina_min']:.2f}   "
            f"(n known={k['n']} ok={k['ok']}, decoy={d['n']} ok={d['ok']})"
        )


if __name__ == "__main__":
    main()
