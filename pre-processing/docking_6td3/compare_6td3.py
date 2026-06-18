#!/usr/bin/env python3
"""Does the 6TD3 docking proxy separate known CDK12-glues from random purine-armed decoys?

The decisive metric is the DDB1 differential (ddb1_dvina = Tier2 - Tier1): a real glue's arm
contacts DDB1 (big negative bonus); a decoy with a random arm should not. If the known set
beats the decoys here, the proxy rewards glue behaviour -- the discrimination CRBN lacked.
"""
import csv
import os
import statistics as st

HERE = os.environ.get("RESULTS_DIR", os.path.dirname(os.path.abspath(__file__)))
FILES = {"known": "known_results.csv", "decoy": "decoy_cdk_results.csv"}


def load(path):
    return list(csv.DictReader(open(os.path.join(HERE, path))))


def col(rows, c):
    out = []
    for r in rows:
        if r["status"] != "ok":
            continue
        v = r.get(c, "")
        if v not in ("", None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def med(xs):
    return st.median(xs) if xs else float("nan")


def frac(xs, pred):
    return (sum(1 for x in xs if pred(x)) / len(xs)) if xs else float("nan")


def summarize(rows):
    return {
        "n": len(rows),
        "ok": sum(1 for r in rows if r["status"] == "ok"),
        "vina_t2": med(col(rows, "vina_t2")),
        "cnnaff_t2": med(col(rows, "cnnaff_t2")),
        "cnnsc_t2": med(col(rows, "cnnsc_t2")),
        "ddb1_dvina": med(col(rows, "ddb1_dvina")),
        "ddb1_dcnnaff": med(col(rows, "ddb1_dcnnaff")),
        "frac_strong_bonus": frac(col(rows, "ddb1_dvina"), lambda x: x < -1.5),
        "frac_vina_lt_-10": frac(col(rows, "vina_t2"), lambda x: x < -10),
        "best_vina": min(col(rows, "vina_t2")) if col(rows, "vina_t2") else float("nan"),
        "best_bonus": min(col(rows, "ddb1_dvina")) if col(rows, "ddb1_dvina") else float("nan"),
    }


def main():
    if not all(os.path.exists(os.path.join(HERE, f)) for f in FILES.values()):
        print(
            "results not present yet:",
            {k: os.path.exists(os.path.join(HERE, v)) for k, v in FILES.items()},
        )
        return
    k = summarize(load(FILES["known"]))
    d = summarize(load(FILES["decoy"]))
    print(f"{'metric':<26}{'known':>12}{'decoy':>12}{'gap(k-d)':>12}")
    print("-" * 62)

    def line(label, kk, dd, pct=False):
        f = (lambda x: f"{x*100:6.1f}%") if pct else (lambda x: f"{x:8.2f}")
        gap = (kk - dd) * (100 if pct else 1)
        print(f"  {label:<24}{f(kk):>12}{f(dd):>12}{gap:>11.2f}{'%' if pct else ''}")

    print(f"  n (ok)                  {k['ok']:>10}  {d['ok']:>10}")
    line("median Tier2 Vina", k["vina_t2"], d["vina_t2"])
    line("median Tier2 CNNaff", k["cnnaff_t2"], d["cnnaff_t2"])
    line("median Tier2 CNN pose", k["cnnsc_t2"], d["cnnsc_t2"])
    print("  --- DDB1 differential (the glue-specific signal) ---")
    line("median DDB1 dVina", k["ddb1_dvina"], d["ddb1_dvina"])
    line("median DDB1 dCNNaff", k["ddb1_dcnnaff"], d["ddb1_dcnnaff"])
    line("frac DDB1 dVina < -1.5", k["frac_strong_bonus"], d["frac_strong_bonus"], pct=True)
    line("frac Tier2 Vina < -10", k["frac_vina_lt_-10"], d["frac_vina_lt_-10"], pct=True)
    print(f"  best Tier2 Vina: known {k['best_vina']:.2f}  decoy {d['best_vina']:.2f}")
    print(f"  best DDB1 bonus: known {k['best_bonus']:.2f}  decoy {d['best_bonus']:.2f}")


if __name__ == "__main__":
    main()
