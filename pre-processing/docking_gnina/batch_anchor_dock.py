#!/usr/bin/env python3
"""Score a random 10% of glutarimide-eligible compounds through the anchored oracle (pass A).

Two phases so the CPU work overlaps the GPU:
  Phase 1 (parallel, CPU): embed N anchored conformers per compound (warhead pinned to
           the CRBN cage) + MMFF arm-relax, write one SDF per compound to scratch.
  Phase 2 (sequential, GPU): gnina --minimize each compound's conformers, select the best
           pose by Vina energy (CLASH-AWARE), checkpoint to CSV.

A compound whose arm cannot avoid a clash (best Vina >= 0) is flagged NO_VALID_POSE rather
than reported with a bogus score.
"""
import csv
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
import anchor_dock as A

SMI = os.path.join(
    HERE, "..", "test-data", "Enamine_CRBN_Molecular_Glue_Library_plated_4560cmpds_20260614.smiles"
)
OUT = os.path.join(HERE, "batch_results.csv")
WORK = "/scratch/markymoo/anchor_batch"
FRACTION = 0.10
SEED = 42
N_CONFS = 100
N_WORKERS = 32

_CORE = None


def _init():
    global _CORE
    _CORE = A.crystal_glutarimide_core()


def largest_frag(m):
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda f: f.GetNumHeavyAtoms()) if len(frags) > 1 else m


def load_eligible(path=SMI):
    rows = []
    with open(path) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            smi, cid = parts[0], (parts[1] if len(parts) > 1 else "")
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            parent = largest_frag(m)
            if parent.HasSubstructMatch(A.GLUT_SMARTS):
                rows.append((cid, Chem.MolToSmiles(parent)))
    return rows


def gen_confs(task):
    """Phase-1 worker: embed anchored conformers, write SDF, return metadata."""
    idx, cid, smi = task
    sdf = os.path.join(WORK, f"c{idx:05d}.sdf")
    try:
        mol = A.build_conformers(smi, _CORE, n_confs=N_CONFS)
        drift = min(A.warhead_drift(mol, _CORE, c.GetId()) for c in mol.GetConformers())
        w = Chem.SDWriter(sdf)
        for c in mol.GetConformers():
            w.write(mol, confId=c.GetId())
        w.close()
        return {
            "idx": idx,
            "cid": cid,
            "smi": smi,
            "sdf": sdf,
            "drift": drift,
            "status": "embedded",
        }
    except Exception as e:
        return {
            "idx": idx,
            "cid": cid,
            "smi": smi,
            "sdf": None,
            "drift": None,
            "status": f"embed_fail:{type(e).__name__}",
        }


def main():
    global N_CONFS, OUT
    # argv: [input.smiles] [output.csv] [fraction] [n_confs]
    in_path = sys.argv[1] if len(sys.argv) > 1 else SMI
    out_path = sys.argv[2] if len(sys.argv) > 2 else OUT
    fraction = float(sys.argv[3]) if len(sys.argv) > 3 else FRACTION
    if len(sys.argv) > 4:
        N_CONFS = int(sys.argv[4])
    OUT = out_path

    os.makedirs(WORK, exist_ok=True)
    eligible = load_eligible(in_path)
    n = len(eligible) if fraction >= 1.0 else round(fraction * len(eligible))
    sample = random.Random(SEED).sample(eligible, n)
    tasks = [(i, cid, smi) for i, (cid, smi) in enumerate(sample, 1)]
    print(
        f"input={os.path.basename(in_path)}  eligible={len(eligible)}  "
        f"scoring {n} compounds (seed={SEED}, {N_CONFS} confs) -> {os.path.basename(out_path)}",
        flush=True,
    )

    t0 = time.time()
    print("phase 1: embedding conformers (parallel)...", flush=True)
    with mp.Pool(N_WORKERS, initializer=_init) as pool:
        gens = pool.map(gen_confs, tasks)
    n_embed = sum(g["status"] == "embedded" for g in gens)
    print(f"phase 1 done: {n_embed}/{n} embedded  ({(time.time()-t0)/60:.1f} min)", flush=True)

    print("phase 2: gnina minimize + clash-aware select (sequential GPU)...", flush=True)
    t1 = time.time()
    out_sdf = os.path.join(WORK, "_min.sdf")
    counts = {"ok": 0, "no_valid_pose": 0, "embed_fail": 0, "gnina_fail": 0}
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "idx",
                "catalog_id",
                "smiles",
                "status",
                "n_confs",
                "n_tethered",
                "n_nonclash",
                "out_drift",
                "vina_min",
                "cnn_score",
                "cnn_affinity",
            ]
        )
        for k, g in enumerate(gens, 1):
            if g["status"] != "embedded":
                counts["embed_fail"] += 1
                w.writerow([g["idx"], g["cid"], g["smi"], g["status"], "", "", "", "", "", "", ""])
            else:
                try:
                    best = A.gnina_local(g["sdf"], out_sdf, core=_CORE, select="vina")
                    on_tether = best.get("on_tether", True)
                    status = (
                        "ok" if (best["minimizedAffinity"] < 0 and on_tether) else "no_valid_pose"
                    )
                    counts[status] += 1
                    w.writerow(
                        [
                            g["idx"],
                            g["cid"],
                            g["smi"],
                            status,
                            best["n_poses"],
                            best["n_tethered"],
                            best["n_nonclash"],
                            f"{best['drift']:.3f}",
                            f"{best['minimizedAffinity']:.3f}",
                            f"{best['CNNscore']:.4f}",
                            f"{best['CNNaffinity']:.4f}",
                        ]
                    )
                except Exception as e:
                    counts["gnina_fail"] += 1
                    w.writerow(
                        [
                            g["idx"],
                            g["cid"],
                            g["smi"],
                            f"gnina_fail:{type(e).__name__}",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )
                if os.path.exists(g["sdf"]):
                    os.remove(g["sdf"])
            fh.flush()
            if k % 20 == 0 or k == n:
                rate = (time.time() - t1) / k
                print(
                    f"  {k}/{n}  ok={counts['ok']} no_valid={counts['no_valid_pose']} "
                    f"fail={counts['embed_fail']+counts['gnina_fail']}  "
                    f"{rate:.1f}s/mol eta={rate*(n-k)/60:.1f}min",
                    flush=True,
                )
    print(f"DONE ({(time.time()-t0)/60:.1f} min total) -> {OUT}", flush=True)
    print(f"  {counts}", flush=True)


if __name__ == "__main__":
    main()
