#!/usr/bin/env python3
"""Dock a set of molecules into the CR8/cyclinK glue pocket and measure the DDB1 cooperativity.

gnina SAMPLES this ATP pocket well (CR8 redocks to 1.23 A, ranked #1), so we use straightforward
global docking -- no warhead anchoring needed. Per molecule:
  1. embed one 3D conformer (gnina re-samples position),
  2. global-dock into Tier 2 (CDK12 + DDB1), autobox around the native CR8,
  3. take the best pose by CNN pose-score, read its Tier 2 Vina / CNNaffinity,
  4. score that SAME pose against Tier 1 (CDK12 only) -> Tier 1 Vina / CNNaffinity,
  5. DDB1 bonus = Tier2 - Tier1  (the glue-specific signal: how much DDB1 contact stabilises it).

Usage: dock_6td3_batch.py <input.csv|.smiles> <output.csv> <set_label> [limit]
  .csv  -> reads the 'SMILES' column (the DDB1_CDK12_Glues set); id from DATAID/Name.
  .smiles -> 'SMILES<TAB>ID' with header (the decoys).
"""
import csv
import multiprocessing as mp
import os
import subprocess
import sys
import time

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
GNINA = "/scratch/markymoo/gnina/run_gnina.sh"
TIER2 = os.path.join(HERE, "6TD3_tier2.pdbqt")
TIER1 = os.path.join(HERE, "6TD3_tier1.pdbqt")
CRYSTAL = os.path.join(HERE, "crystal_RC8.pdb")
WORK = "/scratch/markymoo/dock_6td3"
EXH = 16
NUM_MODES = 9
SEED = 42
N_WORKERS = 32


def largest_frag(m):
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return max(fr, key=lambda f: f.GetNumHeavyAtoms()) if len(fr) > 1 else m


def load_known(path):
    rows, seen = [], set()
    for r in csv.DictReader(open(path)):
        m = Chem.MolFromSmiles(r["SMILES"])
        if m is None:
            continue
        smi = Chem.MolToSmiles(largest_frag(m))
        if smi in seen:
            continue
        seen.add(smi)
        cid = r.get("DATAID") or r.get("Name") or str(len(rows))
        rows.append((cid, smi))
    return rows


def load_smiles(path):
    rows, seen = [], set()
    with open(path) as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            m = Chem.MolFromSmiles(parts[0])
            if m is None:
                continue
            smi = Chem.MolToSmiles(largest_frag(m))
            if smi in seen:
                continue
            seen.add(smi)
            rows.append((parts[1] if len(parts) > 1 else str(len(rows)), smi))
    return rows


def gen_conf(task):
    idx, cid, smi = task
    sdf = os.path.join(WORK, f"m{idx:05d}.sdf")
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(m, randomSeed=SEED) != 0:
            AllChem.EmbedMolecule(m, randomSeed=SEED, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(m)
        w = Chem.SDWriter(sdf)
        w.write(m)
        w.close()
        return {"idx": idx, "cid": cid, "smi": smi, "sdf": sdf, "status": "embedded"}
    except Exception as e:
        return {
            "idx": idx,
            "cid": cid,
            "smi": smi,
            "sdf": None,
            "status": f"embed_fail:{type(e).__name__}",
        }


def _poses(sdf):
    out = []
    for m in Chem.SDMolSupplier(sdf, removeHs=False, sanitize=False):
        if m is None:
            continue
        g = lambda k: float(m.GetProp(k)) if m.HasProp(k) else float("nan")
        out.append(
            {
                "mol": m,
                "vina": g("minimizedAffinity"),
                "cnnsc": g("CNNscore"),
                "cnnaff": g("CNNaffinity"),
            }
        )
    return out


def _score_only(receptor, sdf):
    r = subprocess.run(
        [GNINA, "-r", receptor, "-l", sdf, "--score_only"],
        capture_output=True,
        text=True,
        check=True,
    )
    vals = {}
    for line in r.stdout.splitlines():
        if line.startswith("Affinity:"):
            vals["vina"] = float(line.split()[1])
        elif line.startswith("CNNscore:"):
            vals["cnnsc"] = float(line.split()[1])
        elif line.startswith("CNNaffinity:"):
            vals["cnnaff"] = float(line.split()[1])
    return vals


def dock_one(sdf_in):
    dock = os.path.join(WORK, "_dock.sdf")
    subprocess.run(
        [
            GNINA,
            "-r",
            TIER2,
            "-l",
            sdf_in,
            "--autobox_ligand",
            CRYSTAL,
            "--autobox_add",
            "4",
            "--exhaustiveness",
            str(EXH),
            "--num_modes",
            str(NUM_MODES),
            "--seed",
            str(SEED),
            "-o",
            dock,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    poses = _poses(dock)
    if not poses:
        return None
    best = max(poses, key=lambda p: p["cnnsc"])  # most native-like pose
    best_sdf = os.path.join(WORK, "_best.sdf")
    w = Chem.SDWriter(best_sdf)
    w.write(best["mol"])
    w.close()
    t1 = _score_only(TIER1, best_sdf)
    return {
        "vina_t2": best["vina"],
        "cnnsc_t2": best["cnnsc"],
        "cnnaff_t2": best["cnnaff"],
        "vina_t1": t1.get("vina", float("nan")),
        "cnnaff_t1": t1.get("cnnaff", float("nan")),
        "n_poses": len(poses),
    }


def main():
    in_path, out_path, set_label = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    os.makedirs(WORK, exist_ok=True)
    mols = load_known(in_path) if in_path.endswith(".csv") else load_smiles(in_path)
    if limit:
        mols = mols[:limit]
    tasks = [(i, cid, smi) for i, (cid, smi) in enumerate(mols, 1)]
    n = len(tasks)
    print(
        f"set={set_label}  input={os.path.basename(in_path)}  n={n}  -> {os.path.basename(out_path)}",
        flush=True,
    )

    t0 = time.time()
    print("phase 1: embedding (parallel)...", flush=True)
    with mp.Pool(N_WORKERS) as pool:
        gens = pool.map(gen_conf, tasks)
    n_emb = sum(g["status"] == "embedded" for g in gens)
    print(f"phase 1 done: {n_emb}/{n} embedded ({(time.time()-t0)/60:.1f} min)", flush=True)

    print("phase 2: dock Tier2 + Tier1 rescore (sequential GPU)...", flush=True)
    t1 = time.time()
    counts = {"ok": 0, "fail": 0}
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "idx",
                "id",
                "set",
                "smiles",
                "status",
                "n_poses",
                "vina_t2",
                "cnnsc_t2",
                "cnnaff_t2",
                "vina_t1",
                "cnnaff_t1",
                "ddb1_dvina",
                "ddb1_dcnnaff",
            ]
        )
        for k, g in enumerate(gens, 1):
            if g["status"] != "embedded":
                counts["fail"] += 1
                w.writerow([g["idx"], g["cid"], set_label, g["smi"], g["status"], *[""] * 8])
            else:
                try:
                    r = dock_one(g["sdf"])
                    if r is None:
                        counts["fail"] += 1
                        w.writerow([g["idx"], g["cid"], set_label, g["smi"], "no_pose", *[""] * 8])
                    else:
                        counts["ok"] += 1
                        dv = r["vina_t2"] - r["vina_t1"]
                        dc = r["cnnaff_t2"] - r["cnnaff_t1"]
                        w.writerow(
                            [
                                g["idx"],
                                g["cid"],
                                set_label,
                                g["smi"],
                                "ok",
                                r["n_poses"],
                                f"{r['vina_t2']:.3f}",
                                f"{r['cnnsc_t2']:.4f}",
                                f"{r['cnnaff_t2']:.4f}",
                                f"{r['vina_t1']:.3f}",
                                f"{r['cnnaff_t1']:.4f}",
                                f"{dv:.3f}",
                                f"{dc:.4f}",
                            ]
                        )
                    os.remove(g["sdf"])
                except Exception as e:
                    counts["fail"] += 1
                    w.writerow(
                        [
                            g["idx"],
                            g["cid"],
                            set_label,
                            g["smi"],
                            f"dock_fail:{type(e).__name__}",
                            *[""] * 8,
                        ]
                    )
            fh.flush()
            if k % 10 == 0 or k == n:
                rate = (time.time() - t1) / k
                print(
                    f"  {k}/{n}  ok={counts['ok']} fail={counts['fail']}  "
                    f"{rate:.1f}s/mol eta={rate*(n-k)/60:.1f}min",
                    flush=True,
                )
    print(f"DONE ({(time.time()-t0)/60:.1f} min) -> {out_path}  {counts}", flush=True)


if __name__ == "__main__":
    main()
