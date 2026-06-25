"""6TD3 / CR8 two-tier docking oracle ``O`` for the active-learning loop.

Scores a molecule by the **neosubstrate differential** (the DDB1 cooperativity
bonus) on the 6TD3 system — our validated discrimination metric (see
``Logs/RESEARCH_CONTEXT.md`` and ``Logs/002_...``). For one molecule:

    1. embed + MMFF-optimise a 3D conformer (RDKit);
    2. dock into Tier 2 (CDK12 + DDB1) with gnina, autoboxed on the crystal CR8;
    3. keep the most native-like pose (highest CNNscore — pose selection only);
    4. ``--score_only`` that same pose against Tier 1 (CDK12 alone);
    5. differential = ``Vina(Tier2) - Vina(Tier1)`` (= ``ddb1_dvina``) — *more
       negative* means the arm gains binding once the recruited partner (DDB1) is
       present. This is the **validated** discrimination metric (Log 002: known
       median -2.20 vs decoy -0.60; 85.6% vs 7.3% of molecules below -1.5, the
       +78pt gap). Vina is a binding energy, so **lower is better** ->
       ``higher_is_better = False``. (We use the Vina differential, NOT the
       CNNaffinity differential ``ddb1_dcnnaff``, which does not discriminate;
       CNNscore is used only to pick the pose.)

Provenance / faithfulness:
    This mirrors the **validated** batch pipeline in
    ``research/preprocessing/docking_6td3/dock_cluster.py`` — identical gnina
    flags (``--autobox_ligand`` crystal, ``--autobox_add 4``, exhaustiveness,
    num_modes), identical "best pose = max CNNscore", identical Tier-1
    ``--score_only`` rescoring, and the same ``ddb1_dvina`` differential. The
    difference is orchestration only: this runs one in-process batch (the AL loop
    queries ~hundreds of molecules per round) instead of ``dock_cluster.py``'s
    multi-GPU sharded job. **The two implementations duplicate the docking logic;
    they must be reconciled and cross-validated against each other on Balam**
    (see ``docs/REFACTOR_LOG.md``). ``dock_cluster.py`` is the source of truth.

Environment: gnina, a GPU, and the prepared receptors
(``6TD3_tier{1,2}.pdbqt``, ``crystal_RC8.pdb``) are **Balam-only** — the
``.pdbqt`` files are git-ignored and absent on a laptop. Import is therefore
side-effect-free; all docking I/O is deferred to :meth:`score`, which validates
the toolchain and inputs at call time.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import gin
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from glue.oracles.base import GlueOracle

RDLogger.DisableLog("rdApp.*")

# Default inputs live alongside dock_cluster.py (git-ignored .pdbqt; present on Balam).
_DOCK_DIR = Path(__file__).resolve().parents[2] / "research" / "preprocessing" / "docking_6td3"


def _largest_frag(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda f: f.GetNumHeavyAtoms()) if len(frags) > 1 else mol


@gin.configurable()
class Docking6TD3Oracle(GlueOracle):
    """Expensive 6TD3 docking oracle returning the DDB1 neosubstrate differential."""

    name = "docking_6td3"
    # ddb1_dvina is a Vina binding-energy differential: MORE NEGATIVE = stronger
    # DDB1 cooperativity = better glue. So lower is better.
    higher_is_better = False

    def __init__(
        self,
        tier2_path: Optional[str] = None,
        tier1_path: Optional[str] = None,
        crystal_path: Optional[str] = None,
        gnina: Optional[str] = None,
        exhaustiveness: int = 16,
        num_modes: int = 9,
        autobox_add: float = 4.0,
        n_cpu: int = 8,
        seed: int = 42,
        work_dir: Optional[str] = None,
    ):
        """
        Args mirror ``dock_cluster.py`` defaults so the two stay comparable.

        Args:
            tier2_path / tier1_path / crystal_path: receptor and reference paths;
                default to ``research/preprocessing/docking_6td3/`` (Balam).
            gnina: gnina launcher; defaults to ``$GNINA`` or dock_cluster.py's path.
            exhaustiveness / num_modes / autobox_add / seed: gnina docking params
                (identical defaults to dock_cluster.py).
            n_cpu: CPUs passed to gnina per call.
            work_dir: scratch dir for intermediate SDFs; a temp dir if None.
        """
        self.tier2_path = Path(tier2_path) if tier2_path else _DOCK_DIR / "6TD3_tier2.pdbqt"
        self.tier1_path = Path(tier1_path) if tier1_path else _DOCK_DIR / "6TD3_tier1.pdbqt"
        self.crystal_path = Path(crystal_path) if crystal_path else _DOCK_DIR / "crystal_RC8.pdb"
        self.gnina = gnina or os.environ.get("GNINA", "/scratch/markymoo/gnina/run_gnina.sh")
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.autobox_add = autobox_add
        self.n_cpu = n_cpu
        self.seed = seed
        self.work_dir = work_dir

    # ----------------------------------------------------------------- public API
    def score(self, smiles: List[str]) -> List[float]:
        """Dock a batch and return the DDB1 differential per molecule (nan on failure)."""
        self._check_inputs()
        results = {i: float("nan") for i in range(len(smiles))}
        work = Path(self.work_dir) if self.work_dir else Path(tempfile.mkdtemp(prefix="dock6td3_"))
        work.mkdir(parents=True, exist_ok=True)

        # phase 1: embed (idx -> molblock); idx is the SMILES position in the batch.
        blocks = {}
        for i, smi in enumerate(smiles):
            blk = self._embed(i, smi)
            if blk is not None:
                blocks[i] = blk
        if not blocks:
            return [results[i] for i in range(len(smiles))]

        # write one SDF for the whole batch (titles = idx), matching dock_cluster.py
        batch_sdf = work / "batch.sdf"
        with open(batch_sdf, "w") as fh:
            fh.write("".join(blocks[i] + "$$$$\n" for i in sorted(blocks)))

        # phase 2: dock into Tier 2, then score best poses against Tier 1
        docked = work / "batch_docked.sdf"
        self._run_gnina(
            [
                "-r",
                str(self.tier2_path),
                "-l",
                str(batch_sdf),
                "--autobox_ligand",
                str(self.crystal_path),
                "--autobox_add",
                str(self.autobox_add),
                "--exhaustiveness",
                str(self.exhaustiveness),
                "--num_modes",
                str(self.num_modes),
                "--cpu",
                str(self.n_cpu),
                "--seed",
                str(self.seed),
                "-o",
                str(docked),
            ]
        )

        poses = self._poses(docked)  # idx(str) -> [pose dicts]
        # Pose selection is by CNNscore (most native-like), exactly as
        # dock_cluster.py; the *scored differential* is Vina, not CNNaffinity.
        order, best_mols, vina_t2 = [], [], {}
        for i in sorted(blocks):
            ps = poses.get(str(i))
            if not ps:
                continue
            best = max(ps, key=lambda p: p["cnnsc"])  # most native-like pose
            order.append(i)
            best_mols.append(best["mol"])
            vina_t2[i] = best["vina"]  # Tier-2 Vina affinity of that pose

        if best_mols:
            best_sdf = work / "batch_best.sdf"
            writer = Chem.SDWriter(str(best_sdf))
            for m in best_mols:
                writer.write(m)
            writer.close()
            t1 = self._score_only(self.tier1_path, best_sdf)  # ordered [(Affinity, CNNaffinity)]
            for k, i in enumerate(order):
                if k < len(t1):
                    vina_t1 = t1[k][0]  # Tier-1 Vina affinity of the same pose
                    # ddb1_dvina = Vina(Tier2) - Vina(Tier1); more negative = better
                    # glue (validated discrimination metric, Log 002).
                    results[i] = float(vina_t2[i] - vina_t1)

        return [results[i] for i in range(len(smiles))]

    # ------------------------------------------------------------- gnina plumbing
    def _check_inputs(self) -> None:
        missing = [
            str(p) for p in (self.tier2_path, self.tier1_path, self.crystal_path) if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Docking6TD3Oracle is missing receptor/crystal inputs (expected on Balam): "
                + ", ".join(missing)
            )

    def _embed(self, idx: int, smi: str) -> Optional[str]:
        """3D embed + MMFF optimise; mirrors dock_cluster.embed (title = idx)."""
        try:
            mol = Chem.MolFromSmiles(smi) if smi is not None else None
            if mol is None:
                return None
            mol = Chem.AddHs(_largest_frag(mol))
            if AllChem.EmbedMolecule(mol, randomSeed=self.seed) != 0:
                AllChem.EmbedMolecule(mol, randomSeed=self.seed, useRandomCoords=True)
            AllChem.MMFFOptimizeMolecule(mol)
            mol.SetProp("_Name", str(idx))
            return Chem.MolToMolBlock(mol)
        except Exception:
            return None

    def _run_gnina(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.gnina, *args], capture_output=True, text=True, check=True, env=dict(os.environ)
        )

    @staticmethod
    def _poses(sdf: Path) -> dict:
        """idx(str) -> list of pose dicts; mirrors dock_cluster._poses."""
        groups: dict = {}
        for m in Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False):
            if m is None:
                continue
            idx = m.GetProp("_Name") if m.HasProp("_Name") else None
            g = lambda k: float(m.GetProp(k)) if m.HasProp(k) else float("nan")
            groups.setdefault(idx, []).append(
                {
                    "mol": m,
                    "vina": g("minimizedAffinity"),
                    "cnnsc": g("CNNscore"),
                    "cnnaff": g("CNNaffinity"),
                }
            )
        return groups

    def _score_only(self, receptor: Path, sdf: Path) -> List[tuple]:
        """Ordered [(Affinity, CNNaffinity)]; mirrors dock_cluster._score_only_stream."""
        r = self._run_gnina(["-r", str(receptor), "-l", str(sdf), "--score_only"])
        affs, caffs = [], []
        for line in r.stdout.splitlines():
            if line.startswith("Affinity:"):
                affs.append(float(line.split()[1]))
            elif line.startswith("CNNaffinity:"):
                caffs.append(float(line.split()[1]))
        return list(zip(affs, caffs))
