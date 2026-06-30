"""``RxnFlowActiveLearningLoop`` — the RxnFlow counterpart of
``glue/active_learning/loop.py`` (``[bengio2021gflownet]`` Alg. 1).

Same algorithm as the RGFN and FragGFN loops, step for step::

    Init: proxy M; policy π_θ (RxnFlow synthesis-GFN); oracle O; i = 1
    while i <= N:
        fit M on dataset D_{i-1}
        train π_θ with reward r(x) = M(x)^β            # RxnFlow, not RGFN
        sample query batch B = {x_1..x_b}, x_j ~ π_θ   # + each molecule's route
        evaluate B with O:  D̂_i = {(x_j, O(x_j))}      # via the oracle bridge
        D_i = D̂_i ∪ D_{i-1}
        i = i + 1
    return TopK(D_N)

Two deliberate differences from the RGFN loop, both forced by the env split (see
``validation/generators/rxnflow/README.md``):

  * **Generator.** π_θ is RxnFlow's synthesis GFlowNet (reaction templates over a
    building-block library, via :class:`RxnFlowGlueTrainer`), trained against the
    *same* learned proxy ``M`` as RGFN. Unlike FragGFN, RxnFlow molecules **carry a
    synthesis route** — we extract it per sample and record it in the standard
    candidate dataset (``has_route=1`` + ``routes.jsonl``), the differentiator vs.
    the non-synthesizable baseline and the apples-to-apples match for RGFN.
  * **Oracle call.** ``O`` lives in the ``rgfn`` env (gnina + QuickVina2-GPU +
    ``glue``), unreachable from this env. We label each batch by shelling out to
    ``scripts/score_batch.py`` under ``rgfn`` — the shared scoring standard — and pass
    the per-round routes JSONL via ``--routes`` so the bridge writes them into the
    standard dataset alongside FragGFN/RGFN output.

The invariant is preserved: oracle labels enter training only by refitting ``M``
(never as a direct GFN reward). ``M`` is held by the task and refit in place each
round; ``num_workers=0`` makes the refit visible to sampling immediately.
"""

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

from validation.generators.rxnflow.proxy import AtomMPNNProxy
from validation.generators.rxnflow.task import RxnFlowGlueTrainer


def _canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


class LabelStore:
    """Accumulating ``canonical-SMILES -> label`` store (the dataset ``D``).

    Mirrors ``glue.datasets.OracleLabeledDataset`` (which can't be imported here: it
    lives behind ``glue/__init__`` → ``rgfn``). Lower label = better glue, so Top-K
    sorts ascending (the 6TD3 ``dvina`` differential, like the RGFN/FragGFN loops).
    Identical to FragGFN's ``LabelStore``; duplicated (≈45 lines) to avoid importing
    FragGFN's loop module (which would pull in the fragment env context).
    """

    def __init__(self, lower_is_better: bool = True):
        self._data: Dict[str, float] = {}
        self.lower_is_better = lower_is_better

    def __len__(self) -> int:
        return len(self._data)

    def load_seed(self, csv_path: str, smiles_col: str = "smiles", label_col: str = "label") -> int:
        added = 0
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    y = float(row[label_col])
                except (TypeError, ValueError, KeyError):
                    continue
                if self.add_one(row.get(smiles_col, ""), y):
                    added += 1
        return added

    def add_one(self, smiles: str, label: float) -> bool:
        canon = _canonical(smiles)
        if canon is None or label is None or label != label:  # invalid / NaN
            return False
        self._data[canon] = float(label)
        return True

    def add(self, smiles: List[str], labels: List[float]) -> int:
        return sum(int(self.add_one(s, y)) for s, y in zip(smiles, labels))

    def to_lists(self) -> Tuple[List[str], List[float]]:
        items = list(self._data.items())
        return [s for s, _ in items], [y for _, y in items]

    def top_k(self, k: int) -> List[Tuple[str, float]]:
        return sorted(self._data.items(), key=lambda kv: kv[1], reverse=not self.lower_is_better)[
            :k
        ]

    def save_csv(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["smiles", "label"])
            for s, y in self._data.items():
                w.writerow([s, y])


def extract_route(traj: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Best-effort synthesis route from one RxnFlow trajectory, in the standard schema
    (``glue.active_learning.route`` / ``docs/CANDIDATE_DATASET_FORMAT.md``)::

        {"product_smiles": ..., "num_reactions": N,
         "building_block": {"smiles": ..., "idx": ...},
         "steps": [{"step": 1, "reaction_idx": ..., "reaction_smarts": ...,
                    "reactant": ..., "fragments": [...], "product": ...}, ...]}

    RxnFlow's per-action attribute names are not fully documented, so this reads them
    defensively: it tries the common fields and degrades to a shorter/partial route
    rather than crashing the loop. **Flagged for Balam validation** — confirm the
    action/state attributes against the cloned repo and tighten the field mapping so
    routes are chemically faithful (``docs/REFACTOR_LOG.md``).

    Args:
        traj: one trajectory dict from ``algo.create_training_data_from_own_samples``;
            expected to hold the final object under ``"result"`` and the (state, action)
            sequence under ``"traj"``.
        ctx: the trainer env context (for turning states/actions into SMILES).
    """

    def _smiles_of(obj: Any) -> Optional[str]:
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        for attr in ("smiles", "smi"):
            val = getattr(obj, attr, None)
            if isinstance(val, str):
                return val
        try:  # obj may be an RDKit Mol or a ctx graph
            return Chem.MolToSmiles(obj if isinstance(obj, Chem.Mol) else ctx.graph_to_obj(obj))
        except Exception:
            return None

    building_block: Optional[Dict[str, Any]] = None
    steps: List[Dict[str, Any]] = []
    seq = traj.get("traj") if isinstance(traj, dict) else None
    if seq:
        for state, action in seq:
            if action is None:
                continue
            # First building-block pick -> the route seed; subsequent actions that
            # carry a reaction/template -> committed reaction steps.
            block = getattr(action, "block", None) or getattr(action, "building_block", None)
            rxn = (
                getattr(action, "reaction", None)
                or getattr(action, "template", None)
                or getattr(action, "reaction_smarts", None)
            )
            if building_block is None and block is not None and rxn is None:
                building_block = {
                    "smiles": _smiles_of(block),
                    "idx": getattr(action, "block_idx", None) or getattr(action, "idx", None),
                }
                continue
            if rxn is not None:
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "reaction_idx": getattr(action, "reaction_idx", None)
                        or getattr(action, "template_idx", None),
                        "reaction_smarts": rxn
                        if isinstance(rxn, str)
                        else getattr(rxn, "smarts", None),
                        "reactant": _smiles_of(state),
                        "fragments": [s for s in [_smiles_of(block)] if s],
                        "product": _smiles_of(getattr(action, "product", None)),
                    }
                )

    product_smiles = _smiles_of(traj.get("result")) if isinstance(traj, dict) else None
    return {
        "product_smiles": product_smiles,
        "num_reactions": len(steps),
        "building_block": building_block,
        "steps": steps,
    }


class RxnFlowActiveLearningLoop:
    """Outer active-learning loop driving the RxnFlow synthesis generator."""

    def __init__(
        self,
        trainer: RxnFlowGlueTrainer,
        proxy: AtomMPNNProxy,
        dataset: LabelStore,
        bridge_cmd: List[str],
        run_dir: str,
        n_rounds: int = 3,
        n_train_steps: int = 300,
        query_batch_size: int = 32,
        sample_oversample: float = 4.0,
        top_k: int = 16,
        seed_csv: Optional[str] = None,
        system: Optional[str] = None,
        seed: int = 42,
        oracle_threshold: float = -2.0,
        oracle_higher_is_better: bool = False,
    ):
        """
        Args mirror ``FragGFNActiveLearningLoop``; see that class for the budget knobs.
        The only behavioural addition is route extraction + ``--routes`` to the bridge,
        because RxnFlow molecules are synthesizable.
        """
        self.trainer = trainer
        self.proxy = proxy
        self.dataset = dataset
        self.bridge_cmd = bridge_cmd
        self.run_dir = Path(run_dir)
        self.n_rounds = n_rounds
        self.n_train_steps = n_train_steps
        self.query_batch_size = query_batch_size
        self.sample_oversample = sample_oversample
        self.top_k = top_k
        self.seed_csv = seed_csv
        self.system = system
        self.seed = seed
        self.oracle_threshold = oracle_threshold
        self.oracle_higher_is_better = oracle_higher_is_better
        self._it = 0  # persistent GFN step counter across rounds (for lr schedules)

        if proxy.higher_is_better != oracle_higher_is_better:
            raise ValueError(
                f"Sign mismatch: proxy.higher_is_better={proxy.higher_is_better} but "
                f"oracle_higher_is_better={oracle_higher_is_better}."
            )

    # --------------------------------------------------------------------- driver
    def run(self) -> List[Tuple[str, float]]:
        out_dir = self.run_dir / "active_learning"
        sug_dir = out_dir / "suggestions"
        out_dir.mkdir(parents=True, exist_ok=True)
        sug_dir.mkdir(parents=True, exist_ok=True)

        if len(self.dataset) < 2:
            raise ValueError("RxnFlow AL needs a seed D_0 (>=2 labelled molecules).")
        print(f"[RXN-AL] start: {self.n_rounds} rounds, seed |D_0|={len(self.dataset)}", flush=True)

        for rnd in range(1, self.n_rounds + 1):
            t0 = time.time()
            # 1. fit M on D_{i-1}
            smiles, labels = self.dataset.to_lists()
            fit_metrics = self.proxy.fit(smiles, labels)
            print(
                f"[RXN-AL] round {rnd}: fit M on |D|={len(self.dataset)} -> {fit_metrics}",
                flush=True,
            )

            # 2. train π_θ against r(x) = M(x)^β  (RxnFlow synthesis inner loop)
            t_fit = time.time()
            self._train_steps(self.n_train_steps)
            t_train = time.time()

            # 3. sample a query batch B ~ π_θ, keeping each molecule's synthesis route
            batch, routes = self._sample_query_batch()
            print(f"[RXN-AL] round {rnd}: sampled {len(batch)} unique candidates", flush=True)
            t_sample = time.time()

            # 4. label B with O via the oracle bridge (also writes standard format +
            #    stores this round's routes via --routes)
            scores = self._score_via_bridge(batch, routes, rnd, sug_dir)
            t_score = time.time()

            # 5. D_i = D̂_i ∪ D_{i-1}
            n_added = self.dataset.add(batch, scores)
            self.dataset.save_csv(str(out_dir / f"dataset_round_{rnd:03d}.csv"))
            valid = [s for s in scores if s is not None and s == s]
            best = (
                (min(valid) if not self.oracle_higher_is_better else max(valid))
                if valid
                else float("nan")
            )
            print(
                f"[RXN-AL] round {rnd}: |D|={len(self.dataset)} (+{n_added}); "
                f"oracle mean={sum(valid)/len(valid):.3f} best={best:.3f} "
                f"| t: fit={t_fit-t0:.0f}s train={t_train-t_fit:.0f}s "
                f"sample={t_sample-t_train:.0f}s score={t_score-t_sample:.0f}s"
                if valid
                else f"[RXN-AL] round {rnd}: |D|={len(self.dataset)} (+{n_added}); no valid scores",
                flush=True,
            )

            # Fail loudly if the oracle produced no usable label for the *entire* batch
            # (all NaN) — mirrors the post-014 guard in glue/active_learning/loop.py and
            # the FragGFN loop. This is the signature of a wholesale oracle-backend
            # failure (e.g. a wedged GPU/OpenCL node returning all no_pose, Logs/014): D
            # can't grow, so the next round would refit M on an unchanged D and burn
            # another RxnFlow training run for nothing. Per-round provenance
            # (suggestions/shards, dataset_round CSV) is already written, so the failure
            # stays fully inspectable.
            if batch and not valid:
                raise RuntimeError(
                    f"RxnFlow AL round {rnd}: the oracle returned no usable score for any of "
                    f"the {len(batch)} sampled molecules (all NaN). The dataset cannot grow, so "
                    f"continuing would retrain M on an unchanged D. This usually means the oracle "
                    f"backend failed wholesale (e.g. a wedged GPU/OpenCL node — see Logs/014). "
                    f"Aborting. Inspect {sug_dir} and resubmit on a healthy node."
                )

        # Assemble the canonical standard candidate dataset from per-round shards
        # (joins the stored routes by SMILES -> has_route=1 + routes.jsonl).
        self._finalize_via_bridge(sug_dir)
        top = self.dataset.top_k(self.top_k)
        self._write_top_k(out_dir / "top_k.csv", top)
        print(f"[RXN-AL] done. Top-{self.top_k} written ({len(top)} rows).", flush=True)
        return top

    # ----------------------------------------------------------------- internals
    def _train_steps(self, n: int) -> None:
        """Drive the RxnFlow trainer for ``n`` minibatches against the current ``M``
        (held by the task). Mirrors the inner loop of the gflownet trainer but under
        our control so we can refit ``M`` between rounds (cf. the FragGFN loop)."""
        from gflownet.trainer import cycle

        tr = self.trainer
        tr.model.to(tr.device)
        tr.sampling_model.to(tr.device)
        train_dl = tr.build_training_data_loader()
        start = self._it + 1
        for it, batch in zip(range(start, start + n), cycle(train_dl)):
            info = tr.train_batch(batch.to(tr.device), 0, 0, it)
            if it % max(1, tr.print_every) == 0:
                loss = info.get("loss", float("nan"))
                print(f"[RXN-AL]   gfn step {it}: loss={loss:.3f}", flush=True)
        self._it += n
        del train_dl

    def _sample_query_batch(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Sample unique valid terminal SMILES from the trained policy, with routes.

        Returns parallel lists ``(smiles, routes)``: each route is the synthesis
        pathway reconstructed from that molecule's trajectory (:func:`extract_route`)."""
        tr = self.trainer
        n_sample = int(self.query_batch_size * self.sample_oversample)
        tr.model.to(tr.device)
        tr.model.eval()
        cond_info = tr.task.sample_conditional_information(n_sample, self._it)
        trajs = tr.algo.create_training_data_from_own_samples(
            tr.model, n_sample, cond_info["encoding"].to(tr.device), random_action_prob=0.0
        )
        seen: set = set()
        batch: List[str] = []
        routes: List[Dict[str, Any]] = []
        for t in trajs:
            if not t.get("is_valid", True):
                continue
            try:
                smi = Chem.MolToSmiles(tr.ctx.graph_to_obj(t["result"]))
            except Exception:
                continue
            canon = _canonical(smi)
            if canon is None or canon in seen:
                continue
            seen.add(canon)
            batch.append(canon)
            try:
                routes.append(extract_route(t, tr.ctx))
            except Exception:  # route is provenance — never let it abort sampling
                routes.append(
                    {
                        "product_smiles": canon,
                        "num_reactions": 0,
                        "building_block": None,
                        "steps": [],
                    }
                )
            if len(batch) >= self.query_batch_size:
                break
        return batch, routes

    def _write_routes_jsonl(
        self, path: Path, batch: List[str], routes: List[Dict[str, Any]]
    ) -> None:
        """Write this round's routes as JSONL keyed by SMILES for the bridge --routes."""
        with open(path, "w") as fh:
            for smi, route in zip(batch, routes):
                fh.write(json.dumps({"smiles": smi, **route}) + "\n")

    def _score_via_bridge(
        self, batch: List[str], routes: List[Dict[str, Any]], rnd: int, sug_dir: Path
    ) -> List[float]:
        """Write the batch SMILES + routes, run the oracle bridge under ``rgfn``, read
        labels. The bridge appends the standard-format shard + batch_metrics row and
        stores the routes JSONL (``--routes``) for this round."""
        if not batch:
            return []
        smi_path = sug_dir / f"round_{rnd:03d}_batch.smi"
        lbl_path = sug_dir / f"round_{rnd:03d}_labels.csv"
        routes_path = sug_dir / f"round_{rnd:03d}_routes.jsonl"
        smi_path.write_text("\n".join(batch) + "\n")
        self._write_routes_jsonl(routes_path, batch, routes)
        cmd = list(self.bridge_cmd) + [
            "--in",
            str(smi_path),
            "--out",
            str(lbl_path),
            "--routes",
            str(routes_path),
            "--suggestions-dir",
            str(sug_dir),
            "--step",
            str(rnd),
            "--generator",
            "rxnflow",
            "--seed",
            str(self.seed),
            "--threshold",
            str(self.oracle_threshold),
        ]
        if self.system:
            cmd += ["--system", self.system]
        if self.seed_csv:
            cmd += ["--reference-csv", str(self.seed_csv)]
        print(f"[RXN-AL] round {rnd}: bridge -> {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        return self._read_labels(lbl_path, batch)

    def _finalize_via_bridge(self, sug_dir: Path) -> None:
        cmd = list(self.bridge_cmd) + [
            "--finalize",
            "--suggestions-dir",
            str(sug_dir),
            "--generator",
            "rxnflow",
            "--seed",
            str(self.seed),
        ]
        if self.system:
            cmd += ["--system", self.system]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:  # finalize is provenance, not training
            print(f"[RXN-AL] WARNING finalize failed: {e}", flush=True)

    @staticmethod
    def _read_labels(path: Path, batch: List[str]) -> List[float]:
        """Read labels keyed by SMILES, returned in ``batch`` order (NaN if missing)."""
        by_smi: Dict[str, float] = {}
        if path.exists():
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        by_smi[row["smiles"]] = float(row["label"])
                    except (KeyError, TypeError, ValueError):
                        by_smi[row.get("smiles", "")] = float("nan")
        return [by_smi.get(s, float("nan")) for s in batch]

    @staticmethod
    def _write_top_k(path: Path, rows: List[Tuple[str, float]]) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "smiles", "oracle_label"])
            for rank, (smi, label) in enumerate(rows, start=1):
                w.writerow([rank, smi, label])
