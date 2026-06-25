"""``ActiveLearningLoop`` — the multi-round loop of ``[bengio2021gflownet]`` Alg. 1.

Verbatim algorithm (see ``Logs/RESEARCH_CONTEXT.md``):

    Init: proxy M; policy pi_theta; oracle O; i = 1
    while i <= N:
        fit M on dataset D_{i-1}
        train pi_theta with reward r(x) = M(x)^beta
        sample query batch B = {x_1..x_b}, x_j ~ pi_theta
        evaluate B with O:  D̂_i = {(x_j, O(x_j))}
        D_i = D̂_i ∪ D_{i-1}
        i = i + 1
    return TopK(D_N)

Faithfulness notes / deliberate divergences (flagged for Balam validation in
``docs/REFACTOR_LOG.md``):
    - "Train pi_theta" = one full ``Trainer.train()`` run against the current
      proxy. Each round re-runs the same ``n_iterations`` range, warm-started from
      the previous round's policy + optimizer state (the policy persists across
      rounds, as intended). The paper uses "fewer iterations when fitting the
      generative model" in the multi-round setting — so set ``Trainer.n_iterations``
      smaller in the AL config than a single-shot run.
    - The proxy and the trainer's reward share the *same* ``LearnedGlueProxy``
      instance (a gin singleton), so refitting ``M`` in place updates the reward
      RGFN trains against. Right after ``fit`` we ``clear_cache()`` so stale
      cached predictions are not served.
    - ``O`` labels enter training only by retraining ``M`` — never as a direct
      RGFN reward. This loop preserves that invariant: it calls the oracle solely
      to grow ``D`` for the next ``fit``.
"""

from pathlib import Path
from typing import List, Optional

import gin

from glue.datasets.oracle_labeled import OracleLabeledDataset
from glue.oracles.base import GlueOracle
from glue.proxies.learned_proxy import LearnedGlueProxy
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal


@gin.configurable()
class ActiveLearningLoop:
    """Outer active-learning loop sequencing proxy fit / GFN train / oracle label."""

    def __init__(
        self,
        trainer,
        proxy: LearnedGlueProxy,
        oracle: GlueOracle,
        dataset: OracleLabeledDataset,
        n_rounds: int = 5,
        query_batch_size: int = 200,
        sample_oversample: float = 4.0,
        top_k: int = 100,
        reset_replay_each_round: bool = True,
        run_dir: Optional[str] = None,
    ):
        """
        Args:
            trainer: a configured RGFN ``Trainer`` (its reward must reference the
                same ``proxy`` instance passed here — wire both to ``%train_proxy``).
            proxy: the refit-able ``LearnedGlueProxy`` (the in-loop reward ``M``).
            oracle: the expensive scorer ``O`` (e.g. ``Docking6TD3Oracle``).
            dataset: the accumulating ``(smiles, label)`` store, optionally seeded
                with ``D_0``.
            n_rounds: number of outer rounds ``N``.
            query_batch_size: number of *unique valid* molecules to label per round
                (paper uses 200 in the molecule domain).
            sample_oversample: sample this multiple of ``query_batch_size``
                trajectories to absorb invalid/duplicate terminals before trimming.
            top_k: size of the final Top-K deliverable.
            reset_replay_each_round: clear the replay buffer before each round so
                its sampling priorities are not stale w.r.t. the refit proxy.
            run_dir: where to write per-round dataset/Top-K CSVs; defaults to the
                trainer's run_dir.
        """
        self.trainer = trainer
        self.proxy = proxy
        self.oracle = oracle
        self.dataset = dataset
        self.n_rounds = n_rounds
        self.query_batch_size = query_batch_size
        self.sample_oversample = sample_oversample
        self.top_k = top_k
        self.reset_replay_each_round = reset_replay_each_round
        self.run_dir = Path(run_dir) if run_dir else Path(trainer.run_dir)

    # --------------------------------------------------------------------- driver
    def run(self) -> List[tuple]:
        """Run all rounds and return ``TopK(D_N)`` as ``[(smiles, label), ...]``."""
        out_dir = self.run_dir / "active_learning"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = self.trainer.logger

        print(f"[AL] start: {self.n_rounds} rounds, seed |D_0|={len(self.dataset)}", flush=True)
        if len(self.dataset) < 2:
            raise ValueError(
                "Active learning needs a seed dataset D_0 (>=2 labelled molecules). "
                "Set OracleLabeledDataset.seed_csv in the config."
            )
        # The proxy is fit on the oracle's labels, so their sign conventions must
        # agree or the GFN would be rewarded for the wrong end of the metric.
        if self.proxy.higher_is_better != self.oracle.higher_is_better:
            raise ValueError(
                "Sign mismatch: proxy.higher_is_better="
                f"{self.proxy.higher_is_better} but oracle.higher_is_better="
                f"{self.oracle.higher_is_better}. Set LearnedGlueProxy.higher_is_better "
                "to match the oracle in the gin config."
            )

        for rnd in range(1, self.n_rounds + 1):
            # 1. fit M on D_{i-1}
            smiles, labels = self.dataset.to_lists()
            fit_metrics = self.proxy.fit(smiles, labels)
            self.proxy.clear_cache()  # predictions changed -> drop stale cache
            print(
                f"[AL] round {rnd}: fit M on |D|={len(self.dataset)} -> {fit_metrics}", flush=True
            )

            # 2. train pi_theta against r(x) = M(x)^beta  (RGFN's own inner loop)
            if self.reset_replay_each_round:
                self._reset_replay_buffer()
            self.trainer.train()

            # 3. sample a query batch B ~ pi_theta
            batch = self._sample_query_batch()
            print(f"[AL] round {rnd}: sampled {len(batch)} unique candidates", flush=True)

            # 4. score B with the expensive oracle O
            oracle_scores = self.oracle.score(batch) if batch else []

            # 5. D_i = D̂_i ∪ D_{i-1}
            n_added = self.dataset.add(batch, oracle_scores)
            valid_scores = [s for s in oracle_scores if s is not None and s == s]
            round_metrics = {
                "al_round": rnd,
                "al_dataset_size": len(self.dataset),
                "al_batch_unique": len(batch),
                "al_batch_labelled": n_added,
                "al_batch_oracle_mean": (sum(valid_scores) / len(valid_scores))
                if valid_scores
                else float("nan"),
                "al_batch_oracle_max": max(valid_scores) if valid_scores else float("nan"),
                **fit_metrics,
            }
            logger.log_metrics(metrics=round_metrics, prefix="active_learning")
            self.dataset.save_csv(str(out_dir / f"dataset_round_{rnd:03d}.csv"))
            print(f"[AL] round {rnd}: |D|={len(self.dataset)} (+{n_added})", flush=True)

        top = self.dataset.top_k(self.top_k)
        self._write_top_k(out_dir / "top_k.csv", top)
        print(f"[AL] done. Top-{self.top_k} written ({len(top)} rows).", flush=True)
        return top

    # ----------------------------------------------------------------- internals
    def _sample_query_batch(self) -> List[str]:
        """Sample from the trained forward policy; return unique valid SMILES."""
        sampler = self.trainer.train_forward_sampler
        n_sample = int(self.query_batch_size * self.sample_oversample)
        batch_size = self.trainer.train_batch_size
        seen, batch = set(), []
        for trajectories in sampler.get_trajectories_iterator(n_sample, batch_size):
            for state in trajectories.get_last_states_flat():
                if not isinstance(state, ReactionStateTerminal):
                    continue  # skip early-terminal / invalid molecules
                smi = state.molecule.smiles
                if smi in seen:
                    continue
                seen.add(smi)
                batch.append(smi)
                if len(batch) >= self.query_batch_size:
                    return batch
        return batch

    def _reset_replay_buffer(self) -> None:
        """Best-effort clear of the replay buffer between rounds (priorities go
        stale once M is refit). Guarded against upstream layout changes."""
        rb = getattr(self.trainer, "train_replay_buffer", None)
        if rb is None:
            return
        if hasattr(rb, "states_list") and hasattr(rb, "states_set"):
            rb.states_list = []
            rb.states_set = set()
            if hasattr(rb, "proxy_value_array"):
                rb.proxy_value_array[:] = 0.0

    @staticmethod
    def _write_top_k(path: Path, rows: List[tuple]) -> None:
        import csv

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["rank", "smiles", "oracle_label"])
            for rank, (smi, label) in enumerate(rows, start=1):
                writer.writerow([rank, smi, label])
