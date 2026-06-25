#!/usr/bin/env python
"""Entry point for the RGFN active-learning loop (``[bengio2021gflownet]`` Alg. 1).

Like ``scripts/train.py``, this imports ``glue`` first so gin can resolve our
oracles / proxies / dataset / loop, then parses an ``configs/glue/`` config that
builds an ``ActiveLearningLoop`` and runs it. Unlike ``train.py`` (one inner GFN
run), this drives the *outer* multi-round loop.

    python scripts/active_learning.py --cfg configs/glue/active_learning_mock.gin
    python scripts/active_learning.py --cfg configs/glue/active_learning_6td3.gin --seed 42

The config must define an ``ActiveLearningLoop`` singleton wired to the same
proxy instance the trainer's reward uses (both -> ``%train_proxy``). See
``configs/glue/active_learning_mock.gin`` for the canonical wiring.
"""

import argparse
from pathlib import Path

import gin

import glue  # noqa: F401  (side effect: registers our gin components)
from gin_config import get_time_stamp
from glue.active_learning import ActiveLearningLoop
from rgfn.trainer.trainer import (  # noqa: F401  (registers @Trainer for gin, as upstream train.py does)
    Trainer,
)
from rgfn.utils.helpers import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    config_name = Path(args.cfg).stem
    run_name = f"{config_name}/{get_time_stamp()}"
    gin.parse_config_files_and_bindings([args.cfg], bindings=[f'run_name="{run_name}"'])

    loop = ActiveLearningLoop()
    loop.trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
    loop.trainer.logger.log_to_file(gin.config_str(), "config")
    loop.run()
    loop.trainer.close()
