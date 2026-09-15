"""Persist / restore the RNG streams across a SCENT checkpoint resume.

Why this exists
---------------
`library_io` made a requeued SCENT run **correct** — the promoted vocabulary and its embedding rows
come back exactly. It did not make one **reproducible**: the random-number state is not in the
checkpoint, so a resumed run draws a different sampling stream from the point of the resume and
promotes a *different set of fragments* than an uninterrupted run of the same seed. Measured on the
8-iteration resume harness, both runs reaching 15 promoted fragments with full recipe coverage:

    uninterrupted   chosen_smiles_sha1 7fc4acef...
    resumed         chosen_smiles_sha1 d10044fc...

Same seed, same config, different library. That is the last reason SCENT's arm B was restricted to a
single walltime, and arm B is 320,000 calls = 5,000 iterations, which v1 proved does **not** fit
three days on the docking targets (its cells requeued two and three times). So without this, twelve
of the thirty-six arm-B cells could not be trained at all without carrying a permanent
"not reproducible" asterisk.

THIS IS A CORRECTNESS FIX, NOT AN ALGORITHM CHANGE. For an uninterrupted run it is a no-op: the
sidecar is written and never read. It only takes effect on a resume, and what it does there is make
the trajectory match the one the run would have had if nobody had interrupted it.

What is captured, and why all four
----------------------------------
`seed_everything` (the clone's `rgfn/utils/helpers.py`) seeds Python's `random`, `numpy`,
`torch` CPU and `torch.cuda` — so all four have to come back, or the stream diverges through
whichever one was missed. Capturing three of four would look like it worked right up until the
generator happened to draw from the fourth.

WHEN IT IS RESTORED IS LOAD-BEARING. It must be restored *after* the model is constructed, not
before: construction draws from the RNG for weight initialisation, and on a resume those draws are
thrown away (the weights are overwritten from the checkpoint) but they have still advanced the
stream. Restoring late overwrites whatever construction consumed, so the position is exact
regardless of how much the build happened to draw.

Sidecar: ``rng_state.pt`` beside ``last_gfn.pt``. Binary rather than JSON because torch's RNG state
is a ByteTensor.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

_FILENAME = "rng_state.pt"


def sidecar_path(checkpoint_dir: Path | str) -> Path:
    return Path(checkpoint_dir) / _FILENAME


def save_rng_state(path: Path | str, iteration: Optional[int] = None) -> bool:
    """Capture all four streams. Best-effort: never fail a training run over provenance.

    ``iteration`` is the training iteration this position belongs to -- i.e. the iteration that is
    ABOUT TO RUN. It is not decoration: see ``restore_rng_state``, which refuses a position captured
    for a different iteration than the one being resumed.
    """
    try:
        import numpy as np
        import torch

        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            # A list, one entry per visible device. Absent rather than empty when there is no CUDA,
            # so a CPU-trained cell does not look like a GPU one whose capture failed.
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "n_cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "iteration": iteration,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        tmp.replace(path)  # atomic: a kill mid-write never leaves a truncated sidecar
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[SCENT-FR] WARNING RNG-state save failed: {exc}", flush=True)
        return False


def restore_rng_state(path: Path | str, expect_iteration: Optional[int] = None) -> bool:
    """Put all four streams back. Returns True only if every stream present was restored.

    Call AFTER the model exists — see the module docstring on why late is correct.

    ``expect_iteration`` is the iteration the trainer is about to resume at. A sidecar captured for
    any OTHER iteration is REFUSED rather than applied, because applying it silently reinstates a
    position from the wrong point in the run -- which is exactly the bug this argument exists to
    close (measured 2026-09-12: a position captured inside ``make_checkpoint`` restored as
    a92cb12ed2f14 where the uninterrupted run was at cdf509c5082b5 on the same iteration, and the
    resumed run promoted a different fragment set while every log line looked healthy). A declined
    restore leaves the run CORRECT but not bit-reproducible, and says so; a wrong one leaves it
    looking reproducible and silently is not.
    """
    path = Path(path)
    if not path.is_file():
        return False
    try:
        import numpy as np
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)

        saved_iter = state.get("iteration")
        if expect_iteration is not None and saved_iter != expect_iteration:
            print(
                f"[SCENT-FR] WARNING RNG sidecar was captured for iteration {saved_iter!r} but this "
                f"resume re-enters at iteration {expect_iteration!r}; NOT restoring. The run is "
                f"correct but not bit-reproducible against an uninterrupted run.",
                flush=True,
            )
            return False
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])

        cuda = state.get("torch_cuda")
        if cuda is not None and torch.cuda.is_available():
            # A device-count mismatch means this checkpoint came off a different machine shape.
            # Restoring a truncated list would silently leave later devices on a fresh stream, so
            # say so rather than half-restore.
            have, want = torch.cuda.device_count(), len(cuda)
            if have == want:
                torch.cuda.set_rng_state_all(cuda)
            else:
                print(
                    f"[SCENT-FR] WARNING RNG sidecar holds {want} CUDA device states but this host "
                    f"has {have}; CUDA streams NOT restored — this resume is not bit-reproducible",
                    flush=True,
                )
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[SCENT-FR] WARNING RNG-state restore failed: {exc}", flush=True)
        return False


def fingerprint() -> Optional[str]:
    """A short digest of the CURRENT RNG position, for the resume verification.

    Draws nothing itself — reads the state — so calling it cannot perturb the stream it measures.
    """
    try:
        import hashlib

        import numpy as np
        import torch

        h = hashlib.sha1()
        h.update(repr(random.getstate()).encode())
        h.update(repr(np.random.get_state()).encode())
        h.update(torch.get_rng_state().numpy().tobytes())
        if torch.cuda.is_available():
            for s in torch.cuda.get_rng_state_all():
                h.update(s.numpy().tobytes())
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None
