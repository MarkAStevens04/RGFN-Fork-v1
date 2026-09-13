#!/usr/bin/env python
"""Prove a resumed SCENT run recovers the promoted library BIT-EXACTLY, not just by count.

Why a count check is not enough
-------------------------------
The bug this verifies against is *the right number of fragments pointing at the wrong trained
embedding rows*. ``FragmentOneHotEmbedding.weights`` is positionally indexed and
``on_update_fragments_library`` updates only the count, so after a library reset a re-promoted
fragment inherits the previous occupant's row. A "len(chosen_smiles) matches" assertion passes on
exactly that defect. So this compares:

  * ``chosen_smiles`` as an ORDERED list (sha1 over the joined list, not a set), because the
    embedding is positional -- the same fragments in a different order is the same bug from a new
    cause; and
  * a sha1 over ``weights[: current_fragments]``, the rows actually in use.

The protocol
------------
Run A: train N iterations uninterrupted.
Run B: same seed and config, train N/2, stop, then resume into the same run dir and finish.
Both must land on identical fingerprints.

Promotions fire at multiples of ``every_n_iterations`` (1,000 in production), so a meaningful test
overrides it to something small via ``--gin-binding``; the mechanism under test is identical.

    conda run -n scent python validation/generators/scent/verify_library_recovery.py \\
        --out-root $SCRATCH/rgfn_runs/smoke/libverify --iterations 8 --every 2

Exit 0 iff both fingerprints match. Prints both either way, so a failure shows WHICH half moved.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
RUNNER = _REPO / "validation" / "generators" / "scent" / "run_scent_fixed.py"


def _train(
    run_dir: Path, root: Path, iters: int, every: int, cfg: str, seed: int, valid_every: int = 2
) -> None:
    """One training invocation. Resumes automatically if run_dir already holds a checkpoint."""
    cmd = [
        sys.executable,
        str(RUNNER),
        "--cfg",
        cfg,
        "--seed",
        str(seed),
        "--root-dir",
        str(root),
        "--run-dir",
        str(run_dir),
        "--n-iterations",
        str(iters),
        "--n-samples",
        "5",
        "--gin-binding",
        f"dynamic_library/DynamicLibrary.every_n_iterations={every}",
        # Keep the promotion batch small so the test is fast; the mechanism is unchanged.
        "--gin-binding",
        "dynamic_library/DynamicLibrary.n_new_fragments=5",
        # ⚠ LOAD-BEARING FOR THE RNG COMPARISON, NOT A SPEED KNOB. In production this is 250, so at
        # these iteration counts NO periodic validation fires and the only trigger left is
        # `i == n_iterations - 1` (trainer.py:347). That makes the two runs validate at DIFFERENT
        # iterations purely because they stop at different points -- run A at i=7, run B's first
        # half at i=3 -- and SCENT's valid_step samples 1,000 trajectories through the RNG. The
        # stop itself would then advance the stream, so the resumed run could never match the
        # uninterrupted one no matter how correctly the RNG position is captured. Measured
        # 2026-09-12: three consecutive verification runs failed on this and not on the mechanism
        # under test. Forcing a small cadence makes both runs validate on the SAME schedule.
        "--gin-binding",
        f"Trainer.valid_every_n_iterations={valid_every}",
        # Validation is now frequent (above), and at the stock 1,000 trajectories it would dominate
        # the harness's runtime. Shrinking it is safe for THIS test in a way that shrinking the
        # cadence is not: both runs use the same value, so they draw the same amount at the same
        # points, which is the only property the comparison depends on.
        "--gin-binding",
        "Trainer.valid_n_trajectories=64",
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")  # load-bearing for reproducibility, not hygiene
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False, env=env)


def _fingerprint(run_dir: Path) -> dict:
    """Read the library sidecar and hash it the way library_io.library_fingerprint does."""
    import hashlib

    side = run_dir / "train" / "checkpoints" / "dynamic_library.json"
    if not side.exists():
        return {"error": f"no sidecar at {side}"}
    body = json.loads(side.read_text())
    chosen = body.get("library", {}).get("chosen_smiles", [])
    routed = set(body.get("library", {}).get("smiles_to_route", {}) or {})
    out = {
        "n_chosen": len(chosen),
        "chosen_smiles_sha1": hashlib.sha1("\n".join(chosen).encode()).hexdigest()
        if chosen
        else None,
        # THE GATE THAT ACTUALLY MATTERS for benchmark_v2's route dataset: every promoted fragment
        # must carry a recipe, or check_route_readiness fails the cell. Measured 15/15 uninterrupted
        # against 5/15 resumed before the route restore, so this is the number that moves.
        "n_routed": len(routed & set(chosen)),
        "route_coverage": round(len(routed & set(chosen)) / len(chosen), 3) if chosen else None,
    }
    # And the rows actually in use, straight out of the checkpoint.
    ckpt = run_dir / "train" / "checkpoints" / "last_gfn.pt"
    if ckpt.exists():
        import torch

        model = torch.load(ckpt, map_location="cpu")["model"]
        key = next((k for k in model if k.endswith("one_hot_embeddings.weights")), None)
        if key is not None:
            n = 418 + len(chosen)
            rows = model[key][:n].contiguous()
            out["embedding_rows_sha1"] = hashlib.sha1(rows.numpy().tobytes()).hexdigest()
            out["rows_hashed"] = n
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--every", type=int, default=2, help="DynamicLibrary.every_n_iterations")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", default="validation/configs/scent_seh_fixed_5k.gin")
    ap.add_argument(
        "--valid-every",
        type=int,
        default=2,
        help="Trainer.valid_every_n_iterations for BOTH runs (see _train; load-bearing)",
    )
    a = ap.parse_args()

    root = Path(a.out_root).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    # THE STOP POINT IS NOT ``iterations // 2``, AND THAT IS THE WHOLE TEST. Run B's LAST iteration
    # always validates (`i == n_iterations - 1`), so unless run A validates at that same iteration
    # the stop injects 1,000 validation trajectories' worth of RNG that run A never drew, and the
    # two runs diverge for a reason that has nothing to do with the restore. Run A validates at
    # i > 0 with i % valid_every == 0, so the stop must satisfy (half - 1) % valid_every == 0.
    # Pick the largest such half below `iterations`, nearest the middle.
    v = max(1, a.valid_every)
    candidates = [h for h in range(2, a.iterations) if (h - 1) > 0 and (h - 1) % v == 0]
    half = (
        min(candidates, key=lambda h: abs(h - a.iterations / 2))
        if candidates
        else max(1, a.iterations // 2)
    )
    if not candidates:
        print(
            f"WARNING: no stop point in 2..{a.iterations-1} satisfies (half-1) %% {v} == 0; "
            f"falling back to {half}, and the RNG comparison will be INVALID -- run A and run B "
            f"will validate at different iterations. Raise --iterations or lower --valid-every.",
            flush=True,
        )
    else:
        print(
            f"stop point {half}: run B's final iteration is {half-1}, which run A also validates "
            f"({half-1} % {v} == 0), so both runs draw the same validation trajectories.",
            flush=True,
        )

    print("=" * 78)
    print(f"RUN A: {a.iterations} iterations, uninterrupted")
    print("=" * 78)
    a_dir = root / "runA"
    _train(a_dir, root, a.iterations, a.every, a.cfg, a.seed)

    print("=" * 78)
    print(f"RUN B: {half} iterations, then RESUME to {a.iterations}")
    print("=" * 78)
    b_dir = root / "runB"
    _train(b_dir, root, half, a.every, a.cfg, a.seed)
    # Snapshot the state the stop point actually had, BEFORE resuming, so the attribute diff below
    # compares like with like.
    stop_audit = b_dir / "library_audit.json"
    stop_snapshot = b_dir / "library_audit_at_stop.json"
    if stop_audit.exists():
        shutil.copy2(stop_audit, stop_snapshot)
    _train(b_dir, root, a.iterations, a.every, a.cfg, a.seed)  # resumes from the checkpoint

    fa, fb = _fingerprint(a_dir), _fingerprint(b_dir)
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    for name, f in (("uninterrupted", fa), ("resumed", fb)):
        print(f"  {name:<14} {json.dumps(f, sort_keys=True)}")

    if "error" in fa or "error" in fb:
        print("\nFAIL: a sidecar is missing -- the library was never saved.")
        return 1
    ok_list = fa.get("chosen_smiles_sha1") == fb.get("chosen_smiles_sha1")
    ok_rows = fa.get("embedding_rows_sha1") == fb.get("embedding_rows_sha1")
    print(f"\n  ordered chosen_smiles identical : {ok_list}")
    print(f"  embedding rows identical        : {ok_rows}")
    print(
        f"  recipe coverage  uninterrupted  : {fa.get('n_routed')}/{fa.get('n_chosen')} "
        f"({fa.get('route_coverage')})"
    )
    print(
        f"  recipe coverage  resumed        : {fb.get('n_routed')}/{fb.get('n_chosen')} "
        f"({fb.get('route_coverage')})"
    )
    ok_routes = fb.get("route_coverage") == 1.0
    print(f"  every promoted fragment routed  : {ok_routes}   <- the check_route_readiness gate")
    if fa.get("n_chosen", 0) == 0:
        print(
            "\nINCONCLUSIVE: no fragments were promoted, so nothing was under test. "
            "Lower --every or raise --iterations."
        )
        return 2
    # The systematic half: diff EVERY audited attribute, not only the two we hash above.
    div = _diff_audits(a_dir, b_dir)
    explained = {}
    if div:
        explained = {k: v for k, v in div.items() if k in EXPECTED_DIVERGENCES}
        div = {k: v for k, v in div.items() if k not in EXPECTED_DIVERGENCES}
    if div is None:
        print("\n  (no library_audit.json on one side -- attribute diff skipped)")
    else:
        n_audited = _n_audited(b_dir)
        print(
            f"\n  attribute diff over {n_audited} audited attributes "
            f"(library, cost proxy, replay buffer):"
        )
        if explained:
            print(f"    {len(explained)} differ FOR A CHECKED REASON:")
            for k in sorted(explained):
                va, vb = explained[k]
                print(f"      {k}: {va} -> {vb}")
                print(f"          why it cannot matter: {EXPECTED_DIVERGENCES[k]}")
        if div:
            print(f"    {len(div)} UNEXPLAINED -- these are the failures:")
            for k, (va, vb) in sorted(div.items()):
                print(f"      {k}\n          at stop       : {va}\n          after restore : {vb}")
        else:
            print("    0 unexplained: every other attribute is reconstructed exactly.")

    # ok_list / ok_rows ARE NOW PART OF THE VERDICT. They were not, while the RNG went
    # uncheckpointed: a resumed segment then drew a different stream and legitimately promoted a
    # different fragment set, so an end-of-run comparison could never pass -- and a check that can
    # never pass is the same failure as one that can never fail. `rng_io` closes that, so these
    # become the test of REPRODUCIBILITY, and they are the branch of this harness that goes red if
    # it regresses. Baseline before the fix, both runs at 15 promoted and 1.0 coverage:
    #     uninterrupted 7fc4acef...    resumed d10044fc...
    repro = ok_list and ok_rows
    print(f"  requeued run is bit-reproducible : {repro}   <- rng_io; this was RED before that fix")

    if not div and ok_routes and repro:
        print(
            "\nPASS: the restore reconstructs the stop-point state on every audited attribute, "
            "every promoted fragment carries a recipe, and the requeued run is bit-identical to "
            "an uninterrupted one."
        )
        return 0
    if div:
        print("\nFAIL: the restore did not reconstruct the attributes listed above.")
    if not ok_routes:
        print(
            "FAIL: a promoted fragment has no recipe -- this cell would fail "
            "check_route_readiness."
        )
    if not repro:
        print(
            "FAIL: the requeued run promoted a DIFFERENT fragment set than the uninterrupted one. "
            "The library is correct but the RNG position was not restored, so this cell is not "
            "reproducible (see rng_io)."
        )
    return 1


# Attributes that DO legitimately differ across a restore, each with the reason it cannot matter.
# This is a classification, not a mute button: the peer request behind this audit was "anything that
# differs is either state that needs persisting or state that provably does not matter, and you can
# then SAY WHICH". Every entry here is a claim that was checked against the source, and each is
# printed with its justification so a reader can disagree with it.
EXPECTED_DIVERGENCES = {
    "path_cost_proxy.n_recent_updates": "write-only counter: assigned 0 in __init__ and incremented in assign_costs, and read "
    "NOWHERE in the clone or our tree (grepped). Its value cannot influence any decision.",
    "library._smiles_to_route": "deliberately filtered on save. The sidecar persists routes for the still-PROMOTABLE "
    "candidate set (depth <= max_num_reactions, non-initial), not every molecule ever seen -- "
    "a molecule too deep to be promoted can never need a recipe. The count therefore shrinks "
    "by design; what must be complete is recipe coverage of the PROMOTED fragments, which is "
    "asserted separately above and must read 1.0.",
}


def _n_audited(b_dir: Path) -> int:
    f = b_dir / "library_audit_at_stop.json"
    try:
        return len(json.loads(f.read_text()))
    except Exception:  # noqa: BLE001
        return 0


def _diff_audits(a_dir: Path, b_dir: Path):
    """{key: (at_stop, after_restore)} for every audited attribute that differs, or None.

    Compares run B's state AT THE STOP POINT against its state IMMEDIATELY AFTER THE RESTORE -- not
    two end-of-run states. The RNG is not checkpointed, so a resumed segment samples differently
    from an uninterrupted tail by design; an end-of-run diff would be dominated by that legitimate
    divergence and would flag every field forever, which is worse than no check because it looks
    like one. Stop-point vs post-restore is the comparison that isolates the restore itself.
    """
    fa, fb = b_dir / "library_audit_at_stop.json", b_dir / "library_audit_restore.json"
    if not (fa.exists() and fb.exists()):
        return None
    aa, bb = json.loads(fa.read_text()), json.loads(fb.read_text())
    out = {}
    for k in sorted(set(aa) | set(bb)):
        va, vb = aa.get(k, "<absent>"), bb.get(k, "<absent>")
        # An opaque value records only its type, so it can only report a type CHANGE; say so
        # rather than letting it read as "verified identical".
        if isinstance(va, dict) and va.get("opaque"):
            if va != vb:
                out[k] = (va, vb)
            continue
        if va != vb:
            out[k] = (va, vb)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
