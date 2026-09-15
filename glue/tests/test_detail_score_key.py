"""``detail_score_key`` must name the SAME breakdown key that ``score()`` returns.

WHY THIS TEST EXISTS. The cross-env callers -- ``scripts/score_batch.py`` and the persistent
``glue/oracles/docking_server.py`` -- cannot use an oracle's ``score()``: ``score()`` re-runs
``score_detailed()``, so calling it to pick a column would dock every batch twice. They take the
breakdown and select a key instead, and for a long time BOTH hardcoded ``"dvina"``.

That silently turned every detailed oracle into a dvina oracle regardless of its own ``score()``.
``Docking6TD3BGpuOracle`` scores on ``cnn_vs`` (higher-is-better, roughly [0, 9]); the bridges
handed back ``dvina`` (lower-is-better, negative). A config that CORRECTLY declared
``higher_is_better: true`` then computed ``max(+dvina, 0) = 0.0`` for every molecule -- a perfectly
flat reward, with no exception, no nan, and nothing in any log. Measured on four molecules through
the real oracle: raw values -2.690, -0.006, -3.533, -0.021, every reward 0.000.

The key is now declared on the class, which moves the failure from silent to impossible -- but only
while the declaration and ``score()`` agree. Nothing else enforces that: they are two lines in
different parts of a class, and a subclass that changes one and not the other reintroduces exactly
the original bug. Hence this test, which asserts the pair against a STUBBED breakdown so it needs
no receptor, no GPU and no docking.
"""

import pytest

from glue.oracles.docking_gpu_differential_oracle import (
    Docking6TD3BGpuOracle,
    Docking6TD3GpuOracle,
    GpuDifferentialDockingOracle,
)

# One stub breakdown carrying every key any of these oracles scores on. The values are deliberately
# DIFFERENT and differently-signed, so picking the wrong key cannot coincidentally pass.
_BREAKDOWN = {"dvina": -7.5, "cnn_vs": 6.9, "cnnsc_t2": 0.92, "cnnaff_t2": 7.5}

_ORACLES = [GpuDifferentialDockingOracle, Docking6TD3GpuOracle, Docking6TD3BGpuOracle]


@pytest.mark.parametrize("cls", _ORACLES, ids=lambda c: c.__name__)
def test_declared_key_matches_score(cls):
    """``score()`` must return exactly ``detail_score_key``'s value from the breakdown."""
    obj = object.__new__(cls)  # no __init__: these need receptors we do not want in a unit test
    obj.score_detailed = lambda smiles: [dict(_BREAKDOWN) for _ in smiles]
    got = cls.score(obj, ["C", "CC"])
    want = [_BREAKDOWN[cls.detail_score_key]] * 2
    assert got == want, (
        f"{cls.__name__}.score() returned {got} but detail_score_key="
        f"{cls.detail_score_key!r} implies {want}. The two have diverged, which is what "
        f"silently fed dvina to every cnn_vs config."
    )


@pytest.mark.parametrize("cls", _ORACLES, ids=lambda c: c.__name__)
def test_declared_key_is_present_in_the_breakdown(cls):
    """A key that score_detailed never emits would make every score a nan, not an error."""
    assert cls.detail_score_key in _BREAKDOWN, (
        f"{cls.__name__}.detail_score_key={cls.detail_score_key!r} is not a key this oracle's "
        f"breakdown is known to carry; the bridges would read nan for every molecule."
    )


def test_6td3b_and_6td3_disagree():
    """The regression in one line: these two MUST NOT score on the same key.

    6TD3-B exists precisely to score on cnn_vs where its parent scores on dvina. If a refactor
    ever collapses them, every 6TD3-B cell silently becomes a 6TD3 cell with an inverted sign.
    """
    assert Docking6TD3BGpuOracle.detail_score_key == "cnn_vs"
    assert Docking6TD3GpuOracle.detail_score_key == "dvina"
    assert Docking6TD3BGpuOracle.detail_score_key != Docking6TD3GpuOracle.detail_score_key


def test_orientation_matches_the_key():
    """higher_is_better must agree with the metric the key selects."""
    # cnn_vs = CNNaffinity x CNNscore -- larger is a better predicted binder.
    assert Docking6TD3BGpuOracle.higher_is_better is True
    # dvina is a Vina energy differential -- more negative is better.
    assert Docking6TD3GpuOracle.higher_is_better is False
    assert GpuDifferentialDockingOracle.higher_is_better is False
