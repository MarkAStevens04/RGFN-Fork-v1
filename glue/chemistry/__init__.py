"""Chemistry — the standard, swappable fragment + reaction library.

This subpackage owns the two *inputs* every reaction-GFlowNet builds molecules from
— a building-block set and a reaction-template set — plus an optional cost
annotation (per-block price, per-reaction yield). The point is a *single standard
set* that RGFN, RxnFlow, and SCENT can all consume, so a benchmark run varies the
**generator** while holding the **chemistry** fixed. Full design + build order:
``docs/CHEM_LIBRARY_FORMAT.md``.

Split of responsibilities:
    - ``library.py``              = "what is the library" (format-agnostic, no RGFN
                                     plumbing; loaders in, exporters out, cost maps).
    - ``reaction_data_factory.py`` = "how does that library become RGFN's action
                                     space" (subclasses upstream ``ReactionDataFactory``
                                     so ``rgfn/`` stays pristine).

Cost enters generation two ways (see the doc): natively for SCENT (its own
``PathCostProxy``), retroactively for everyone else (``validation/harness/cost.py``
prices recorded routes with the same numbers + formula).

**Status: LIVE** (corrected 2026-09-12 — this docstring previously said "non-functional
stubs, every method raises NotImplementedError", which stopped being true when the loaders
landed and was never updated). ``ChemLibrary`` loads and exports real libraries, and
``GlueReactionDataFactory`` is selected by four configs in ``configs/glue/`` against
``data/libraries/glue_standard_v1`` (418 priced fragments, 112 templates with yields).

Imported here so ``glue.registry`` registers ``GlueReactionDataFactory`` with gin.
"""

from glue.chemistry.library import ChemLibrary, FragmentSpec, ReactionSpec  # noqa: F401
from glue.chemistry.reaction_data_factory import GlueReactionDataFactory  # noqa: F401

__all__ = [
    "ChemLibrary",
    "FragmentSpec",
    "ReactionSpec",
    "GlueReactionDataFactory",
]
