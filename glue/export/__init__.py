"""Library/route exporter for the LSD-Flow route dataset (``docs/ROUTE_DATASET_SCHEMA.md``).

Turns a finished selection -- a :class:`~glue.samplers.lsdflow.campaign.CampaignResult`, which both
selection strategies return -- into the chemist-facing tree the schema specifies: one directory per
cell, one subdirectory per strategy, and inside it ``molecules.csv`` / ``batches.csv`` /
``routes.json`` / ``steps.csv`` plus a ``batch_NN/`` per plate holding ``buy_list.csv``,
``protocol.md`` and ``scheme.png``.

WHY IT LIVES IN ``glue/`` AND TAKES NO PATHS. The schema is a decision about a published artifact,
not about one run: it must survive the ``benchmark_v2`` re-run that replaces every input path, every
gate and the selection config (``docs/RETRAIN_RUNBOOK.md`` §0/§6). So nothing in this package opens
a run directory or knows what a cell is called. It is given objects -- a campaign result, a map of
logged routes, a catalogue -- and returns files. Locating artifacts is the driver's job
(``experiments/lsd_hubs/campaign/export_library.py``), which is also where a v2 driver would sit
beside the v1 one without either touching this code. The package name is deliberately neutral so it
can migrate to the publication repo as ``hubbatching.export``.

Layout::

    naming.py    template string -> named reaction (the ONE table; re-exported by experiments/)
    routes.py    route assembly + the AiZynth tree and flat-table views, and the §4.4 cross-check
    library.py   a CampaignResult resolved into molecules, batches and routes
    writers.py   the on-disk tree of §3, plus the per-cell provenance README of §5
    scheme.py    batch_NN/scheme.png -- the only module that needs RDKit drawing, imported lazily

Heavy dependencies are imported inside functions: the CSV and markdown half of a dataset is
correct and complete on a machine with no chemistry stack at all.
"""

from glue.export.library import (
    Batch,
    LoggedRouteSource,
    Molecule,
    StrategyLibrary,
    build_library,
    select_within_budget,
)
from glue.export.naming import named_reaction, parse_template
from glue.export.routes import (
    LeafAudit,
    RouteAssembler,
    RouteStep,
    RouteViewMismatch,
    cross_check,
    leaf_audit,
    linearize,
    to_aizynth_tree,
)
from glue.export.writers import (
    CellProvenance,
    write_cell_readme,
    write_dataset_readme,
    write_strategy,
)

__all__ = [
    "Batch",
    "CellProvenance",
    "LeafAudit",
    "LoggedRouteSource",
    "Molecule",
    "RouteAssembler",
    "RouteStep",
    "RouteViewMismatch",
    "StrategyLibrary",
    "build_library",
    "cross_check",
    "leaf_audit",
    "linearize",
    "named_reaction",
    "parse_template",
    "select_within_budget",
    "to_aizynth_tree",
    "write_cell_readme",
    "write_dataset_readme",
    "write_strategy",
]
