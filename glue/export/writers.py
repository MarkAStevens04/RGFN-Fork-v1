"""Serialise a resolved library to the on-disk tree ``docs/ROUTE_DATASET_SCHEMA.md`` §3 specifies.

Every file here is generated from the same in-memory :class:`~glue.export.library.StrategyLibrary`,
which is the schema's own requirement (§4.4) and the reason the four views cannot drift apart. The
column sets are the schema's, verbatim and in its order: a reader should be able to diff §4 against
:meth:`Molecule.row` / :meth:`Batch.row` / :func:`write_steps` and find nothing.

WHAT THE README IS FOR (§5, §7). "A dataset that cannot prove which enumeration it came from cannot
support the comparison it is published to make." So the per-cell README carries the absolute path
AND md5 of every input, the gate and its direction, the budget and the diversity cutoff, what the
cell delivered, and its known limitations -- and :func:`write_cell_readme` refuses to invent any of
them: they are fields of :class:`CellProvenance`, supplied by the driver that actually read the
files.

Markdown and CSV only; no chemistry stack. The one drawing step lives in
:mod:`glue.export.scheme` and is optional, so a machine without a rendering stack still produces a
complete, correct dataset minus the PNGs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from glue.export.library import Batch, StrategyLibrary
from glue.export.routes import LeafAudit, RouteStep, step_rows

MOLECULE_COLUMNS = [
    "mol_id",
    "smiles",
    "smiles_flat",
    "has_unassigned_stereo",
    "batch_id",
    "reward",
    "reward_name",
    "selection_step",
    "cum_reactions",
    "reactions_added",
    "n_steps",
]
BATCH_COLUMNS = [
    "batch_id",
    "shared_intermediate",
    "n_molecules",
    "prefix_reactions",
    "total_reactions",
    "reactions_per_molecule",
]
STEP_COLUMNS = ["mol_id", "step", "reaction", "named_reaction", "reactants", "product", "is_shared"]
BUY_COLUMNS = ["smiles", "role", "n_molecules_needing_it"]

CAVEAT = (
    "These are **logged assembly routes** -- the exact sequence of reactions the generative model "
    "used to build each molecule, recorded as it happened -- not a claim of optimal retrosynthesis. "
    "A chemist may well know a shorter or more robust route. Reaction conditions, solvents, "
    "temperatures, yields and stereochemical outcomes are **not** included: this is a "
    "connectivity-level plan, not a procedure."
)


def file_md5(path: Path, chunk: int = 1 << 22) -> str:
    """Streaming md5 -- these inputs run to hundreds of MB (§7)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------- the four per-strategy files
def write_molecules(path: Path, lib: StrategyLibrary) -> None:
    _write_csv(path, MOLECULE_COLUMNS, [m.row() for m in lib.molecules])


def write_batches(path: Path, lib: StrategyLibrary) -> None:
    _write_csv(path, BATCH_COLUMNS, [b.row() for b in lib.batches])


def write_routes_json(path: Path, lib: StrategyLibrary) -> None:
    """``{smiles: AiZynth route tree}``, keyed stereo-aware (§6: identity across files is ``smiles``)."""
    path.write_text(json.dumps(lib.trees, indent=1))


def write_steps(path: Path, lib: StrategyLibrary) -> None:
    rows: List[Dict] = []
    for m in lib.molecules:
        rows.extend(step_rows(m.mol_id, m.steps, m.n_shared))
    _write_csv(path, STEP_COLUMNS, rows)


# ----------------------------------------------------------------- per-batch files
def _step_line(n: int, st: RouteStep, made: set) -> str:
    """One numbered protocol line. ``buy``/``make`` is decided by whether an EARLIER step in this
    batch produced the molecule, not by catalogue membership -- a block that is purchasable but was
    also built on the way is still something you already have in a flask.

    A single-reactant step (a functional-group interconversion) is written with nothing over the
    arrow rather than ``+ -``; the template says how many reactants it consumes, so the line follows
    the chemistry instead of a fixed shape.
    """

    def tag(s: str) -> str:
        return "make" if s in made else "buy"

    parts = [f"`{x}` ({tag(x)})" for x in st.all_reactants()]
    lhs = " + ".join(parts) if parts else "-"
    return f"{n}. **{st.named}** [`t{st.template_id or '?'}`] - {lhs} -> `{st.product}`"


def write_protocol(path: Path, batch: Batch, header: str) -> None:
    """``protocol.md`` (§4.5): the shared prefix stated once, then each molecule's divergence.

    Generated from the same route object as every other file, never hand-written, so a protocol
    cannot describe a molecule the CSVs do not contain.
    """
    made = {st.product for m in batch.members for st in m.steps}
    out: List[str] = [f"# {header} - {batch.batch_id}", ""]
    solo = batch.n_molecules == 1
    if batch.shared_intermediate:
        lead = (
            "**Final substrate** (this batch is one molecule, so nothing is shared with anything):"
            if solo
            else f"**Shared intermediate** (made once, then split {batch.n_molecules} ways):"
        )
        out += [lead, f"`{batch.shared_intermediate}`", ""]
    else:
        out += ["**No shared intermediate** - every step below is this molecule's own.", ""]
    out += [
        f"{batch.n_molecules} molecule{'' if solo else 's'} | {batch.prefix_reactions} "
        f"{'preparatory' if solo else 'shared'} reaction(s) + "
        f"{batch.total_reactions - batch.prefix_reactions} final = "
        f"{batch.total_reactions} reactions "
        f"({batch.reactions_per_molecule:.2f} per molecule).",
        "",
        CAVEAT,
        "",
        "## Order",
        "",
        "| SMILES | role | molecules needing it |",
        "|---|---|---|",
    ]
    for r in batch.buy_rows():
        out.append(f"| `{r['smiles']}` | {r['role']} | {r['n_molecules_needing_it']} |")
    out += [
        "",
        "## Prepare the intermediates"
        if solo
        else "## Shared prefix - run ONCE for the whole batch",
        "",
    ]
    if batch.prefix:
        for i, st in enumerate(batch.prefix, 1):
            out.append(_step_line(i, st, made))
    else:
        out.append("_Nothing shared: every step below is per-molecule._")
    out += ["", "## Final step" if solo else "## Diversification - one entry per molecule", ""]
    for m in batch.members:
        out.append(
            f"### {m.mol_id} - {m.reward_name} {m.reward:.4g}"
            + (
                "  *(unassigned stereocentre - see the cell README)*"
                if m.has_unassigned_stereo
                else ""
            )
        )
        out.append("")
        out.append(f"`{m.smiles}`")
        out.append("")
        for i, st in enumerate(m.diverging, batch.prefix_reactions + 1):
            out.append(_step_line(i, st, made))
        out.append("")
    path.write_text("\n".join(out) + "\n")


def write_batch_dir(
    root: Path, batch: Batch, lib: StrategyLibrary, header: str, scheme: bool = True
) -> Optional[str]:
    """``batch_NN/{buy_list.csv, protocol.md, scheme.png}``. Returns a scheme error, or None."""
    d = root / batch.dirname
    d.mkdir(parents=True, exist_ok=True)
    _write_csv(d / "buy_list.csv", BUY_COLUMNS, batch.buy_rows())
    write_protocol(d / "protocol.md", batch, header)
    if not scheme:
        return None
    from glue.export.scheme import render_batch_scheme  # lazy: RDKit + matplotlib

    return render_batch_scheme(
        d / "scheme.png",
        batch,
        title=f"{header} - {batch.batch_id}",
        purchasable=lib.purchasable,
    )


def write_strategy(
    root: Path, lib: StrategyLibrary, header: str, *, schemes: bool = True, max_batch_dirs: int = 0
) -> Dict:
    """One ``<cell>/<strategy>/`` directory, top to bottom. Returns a small summary for the README.

    ``max_batch_dirs`` caps how many ``batch_NN/`` directories are written (0 = all). It never
    touches the four CSV/JSON files, which always describe the whole library -- a smoke test that
    quietly shipped a truncated ``molecules.csv`` would be worse than no smoke test.
    """
    root.mkdir(parents=True, exist_ok=True)
    write_molecules(root / "molecules.csv", lib)
    write_batches(root / "batches.csv", lib)
    write_routes_json(root / "routes.json", lib)
    write_steps(root / "steps.csv", lib)
    errors: List[str] = []
    drawn = lib.batches[:max_batch_dirs] if max_batch_dirs else lib.batches
    for b in drawn:
        err = write_batch_dir(root, b, lib, header, scheme=schemes)
        if err:
            errors.append(f"{b.batch_id}: {err}")
    return {
        "batch_dirs_written": len(drawn),
        "strategy": lib.strategy,
        "n_molecules": len(lib.molecules),
        "n_batches": len(lib.batches),
        "reactions_used": lib.reactions_used,
        "budget_reactions": lib.budget_reactions,
        "stop_reason": lib.stop_reason,
        "largest_batch": max((b.n_molecules for b in lib.batches), default=0),
        "reactions_per_molecule": round(lib.reactions_used / len(lib.molecules), 3)
        if lib.molecules
        else None,
        "scheme_errors": errors,
        **lib.diagnostics,
    }


# ----------------------------------------------------------------- cell README (§5)
@dataclass
class CellProvenance:
    """Everything §5 requires of a cell, supplied by whoever actually read the inputs.

    ``artifacts`` maps a human label -> absolute path; the md5 is computed here, or taken from
    ``artifact_md5`` when a sidecar digest already exists beside the file (the frozen enumeration
    ships ``enum_children.md5``, and re-hashing 372 MB to reprint a number the snapshot already
    recorded is both slow and a chance to disagree with it).
    """

    cell: str
    generator: str
    target: str
    reward_name: str
    hit_threshold: float
    higher_is_better: bool
    budget_reactions: int
    similarity_cutoff: float
    similarity_metric: str
    artifacts: Dict[str, str] = field(default_factory=dict)
    artifact_md5: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, object] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)

    def resolved_md5(self) -> Dict[str, str]:
        out = {}
        for label, p in self.artifacts.items():
            if label in self.artifact_md5:
                out[label] = self.artifact_md5[label]
            else:
                path = Path(p)
                out[label] = file_md5(path) if path.is_file() else "(not a file)"
        return out


def _stereo_sentence(summaries: Sequence[Mapping]) -> str:
    """Bought or made? -- answered from THIS export, never carried forward from the schema.

    §6's stereo-aware policy rests on a measurement: no step in the reaction set creates an sp3
    centre, so every centre is one you bought and a squiggle would invent a separation problem that
    does not exist. That is a property of the template set, so a README that restated it while the
    export had just measured otherwise would be the one contradiction this file exists to prevent.
    """
    created = sum(int(s.get("n_steps_creating_a_stereocentre") or 0) for s in summaries)
    total = sum(int(s.get("n_steps") or 0) for s in summaries)
    flagged = sum(int(s.get("n_molecules_with_unassigned_stereo") or 0) for s in summaries)
    if not created:
        return (
            f"**Stereocentres here are bought, not made.** Measured on this export: **0 of {total} "
            "steps create a stereocentre**, so the configuration shown is the one you order and "
            "nothing needs resolving. `has_unassigned_stereo` flags the "
            f"{flagged} molecule(s) carrying a genuinely undefined centre - inherited from a "
            "catalogue block sold that way - and those are the only places a squiggle is the "
            "honest depiction."
        )
    reactions = sorted(
        {r for s in summaries for r in (s.get("stereocentre_creating_reactions") or [])}
    )
    return (
        f"**Stereocentres here are almost all bought, not made** - but not all: measured on this "
        f"export, **{created} of {total} steps CREATE an undefined stereocentre** "
        f"({', '.join(reactions)}). Those products come out racemic and the usual separation "
        "problem is real for them. Every other centre is inherited from a catalogue block sold in "
        f"that configuration, so it costs nothing to obtain. All {flagged} molecule(s) with an "
        "undefined centre - created or inherited - are flagged by `has_unassigned_stereo` in "
        "`molecules.csv`, and those are the only places a squiggle is the honest depiction."
    )


def write_cell_readme(
    path: Path,
    prov: CellProvenance,
    summaries: Sequence[Mapping],
    leaf_audits: Mapping[str, LeafAudit],
) -> None:
    direction = "higher is better" if prov.higher_is_better else "lower is better"
    md5s = prov.resolved_md5()
    out: List[str] = [
        f"# {prov.cell}",
        "",
        f"Library of molecular candidates for **{prov.target}** generated by **{prov.generator}**, "
        f"with the full logged synthesis route for every molecule and the batch it is made in. "
        f"Two selection strategies are shipped on **identical chemistry** (same enumeration, same "
        f"hit threshold, same budget - only the chooser differs), so they can be compared directly.",
        "",
        CAVEAT,
        "",
        "## Selection",
        "",
        "| | |",
        "|---|---|",
        f"| generator | `{prov.generator}` |",
        f"| target | `{prov.target}` |",
        f"| oracle | `{prov.reward_name}` |",
        f"| hit threshold | **{prov.hit_threshold}** ({direction}) |",
        f"| reaction budget | {prov.budget_reactions} reactions |",
        f"| diversity cutoff tau | {prov.similarity_cutoff} |",
        f"| similarity metric | {prov.similarity_metric} |",
    ]
    for k, v in prov.config.items():
        out.append(f"| {k} | `{v}` |")
    out += [
        "",
        "## Provenance (§7 - a dataset that cannot prove which enumeration it came from cannot "
        "support the comparison it is published to make)",
        "",
        "| artifact | path | md5 |",
        "|---|---|---|",
    ]
    for label, p in prov.artifacts.items():
        out.append(f"| {label} | `{p}` | `{md5s.get(label, '')}` |")
    out += [
        "",
        "## Delivered",
        "",
        "| strategy | molecules | batches | reactions spent | largest batch | stop reason |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        out.append(
            f"| `{s['strategy']}` | {s['n_molecules']} | {s['n_batches']} | "
            f"{s['reactions_used']} / {s['budget_reactions']} | {s['largest_batch']} | "
            f"{s['stop_reason']} |"
        )
    out += [
        "",
        "### Route integrity",
        "",
        "| strategy | routes | leaves | leaves NOT purchasable |",
        "|---|---|---|---|",
    ]
    for name, audit in leaf_audits.items():
        out.append(
            f"| `{name}` | {audit.n_routes} | {audit.n_leaves} | "
            f"{audit.n_unpurchasable}"
            + ("" if audit.solved else f" ({', '.join(audit.examples[:3])})")
            + " |"
        )
    out += [
        "",
        "Every route must bottom out in catalogue building blocks (§2). A non-zero count in the "
        "last column means a route says *buy X* for an X that cannot be bought, and the cell should "
        "not have shipped.",
        "",
        "## Stereochemistry (§6)",
        "",
        "`smiles` is **stereo-aware** - the structure you would actually make. `smiles_flat` is the "
        "stereo-stripped form, carried because it is the key the diversity metric and the selection "
        "ran on. `buy_list.csv` names the stereo-defined catalogue entry - ordering the flat form is "
        "ordering the wrong thing.",
        "",
        _stereo_sentence(summaries),
        "",
        "## Known limitations",
        "",
    ]
    for lim in prov.limitations:
        out.append(f"* {lim}")
    out.append("")
    path.write_text("\n".join(out) + "\n")


# ----------------------------------------------------------------- dataset root
def write_dataset_readme(path: Path, schema_src: Optional[Path] = None) -> None:
    """The root ``README.md`` (+ a copy of ``SCHEMA.md``) of §3's tree."""
    path.mkdir(parents=True, exist_ok=True)
    if schema_src and Path(schema_src).is_file():
        (path / "SCHEMA.md").write_text(Path(schema_src).read_text())
    (path / "README.md").write_text(
        "# LSD-Flow route dataset\n"
        "\n"
        "For each published library, every molecule's full synthesis route and the batch it "
        "belongs to. A **batch** is a set of molecules that share a synthetic intermediate, so they "
        "are made together off one common prefix - the thing a chemist would run as one plate.\n"
        "\n"
        "Libraries are produced under a **fixed reaction budget**: *I can run N reactions - how "
        "many genuinely different molecules do I get, and how do I make them?* Two selection "
        "strategies (`hub_batching`, `best_candidate`) are shipped per cell on identical "
        "chemistry.\n"
        "\n"
        f"{CAVEAT}\n"
        "\n"
        "`SCHEMA.md` is the authoritative file format. One directory per cell = "
        "(generator, target, seed); cells are self-contained and nothing in one refers to another. "
        "Each cell's `README.md` records the gate, the budget, the inputs by absolute path and md5, "
        "and that cell's own limitations.\n"
        "\n"
        "Route trees are in AiZynthFinder's format (`genheden2020aizynth`), so existing "
        "retrosynthesis tooling reads them unmodified. The cross-molecule batch layer "
        "(`batches.csv`, and `is_shared` in `steps.csv`) is ours: we know of no standard format "
        "that encodes one, and it is kept deliberately thin so the per-molecule routes stay "
        "readable by tools that ignore it.\n"
    )
