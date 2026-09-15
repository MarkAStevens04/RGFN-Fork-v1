"""A selected library, resolved into molecules, batches and routes (``docs/ROUTE_DATASET_SCHEMA.md``).

This is the middle layer of the exporter: it takes the object the campaign already produces
(:class:`~glue.samplers.lsdflow.campaign.CampaignResult` -- the same object both strategies return)
plus a way to look up each accepted molecule's logged route, and produces the in-memory library that
:mod:`glue.export.writers` serialises. Nothing here reads a file or knows what a run directory looks
like; that is the driver's job, which is what keeps the exporter usable on the next data generation
without edits.

THREE DEFINITIONS THIS MODULE OWNS, because getting them wrong is how a route dataset lies.

1. **What the library is.** The campaign is run to a fixed REACTION budget (CLAUDE.md: 100 reactions
   is the primary readout, "I have 100 reactions, how many distinct high-reward molecules do I
   get?"). The strategy stops as soon as ``cum_reactions >= R``, so its LAST accepted mode usually
   pushes the total past R. :func:`select_within_budget` drops that overshoot, so the shipped library
   is one a chemist could actually run at R -- and it records ``stop_reason`` and the reactions
   actually spent, because a library that stopped because the pool ran out is not the same claim as
   one that stopped because the budget bound.

2. **What a batch is.** §1: "a set of molecules that share a synthetic intermediate, so they are made
   together off one common prefix". Members are grouped by the hub the cost model charged them
   against (``source_hub``), but the PREFIX itself is measured as the longest run of leading steps
   the members' logged routes actually agree on. Those two can disagree: for ``best_candidate`` the
   assigned hub is the cheapest valid shared prefix for *accounting* and need not be a prefix of the
   logged route at all (see ``campaign.py`` on accidental hub sharing). Taking the assignment on
   faith would print a ``shared_intermediate`` the chemist never weighs out;
   :attr:`Batch.prefix_is_assigned_hub` records whether the two agreed.

   Every member keeps at least one diverging step, so a singleton batch is "make the precursor, then
   the last step", never "the whole route is shared" -- which is what makes ``reactions_per_molecule``
   read as the batching efficiency the schema says it is.

3. **Which SMILES is the molecule.** §6: ``smiles`` is stereo-aware (the structure you would make),
   ``smiles_flat`` is the stereo-stripped key the selection and the diversity metric ran on. The
   stereo-aware form is taken from the logged route's final product -- the log recorded it; nothing
   is re-perceived. ``has_unassigned_stereo`` flags the few molecules carrying a genuinely undefined
   centre, which per §6 are the only ones a squiggle would honestly depict.

RDKit is imported lazily and only for ``has_unassigned_stereo``; everything else here is text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from glue.export.routes import (
    RouteAssembler,
    RouteStep,
    common_prefix_len,
    cross_check,
    to_aizynth_tree,
)
from glue.samplers.lsdflow.campaign import CampaignPoint, CampaignResult


# ----------------------------------------------------------------- route lookup
class LoggedRouteSource:
    """Resolves an accepted mode into ``(stereo-aware smiles, ordered steps)``.

    One class covers both strategies because the difference is only what is known about the final
    step:

    * ``best_candidate`` accepts **sampled terminals**, which carry their own complete logged route.
      Nothing extra is needed: ``assemble(target=...)`` linearizes it.
    * ``hub_batching`` accepts **enumerated children**, which exist only as "this hub, plus this one
      logged reaction". The caller supplies that reaction (``final_steps``) and the promoted
      fragments it attaches (``extra_builds``), keyed ``(hub_key, child_smiles)``; the hub's own
      route is linearized from the same ``routes`` map, so the child's route is complete rather than
      starting from an unbuyable scaffold.

    Falls back to the sampled-terminal path whenever no final step is registered for a point, so a
    mixed or partially-indexed input degrades to "the route we do have" instead of raising.
    """

    def __init__(
        self,
        assembler: RouteAssembler,
        *,
        final_steps: Optional[Mapping[Tuple[str, str], Sequence[Mapping]]] = None,
        extra_builds: Optional[Mapping[Tuple[str, str], Sequence[str]]] = None,
    ):
        self.assembler = assembler
        self.final_steps = final_steps or {}
        self.extra_builds = extra_builds or {}

    def route_for(self, point: CampaignPoint) -> Optional[Tuple[str, List[RouteStep]]]:
        key = (point.source_hub or "", point.smiles)
        finals = self.final_steps.get(key)
        if finals:
            steps = self.assembler.assemble(
                target=point.smiles,
                prefix=point.source_hub,
                extra_builds=tuple(self.extra_builds.get(key, ())),
                final_steps=finals,
            )
        else:
            steps = self.assembler.assemble(target=point.smiles)
        if not steps:
            return None
        return steps[-1].product or point.smiles, steps


# ----------------------------------------------------------------- the library
@dataclass
class Molecule:
    """One row of ``molecules.csv``, plus the route the other three files are views of."""

    mol_id: str
    smiles: str  # stereo-aware -- the structure to make (§6)
    smiles_flat: str  # stereo-stripped -- the key selection and the diversity metric ran on
    has_unassigned_stereo: bool
    batch_id: str
    reward: float
    reward_name: str
    selection_step: int
    cum_reactions: int
    reactions_added: int
    steps: List[RouteStep] = field(default_factory=list)
    source_hub: str = ""
    n_shared: int = 0  # leading steps that are the batch prefix (filled when batches are built)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def diverging(self) -> List[RouteStep]:
        return self.steps[self.n_shared :]

    def row(self) -> Dict:
        """Exactly the columns of §4.1, in the schema's order."""
        return {
            "mol_id": self.mol_id,
            "smiles": self.smiles,
            "smiles_flat": self.smiles_flat,
            "has_unassigned_stereo": self.has_unassigned_stereo,
            "batch_id": self.batch_id,
            "reward": self.reward,
            "reward_name": self.reward_name,
            "selection_step": self.selection_step,
            "cum_reactions": self.cum_reactions,
            "reactions_added": self.reactions_added,
            "n_steps": self.n_steps,
        }


@dataclass
class Batch:
    """One row of ``batches.csv`` (§4.2) and one ``batch_NN/`` directory."""

    batch_id: str
    members: List[Molecule]
    n_shared: int
    assigned_hub: str = ""

    @property
    def prefix(self) -> List[RouteStep]:
        return self.members[0].steps[: self.n_shared] if self.members else []

    @property
    def shared_intermediate(self) -> str:
        """The SMILES every member is built from (§4.2).

        Normally the product of the last shared step. When there is no shared STEP, the members can
        still share a substrate -- a **purchasable hub**, which the cost model correctly prices at
        zero reactions because a chemist buys it (``shallow_couplings(depth=0) == 0``). Reporting
        that batch as having no shared intermediate would hide the very case where batching is
        cheapest, so the common substrate of the members' final steps is used instead. It is empty
        only when the members genuinely have nothing in common, which is what a best-candidate
        library of singletons looks like and is meant to look like.
        """
        if self.prefix:
            return self.prefix[-1].product
        inputs = {m.steps[-1].input for m in self.members if m.steps}
        return inputs.pop() if len(inputs) == 1 else ""

    @property
    def prefix_is_assigned_hub(self) -> bool:
        """Did the accounting hub turn out to be the real shared prefix? (Diagnostic, not a column.)"""
        return bool(self.assigned_hub) and self.shared_intermediate == self.assigned_hub

    @property
    def n_molecules(self) -> int:
        return len(self.members)

    @property
    def prefix_reactions(self) -> int:
        return self.n_shared

    @property
    def total_reactions(self) -> int:
        """Prefix once + every distinct diversifying step, counted once **within this batch**.

        Count-once by product identity: two members that both build the same promoted fragment after
        the prefix pay for it once, the way the campaign's global cost model does. It is batch-local
        on purpose -- a fragment shared with a DIFFERENT batch is charged in both, so the batch
        totals sum to more than the library's reaction count. The library-wide count-once number is
        ``cum_reactions`` in ``molecules.csv``; this column is what this plate costs to run.
        """
        distinct = {st.product for m in self.members for st in m.diverging}
        return self.n_shared + len(distinct)

    @property
    def reactions_per_molecule(self) -> float:
        return self.total_reactions / self.n_molecules if self.members else float("nan")

    @property
    def dirname(self) -> str:
        return f"batch_{self.batch_id[1:]}"

    def row(self) -> Dict:
        """Exactly the columns of §4.2, in the schema's order."""
        return {
            "batch_id": self.batch_id,
            "shared_intermediate": self.shared_intermediate,
            "n_molecules": self.n_molecules,
            "prefix_reactions": self.prefix_reactions,
            "total_reactions": self.total_reactions,
            "reactions_per_molecule": round(self.reactions_per_molecule, 3),
        }

    # -------------------------------------------------------------- buy list (§4.5)
    def buy_rows(self) -> List[Dict]:
        """``smiles, role, n_molecules_needing_it`` -- what to order for this plate.

        A leaf is anything consumed that no step in that molecule's own route produces. ``role`` is
        ``shared`` when the prefix consumes it (order it once for the plate, in N molecules' worth)
        and ``diverging`` otherwise (one per well). SMILES are passed through verbatim so the entry
        keeps its catalogue stereochemistry -- §6: ordering the flat form is ordering the wrong
        thing.
        """
        prefix_products = {st.product for st in self.prefix}
        shared: Set[str] = set()
        for st in self.prefix:
            for r in st.all_reactants():
                if r and r not in prefix_products:
                    shared.add(r)
        diverging: Dict[str, int] = {}
        for m in self.members:
            made = {st.product for st in m.steps}
            used: Set[str] = set()
            for st in m.diverging:
                for r in st.all_reactants():
                    if r and r not in made and r not in shared:
                        used.add(r)
            for r in used:
                diverging[r] = diverging.get(r, 0) + 1
        rows = [
            {"smiles": s, "role": "shared", "n_molecules_needing_it": self.n_molecules}
            for s in sorted(shared)
        ]
        rows += [
            {"smiles": s, "role": "diverging", "n_molecules_needing_it": n}
            for s, n in sorted(diverging.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return rows


@dataclass
class StrategyLibrary:
    """Everything one ``<cell>/<strategy>/`` directory holds."""

    strategy: str
    molecules: List[Molecule]
    batches: List[Batch]
    trees: Dict[str, Dict]  # stereo-aware smiles -> AiZynth route tree
    purchasable: Set[str]
    budget_reactions: int
    reactions_used: int
    stop_reason: str
    total_modes_before_trim: int
    dropped_no_route: List[str] = field(default_factory=list)
    diagnostics: Dict = field(default_factory=dict)


# ----------------------------------------------------------------- assembly
def select_within_budget(
    result: CampaignResult, budget_reactions: Optional[int]
) -> List[CampaignPoint]:
    """The prefix of the selection that actually fits the budget.

    The strategies stop when ``cum_reactions >= R``, so the last accepted mode typically overshoots.
    Shipping it would describe a library that cost more than the budget it is published against;
    dropping it leaves one that spent slightly less, which is the honest reading and the same slice
    ``parallel_groups.py`` takes (``cum_reactions <= budget``). Prefix-stable for both strategies
    (CLAUDE.md: greedy mode selection and our hub-batching are prefix-stable), so this is an exact
    re-read of the same ordering, not a different experiment.
    """
    if budget_reactions is None:
        return list(result.accepted)
    return [p for p in result.accepted if p.cum_reactions <= budget_reactions]


@lru_cache(maxsize=None)
def _n_unassigned_centres(smiles: str) -> int:
    """Unassigned tetrahedral centres, EXCLUDING bridgeheads -- the squiggle-worthy count (§6).

    A bridgehead (3 or more ring bonds) is reported by RDKit's modern stereo perception but is not
    a configuration anyone chooses: the cage fixes it, and the other one is not for sale. Measured
    on this project's 418-block catalogue, not filtering turns 3 genuinely undefined blocks into 5
    by adding quinuclidine's bridgehead carbon AND its bridgehead nitrogen -- and a squiggle drawn
    on those would invent a separation problem that does not exist, which is precisely what §6
    exists to prevent. The rule can under-flag a cis/trans ring-FUSION centre, which also carries 3
    ring bonds; blocks sold with such a centre carry it specified, so it does not arise here.

    RDKit is imported here so the CSV/markdown half of the exporter runs without a chemistry stack;
    unavailable RDKit reports 0, and the caller records that the flag was not computed.
    """
    try:
        from rdkit import Chem
    except Exception:  # pragma: no cover - RDKit present in-env
        return 0
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return 0
    try:
        info = Chem.FindPotentialStereo(mol)
    except Exception:  # pragma: no cover
        return 0
    n = 0
    for entry in info:
        if entry.type != Chem.StereoType.Atom_Tetrahedral:
            continue
        if entry.specified != Chem.StereoSpecified.Unspecified:
            continue
        atom = mol.GetAtomWithIdx(entry.centeredOn)
        if sum(1 for b in atom.GetBonds() if b.IsInRing()) >= 3:
            continue  # bridgehead: fixed by the cage, not a choice when ordering
        n += 1
    return n


def _has_unassigned_stereo(smiles: str) -> bool:
    """True iff a stereocentre is genuinely undefined (§6) -- the only squiggle-worthy case."""
    return _n_unassigned_centres(smiles) > 0


def count_stereocentre_creating_steps(
    molecules: Sequence["Molecule"],
) -> List[Tuple[str, int, str]]:
    """Steps whose product carries an undefined centre none of its reactants had.

    §6 is built on a measurement -- "0 of 10,248 route steps CREATE a stereocentre" -- and the whole
    stereo-aware policy follows from it: if no step makes a centre, every centre is one you bought
    and a squiggle would be a lie. That claim is a property of the TEMPLATE SET, so it has to be
    re-measured on whatever chemistry is being exported rather than carried forward. This is that
    measurement, run per export, and any non-zero result is a caveat the cell README must carry:
    those molecules really do come out racemic.
    """
    out: List[Tuple[str, int, str]] = []
    for m in molecules:
        for i, st in enumerate(m.steps, 1):
            before = sum(_n_unassigned_centres(r) for r in st.all_reactants())
            if _n_unassigned_centres(st.product) > before:
                out.append((m.mol_id, i, st.named))
    return out


def build_library(
    result: CampaignResult,
    source: LoggedRouteSource,
    purchasable: Set[str],
    *,
    reward_name: str,
    budget_reactions: Optional[int] = 100,
    stereo_check: Optional[Mapping[str, str]] = None,
) -> StrategyLibrary:
    """Resolve a :class:`CampaignResult` into the library the writers serialise.

    ``stereo_check`` is an INDEPENDENT record of each flat SMILES' stereo-aware form (the sampler's
    ``child_stereo_key`` column). It is not used to build anything -- it is compared against the form
    recovered from the logged route, so the claim "``smiles`` is the structure you would make" has a
    failing case rather than being asserted. Disagreements are counted, reported, and do not change
    the output: the route is the thing being shipped, so the route's own product wins.
    """
    points = select_within_budget(result, budget_reactions)
    molecules: List[Molecule] = []
    dropped: List[str] = []
    stereo_mismatch: List[str] = []
    for p in points:
        resolved = source.route_for(p)
        if resolved is None:
            dropped.append(p.smiles)
            continue
        smiles, steps = resolved
        if stereo_check is not None:
            expect = stereo_check.get(p.smiles)
            if expect and expect != smiles:
                stereo_mismatch.append(p.smiles)
        molecules.append(
            Molecule(
                mol_id=f"M{len(molecules) + 1:04d}",
                smiles=smiles,
                smiles_flat=p.smiles,
                has_unassigned_stereo=_has_unassigned_stereo(smiles),
                batch_id="",
                reward=p.reward,
                reward_name=reward_name,
                selection_step=len(molecules) + 1,
                cum_reactions=p.cum_reactions,
                reactions_added=p.reactions_added,
                steps=steps,
                source_hub=p.source_hub or "",
            )
        )

    batches = _build_batches(molecules)
    trees: Dict[str, Dict] = {}
    for m in molecules:
        tree = to_aizynth_tree(m.steps, m.smiles, purchasable)
        cross_check(m.mol_id, m.smiles, m.steps, tree)  # §4.4: a mismatch is a build failure
        trees[m.smiles] = tree

    created = count_stereocentre_creating_steps(molecules)
    used = molecules[-1].cum_reactions if molecules else 0
    return StrategyLibrary(
        strategy=result.strategy,
        molecules=molecules,
        batches=batches,
        trees=trees,
        purchasable=set(purchasable),
        budget_reactions=budget_reactions if budget_reactions is not None else 0,
        reactions_used=used,
        stop_reason=result.stop_reason,
        total_modes_before_trim=result.total_modes,
        dropped_no_route=dropped,
        diagnostics={
            "n_selected_before_trim": result.total_modes,
            "n_within_budget": len(points),
            "n_dropped_no_route": len(dropped),
            "n_batches_prefix_is_assigned_hub": sum(1 for b in batches if b.prefix_is_assigned_hub),
            "n_batches_with_assigned_hub": sum(1 for b in batches if b.assigned_hub),
            "n_stereo_disagreements_vs_records": len(stereo_mismatch),
            "stereo_disagreement_examples": stereo_mismatch[:5],
            "n_molecules_with_unassigned_stereo": sum(
                1 for m in molecules if m.has_unassigned_stereo
            ),
            "n_molecules_with_defined_stereo": sum(1 for m in molecules if "@" in m.smiles),
            "n_steps": sum(len(m.steps) for m in molecules),
            "n_steps_creating_a_stereocentre": len(created),
            "stereocentre_creating_reactions": sorted({r for _m, _i, r in created}),
        },
    )


def _build_batches(molecules: Sequence[Molecule]) -> List[Batch]:
    """Group into plates and measure each plate's real shared prefix.

    Grouping key is the accounting hub, so ``batches.csv`` describes the library the cost model
    priced. A molecule the cost model charged independently (``source_hub`` empty -- every
    best-candidate pick that shares nothing) is its own batch, which is why best-candidate's file is
    mostly singletons: §4.2 is explicit that this IS the result and the format should state it
    rather than hide it by omitting the file.
    """
    order: List[str] = []
    groups: Dict[str, List[Molecule]] = {}
    for i, m in enumerate(molecules):
        key = m.source_hub or f"\x00solo{i}"  # unshareable key: a solo pick is its own plate
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(m)

    batches: List[Batch] = []
    for key in order:
        members = groups[key]
        routes = [m.steps for m in members]
        n_shared = common_prefix_len(routes)
        # Every member keeps at least one diverging step: a plate where nothing diverges is not a
        # plate, and for a singleton the "prefix" is everything up to its final reaction.
        n_shared = max(0, min(n_shared, min(len(r) for r in routes) - 1))
        batch = Batch(
            batch_id=f"B{len(batches) + 1:02d}",
            members=members,
            n_shared=n_shared,
            assigned_hub="" if key.startswith("\x00solo") else key,
        )
        for m in members:
            m.batch_id = batch.batch_id
            m.n_shared = n_shared
        batches.append(batch)
    return batches
