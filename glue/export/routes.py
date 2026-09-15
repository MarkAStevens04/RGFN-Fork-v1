"""Route assembly and the two views of it that ``docs/ROUTE_DATASET_SCHEMA.md`` ships.

ONE in-memory route, two writable views. §4.4 of the schema is explicit that ``routes.json`` (an
AiZynthFinder tree, §4.3) and ``steps.csv`` (a flat table) are "two views of one object", derived
from the same route and **asserted to agree before writing**. That assertion lives here
(:func:`cross_check`) rather than in the writers, so it cannot be skipped by a caller that only
wants one of the two files.

WHAT A ROUTE IS (§2). The *logged assembly route* -- the sequence of reaction templates that
actually fired when the generative model built the molecule -- not a retrosynthesis proposal. Every
step carries the template that produced it. Where a reactant is itself something the model built (a
promoted dynamic-library fragment), its own logged recipe is expanded inline by :func:`linearize`,
so no step ever says *buy X* for an X that cannot be bought. :func:`leaf_audit` measures whether
that actually held, per export; an unpurchasable leaf is a build failure, not a footnote.

THE ORDER MATTERS, because the batch layer reads it. Steps are emitted dependencies-first, and for a
hub-derived molecule the shared scaffold is linearized BEFORE the fragment it is decorated with.
That makes every member of a batch share a genuine leading run of identical steps, which is what
:func:`common_prefix_len` measures and what ``steps.csv``'s ``is_shared`` flag marks. Grouping
instead on the cost model's assigned hub would answer a different question -- see
``glue/samplers/lsdflow/campaign.py`` on accidental hub sharing, where the assigned hub is the
cheapest valid *accounting* prefix and need not be a prefix of the logged route at all.

Dependency-free (no RDKit): a route is connectivity recorded by the generator, and re-perceiving it
would be a chance to disagree with the log. SMILES are passed through verbatim -- they are already
RDKit-canonical where they were written, and re-canonicalising a catalogue entry is how a buy list
starts naming a compound nobody sells (§6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from glue.export.naming import named_reaction, parse_template


@dataclass(frozen=True)
class RouteStep:
    """One reaction that fired: ``input`` + ``reactants`` --template--> ``product``.

    ``input`` is the substrate the model was carrying (the growing molecule); ``reactants`` are what
    it was reacted with. The distinction is the log's, and it is kept because a protocol reads
    better when the thing you already made is named as the substrate -- but for the reaction-SMILES
    convention in ``steps.csv`` the two are joined back together (:meth:`all_reactants`), because
    chemically they are just the reactants.
    """

    template: str  # raw ``R'(A.B >> P, flag, id)``; "" when the step's template was not logged
    template_id: str
    named: str
    input: str
    reactants: Tuple[str, ...]
    product: str

    @staticmethod
    def from_logged(step: Mapping) -> "RouteStep":
        """Build from one logged step dict (``routes.json`` / ``fragments_<N>.json`` /
        ``enum_children.json`` all use the same shape -- see ``glue/samplers/lsdflow/route_steps.py``).
        """
        tmpl = step.get("reaction") or ""
        tid, _ = parse_template(tmpl)
        return RouteStep(
            template=tmpl,
            template_id=tid,
            named=named_reaction(tmpl) if tmpl else "?",
            input=step.get("input") or "",
            reactants=tuple(step.get("reactants") or ()),
            product=step.get("product") or "",
        )

    def all_reactants(self) -> Tuple[str, ...]:
        """Every molecule consumed, substrate first -- the ``.``-joined field of ``steps.csv``."""
        return ((self.input,) if self.input else ()) + self.reactants


# ----------------------------------------------------------------- assembly
def linearize(
    target: str,
    routes: Mapping[str, dict],
    built: Set[str],
    out: List[dict],
) -> None:
    """Ordered logged steps that build ``target``, dependencies first, each intermediate once.

    Lifted unchanged from ``experiments/lsd_hubs/campaign/synthesis_routes.py`` (Logs/032), which is
    where the recursion was worked out: a step's ``input`` or any of its ``reactants`` may itself be
    a promoted fragment with its own logged recipe, so those are expanded before the step that
    consumes them. ``built`` is shared across successive calls on purpose -- linearizing a hub and
    then a fragment that reuses part of it emits the shared piece once, which is exactly the
    count-once view the cost model takes.

    A ``target`` absent from ``routes`` contributes nothing: it is a purchasable leaf.
    """
    if target in built or target not in routes:
        return
    for st in routes[target].get("steps") or ():
        for dep in [st.get("input")] + list(st.get("reactants") or ()):
            if dep and dep != target and dep in routes and dep not in built:
                linearize(dep, routes, built, out)
        out.append(st)
        built.add(st.get("product"))
    built.add(target)


class RouteAssembler:
    """Turns "which molecule" into "the ordered steps that make it, bottoming out in purchasables".

    ``routes`` is one flat map ``{product_smiles: {"steps": [...]}}``. The caller merges its sources
    into it -- for the LSD-Flow cells that is the sampler's ``routes.json`` (so a hub linearizes
    instead of appearing as a bought leaf) plus the recipe snapshot's ``smiles_to_route`` (the
    promoted-fragment builds). Merge order is the caller's business; the assembler only reads.

    ``purchasable`` is the run's own catalogue (``initial_smiles_set``), used for ``in_stock`` and
    for the buy lists. It is NOT used to decide what to expand -- expansion follows the logged
    routes, so a block that is both purchasable and logged as built still shows its logged route.
    """

    def __init__(self, routes: Mapping[str, dict], purchasable: Iterable[str] = ()):
        self.routes = routes
        self.purchasable: Set[str] = set(purchasable)

    def assemble(
        self,
        *,
        target: str,
        prefix: Optional[str] = None,
        extra_builds: Sequence[str] = (),
        final_steps: Sequence[Mapping] = (),
    ) -> List[RouteStep]:
        """The full route for one library member.

        Two shapes, one function, because the two selection strategies deliver their molecules
        differently:

        * **hub-derived** (``prefix`` + ``final_steps``): build the shared scaffold, build whatever
          promoted fragments the last reaction attaches (``extra_builds``), then run the logged
          final step(s). The prefix is linearized FIRST so batch members share a leading run.
        * **sampled terminal** (neither): linearize ``target`` from its own logged route.

        Returns ``[]`` when nothing is known -- a caller that gets an empty route must drop the
        molecule rather than ship a member with no way to make it.
        """
        raw: List[dict] = []
        built: Set[str] = set()
        if prefix or final_steps:
            if prefix:
                linearize(prefix, self.routes, built, raw)
            for frag in extra_builds:
                linearize(frag, self.routes, built, raw)
            raw.extend(final_steps)
        else:
            linearize(target, self.routes, built, raw)
        return [RouteStep.from_logged(s) for s in raw]


def common_prefix_len(routes: Sequence[Sequence[RouteStep]]) -> int:
    """How many leading steps every one of these routes shares, compared on product identity.

    Product identity rather than the whole step, because the same intermediate reached by the same
    logged step is the same flask; comparing template strings too would only add ways to disagree.
    """
    if not routes:
        return 0
    shortest = min(len(r) for r in routes)
    n = 0
    while n < shortest:
        p = routes[0][n].product
        if any(r[n].product != p for r in routes[1:]):
            break
        n += 1
    return n


# ----------------------------------------------------------------- view 1: AiZynth tree (§4.3)
def to_aizynth_tree(steps: Sequence[RouteStep], target: str, purchasable: Set[str]) -> Dict:
    """The route as an AiZynthFinder route tree (``genheden2020aizynth``), §4.3.

    Conventions the schema fixes, and why they are implemented the way they are:

    * ``in_stock: true`` marks a purchasable leaf, and a tree is well-formed iff every leaf is
      ``in_stock`` -- the same "solved" test AiZynth uses. We report ``in_stock`` honestly for every
      mol node, including intermediates, so the test is exactly "are the leaves buyable".
    * ``metadata.template`` is the template that actually fired. AiZynth has no first-class slot for
      it, so it rides in ``metadata`` where a foreign reader can ignore it. A route RECOVERED rather
      than logged has no template, and the key is then **absent, not empty**, so the two provenances
      stay distinguishable by a reader that never saw this docstring.
    * ``named_reaction`` is a reading aid; ``template_id`` is the identifier.

    Expansion is depth-first over the logged steps with a guard on the recursion stack: a route is a
    tree by construction here, but the guard means a malformed input degrades to a leaf instead of
    recursing forever.
    """
    by_product: Dict[str, RouteStep] = {}
    for st in steps:
        by_product.setdefault(st.product, st)

    def mol(smiles: str, stack: Tuple[str, ...]) -> Dict:
        node = {
            "type": "mol",
            "smiles": smiles,
            "in_stock": smiles in purchasable,
            "children": [],
        }
        st = by_product.get(smiles)
        if st is None or smiles in stack:
            return node
        meta = {"template_id": st.template_id, "named_reaction": st.named}
        if st.template:  # absent, not empty, for a recovered route
            meta["template"] = st.template
        node["children"] = [
            {
                "type": "reaction",
                # AiZynth REQUIRES a ``smiles`` on a reaction node and reads it in the RETRO
                # direction (``product>>reactants``): ``ReactionTreeFromDict._parse_tree_dict``
                # does ``rxn_tree_dict["smiles"]`` with no default, so a tree without it raises
                # ``KeyError`` on load. §4.3's illustrative JSON omits the field; a file that
                # followed the example literally would not be readable by the tool the section
                # exists to interoperate with, which was measured: 0 of 75 trees parsed before this
                # was added, 75 of 75 after.
                "smiles": f"{smiles}>>{'.'.join(st.all_reactants())}",
                "metadata": meta,
                "children": [mol(r, stack + (smiles,)) for r in st.all_reactants()],
            }
        ]
        return node

    return mol(target, ())


def tree_leaves(node: Mapping) -> List[Dict]:
    """Every ``mol`` node with no reaction under it -- the things you buy."""
    kids = node.get("children") or []
    if not kids:
        return [dict(node)]
    out: List[Dict] = []
    for rxn in kids:
        for child in rxn.get("children") or []:
            out.extend(tree_leaves(child))
    return out


def tree_reaction_count(node: Mapping) -> int:
    kids = node.get("children") or []
    n = 0
    for rxn in kids:
        n += 1
        for child in rxn.get("children") or []:
            n += tree_reaction_count(child)
    return n


def tree_products(node: Mapping) -> Set[str]:
    """Every mol node that something makes -- the set ``steps.csv`` must agree with."""
    out: Set[str] = set()
    kids = node.get("children") or []
    if kids:
        out.add(node.get("smiles"))
    for rxn in kids:
        for child in rxn.get("children") or []:
            out |= tree_products(child)
    return out


# ----------------------------------------------------------------- view 2: the flat table (§4.4)
def step_rows(mol_id: str, steps: Sequence[RouteStep], n_shared: int) -> List[Dict]:
    """``steps.csv`` rows for one molecule. ``is_shared`` marks the batch prefix (run ONCE)."""
    return [
        {
            "mol_id": mol_id,
            "step": i + 1,
            "reaction": st.template_id,
            "named_reaction": st.named,
            "reactants": ".".join(st.all_reactants()),
            "product": st.product,
            "is_shared": i < n_shared,
        }
        for i, st in enumerate(steps)
    ]


class RouteViewMismatch(RuntimeError):
    """The tree and the flat table disagree. §4.4: a build failure, not a warning."""


def cross_check(mol_id: str, smiles: str, steps: Sequence[RouteStep], tree: Mapping) -> None:
    """Assert the two views describe the same route, before either is written (§4.4).

    Checks the root is the molecule, the reaction count matches the step count, and the set of
    things made matches. Cheap, and it is the only thing standing between a reader and a
    ``routes.json`` that silently describes a different molecule than the row beside it.
    """
    if tree.get("smiles") != smiles:
        raise RouteViewMismatch(f"{mol_id}: tree root {tree.get('smiles')!r} != smiles {smiles!r}")
    n_rxn = tree_reaction_count(tree)
    if n_rxn != len(steps):
        raise RouteViewMismatch(f"{mol_id}: tree has {n_rxn} reactions, steps.csv has {len(steps)}")
    want = {st.product for st in steps}
    got = tree_products(tree)
    if want != got:
        raise RouteViewMismatch(
            f"{mol_id}: products differ (only in steps: {sorted(want - got)[:3]}; "
            f"only in tree: {sorted(got - want)[:3]})"
        )


# ----------------------------------------------------------------- provenance the README must carry
@dataclass
class LeafAudit:
    """Did every route actually bottom out in the catalogue? (§2 "Leaves are purchasable".)"""

    n_routes: int = 0
    n_leaves: int = 0
    n_unpurchasable: int = 0
    examples: List[str] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return self.n_unpurchasable == 0


def leaf_audit(trees: Mapping[str, Mapping], purchasable: Set[str]) -> LeafAudit:
    audit = LeafAudit(n_routes=len(trees))
    seen: Set[str] = set()
    for tree in trees.values():
        for leaf in tree_leaves(tree):
            audit.n_leaves += 1
            smi = leaf.get("smiles")
            if smi not in purchasable:
                audit.n_unpurchasable += 1
                if smi not in seen and len(audit.examples) < 10:
                    seen.add(smi)
                    audit.examples.append(smi)
    return audit
