"""``batch_NN/scheme.png`` -- one batch drawn the way a chemist reads it (§4.5).

Two sections, because a batch is two different jobs: **make the shared intermediate once**, then
**split it N ways**. Each row is one reaction, laid out substrate + reagent -> product, and every
structure is coloured by the only distinction that changes what you do next:

    green  = purchasable building block (order it)
    blue   = made in an earlier step
    maroon = the shared intermediate itself
    dark   = a final library member

"Purchasable" is the run's own catalogue, passed in by the caller, so the figure cannot disagree
with what the model was allowed to buy. Stereocentres are drawn wedge/dash with R/S annotation
rather than flattened: they are INHERITED from chiral-pool blocks (§6 -- no step in this template
set creates one), so the centre visible in the product is the one you bought, not one you have to
resolve. Flattening it would invent a separation problem the route does not have.

This is the same rendering the prototype ``experiments/lsd_hubs/campaign/render_route_scheme.py``
was written to test, generalised from one molecule to one batch. The publication-grade renderer with
hand-tuned poses, cropped ink and yield labels is
``experiments/lsd_hubs/campaign/render_batch_scheme.py``; it is a figure script bound to specific
run paths, so it is not what the dataset ships. If the two ever disagree about chemistry, the
dataset's CSVs are the authority -- both draw from the same logged steps.

ALL heavy imports are inside the functions. A machine without RDKit or matplotlib still produces a
complete dataset: :func:`render_batch_scheme` returns the reason it could not draw, and the caller
records it instead of failing the export.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from glue.export.library import Batch
from glue.export.routes import RouteStep

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BUY, MAKE, HUB, PROD = "#1baf7a", "#2a78d6", "#8a4b08", "#0b0b0b"
PANEL = (300, 210)

# A batch can hold dozens of members; a row per member would produce a figure nobody opens. The cap
# is on ROWS DRAWN, and the figure says so -- the complete set is always in protocol.md/steps.csv.
DEFAULT_MAX_MEMBER_ROWS = 16


def _draw(smiles: str, size: Tuple[int, int] = PANEL) -> bytes:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Draw import rdMolDraw2D

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return b""
    Chem.rdDepictor.Compute2DCoords(mol)
    try:
        Chem.rdDepictor.StraightenDepiction(mol)
    except Exception:  # pragma: no cover - older RDKit
        pass
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = d.drawOptions()
    opts.clearBackground = False
    opts.addStereoAnnotation = True  # the centre is bought, so name it
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return d.GetDrawingText()


def render_batch_scheme(
    path: Path,
    batch: Batch,
    *,
    title: str = "",
    purchasable: Optional[set] = None,
    max_member_rows: int = DEFAULT_MAX_MEMBER_ROWS,
) -> Optional[str]:
    """Draw ``batch``; return ``None`` on success or a short reason it could not be drawn."""
    try:
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - drawing stack optional by design
        return f"no drawing stack ({exc})"

    made = {st.product for m in batch.members for st in m.steps}
    buyable = set(purchasable or ())
    hub = batch.shared_intermediate
    finals = {m.smiles for m in batch.members}

    rows: List[Tuple[str, RouteStep, str]] = []
    for st in batch.prefix:
        rows.append(("Make the shared intermediate", st, ""))
    shown = batch.members[:max_member_rows]
    for m in shown:
        for st in m.diverging:
            rows.append((f"-> {m.mol_id}  ({m.reward_name} {m.reward:.4g})", st, m.mol_id))
    if not rows:
        return "no steps"

    try:
        fig, axes = plt.subplots(
            len(rows), 3, figsize=(14.4, 3.05 * len(rows)), facecolor=SURFACE, squeeze=False
        )
        last_section = None
        for i, (section, st, _mol) in enumerate(rows):
            reagent = st.reactants[0] if st.reactants else ""
            for j, smi in enumerate((st.input, reagent, st.product)):
                ax = axes[i][j]
                ax.set_facecolor(SURFACE)
                ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])
                if not smi:
                    continue
                if smi == hub:
                    col, tag = HUB, "shared intermediate"
                elif smi in finals:
                    col, tag = PROD, "library member"
                elif smi in made:
                    col, tag = MAKE, "make"
                else:
                    col, tag = BUY, ("buy" if not buyable or smi in buyable else "NOT in catalogue")
                png = _draw(smi)
                if png:
                    ax.imshow(mpimg.imread(io.BytesIO(png), format="png"))
                ax.set_title(tag, fontsize=9.5, color=col, fontweight="semibold", pad=2)
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(col)
                    sp.set_linewidth(1.6)
            # Row labels are the longest text on the page (a named reaction plus a section
            # banner), and they sit OUTSIDE the axes. Wrapped to a fixed width and given a left
            # margin sized for that width, because an unwrapped one is silently clipped by the
            # figure edge -- the failure is invisible in the code and obvious only in the PNG.
            label = "\n".join(textwrap.wrap(st.named, 26) + [f"[t{st.template_id or '?'}]"])
            if section != last_section:
                label = "\n".join(textwrap.wrap(section, 26)) + "\n\n" + label
                last_section = section
            axes[i][0].text(
                -0.05,
                0.5,
                label,
                transform=axes[i][0].transAxes,
                ha="right",
                va="center",
                fontsize=8.6,
                color=INK,
                fontweight="semibold",
            )
        head = title or batch.batch_id
        fig.text(0.5, 0.99, head, ha="center", va="top", fontsize=13, color=INK, fontweight="bold")
        sub = (
            f"{batch.n_molecules} molecule(s) off one intermediate  ·  "
            f"{batch.prefix_reactions} shared + "
            f"{batch.total_reactions - batch.prefix_reactions} diversifying reactions  ·  "
            f"{batch.reactions_per_molecule:.2f} reactions/molecule"
        )
        if len(shown) < batch.n_molecules:
            sub += f"   ({len(shown)} of {batch.n_molecules} members drawn; all are in protocol.md)"
        fig.text(0.5, 0.968, sub, ha="center", va="top", fontsize=9, color=INK2)
        fig.text(
            0.008,
            0.008,
            "green = purchasable building block (order it)  ·  blue = made in an earlier step  ·  "
            "maroon = the shared intermediate  ·  black = a library member.\n"
            "Stereocentres are drawn wedge/dash with R/S annotation: they are INHERITED from a "
            "chiral-pool block, not created, so the configuration shown is the one you buy.\n"
            "Logged assembly route, not a claimed optimal retrosynthesis. Conditions and yields are "
            "not included.",
            fontsize=8,
            color=INK3,
            ha="left",
            va="bottom",
            linespacing=1.5,
        )
        fig.subplots_adjust(
            left=0.205, right=0.99, top=0.945, bottom=0.075, hspace=0.30, wspace=0.04
        )
        fig.savefig(path, dpi=170, facecolor=SURFACE)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - one bad structure must not fail an export
        try:
            plt.close("all")
        except Exception:
            pass
        return f"{type(exc).__name__}: {exc}"
    return None
