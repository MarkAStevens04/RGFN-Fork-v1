# References — orientation sheet

**Purpose:** a fast way for an agent (or a new contributor) to orient itself on the
handful of papers this project is *directly* built on — what the method is, why we
chose these systems, and which paper to open when a question comes up. This is **not**
an exhaustive bibliography; it's the short shelf of things you should actually know.

If you find yourself unsure what RGFN is, how a GFlowNet works, how the training loop is
supposed to run, or why we use the 6TD3 system, the answer is here — read the relevant
entry before reasoning further.

## Conventions

- **Cite by key, don't paraphrase.** In logs/docs write `[koziarski2024rgfn]`. The one
  authoritative citation string lives in `references.bib`.
- **PDFs are in `pdfs/<key>.pdf`** and are **git-ignored** (copyrighted + binary; keeps
  the repo clean for the code release). Open them locally by path. If one is
  missing, fetch it from the arXiv/DOI link below and name it after its key.

---

## What we're building (in one paragraph)

We fork **RGFN** (`[koziarski2024rgfn]`) as the generator and train it through the
**multi-round active-learning loop** from the original GFlowNet paper
(`[bengio2021gflownet]`, §4.3 / Alg. 1): RGFN samples molecules proportional to a fast
**proxy** reward; an expensive **oracle** (our docking-based *neosubstrate differential*,
MD later) scores only the per-round query batch; the proxy is refit on those labels;
repeat. The novel piece is **the oracle itself** — a two-tier, same-pose docking
differential that scores whether a candidate's *arm* recruits the second protein of the
ternary complex. Code: the generator lives in RGFN's `rgfn/gfns/reaction_gfn/`; our
oracle/proxy/sampler code lives in the `glue/` package and plugs into RGFN's reward
interface; configs in `configs/glue/`. See `RESEARCH_CONTEXT.md` for the full picture.

---

## Method — what we're building with

### `[bengio2021gflownet]` — Flow Network based Generative Models (NeurIPS 2021)
The original **GFlowNet**. Two ideas to internalize:
1. A GFlowNet learns to *build an object step by step* and sample it with probability
   **proportional to a reward `R(x)`** — so you get many diverse high-reward samples, not
   one optimum.
2. **The training loop we use is theirs.** §4.3 ("Multi-Round Experiments") and
   **Algorithm 1** in Appendix A.5 define the active-learning loop: a learned **proxy `M`**
   is the in-loop reward, RGFN trains against `M(x)^β`, a query batch is sampled and scored
   by the expensive **oracle `O`**, and `M` is refit on the new labels each round. Their
   molecule instantiation (A.5.2) is an MPNN proxy predicting AutoDock scores, refit on
   ~200 freshly docked molecules per round — the direct template for our setup, with our
   docking differential in the role of `O`. **`O` scores enter only by retraining `M`,
   never as a direct RGFN reward.** &nbsp;`pdfs/bengio2021gflownet.pdf` · arXiv:2106.04399

### `[koziarski2024rgfn]` — RGFN: Synthesizable Molecular Generation (NeurIPS 2024)
**The paper this whole fork builds on.** RGFN = Reaction-GFlowNet: instead of growing a
molecule atom-by-atom, it assembles it through a **DAG of chemical reactions over a
building-block library**, so every generated molecule is synthesizable by construction.
The entire `rgfn/gfns/reaction_gfn/` package implements this; `data/chemistry.xlsx` is the
building-block/reaction library; our `glue/` oracles plug into its proxy/reward interface.
When extending the model, "does this match RGFN?" is answered here.
&nbsp;`pdfs/koziarski2024rgfn.pdf` · arXiv:2406.08506

---

## Domain — systems, glue design & evaluation

### `[koziarski2024rgfn]` is method; these explain the *chemistry* we're scoring.

### `[bengeoffrey2025molde]` — Molecular Glue-Design-Evaluator (ACS Omega 2025)
In-silico method for **designing and scoring molecular glues**. Consult this when
reasoning about glue design or oracle scoring choices — and note it's the reference
behind our choice of the **6TD3** system as the testbed (`Logs/000_TEMPLATE.md`).
&nbsp;`pdfs/bengeoffrey2025molde.pdf` · doi:10.1021/acsomega.4c08049

### `[slabicki2020cr8]` — CR8 is a molecular glue degrader of cyclin K (Nature 2020)
Source of the **6TD3** system: DDB1·CDK12–cyclinK·CR8 ternary complex — our **validated
oracle** (78-pp separation on the neosubstrate differential). Cited in logs 002, 003, 005.
&nbsp;doi:10.1038/s41586-020-2133-z &nbsp;*(no PDF — paywalled; drop in if obtained)*

### `[matyskiela2016cc885]` — Cereblon modulator recruits GSPT1 (Nature 2016)
Source of the **5HXB** system: CRBN·DDB1·GSPT1·CC-885 — the **ceiling-hit** system where
docking can't separate real glues from decoys. Cited in logs 001, 003, 005.
&nbsp;doi:10.1038/nature18611 &nbsp;*(no PDF — paywalled; drop in if obtained)*
> ⚠️ Logs 001 & 003 cite this as "Science 2016" — it is **Nature 535:252–257 (2016)**.
> Fix when those logs are next touched.

---

## Adding a paper

Keep this sheet short — add a paper only if work genuinely builds on it.
1. Drop the PDF in `pdfs/` named `<citekey>.pdf` (won't be committed — fine).
2. Add the BibTeX entry to `references.bib` (key = `<firstauthor><year><tag>`).
3. Add a 2–4 line entry here: what it is, why we care, and a pointer if useful.
4. Cite it by key from logs/docs — never restate the citation inline.
