# References — orientation sheet

**Purpose:** a fast way for an agent (or a new contributor) to orient itself on the
handful of papers this project is *directly* built on — what the method is, why we
chose these systems, and which paper to open when a question comes up. This is **not**
an exhaustive bibliography; it's the short shelf of things you should actually know.

If you find yourself unsure what RGFN is, how a GFlowNet works, or why we use the 6TD3
system, the answer is here — read the relevant entry before reasoning further.

## Conventions

- **Cite by key, don't paraphrase.** In logs/docs write `[koziarski2024rgfn]`. The one
  authoritative citation string lives in `references.bib`.
- **PDFs are in `pdfs/<key>.pdf`** and are **git-ignored** (copyrighted + binary; keeps
  the repo clean for the NeurIPS code release). Open them locally by path. If one is
  missing, fetch it from the arXiv/DOI link below and name it after its key.

---

## Method — what we're building with

### `[bengio2021gflownet]` — Flow Network based Generative Models (NeurIPS 2021)
The original **GFlowNet**. The one idea to internalize: a GFlowNet learns to *build an
object step by step* and sample it with probability **proportional to a reward `R(x)`** —
so you get many diverse high-reward samples, not one optimum. **Our oracle is that
reward.** &nbsp;`pdfs/bengio2021gflownet.pdf` · arXiv:2106.04399

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
