# GeologicDome — Robot Everest 2026 real2sim2real pipeline

Everest footage → LingBot-Map reconstruction → Open3D → MuJoCo/MJX terrain → fine-tuned Unitree G1 policy → DimOS.

## Start here

**Read `notes/overview.md` first on every cold start.** Before any nontrivial task, read all of `notes/` — it's nine short files, flat, and written to be readable standalone in under 5 minutes each.

## The vault (`notes/`)

`notes/` is an Obsidian vault whose only purpose is giving an agent (or Aditya) enough context to work on this pipeline. It is not a journal and not a PKM system.

Rules:

- Every file has frontmatter: `title`, `updated`, `status` (`current`/`stale`).
- One fact, one home — link with `[[wikilinks]]` instead of duplicating.
- Delete stale content instead of appending updates; git holds history. Bump `updated` when you edit.
- Exception: `notes/experiments.md` rows are **never** deleted — failed runs with takeaways are the point.
- Do **not** create new files in `notes/` without asking. Split a file only past ~500 lines.
- Write valid Obsidian-flavored markdown — use the `obsidian-markdown` skill in `.claude/skills/`.

Agent behavior:

- Update the relevant note **in the same commit** as the code change it describes.
- When a phase demo lands, flip its status in the milestone table in `notes/overview.md`.
- Log every training/reconstruction run as a row in `notes/experiments.md` automatically, with the short commit hash of the code that ran.
- `notes/setup.md` must reflect the machine's *current* state — update it whenever something gets installed.

## Lab notebook (`lab-notebook/`)

The weekly lab notebook lives in `lab-notebook/`, **outside** the vault — one markdown file per week (`YYYY-Www.md`): what was tried, what broke, screenshots. Narrative and failures go there; only distilled, current facts go in `notes/`. Don't mix them.
