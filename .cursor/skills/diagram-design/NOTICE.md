# Notice

This directory is a vendored copy of the `skills/diagram-design/` skill from
https://github.com/cathrynlavery/diagram-design (MIT License, Copyright (c) 2025 Cathryn
Lavery; full text in `LICENSE`, bundled third-party notices in `THIRD_PARTY_LICENSES.md`).
Source: upstream commit `0eeda75a708f419560d5e032693618bb9954f17b`, packaged with the
upstream `scripts/build-claude-skill-zip.py` (bounded build: SKILL.md, references, packaged
scripts, templates, default-light examples; no dark / full / animated example variants).

RetryVLA modifications:

- `SKILL.md`: frontmatter `description` rewritten (routing vs the project `flow-diagram`
  skill, EN + VI trigger terms); new section "RetryVLA local adaptation" (project defaults,
  cluster rendering, fallback). Upstream body otherwise unchanged.
- Added `scripts/pw_env.sh`, `scripts/setup_playwright.sh`, `scripts/render_png.sh`
  (user-local Playwright + Chromium with conda-forge shim libs; cairosvg fallback).
- Added `LICENSE`, `THIRD_PARTY_LICENSES.md`, this `NOTICE.md`.
