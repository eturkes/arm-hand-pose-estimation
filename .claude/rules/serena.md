---
paths:
  - ".serena/project.yml"
  - "analysis/*.Rmd"
---

# Session launch cost — Serena indexing

`headroom wrap claude` blocks on `uvx … serena project index` (`cli/wrap.py:_index_serena_project`, 300 s cap) before Claude Code starts → anything that stalls Serena's indexer is felt as launch latency, and only in the repo that holds it.

- Budget: full cold index ≈ 12 s (109 files, 7 language servers), warm ≈ 6 s. A launch stalling far past that means one file is eating an LS request timeout (`serena_config.yml` `tool_timeout` 240 − 5 = 235 s each) → `.serena/logs/indexing.txt` names the file.
- **`**/*.Rmd` is excluded in `.serena/project.yml` for exactly that reason** — the R language server never answers `documentSymbol` for R Markdown, while `.R` files index at ~4 files/s. **Reach `analysis/analysis_summary.Rmd` by `Read`/`rg`; Serena's symbol and search tools do not see it.**
- Serena's own session start is repo-independent (~3.5 s, dominated by the bash LS) and asynchronous — it never blocks the MCP handshake.
