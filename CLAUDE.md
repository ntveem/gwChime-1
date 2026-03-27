# gwChime — Population Inference for Gravitational-Wave Events

Hierarchical Bayesian inference of BBH population properties (mass distribution,
spin distribution, merger rate) from GWTC event posteriors + injection campaigns.
Uses dynesty nested sampling with JAX/JIT for performance.

## Branch Safety
- Working branch: `tejaswi/dev` (push to `fork` remote).
- `main` tracks upstream (`origin` = `isha1810/gwChime`). Do not commit on `main`.
- Run `git branch --show-current` before making edits.

## Conda Environment
- Set via `.env` file at repo root (`GW_CONDA_ENV=...`). Differs per machine.
- The conda-python hook auto-wraps bare `python`/`pip` calls with `conda run`.

## Serena MCP
- Hooks enforce Serena-first for project file operations (read, edit, search, shell).
- Built-in tools (Read, Grep, Glob, Edit, Write, Bash) are redirected to Serena equivalents.
- Exceptions: git, gh, conda, brew, chmod, mkdir pass through to Bash directly.
- If Serena fails, fall back to built-in tools.

## Agent Teams SDK
- Full 3-phase pipeline: planning → execution → dreaming.
- Config: `.claude/agent_teams_config.json`
- Architecture: `.claude/AGENT_TEAMS_ARCHITECTURE.md`
- Crew prompts: `.claude/crew/*.md`
- Commands: `/architect`, `/build`, `/check`, `/consult`, `/tidy`, `/dream`
- SDK source: `.claude/sdk/`

## Population Inference Invariants
- PDFs must be properly normalized (or explicitly documented as unnormalized).
- Jacobians must be consistent when transforming between coordinate systems.
- Source frame vs detector frame: always document which frame variables are in.
- Selection effects (VT): prior reweighting must account for all coordinate transforms.
- Log-space arithmetic: use logsumexp for numerical stability.
- JAX/JIT: no Python control flow in jitted paths; use `jnp.where`, `jax.lax.cond`.
