# CLAUDE.md

Before any design or code change, read:
1. `.specify/memory/constitution.md`: non-negotiable principles. Never violate them silently.
2. `.kiro/steering/product.md`, `tech.md`, `structure.md`: project context and conventions.
3. The spec for the area you're touching in `.kiro/specs/<nn-name>/` (requirements → design → tasks), and `.kiro/specs/README.md` for the plan.

Working rules
- Implement tasks in spec order. Reference requirement ids (`R3.5`) in commits and tests. Tick tasks in `tasks.md` when done.
- If implementation diverges from a spec, update the spec in the same change.
- Code in `src/finint/core/` must not import LangChain/LangGraph.
- Never send real personal data or non-injected entities to web search.
- Repo docs and code are in English.
