# AGENTS.md

## Commands

- Use `pnpm@11.6.0`.
- Check order: `pnpm run typecheck`, `pnpm run test`, `pnpm run build`.
- Tests use `vitest run`; SQLite binding order is `bun:sqlite` -> `node:sqlite` -> `better-sqlite3`.
- No lint or formatter script exists. Do not invent one.

## Map

- `src/index.ts`: OpenCode plugin entry.
- `src/compact.ts`: request wrapping, compaction checkpoint injection, hook logic.
- `src/state.ts`: SQLite checkpoint store. DB version uses `PRAGMA user_version`.
- `src/sqlite.ts`: SQLite binding bootstrap and adapter.
- `src/config.ts`: JSONC config loading and default config file creation.
- Config files are `openai-compact.json` / `openai-compact.jsonc`, not `opencode.json`.
- Config shape lives in `src/schema.ts`, `configSchema.json`, and README. Change all three or change none.
- Runtime DB is fixed at global OpenCode config dir: `openai-compact/checkpoints.db`.

## Sharp Edges

- ESM `NodeNext`: relative TypeScript imports use `.js` extensions.
- Keep `state.retentionDays` default at 30 and configurable.
