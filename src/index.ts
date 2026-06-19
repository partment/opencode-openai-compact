import type { Plugin } from "@opencode-ai/plugin"
import { loadConfig } from "./config.js"
import { createCompactHooks } from "./compact.js"
import { getDatabasePath } from "./paths.js"
import { CheckpointStore } from "./state.js"

export const server: Plugin = async ({ directory, worktree }) => {
  const config = await loadConfig({ directory, worktree })
  if (!config.enabled) return {}

  const store = await CheckpointStore.open(getDatabasePath())
  store.prune(config.state.retentionDays)

  return createCompactHooks(config, store)
}

export default {
  id: "opencode-openai-compact",
  server,
}

export { createCompactHooks, loadConfig }
export { CheckpointStore, currentSchemaVersion, type Checkpoint } from "./state.js"
export type { OpenAICompactConfig } from "./schema.js"
