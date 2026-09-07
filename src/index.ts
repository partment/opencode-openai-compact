import type { Plugin } from "@opencode-ai/plugin"
import { loadConfig } from "./config.js"
import { createCompactHooks } from "./compact.js"
import { getDatabasePath } from "./paths.js"
import { CheckpointStore } from "./state.js"

export const server: Plugin = async ({ client, directory, worktree }) => {
  const config = await loadConfig({ directory, worktree })
  if (!config.enabled || !Object.values(config.providers).some((provider) => provider.enabled)) return {}

  const store = await CheckpointStore.open(getDatabasePath())
  try {
    store.prune(config.state.retentionDays)

    return createCompactHooks(config, store, fetch, {
      async getSessionMessages(sessionID) {
        const result = await client.session.messages({ path: { id: sessionID } })
        return result.data
      },
      async getSessionStatus(sessionID) {
        const result = await client.session.status()
        const statuses = result.data
        if (!statuses || typeof statuses !== "object" || Array.isArray(statuses)) return undefined
        if (Object.values(statuses).some((status) => !status || Array.isArray(status) ||
          (status.type !== "idle" && status.type !== "busy" && status.type !== "retry"))) return undefined
        return Object.hasOwn(statuses, sessionID) ? statuses[sessionID].type : "idle"
      },
      async setOpenAIAuth(auth) {
        await client.auth.set({ path: { id: "openai" }, body: auth as any })
      },
    })
  } catch (error) {
    store.close()
    throw error
  }
}

export default {
  id: "opencode-openai-compact",
  server,
}

export { createCompactHooks, loadConfig }
export { CheckpointStore, currentSchemaVersion, type Checkpoint } from "./state.js"
export type { OpenAICompactConfig } from "./schema.js"
