import { describe, expect, test } from "vitest"
import { mkdtemp, rm, stat } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { CheckpointStore, currentSchemaVersion } from "../src/state.js"

const dayMs = 24 * 60 * 60 * 1000

async function exists(file: string) {
  try {
    await stat(file)
    return true
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false
    throw error
  }
}

describe("CheckpointStore", () => {
  test("creates versioned schema", () => {
    const store = CheckpointStore.openMemory()
    try {
      expect(store.version()).toBe(currentSchemaVersion)
    } finally {
      store.close()
    }
  })

  test("saves, loads, prunes, and deletes checkpoints", () => {
    const store = CheckpointStore.openMemory()
    try {
      store.upsert("ses", {
        providerID: "openai",
        responseID: "resp_old",
        afterMessageID: "msg_old",
        afterCreatedAt: 1,
        createdAt: Date.now() - 2 * dayMs,
        items: [{ type: "compaction", encrypted_content: "old" }],
      })
      store.upsert("ses", {
        providerID: "openai",
        responseID: "resp_new",
        afterMessageID: "msg_new",
        afterCreatedAt: 2,
        createdAt: Date.now(),
        items: [{ type: "compaction", encrypted_content: "new" }],
      })

      expect(store.loadAll().filter((entry) => entry.sessionID === "ses").map((entry) => entry.checkpoint.responseID)).toEqual([
        "resp_old",
        "resp_new",
      ])

      store.prune(1)
      expect(store.loadAll().filter((entry) => entry.sessionID === "ses").map((entry) => entry.checkpoint.responseID)).toEqual([
        "resp_new",
      ])

      store.upsert("ses", {
        providerID: "custom-openai",
        responseID: "resp_new",
        afterMessageID: "msg_custom",
        afterCreatedAt: 3,
        createdAt: Date.now(),
        items: [{ type: "compaction", encrypted_content: "custom" }],
      })
      expect(store.loadAll().filter((entry) => entry.sessionID === "ses")).toHaveLength(2)

      store.deleteSession("ses")
      expect(store.loadAll().some((entry) => entry.sessionID === "ses")).toBe(false)
    } finally {
      store.close()
    }
  })

  test("uses WAL mode with sidecar files for file databases", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "opencode-openai-compact-"))
    const file = path.join(root, "checkpoints.db")
    const store = await CheckpointStore.open(file)

    try {
      store.upsert("ses", {
        providerID: "openai",
        responseID: "resp_wal",
        afterMessageID: "msg_wal",
        afterCreatedAt: 1,
        createdAt: Date.now(),
        items: [{ type: "compaction", encrypted_content: "wal" }],
      })

      expect(await exists(`${file}-wal`)).toBe(true)
      expect(await exists(`${file}-shm`)).toBe(true)
    } finally {
      store.close()
      await rm(root, { recursive: true, force: true })
    }
  })

})
