import { afterEach, describe, expect, test, vi } from "vitest"
import { mkdtemp, rm } from "node:fs/promises"
import os from "node:os"
import path from "node:path"

const mocks = vi.hoisted(() => ({ openSQLiteDatabase: vi.fn() }))
vi.mock("../src/sqlite.js", () => ({ openSQLiteDatabase: mocks.openSQLiteDatabase }))

import { CheckpointStore } from "../src/state.js"

const roots: string[] = []
afterEach(async () => {
  vi.clearAllMocks()
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

function fakeDatabase(options: { configureFailure?: boolean; migrationFailure?: boolean; userVersion?: number } = {}) {
  const close = vi.fn()
  const db = {
    exec: vi.fn((sql: string) => {
      if (options.migrationFailure && sql.includes("create table")) throw new Error("migration failed")
    }),
    query: vi.fn((sql: string) => ({
      get: vi.fn(() => {
        if (options.configureFailure && sql.includes("journal_mode")) throw new Error("configuration failed")
        if (sql.includes("user_version")) return { user_version: options.userVersion ?? 0 }
        return null
      }),
      all: vi.fn(() => []),
      run: vi.fn(),
    })),
    transaction: vi.fn((run: () => unknown) => run()),
    close,
  }
  mocks.openSQLiteDatabase.mockReturnValue(db)
  return { db, close }
}

async function databaseFile() {
  const root = await mkdtemp(path.join(os.tmpdir(), "openai-compact-open-failure-"))
  roots.push(root)
  return path.join(root, "checkpoints.db")
}

describe("CheckpointStore open failure cleanup", () => {
  test("closes a file database when PRAGMA configuration fails", async () => {
    const { close } = fakeDatabase({ configureFailure: true })
    await expect(CheckpointStore.open(await databaseFile())).rejects.toThrow("configuration failed")
    expect(close).toHaveBeenCalledOnce()
  })

  test("closes a file database when migration fails", async () => {
    const { close } = fakeDatabase({ migrationFailure: true })
    await expect(CheckpointStore.open(await databaseFile())).rejects.toThrow("migration failed")
    expect(close).toHaveBeenCalledOnce()
  })

  test("closes a future-schema database before rejecting it", async () => {
    const { close } = fakeDatabase({ userVersion: 99 })
    await expect(CheckpointStore.open(await databaseFile())).rejects.toThrow("Unsupported openai-compact database schema version: 99")
    expect(close).toHaveBeenCalledOnce()
  })

  test("closes an in-memory database when migration fails", () => {
    const { close } = fakeDatabase({ migrationFailure: true })
    expect(() => CheckpointStore.openMemory()).toThrow("migration failed")
    expect(close).toHaveBeenCalledOnce()
  })
})
