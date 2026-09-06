import { beforeEach, describe, expect, test, vi } from "vitest"
import { createCompactHooks } from "../src/compact.js"
import { loadConfig } from "../src/config.js"
import { server } from "../src/index.js"
import { defaultConfig } from "../src/schema.js"
import { CheckpointStore } from "../src/state.js"

vi.mock("../src/compact.js", () => ({ createCompactHooks: vi.fn(() => ({})) }))
vi.mock("../src/config.js", () => ({ loadConfig: vi.fn() }))
vi.mock("../src/paths.js", () => ({ getDatabasePath: () => ":memory:" }))

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(loadConfig).mockResolvedValue(defaultConfig)
})

describe("plugin store lifecycle", () => {
  test("prunes once before handing the store to hooks", async () => {
    const store = CheckpointStore.openMemory()
    const open = vi.spyOn(CheckpointStore, "open").mockResolvedValue(store)
    const prune = vi.spyOn(store, "prune")
    try {
      await server({ client: { session: { status: async () => ({ data: {} }) } }, directory: ".", worktree: "." } as any)
      expect(prune).toHaveBeenCalledOnce()
      expect(prune).toHaveBeenCalledWith(defaultConfig.state.retentionDays)
      expect(createCompactHooks).toHaveBeenCalledOnce()
    } finally { open.mockRestore(); store.close() }
  })

  test.each(["prune", "hooks"])("closes the store when %s initialization fails", async (failure) => {
    const store = CheckpointStore.openMemory()
    const open = vi.spyOn(CheckpointStore, "open").mockResolvedValue(store)
    const close = vi.spyOn(store, "close").mockImplementation(() => {})
    if (failure === "prune") vi.spyOn(store, "prune").mockImplementation(() => { throw new Error("prune failed") })
    else vi.mocked(createCompactHooks).mockImplementationOnce(() => { throw new Error("hooks failed") })
    try {
      await expect(server({ client: { session: {} }, directory: ".", worktree: "." } as any)).rejects.toThrow(`${failure} failed`)
      expect(close).toHaveBeenCalledOnce()
    } finally { open.mockRestore(); close.mockRestore(); store.close() }
  })
})

describe("session status SDK adapter", () => {
  test.each([
    ["busy", { data: { ses_test: { type: "busy" } } }, "busy"],
    ["retry", { data: { ses_test: { type: "retry", attempt: 2, next: 123 } } }, "retry"],
    ["idle", { data: { ses_test: { type: "idle" } } }, "idle"],
    ["absent session", { data: {} }, "idle"],
    ["missing response", {}, undefined],
    ["API error", { error: { message: "unavailable" } }, undefined],
    ["null", { data: null }, undefined],
    ["array", { data: [] }, undefined],
    ["malformed entry", { data: { ses_test: null } }, undefined],
    ["array entry", { data: { ses_test: [] } }, undefined],
    ["invalid table with absent session", { data: { other: "invalid" } }, undefined],
    ["unknown status", { data: { ses_test: { type: "unexpected" } } }, undefined],
  ])("maps %s without treating failures as idle", async (_name, result, expected) => {
    const store = CheckpointStore.openMemory()
    const open = vi.spyOn(CheckpointStore, "open").mockResolvedValue(store)
    try {
      const status = vi.fn(async () => result)
      await server({ client: { session: { status } }, directory: ".", worktree: "." } as any)
      const options = vi.mocked(createCompactHooks).mock.calls[0][3]!
      expect(await options.getSessionStatus!("ses_test")).toBe(expected)
      expect(status).toHaveBeenCalledWith()
    } finally { open.mockRestore(); store.close() }
  })

  test("does not reinterpret a rejected SDK request as idle", async () => {
    const store = CheckpointStore.openMemory()
    const open = vi.spyOn(CheckpointStore, "open").mockResolvedValue(store)
    try {
      await server({ client: { session: { status: async () => { throw new Error("offline") } } }, directory: ".", worktree: "." } as any)
      const options = vi.mocked(createCompactHooks).mock.calls[0][3]!
      await expect(options.getSessionStatus!("ses_test")).rejects.toThrow("offline")
    } finally { open.mockRestore(); store.close() }
  })
})
