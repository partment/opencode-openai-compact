import { describe, expect, test, vi } from "vitest"
import { createCompactHooks } from "../src/compact.js"
import { defaultConfig } from "../src/schema.js"
import { CheckpointStore } from "../src/state.js"
import { compactionFixture } from "./compaction-fixture.js"

const sessionID = "ses_persistence"
const url = "https://proxy.test/v1/responses"

function completed(id = "resp_new") {
  return new Response([
    { type: "response.output_item.done", item: { type: "compaction", encrypted_content: `checkpoint-${id}` } },
    { type: "response.completed", response: { id, model: "gpt" } },
  ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""))
}

function jsonBody(init: RequestInit | undefined) {
  return JSON.parse(typeof init?.body === "string" ? init.body : "{}")
}

async function setup() {
  const store = CheckpointStore.openMemory()
  const now = Date.now()
  const oldItems = [
    { role: "user", content: "old durable history" },
    { type: "compaction", encrypted_content: "old-checkpoint" },
  ]
  store.upsert(sessionID, {
    providerID: "openai", responseID: "resp_old", afterMessageID: "msg_old_boundary",
    afterCreatedAt: now, createdAt: now, items: oldItems,
  })
  const fixture = compactionFixture()
  const network = vi.fn(async () => completed())
  const hooks = createCompactHooks(defaultConfig, store, network as typeof fetch, {
    getSessionMessages: fixture.getSessionMessages,
    getSessionStatus: fixture.getSessionStatus,
  })
  const cfg: any = {}
  await hooks.config?.(cfg)
  const history = [
    {
      info: {
        id: "msg_old_boundary", sessionID, role: "user", time: { created: now },
        model: { providerID: "openai", modelID: "gpt" },
      },
      parts: [{ type: "compaction", messageID: "msg_old_boundary", sessionID }],
    },
    {
      info: {
        id: "msg_tail", sessionID, role: "user", time: { created: now + 1 },
        model: { providerID: "openai", modelID: "gpt" },
      },
      parts: [{ type: "text", text: "new durable constraint", messageID: "msg_tail", sessionID }],
    },
  ]
  const headers = await fixture.capture(hooks, {
    sessionID, history, model: { providerID: "openai", id: "gpt" },
    boundary: { id: "msg_new_boundary", time: { created: now + 2 } },
  })
  const body = JSON.stringify({
    model: "gpt",
    instructions: "You are an anchored context summarization assistant for coding sessions.",
    input: [{ role: "user", content: "flattened history must be replaced" }],
  })
  const send = () => cfg.provider.openai.options.fetch(url, { method: "POST", headers, body }) as Promise<Response>
  return { store, fixture, network, hooks, cfg, history, headers, body, oldItems, send }
}

describe("checkpoint persistence boundary", () => {
  test("keeps the old active checkpoint until durable commit and retries only SQLite", async () => {
    const f = await setup()
    const commit = vi.spyOn(f.store, "commitCheckpoint")
    commit.mockImplementationOnce(() => { throw new Error("disk full") })
    try {
      const failed = await f.send()
      expect(failed.status).toBe(503)
      expect(await failed.text()).toContain("could not be persisted")
      expect(f.network).toHaveBeenCalledTimes(1)
      expect(f.store.loadAll().map(({ checkpoint }) => checkpoint.responseID)).toEqual(["resp_old"])

      f.network.mockResolvedValueOnce(new Response("ordinary"))
      await f.cfg.provider.openai.options.fetch(url, {
        method: "POST",
        headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "after failed commit" }] }),
      })
      expect(jsonBody(f.network.mock.calls.at(-1)![1]).input).toEqual([
        ...f.oldItems,
        { role: "user", content: "after failed commit" },
      ])
      expect(f.store.loadAll().map(({ checkpoint }) => checkpoint.responseID)).toEqual(["resp_old"])

      const retried = await f.send()
      expect(retried.status).toBe(200)
      expect(f.network).toHaveBeenCalledTimes(2)
      expect(commit).toHaveBeenCalledTimes(2)
      expect(f.store.loadAll().map(({ checkpoint }) => checkpoint.responseID)).toEqual(["resp_old", "resp_new"])
    } finally { f.store.close() }
  })

  test("recognizes an exact checkpoint when commit succeeded before the caller saw an error", async () => {
    const f = await setup()
    const original = f.store.commitCheckpoint.bind(f.store)
    const commit = vi.spyOn(f.store, "commitCheckpoint")
    commit.mockImplementationOnce((...args) => {
      original(...args)
      throw new Error("connection failed after commit")
    })
    try {
      expect((await f.send()).status).toBe(503)
      expect(f.store.loadAll().map(({ checkpoint }) => checkpoint.responseID)).toEqual(["resp_old", "resp_new"])
      expect(f.network).toHaveBeenCalledOnce()

      expect((await f.send()).status).toBe(200)
      expect(f.network).toHaveBeenCalledOnce()
      expect(commit).toHaveBeenCalledTimes(2)
    } finally { f.store.close() }
  })

  test("does not accept an exact post-commit retry after a later session mutation", async () => {
    const f = await setup()
    const original = f.store.commitCheckpoint.bind(f.store)
    vi.spyOn(f.store, "commitCheckpoint").mockImplementationOnce((...args) => {
      original(...args)
      throw new Error("connection failed after commit")
    })
    try {
      expect((await f.send()).status).toBe(503)
      f.store.invalidateSessionState(sessionID)

      expect((await f.send()).status).toBe(409)
      expect(f.network).toHaveBeenCalledOnce()
    } finally { f.store.close() }
  })

  test("rejects a pending local commit when persisted session state changed", async () => {
    const f = await setup()
    const commit = vi.spyOn(f.store, "commitCheckpoint")
    commit.mockImplementationOnce(() => { throw new Error("temporarily unavailable") })
    try {
      expect((await f.send()).status).toBe(503)
      expect(f.network).toHaveBeenCalledOnce()
      f.store.invalidateSessionState(sessionID)

      const retried = await f.send()
      expect(retried.status).toBe(409)
      expect(await retried.text()).toContain("session state changed")
      expect(f.network).toHaveBeenCalledOnce()
      expect(f.store.loadAll().map(({ checkpoint }) => checkpoint.responseID)).toEqual(["resp_old"])
    } finally { f.store.close() }
  })

  test("does not prune during hook creation or checkpoint commit", async () => {
    const f = await setup()
    const prune = vi.spyOn(f.store, "prune")
    try {
      expect((await f.send()).status).toBe(200)
      expect(prune).not.toHaveBeenCalled()
    } finally { f.store.close() }
  })

  test("retries a deletion tombstone after a transient database failure", async () => {
    const store = CheckpointStore.openMemory()
    store.upsert(sessionID, {
      providerID: "openai", responseID: "resp_keep", afterMessageID: "msg_boundary",
      afterCreatedAt: Date.now(), createdAt: Date.now(),
      items: [{ type: "compaction", encrypted_content: "keep" }],
    })
    const hooks = createCompactHooks(defaultConfig, store)
    const invalidate = vi.spyOn(store, "invalidateSessionState")
    invalidate.mockImplementationOnce(() => { throw new Error("database unavailable") })
    const event = { type: "session.deleted", properties: { sessionID } } as any
    try {
      await expect(hooks.event?.({ event })).rejects.toThrow("could not be refreshed")
      expect(store.loadSession(sessionID, 30).deleted).toBe(false)

      await hooks.event?.({ event })
      expect(store.loadSession(sessionID, 30).deleted).toBe(true)
      expect(store.loadAll()).toEqual([])
    } finally { store.close() }
  })

  test("does not remove a control turn when its identity cannot be persisted", async () => {
    const store = CheckpointStore.openMemory()
    const commit = vi.spyOn(store, "commitControlMessages").mockImplementation(() => { throw new Error("disk full") })
    const hooks = createCompactHooks(defaultConfig, store)
    const message = {
      info: {
        id: "msg_control", sessionID, role: "user", time: { created: Date.now() },
        model: { providerID: "openai", modelID: "gpt" },
      },
      parts: [{
        type: "text", text: "continue", synthetic: true, metadata: { compaction_continue: true },
        messageID: "msg_control", sessionID,
      }],
    }
    try {
      const messages = [structuredClone(message)]
      await hooks["experimental.chat.messages.transform"]?.({}, { messages } as any)
      expect(messages).toEqual([message])
      expect(store.loadControlMessages()).toEqual([])
      expect(commit).toHaveBeenCalledOnce()
    } finally { store.close() }
  })
})
