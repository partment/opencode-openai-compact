import { afterEach, describe, expect, test, vi } from "vitest"
import { mkdtemp, rm } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { createCompactHooks } from "../src/compact.js"
import { defaultConfig, OpenAICompactConfigSchema } from "../src/schema.js"
import { CheckpointStore, type Checkpoint } from "../src/state.js"
import { compactionFixture } from "./compaction-fixture.js"

const roots: string[] = []
const sessionID = "ses_shared"
const url = "https://proxy.test/v1/responses"
const dayMs = 24 * 60 * 60 * 1000

afterEach(async () => {
  vi.useRealTimers()
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

function checkpoint(createdAt = Date.now(), responseID = "resp_shared"): Checkpoint {
  return {
    providerID: "openai", responseID, afterMessageID: "msg_boundary",
    afterCreatedAt: createdAt, createdAt,
    items: [
      { role: "user", content: "shared history" },
      { type: "compaction", encrypted_content: responseID },
    ],
  }
}

function history(markedControl = false) {
  return [
    {
      info: {
        id: "msg_boundary", sessionID, role: "user", time: { created: Date.now() },
        model: { providerID: "openai", modelID: "gpt" },
      },
      parts: [{ type: "compaction", messageID: "msg_boundary", sessionID }],
    },
    ...(markedControl ? [{
      info: {
        id: "msg_control", sessionID, role: "user", time: { created: Date.now() + 1 },
        model: { providerID: "openai", modelID: "gpt" },
      },
      parts: [{
        type: "text", text: "continue", synthetic: true, metadata: { compaction_continue: true },
        messageID: "msg_control", sessionID,
      }],
    }] : []),
    {
      info: {
        id: "msg_current", sessionID, role: "user", time: { created: Date.now() + 2 },
        model: { providerID: "openai", modelID: "gpt" },
      },
      parts: [{ type: "text", text: "current question", messageID: "msg_current", sessionID }],
    },
  ]
}

function completed(id: string) {
  return new Response([
    { type: "response.output_item.done", item: { type: "compaction", encrypted_content: id } },
    { type: "response.completed", response: { id, model: "gpt" } },
  ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""))
}

async function stores() {
  const root = await mkdtemp(path.join(os.tmpdir(), "openai-compact-multi-"))
  roots.push(root)
  const file = path.join(root, "checkpoints.db")
  return {
    a: await CheckpointStore.open(file),
    b: await CheckpointStore.open(file),
  }
}

async function hook(store: CheckpointStore, network = vi.fn(async () => new Response("ok")), options: any = {}) {
  const hooks = createCompactHooks(defaultConfig, store, network as typeof fetch, options)
  const cfg: any = {}
  await hooks.config?.(cfg)
  return { hooks, cfg, network }
}

async function select(hooks: ReturnType<typeof createCompactHooks>, messages = history()) {
  const transformed = structuredClone(messages)
  await hooks["experimental.chat.messages.transform"]?.({}, { messages: transformed } as any)
  return transformed
}

describe("multi-instance SQLite consistency", () => {
  test("sees a checkpoint added by another instance before the session is used", async () => {
    const s = await stores()
    const b = await hook(s.b)
    try {
      s.a.upsert(sessionID, checkpoint())
      expect((await select(b.hooks)).map((message) => message.info.id)).toEqual(["msg_current"])
      await b.cfg.provider.openai.options.fetch(url, {
        method: "POST", headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "current question" }] }),
      })
      expect(JSON.parse(b.network.mock.calls[0][1]!.body as string).input).toEqual([
        ...checkpoint().items,
        { role: "user", content: "current question" },
      ])
    } finally { s.b.close(); s.a.close() }
  })

  test("keeps an already selected older checkpoint when another instance adds a newer one", async () => {
    const s = await stores()
    const b = await hook(s.b)
    try {
      const old = checkpoint(Date.now(), "resp_old")
      s.a.upsert(sessionID, old)
      await select(b.hooks)
      s.a.upsert(sessionID, {
        ...checkpoint(Date.now() + 10, "resp_newer"),
        afterMessageID: "msg_other_boundary",
        items: [{ type: "compaction", encrypted_content: "newer" }],
      })

      await b.cfg.provider.openai.options.fetch(url, {
        method: "POST", headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "current question" }] }),
      })
      expect(JSON.parse(b.network.mock.calls[0][1]!.body as string).input).toEqual([
        ...old.items,
        { role: "user", content: "current question" },
      ])
    } finally { s.b.close(); s.a.close() }
  })

  test("does not inject a cached checkpoint after another instance prunes it", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"))
    const s = await stores()
    const b = await hook(s.b)
    try {
      s.a.upsert(sessionID, checkpoint())
      expect((await select(b.hooks)).map((message) => message.info.id)).toEqual(["msg_current"])
      vi.setSystemTime(new Date(Date.now() + 31 * dayMs))
      s.a.prune(30)

      await b.cfg.provider.openai.options.fetch(url, {
        method: "POST", headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "current question" }] }),
      })
      expect(JSON.parse(b.network.mock.calls[0][1]!.body as string).input).toEqual([
        { role: "user", content: "current question" },
      ])
    } finally { s.b.close(); s.a.close() }
  })

  test.each([true, false])("propagates deletion when deleteOnSessionDeleted is %s", async (deleteData) => {
    const s = await stores()
    const config = OpenAICompactConfigSchema.parse({ state: { deleteOnSessionDeleted: deleteData } })
    const network = vi.fn(async () => new Response("should not run"))
    const hooks = createCompactHooks(config, s.b, network as typeof fetch)
    const cfg: any = {}
    await hooks.config?.(cfg)
    try {
      s.a.upsert(sessionID, checkpoint())
      await select(hooks)
      s.a.invalidateSessionState(sessionID, { deleted: true, deleteData })

      const response = await cfg.provider.openai.options.fetch(url, {
        method: "POST", headers: { [config.headers.session]: sessionID },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "current question" }] }),
      })
      expect(response.status).toBe(400)
      expect(network).not.toHaveBeenCalled()
      expect(s.a.loadAll()).toHaveLength(deleteData ? 0 : 1)
      expect(s.b.loadSession(sessionID, 30).deleted).toBe(true)
    } finally { s.b.close(); s.a.close() }
  })

  test("reloads external control identities and removals", async () => {
    const s = await stores()
    const raw = history(true)
    const b = await hook(s.b, undefined, { getSessionMessages: async () => raw })
    try {
      s.a.upsert(sessionID, checkpoint())
      s.a.upsertControlMessage({
        providerID: "openai", sessionID, messageID: "msg_control",
        createdAt: raw[1].info.time.created, contentText: "continue",
      })
      expect((await select(b.hooks, raw)).map((message) => message.info.id)).toEqual(["msg_current"])

      s.a.deleteControlMessage(sessionID, "msg_control")
      const markerless = structuredClone(raw)
      delete markerless[1].parts[0].synthetic
      delete markerless[1].parts[0].metadata
      expect((await select(b.hooks, markerless)).map((message) => message.info.id)).toEqual(["msg_control", "msg_current"])
    } finally { s.b.close(); s.a.close() }
  })

  test("fails a transform instead of trimming with stale cache when refresh fails", async () => {
    const s = await stores()
    const b = await hook(s.b)
    try {
      s.a.upsert(sessionID, checkpoint())
      await select(b.hooks)
      vi.spyOn(s.b, "loadSession").mockImplementation(() => { throw new Error("database unavailable") })
      await expect(select(b.hooks)).rejects.toThrow("OpenAI compact session state could not be refreshed safely")
    } finally { s.b.close(); s.a.close() }
  })

  test("fails closed instead of using cache when refresh fails", async () => {
    const s = await stores()
    const b = await hook(s.b)
    try {
      s.a.upsert(sessionID, checkpoint())
      await select(b.hooks)
      vi.spyOn(s.b, "loadSession").mockImplementation(() => { throw new Error("database unavailable") })

      const response = await b.cfg.provider.openai.options.fetch(url, {
        method: "POST", headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "current question" }] }),
      })
      expect(response.status).toBe(503)
      expect(await response.text()).toContain("could not be refreshed")
      expect(b.network).not.toHaveBeenCalled()
    } finally { s.b.close(); s.a.close() }
  })

  test("allows both processes to reach the API but only the current revision can commit", async () => {
    const s = await stores()
    const fixtureA = compactionFixture()
    const fixtureB = compactionFixture()
    let resolveA!: (response: Response) => void
    let startedA!: () => void
    const started = new Promise<void>((resolve) => { startedA = resolve })
    const networkA = vi.fn(async () => {
      startedA()
      return new Promise<Response>((resolve) => { resolveA = resolve })
    })
    const networkB = vi.fn(async () => completed("resp_B"))
    const a = await hook(s.a, networkA, {
      getSessionMessages: fixtureA.getSessionMessages, getSessionStatus: fixtureA.getSessionStatus,
    })
    const b = await hook(s.b, networkB, {
      getSessionMessages: fixtureB.getSessionMessages, getSessionStatus: fixtureB.getSessionStatus,
    })
    const input = [{
      info: {
        id: "msg_user", sessionID, role: "user", time: { created: Date.now() },
        model: { providerID: "openai", modelID: "gpt" },
      },
      parts: [{ type: "text", text: "history", messageID: "msg_user", sessionID }],
    }]
    const body = JSON.stringify({
      model: "gpt", instructions: "You are an anchored context summarization assistant for coding sessions.",
      input: [{ role: "user", content: "history" }],
    })
    try {
      const headersA = await fixtureA.capture(a.hooks, {
        sessionID, history: structuredClone(input), boundary: { id: "boundary_A" },
        model: { providerID: "openai", id: "gpt" },
      })
      const pendingA = a.cfg.provider.openai.options.fetch(url, { method: "POST", headers: headersA, body }) as Promise<Response>
      await started

      const headersB = await fixtureB.capture(b.hooks, {
        sessionID, history: structuredClone(input), boundary: { id: "boundary_B", time: { created: Date.now() + 10 } },
        model: { providerID: "openai", id: "gpt" },
      })
      expect((await b.cfg.provider.openai.options.fetch(url, { method: "POST", headers: headersB, body })).status).toBe(200)
      resolveA(completed("resp_A"))
      expect((await pendingA).status).toBe(409)

      expect(networkA).toHaveBeenCalledOnce()
      expect(networkB).toHaveBeenCalledOnce()
      expect(s.a.loadAll().map(({ checkpoint }) => checkpoint.responseID)).toEqual(["resp_B"])
    } finally { s.b.close(); s.a.close() }
  })
})
