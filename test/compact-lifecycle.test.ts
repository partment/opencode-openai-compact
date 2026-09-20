import { describe, expect, test, vi } from "vitest"
import { createCompactHooks } from "../src/compact.js"
import { openAIOAuthDummyKey } from "../src/oauth.js"
import { OpenAICompactConfigSchema } from "../src/schema.js"
import { CheckpointStore } from "../src/state.js"
import { compactionFixture } from "./compaction-fixture.js"

const sessionID = "ses_lifecycle"
const url = "https://proxy.test/v1/responses"
const operationHeader = "x-opencode-openai-compact-operation"
const body = JSON.stringify({
  model: "gpt", instructions: "You are an anchored context summarization assistant for coding sessions.",
  input: [{ role: "user", content: "Here is the conversation so far:\n<conversation>\noriginal history\n</conversation>" }],
})

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

function completed(id = "resp_A") {
  return new Response([
    { type: "response.output_item.done", item: { type: "compaction", encrypted_content: `checkpoint-${id}` } },
    { type: "response.completed", response: { id } },
  ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""))
}

async function setup(options: {
  deletion?: boolean; seed?: boolean; native?: boolean; attachments?: boolean; tokenFetch?: typeof fetch;
} = {}) {
  const fixture = compactionFixture()
  const config = OpenAICompactConfigSchema.parse({
    providers: { openai: {}, other: {} }, state: { deleteOnSessionDeleted: options.deletion ?? true },
  })
  const store = CheckpointStore.openMemory()
  const now = Date.now()
  if (options.seed || options.native) {
    store.upsert(sessionID, {
      providerID: "openai", responseID: "resp_existing", afterMessageID: "msg_existing",
      afterCreatedAt: now - 100, createdAt: now,
      items: [
        { role: "user", content: options.native
          ? "Here is the conversation so far:\n<conversation>\nold history\n</conversation>" : "existing history" },
        { type: "compaction", encrypted_content: "existing-checkpoint" },
      ],
    })
    store.upsertControlMessage({ providerID: "openai", sessionID, messageID: "msg_control", createdAt: now, contentText: "control" })
  }
  const source = { read: fixture.getSessionMessages, status: fixture.getSessionStatus }
  const network = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => completed())
  const hooks = createCompactHooks(config, store, network as typeof fetch, {
    getSessionMessages: (id) => source.read(id), getSessionStatus: (id) => source.status(id), tokenFetch: options.tokenFetch,
  })
  const cfg: any = {}
  await hooks.config?.(cfg)
  let count = 0
  async function prepare(providerID = "openai") {
    count++
    return fixture.prepare({
      sessionID, model: { providerID, id: "gpt" },
      boundary: { id: `boundary_${count}`, time: { created: now + count * 10 } },
      history: [{
        info: {
          id: `msg_user_${count}`, sessionID, role: "user", time: { created: now + count * 10 - 1 },
          model: { providerID, modelID: "gpt" },
        },
        parts: [
          { type: "text", text: `history ${count}` },
          ...(options.attachments ? [{ type: "file", mime: "image/png", url: "data:image/png;base64,AA==" }] : []),
        ],
      }],
    })
  }
  async function headers(prepared: Awaited<ReturnType<typeof prepare>>) {
    const output = { headers: {} as Record<string, string> }
    await hooks["chat.headers"]?.(prepared.request as any, output)
    return output.headers
  }
  async function begin(providerID = "openai") {
    const prepared = await prepare(providerID)
    await hooks["experimental.session.compacting"]?.({ sessionID }, { context: [] })
    await hooks["experimental.chat.messages.transform"]?.({}, { messages: prepared.history } as any)
    fixture.addSummary(prepared)
    return { prepared, headers: await headers(prepared) }
  }
  const send = (headers: Record<string, string>, providerID = "openai", extra: RequestInit = {}) =>
    cfg.provider[providerID].options.fetch(url, { method: "POST", body, headers, ...extra }) as Promise<Response>
  const event = (type: string, properties: Record<string, unknown> = {}) =>
    hooks.event!({ event: { type, properties: { sessionID, ...properties } } as any })
  const conflict = () => hooks["experimental.session.compacting"]!({ sessionID }, { context: [] })
  return { fixture, config, store, source, network, hooks, cfg, prepare, headers, begin, send, event, conflict, now }
}

describe("session compaction lifecycle", () => {
  test.each(["capture", "headers", "API", "stream", "429", "network", "API-completed"])(
    "rejects B without replacing A while A is in %s", async (phase) => {
      const f = await setup()
      const gate = deferred<Response>()
      const started = deferred<void>()
      let pending: Promise<Response> | undefined
      let stream!: ReadableStreamDefaultController<Uint8Array>
      try {
        const prepared = await f.prepare()
        const raw = f.fixture.sessions.get(sessionID)
        const readGate = deferred<unknown>()
        if (phase === "capture") f.source.read = async () => { started.resolve(); return readGate.promise }
        const capture = f.hooks["experimental.session.compacting"]!({ sessionID }, { context: [] })
        if (phase === "capture") {
          await started.promise
          await expect(f.conflict()).rejects.toThrow("already in progress")
          f.source.read = f.fixture.getSessionMessages
          readGate.resolve(raw)
        }
        await capture
        await f.hooks["experimental.chat.messages.transform"]?.({}, { messages: prepared.history } as any)
        f.fixture.addSummary(prepared)
        if (phase === "headers") await expect(f.conflict()).rejects.toThrow("already in progress")
        const headers = await f.headers(prepared)
        expect(headers[operationHeader]).toBeTruthy()
        if (phase === "API") {
          f.network.mockImplementationOnce(async () => { started.resolve(); return gate.promise })
          pending = f.send(headers)
          await started.promise
        }
        if (phase === "stream") {
          f.network.mockImplementationOnce(async () => new Response(new ReadableStream({
            start(controller) { stream = controller; started.resolve() },
          })))
          pending = f.send(headers)
          await started.promise
        }
        if (phase === "429") {
          f.network.mockResolvedValueOnce(new Response("limited", { status: 429 }))
          expect((await f.send(headers)).status).toBe(429)
          f.fixture.statuses.set(sessionID, "retry")
        }
        if (phase === "network") {
          f.network.mockRejectedValueOnce(new TypeError("connection reset"))
          await expect(f.send(headers)).rejects.toThrow("connection reset")
        }
        if (phase === "API-completed") expect((await f.send(headers)).status).toBe(200)
        const callCount = f.network.mock.calls.length
        await expect(f.conflict()).rejects.toThrow("already in progress")
        expect(await f.headers(prepared)).toEqual(headers)
        expect(f.network).toHaveBeenCalledTimes(callCount)
        expect(f.network.mock.calls.every((call) => !call[1]?.signal?.aborted)).toBe(true)
        gate.resolve(completed())
        if (phase === "stream") {
          stream.enqueue(new TextEncoder().encode(await completed().text()))
          stream.close()
        }
        expect((await (pending ?? f.send(headers))).status).toBe(200)
        expect(f.store.loadAll()[0].checkpoint.afterMessageID).toBe(prepared.request.message.id)
        await f.fixture.finish(f.hooks, sessionID)
        const next = await f.begin("other")
        f.network.mockResolvedValueOnce(completed("resp_B"))
        expect((await f.send(next.headers, "other")).status).toBe(200)
        expect(f.store.loadAll().map(({ checkpoint }) => [checkpoint.responseID, checkpoint.afterMessageID])).toEqual([
          ["resp_A", prepared.request.message.id], ["resp_B", next.prepared.request.message.id],
        ])
        expect((await f.send(headers)).status).toBe(400)
      } finally { f.store.close() }
    },
  )

  test("a persisted B boundary does not prevent binding or change the boundary of A", async () => {
    const f = await setup()
    try {
      const prepared = await f.prepare()
      await f.hooks["experimental.session.compacting"]?.({ sessionID }, { context: [] })
      await f.hooks["experimental.chat.messages.transform"]?.({}, { messages: prepared.history } as any)
      f.fixture.addSummary(prepared)
      f.fixture.sessions.get(sessionID)!.push({
        info: { ...prepared.request.message, id: "boundary_B", time: { created: f.now + 100 } },
        parts: [{ type: "compaction", sessionID, messageID: "boundary_B" }],
      })
      await expect(f.conflict()).rejects.toThrow("already in progress")
      const headers = await f.headers(prepared)
      expect(headers[operationHeader]).toBeTruthy()
      expect((await f.send(headers)).status).toBe(200)
      expect(f.store.loadAll()[0].checkpoint.afterMessageID).toBe(prepared.request.message.id)
    } finally { f.store.close() }
  })

  test.each(["busy", "retry", "unknown", "failure"])("idle hints cannot unlock a session whose current status is %s", async (status) => {
    const f = await setup()
    try {
      const a = await f.begin()
      f.source.status = async () => {
        if (status === "failure") throw new Error("status unavailable")
        return status === "unknown" ? undefined as any : status as "busy" | "retry"
      }
      for (const type of ["session.idle", "session.status", "session.error", "session.compacted"]) {
        await f.event(type, { status: { type: "idle" } })
        await expect(f.conflict()).rejects.toThrow("already in progress")
      }
      expect((await f.send(a.headers)).status).toBe(200)
    } finally { f.store.close() }
  })

  test.each(["before-summary", "after-summary", "final-failure"])("releases only the cancelled or failed A (%s)", async (phase) => {
    const f = await setup()
    const gate = deferred<Response>()
    const started = deferred<void>()
    try {
      let headers: Record<string, string> | undefined
      let pending: Promise<Response> | undefined
      if (phase === "before-summary") {
        await f.prepare()
        await f.hooks["experimental.session.compacting"]?.({ sessionID }, { context: [] })
      } else {
        headers = (await f.begin()).headers
        f.network.mockImplementationOnce(async () => { started.resolve(); return gate.promise })
        pending = f.send(headers)
        await started.promise
      }
      if (phase === "final-failure") await f.fixture.finish(f.hooks, sessionID, { name: "APIError" })
      else await f.fixture.cancel(f.hooks, sessionID)
      if (pending) expect(f.network.mock.calls[0][1]?.signal?.aborted).toBe(true)
      const b = await f.begin()
      gate.resolve(completed("resp_late_A"))
      if (pending) expect((await pending).status).toBe(400)
      if (headers) expect((await f.send(headers)).status).toBe(400)
      expect(f.store.count()).toBe(0)
      expect((await f.send(b.headers)).status).toBe(200)
      expect(f.store.loadAll()[0].checkpoint.afterMessageID).toBe(b.prepared.request.message.id)
    } finally { f.store.close() }
  })

  test("recovers a missed summary completion event before admitting B", async () => {
    const f = await setup()
    try {
      const a = await f.begin()
      await f.send(a.headers)
      const summary = f.fixture.sessions.get(sessionID)!.at(-1).info
      summary.finish = "stop"
      summary.time.completed = Date.now()
      // Reconcile before preparing B, since the test fixture replaces raw history on prepare.
      await f.event("session.compacted")
      const b = await f.begin()
      expect((await f.send(b.headers)).status).toBe(200)
    } finally { f.store.close() }
  })

  test("a late status query cannot release B or revive a deleted session", async () => {
    const f = await setup()
    try {
      await f.begin()
      const gate = deferred<"idle">()
      const started = deferred<void>()
      f.source.status = async () => { started.resolve(); return gate.promise }
      const pending = f.event("session.idle")
      await started.promise
      await f.event("session.updated", { info: { id: sessionID, revert: { messageID: "msg_undo" } } })
      f.source.status = f.fixture.getSessionStatus
      const b = await f.begin()
      gate.resolve("idle")
      await pending
      await expect(f.conflict()).rejects.toThrow("already in progress")
      expect((await f.send(b.headers)).status).toBe(200)
      await f.event("session.deleted")
      await expect(f.conflict()).rejects.toThrow("no longer valid")
      expect(f.store.count()).toBe(0)
    } finally { f.store.close() }
  })

  test.each([true, false])("deletion prevents late writes while preserving configured retention (%s)", async (deletion) => {
    const f = await setup({ deletion, seed: true })
    const gate = deferred<Response>()
    const started = deferred<void>()
    try {
      const before = f.store.loadAll()
      const controls = f.store.loadControlMessages()
      const a = await f.begin()
      const upsert = vi.spyOn(f.store, "commitCheckpoint")
      f.network.mockImplementationOnce(async () => { started.resolve(); return gate.promise })
      const pending = f.send(a.headers)
      await started.promise
      await f.event("session.deleted")
      expect(f.network.mock.calls[0][1]?.signal?.aborted).toBe(true)
      expect(f.store.loadAll()).toEqual(deletion ? [] : before)
      gate.resolve(completed()) // Deliberately ignores AbortSignal.
      expect((await pending).status).toBe(400)
      await f.event("message.part.updated", { part: { type: "text", text: f.config.summary, messageID: "late" }, time: Date.now() })
      await f.hooks["experimental.compaction.autocontinue"]?.(a.prepared.request as any, { enabled: true })
      await f.hooks["experimental.chat.messages.transform"]?.({}, { messages: a.prepared.history } as any)
      await f.hooks["chat.message"]?.({ sessionID } as any, { message: a.prepared.request.message, parts: [] } as any)
      expect(f.store.loadAll()).toEqual(deletion ? [] : before)
      expect(f.store.loadControlMessages()).toEqual(deletion ? [] : controls)
      expect(upsert).not.toHaveBeenCalled()
      expect((await f.send({ [f.config.headers.session]: sessionID })).status).toBe(400)
    } finally { f.store.close() }
  })

  test.each(["deleted", "dispose", "revert", "message-removed", "cancel"])("an already-open SSE cannot commit after %s", async (action) => {
    const f = await setup()
    let controller!: ReadableStreamDefaultController<Uint8Array>
    const started = deferred<void>()
    let disposed = false
    try {
      const a = await f.begin()
      const upsert = vi.spyOn(f.store, "commitCheckpoint")
      f.network.mockImplementationOnce(async () => new Response(new ReadableStream({
        start(value) { controller = value; started.resolve() },
      })))
      const pending = f.send(a.headers)
      await started.promise
      if (action === "deleted") await f.event("session.deleted")
      if (action === "cancel") await f.fixture.cancel(f.hooks, sessionID)
      if (action === "dispose") { await f.hooks.dispose?.(); disposed = true }
      if (action === "revert") await f.event("session.updated", { info: { id: sessionID, revert: { messageID: "msg_before" } } })
      if (action === "message-removed") await f.event("message.removed", { messageID: a.prepared.request.message.id })
      controller.enqueue(new TextEncoder().encode(await completed().text()))
      controller.close()
      expect((await pending).status).toBe(400)
      expect(upsert).not.toHaveBeenCalled()
      expect(f.network.mock.calls[0][1]?.signal?.aborted).toBe(true)
    } finally { if (!disposed) f.store.close() }
  })

  test.each(["attempt", "busy", "retry"])("a resumed %s makes an old idle lookup stale without changing operation identity", async (activity) => {
    const f = await setup()
    try {
      const a = await f.begin()
      f.network.mockResolvedValue(new Response("limited", { status: 429 }))
      await f.send(a.headers)
      const gate = deferred<"idle">()
      const started = deferred<void>()
      f.source.status = async () => { started.resolve(); return gate.promise }
      const pending = f.event("session.idle")
      await started.promise
      if (activity === "attempt") await f.send(a.headers)
      else await f.event("session.status", { status: { type: activity } })
      gate.resolve("idle")
      await pending
      f.source.status = f.fixture.getSessionStatus
      await expect(f.conflict()).rejects.toThrow("already in progress")
      expect(await f.headers(a.prepared)).toEqual(a.headers)
      f.network.mockResolvedValueOnce(completed())
      expect((await f.send(a.headers)).status).toBe(200)
      expect(new Set(f.network.mock.calls.map((call) => call[1]?.body)).size).toBe(1)
    } finally { f.store.close() }
  })

  test("a synchronous history/status callback failure is treated as unknown", async () => {
    const f = await setup()
    try {
      const a = await f.begin()
      f.source.read = () => { throw new Error("synchronous read") }
      f.source.status = () => { throw new Error("synchronous status") }
      await f.event("session.idle")
      await expect(f.conflict()).rejects.toThrow("already in progress")
      expect((await f.send(a.headers)).status).toBe(200)
    } finally { f.store.close() }
  })

  test("deletion during attachment rejection inspection cannot start the fallback", async () => {
    const f = await setup({ attachments: true })
    let controller!: ReadableStreamDefaultController<Uint8Array>
    const started = deferred<void>()
    try {
      const a = await f.begin()
      f.network.mockImplementationOnce(async () => new Response(new ReadableStream({
        start(value) { controller = value; started.resolve() },
      }), { status: 400 }))
      const pending = f.send(a.headers)
      await started.promise
      expect(f.network.mock.calls[0][1]?.body).toContain("input_image")
      await f.event("session.deleted")
      controller.enqueue(new TextEncoder().encode("input_image is not supported"))
      controller.close()
      expect((await pending).status).toBe(400)
      expect(f.network).toHaveBeenCalledTimes(1)
      expect(f.store.count()).toBe(0)
    } finally { f.store.close() }
  })

  test("deletion during OAuth refresh prevents the compaction request", async () => {
    const gate = deferred<Response>()
    const started = deferred<void>()
    const f = await setup({ tokenFetch: (async () => { started.resolve(); return gate.promise }) as typeof fetch })
    try {
      const a = await f.begin()
      await f.hooks.auth?.loader?.(async () => ({
        type: "oauth", access: "expired-access", refresh: "test-refresh", expires: 0, accountId: "test-account",
      }), {} as any)
      const pending = f.send({ ...a.headers, authorization: `Bearer ${openAIOAuthDummyKey}` })
      await started.promise
      await f.event("session.deleted")
      gate.resolve(Response.json({ access_token: "new", refresh_token: "rotated", expires_in: 3600 }))
      expect((await pending).status).toBe(400)
      expect(f.network).not.toHaveBeenCalled()
      expect(f.store.count()).toBe(0)
    } finally { f.store.close() }
  })

  test("deleting A does not invalidate another session's in-flight compaction", async () => {
    const f = await setup()
    const gate = deferred<Response>()
    const started = deferred<void>()
    try {
      const a = await f.begin()
      f.network.mockImplementationOnce(async () => { started.resolve(); return gate.promise })
      const pending = f.send(a.headers)
      await started.promise
      const other = "ses_other"
      const headers = await f.fixture.capture(f.hooks, {
        sessionID: other,
        history: [{ info: { id: "other_user", sessionID: other, role: "user" }, parts: [{ type: "text", text: "other history" }] }],
      })
      const otherGate = deferred<Response>()
      const otherStarted = deferred<void>()
      f.network.mockImplementationOnce(async () => { otherStarted.resolve(); return otherGate.promise })
      const otherPending = f.send(headers)
      await otherStarted.promise
      await f.event("session.deleted")
      gate.resolve(completed("resp_deleted"))
      expect((await pending).status).toBe(400)
      otherGate.resolve(completed("resp_other"))
      expect((await otherPending).status).toBe(200)
      expect(f.store.loadAll().map((row) => row.sessionID)).toEqual([other])
      expect(f.network.mock.calls[1][1]?.signal?.aborted).toBe(false)
    } finally { f.store.close() }
  })

  test.each(["A-first", "B-first"])("a cancelled A cannot affect B in %s completion order", async (order) => {
    const f = await setup()
    const gate = deferred<Response>()
    const started = deferred<void>()
    try {
      const a = await f.begin()
      const oldSummary = structuredClone(f.fixture.sessions.get(sessionID)!.at(-1).info)
      f.network.mockImplementationOnce(async () => { started.resolve(); return gate.promise })
      const pending = f.send(a.headers)
      await started.promise
      await f.fixture.cancel(f.hooks, sessionID)
      const b = await f.begin()
      f.network.mockResolvedValueOnce(completed("resp_B"))
      const upsert = vi.spyOn(f.store, "commitCheckpoint")
      if (order === "B-first") expect((await f.send(b.headers)).status).toBe(200)
      gate.resolve(completed("resp_late_A"))
      expect((await pending).status).toBe(400)
      if (order === "A-first") expect((await f.send(b.headers)).status).toBe(200)
      await f.event("message.updated", { info: { ...oldSummary, finish: "stop", time: { ...oldSummary.time, completed: Date.now() } } })
      await f.hooks["experimental.compaction.autocontinue"]?.(a.prepared.request as any, { enabled: true })
      expect((await f.headers(a.prepared))[operationHeader]).toBeUndefined()
      expect(upsert).toHaveBeenCalledTimes(1)
      expect(f.store.loadAll().map(({ checkpoint }) => [checkpoint.responseID, checkpoint.afterMessageID])).toEqual([
        ["resp_B", b.prepared.request.message.id],
      ])
      f.network.mockResolvedValueOnce(new Response("ordinary answer"))
      await f.send({ [f.config.headers.session]: sessionID }, "openai", {
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "after B" }] }),
      })
      const sent = JSON.parse(f.network.mock.calls.at(-1)![1]!.body as string)
      expect(sent.input.some((item: any) => item.encrypted_content === "checkpoint-resp_B")).toBe(true)
      expect(sent.input.at(-1)).toEqual({ role: "user", content: "after B" })
    } finally { f.store.close() }
  })

  test("native fetch cannot mark success if the caller aborted but fetch ignored it", async () => {
    const f = await setup({ native: true })
    const gate = deferred<Response>()
    const started = deferred<void>()
    try {
      const a = await f.begin()
      f.network.mockImplementationOnce(async () => { started.resolve(); return gate.promise })
      const controller = new AbortController()
      const pending = f.send(a.headers, "openai", { signal: controller.signal })
      const rejected = expect(pending).rejects.toThrow("cancelled attempt")
      await started.promise
      controller.abort(new Error("cancelled attempt"))
      gate.resolve(new Response("ignored abort"))
      await rejected
      await f.fixture.finish(f.hooks, sessionID)
      expect(f.store.count()).toBe(1)
    } finally { f.store.close() }
  })

  test("an unbound capture releases ownership without trimming its native history", async () => {
    const f = await setup({ seed: true })
    try {
      const prepared = await f.prepare()
      prepared.history.unshift({
        info: { id: "msg_existing", sessionID, role: "user", time: { created: f.now - 100 } },
        parts: [{ type: "compaction", sessionID, messageID: "msg_existing" }],
      })
      f.source.read = async () => { throw new Error("unavailable history") }
      await f.hooks["experimental.session.compacting"]?.({ sessionID }, { context: [] })
      const original = structuredClone(prepared.history)
      await f.hooks["experimental.chat.messages.transform"]?.({}, { messages: prepared.history } as any)
      expect(prepared.history).toEqual(original)
      expect((await f.headers(prepared))[operationHeader]).toBeUndefined()
      f.source.read = f.fixture.getSessionMessages
      const next = await f.begin()
      expect(next.headers[operationHeader]).toBeTruthy()
    } finally { f.store.close() }
  })

  test("ordinary session metadata does not cancel the owner", async () => {
    const f = await setup()
    try {
      const a = await f.begin()
      await f.event("session.updated", { info: { id: sessionID, title: "renamed", time: { updated: Date.now() } } })
      await expect(f.conflict()).rejects.toThrow("already in progress")
      expect((await f.send(a.headers)).status).toBe(200)
    } finally { f.store.close() }
  })

  test.each(["openai", "other"])("native fallback owns the session against %s until its summary ends", async (providerID) => {
    const f = await setup({ native: true })
    try {
      const a = await f.begin()
      expect(a.headers[f.config.headers.compact]).toBe("native")
      await expect(f.conflict()).rejects.toThrow("already in progress")
      expect((await f.send({ [f.config.headers.session]: sessionID, [f.config.headers.compact]: "1" }, providerID)).status).toBe(400)
      f.network.mockResolvedValueOnce(new Response("native summary"))
      expect((await f.send(a.headers)).status).toBe(200)
      await expect(f.conflict()).rejects.toThrow("already in progress")
      await f.fixture.finish(f.hooks, sessionID)
      expect(f.store.count()).toBe(0)
      const b = await f.begin(providerID)
      expect(b.headers[f.config.headers.compact]).toBe("1")
    } finally { f.store.close() }
  })

  test("native cancellation aborts the network without clearing its checkpoint", async () => {
    const f = await setup({ native: true })
    const started = deferred<void>()
    try {
      const a = await f.begin()
      f.network.mockImplementationOnce(async (_input, init) => {
        started.resolve()
        return new Promise((_resolve, reject) => init!.signal!.addEventListener("abort", () => reject(init!.signal!.reason), { once: true }))
      })
      const pending = f.send(a.headers)
      await started.promise
      await f.fixture.cancel(f.hooks, sessionID)
      expect((await pending).status).toBe(400)
      expect(f.store.count()).toBe(1)
      expect((await f.begin()).headers[f.config.headers.compact]).toBe("native")
    } finally { f.store.close() }
  })

  test.each(["AB", "BA"])("legacy callbacks in %s order cannot move fixed synthetic boundaries", async (order) => {
    const f = await setup()
    try {
      const headers = { [f.config.headers.session]: sessionID, [f.config.headers.compact]: "1" }
      const upsert = vi.spyOn(f.store, "commitCheckpoint")
      await f.send(headers)
      const first = f.store.loadAll()[0].checkpoint
      expect(first.afterMessageID).toMatch(/^msg_compact_/)
      f.network.mockResolvedValueOnce(completed("resp_B"))
      await f.send(headers, "openai", { body: `${body} ` })
      const original = f.store.loadAll()
      for (const name of order) {
        await f.event("message.part.updated", {
          part: { type: "text", messageID: `boundary_${name}`, text: f.config.summary }, time: f.now + 100,
        })
        await f.hooks["experimental.compaction.autocontinue"]?.({
          sessionID, model: { providerID: "openai" }, message: { id: `boundary_${name}`, time: { created: f.now + 100 } },
        } as any, { enabled: true })
        await f.event("session.compacted")
      }
      expect(f.store.loadAll()).toEqual(original)
      expect(new Set(original.map(({ checkpoint }) => checkpoint.afterMessageID)).size).toBe(2)
      expect(upsert).toHaveBeenCalledTimes(2)
      const output = { headers: {} as Record<string, string> }
      await f.hooks["chat.headers"]?.({
        sessionID, model: { providerID: "openai" }, agent: "build",
        message: { id: "new_question", sessionID, role: "user", agent: "build", time: { created: f.now + 200 } },
      } as any, output)
      f.network.mockResolvedValueOnce(new Response("ordinary answer"))
      const response = await f.send(output.headers, "openai", {
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "genuine new question" }] }),
      })
      expect(await response.text()).toBe("ordinary answer")
      expect(JSON.parse(f.network.mock.calls.at(-1)![1]!.body as string).input.at(-1)).toEqual({ role: "user", content: "genuine new question" })
    } finally { f.store.close() }
  })

  test.each([
    "child-read-delete", "source-read-delete-child", "source-read-delete-source", "source-read-remove-source",
    "source-part-update", "child-part-update", "revert", "dispose", "valid",
  ])(
    "fork inheritance without a capture checks session generations and identity revisions (%s)", async (action) => {
      const store = CheckpointStore.openMemory()
      const config = OpenAICompactConfigSchema.parse({})
      const now = Date.now()
      const history = (id: string) => [
        { info: { id: `${id}_boundary`, sessionID: id, role: "user", model: { providerID: "openai", modelID: "gpt" }, time: { created: now } },
          parts: [{ type: "compaction", sessionID: id, messageID: `${id}_boundary` }] },
        { info: { id: `${id}_summary`, sessionID: id, role: "assistant", summary: true, parentID: `${id}_boundary`, time: { created: now + 1 } },
          parts: [{ type: "text", text: config.summary }] },
        { info: { id: `${id}_control`, sessionID: id, role: "user", model: { providerID: "openai", modelID: "gpt" }, time: { created: now + 2 } },
          parts: [{ type: "text", text: "markerless control", synthetic: true, metadata: { compaction_continue: true } }] },
      ]
      store.upsert("parent", {
        providerID: "openai", responseID: "resp_parent", afterMessageID: "parent_boundary",
        afterCreatedAt: now, createdAt: now, items: [{ type: "compaction", encrypted_content: "parent" }],
      })
      store.upsertControlMessage({ providerID: "openai", sessionID: "parent", messageID: "parent_control", createdAt: now + 2, contentText: "markerless control" })
      const gate = deferred<unknown>()
      const started = deferred<void>()
      const delayedSession = action === "child-read-delete" ? "child" : "parent"
      const network = vi.fn(async () => new Response("ordinary answer"))
      const hooks = createCompactHooks(config, store, network as typeof fetch, {
        async getSessionMessages(id) {
          if (id === delayedSession) { started.resolve(); return gate.promise }
          return history(id)
        },
      })
      let disposed = false
      try {
        const cfg: any = {}
        await hooks.config?.(cfg)
        const messages = history("child")
        const commit = vi.spyOn(store, "commitForkState")
        const pending = hooks["experimental.chat.messages.transform"]!({}, { messages } as any)
        await started.promise
        if (action.includes("delete")) await hooks.event?.({ event: {
          type: "session.deleted", properties: { sessionID: action.endsWith("source") ? "parent" : "child" },
        } as any })
        if (action === "source-read-remove-source") await hooks.event?.({ event: {
          type: "message.removed", properties: { sessionID: "parent", messageID: "parent_boundary" },
        } as any })
        if (action === "source-part-update" || action === "child-part-update") {
          const id = action === "source-part-update" ? "parent" : "child"
          await hooks.event?.({ event: { type: "message.part.updated", properties: {
            sessionID: id, part: { type: "text", messageID: `${id}_control`, sessionID: id, text: "now a real constraint" },
          } } as any })
        }
        if (action === "revert") await hooks.event?.({ event: {
          type: "session.updated", properties: { sessionID: "child", info: { id: "child", revert: { messageID: "child_boundary" } } },
        } as any })
        if (action === "dispose") { await hooks.dispose?.(); disposed = true }
        gate.resolve(history(delayedSession))
        await pending
        if (action === "valid") {
          expect(store.loadAll().find((entry) => entry.sessionID === "child")?.checkpoint.afterMessageID).toBe("child_boundary")
          expect(store.loadControlMessages().some((item) => item.sessionID === "child")).toBe(true)
          expect(messages).toEqual([])
        } else {
          expect(commit).not.toHaveBeenCalled()
          expect(messages).toEqual(history("child"))
          if (!disposed) {
            const response = await cfg.provider.openai.options.fetch(url, {
              method: "POST", headers: { [config.headers.session]: "child" },
              body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "new question" }] }),
            })
            if (action === "child-read-delete" || action === "source-read-delete-child") expect(response.status).toBe(400)
            else expect(JSON.parse((network.mock.calls[0] as any)[1].body).input).toEqual([{ role: "user", content: "new question" }])
          }
        }
      } finally { if (!disposed) store.close() }
    },
  )

  test("legacy retries retain ownership across providers", async () => {
    const f = await setup()
    try {
      const headers = { [f.config.headers.session]: sessionID, [f.config.headers.compact]: "1" }
      f.network.mockResolvedValueOnce(new Response("limited", { status: 429 }))
      expect((await f.send(headers)).status).toBe(429)
      expect((await f.send(headers, "other")).status).toBe(409)
      await expect(f.conflict()).rejects.toThrow("already in progress")
      expect((await f.send(headers)).status).toBe(200)
      expect(f.network).toHaveBeenCalledTimes(2)
    } finally { f.store.close() }
  })
})
