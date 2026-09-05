import { describe, expect, test, vi } from "vitest"
import { createCompactHooks } from "../src/compact.js"
import { OpenAICompactConfigSchema } from "../src/schema.js"
import { CheckpointStore } from "../src/state.js"
import { compactionFixture } from "./compaction-fixture.js"

const operationHeader = "x-opencode-openai-compact-operation"
const url = "https://proxy.test/v1/responses"
const sessionID = "ses_isolation"
const targetModel = (providerID: string) => ({ providerID, id: `${providerID}-model` })
const compactBody = JSON.stringify({
  model: "openai-model", instructions: "You are an anchored context summarization assistant for coding sessions.",
  input: [{ role: "user", content: "Here is the conversation so far:\n<conversation>\nflattened\n</conversation>" }],
})

function completed() {
  return new Response([
    { type: "response.output_item.done", item: { type: "compaction", encrypted_content: "new-checkpoint" } },
    { type: "response.completed", response: { id: "resp_isolation", created_at: 1 } },
  ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""))
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

async function setup(invalid = false) {
  const config = OpenAICompactConfigSchema.parse({ providers: { openai: {}, other: {} } })
  const store = CheckpointStore.openMemory()
  const fixture = compactionFixture()
  const source = { read: fixture.getSessionMessages }
  const sent: Array<{ input: RequestInfo | URL; init?: RequestInit }> = []
  const network = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    sent.push({ input, init })
    return JSON.stringify(init?.body).includes("compaction_trigger") ? completed() : new Response("ordinary answer")
  })
  const now = Date.now()
  const history: any[] = [{
    info: {
      id: "msg_history", sessionID, role: "user", agent: "build", time: { created: now - 20 },
      model: { providerID: "anthropic", modelID: "anthropic-model" },
    },
    parts: [{ type: "text", text: "private captured history" }],
  }, {
    info: {
      id: "msg_history_answer", sessionID, role: "assistant", time: { created: now - 10 },
      providerID: "anthropic", modelID: "anthropic-model",
    },
    parts: [{ type: "text", text: "private captured answer" }],
  }]
  if (invalid) store.upsert(sessionID, {
    providerID: "openai", responseID: "resp_invalid", afterMessageID: "msg_previous",
    afterCreatedAt: now - 100, createdAt: now,
    items: [
      { role: "user", content: "Here is the conversation so far:\n<conversation>\nold\n</conversation>" },
      { type: "compaction", encrypted_content: "invalid" },
    ],
  })
  const hooks = createCompactHooks(config, store, network as typeof fetch, {
    getSessionMessages: (id) => source.read(id),
    getSessionStatus: fixture.getSessionStatus,
  })
  const cfg: any = {}
  await hooks.config?.(cfg)
  let generation = 0
  async function prepare(providerID = "anthropic", messages = history) {
    generation++
    return fixture.prepare({
      sessionID, history: messages, model: targetModel(providerID),
      boundary: {
        id: `msg_boundary_${generation}`, time: { created: now + generation * 10 },
        model: { providerID: "anthropic", modelID: "anthropic-model" },
      },
    })
  }
  async function begin(providerID = "anthropic", messages = history) {
    const prepared = await prepare(providerID, messages)
    await hooks["experimental.session.compacting"]?.({ sessionID }, { context: [] })
    await hooks["experimental.chat.messages.transform"]?.({}, { messages: prepared.history } as any)
    fixture.addSummary(prepared)
    return prepared
  }
  async function headers(input: any) {
    const output = { headers: {} as Record<string, string> }
    await hooks["chat.headers"]?.(input, output)
    return output.headers
  }
  const user = {
    id: "msg_question", sessionID, role: "user", agent: "build", time: { created: now + 1000 },
    model: { providerID: "openai", modelID: "openai-model" },
  }
  const body = { model: "openai-model", input: [{ role: "user", content: "new genuine question" }] }
  async function ordinary(chatMessage = false, transform = false) {
    if (chatMessage) await hooks["chat.message"]?.(
      { sessionID, messageID: user.id, model: user.model } as any,
      { message: user, parts: [{ type: "text", text: "new genuine question" }] } as any,
    )
    if (transform) await hooks["experimental.chat.messages.transform"]?.({}, {
      messages: [{ info: user, parts: [{ type: "text", text: "new genuine question" }] }],
    } as any)
    const output = await headers({ sessionID, model: targetModel("openai"), agent: "build", message: user })
    expect(output[config.headers.compact]).toBeUndefined()
    expect(output[operationHeader]).toBeUndefined()
    const response = await cfg.provider.openai.options.fetch(url, {
      method: "POST", headers: output, body: JSON.stringify(body),
    })
    expect(await response.text()).toBe("ordinary answer")
    expect(JSON.parse(sent.at(-1)!.init!.body as string)).toEqual(body)
    expect(store.count()).toBe(0)
    return output
  }
  async function send(headers: Record<string, string>, providerID = "openai") {
    return cfg.provider[providerID].options.fetch(url, { method: "POST", headers, body: compactBody }) as Promise<Response>
  }
  return { config, store, hooks, cfg, fixture, source, network, sent, history, prepare, begin, headers, ordinary, send, user }
}

describe("compaction capture ownership", () => {
  test.each([false, true])("unsupported compaction cannot affect OpenAI (chat.message: %s)", async (chatMessage) => {
    const f = await setup()
    try {
      const prepared = await f.begin()
      expect(await f.headers(prepared.request)).toEqual({})
      await f.ordinary(chatMessage)
      expect(f.network).toHaveBeenCalledTimes(1)
    } finally { f.store.close() }
  })

  test.each(["missing", "delayed", "manual", "replay", "failed-clone", "failed-read"])(
    "unsupported capture is isolated with %s hooks/history", async (kind) => {
      const f = await setup()
      try {
        if (kind === "failed-read") f.source.read = async () => { throw new Error("unavailable") }
        if (kind === "failed-clone") f.history[0].parts[0].metadata = { uncloneable() {} }
        const prepared = await f.begin()
        if (["manual", "replay", "failed-clone", "failed-read"].includes(kind)) {
          expect(await f.headers(prepared.request)).toEqual({})
        }
        await f.ordinary()
        if (kind === "delayed") expect(await f.headers(prepared.request)).toEqual({})
        await f.ordinary(false, true)
        expect(f.network).toHaveBeenCalledTimes(2)
      } finally { f.store.close() }
    },
  )

  test.each(["wrong-message", "wrong-session", "wrong-time", "wrong-provider", "wrong-model", "utility"])(
    "%s headers cannot claim a configured capture", async (kind) => {
      const f = await setup()
      try {
        const prepared = await f.begin("openai")
        const input = structuredClone(prepared.request)
        if (kind === "wrong-message") input.message = f.user
        if (kind === "wrong-session") input.message.sessionID = "ses_foreign"
        if (kind === "wrong-time") input.message.time.created++
        if (kind === "wrong-provider") input.model = targetModel("other")
        if (kind === "wrong-model") input.model.id = "different-model"
        if (kind === "utility") input.agent = "title"
        const output = await f.headers(input)
        expect(output[f.config.headers.compact]).toBeUndefined()
        expect(output[operationHeader]).toBeUndefined()
        const real = await f.headers(prepared.request)
        expect(real[operationHeader]).toBeTruthy()
        expect((await f.send(real)).status).toBe(200)
        expect(f.sent[0].init!.body).toContain("private captured history")
        expect(f.sent[0].init!.body).not.toContain("flattened")
        expect(f.store.loadAll()[0].checkpoint.afterMessageID).toBe(prepared.request.message.id)
      } finally { f.store.close() }
    },
  )

  test.each(["missing-part", "foreign-part", "missing-id", "invalid-time", "duplicate", "foreign-session", "newer-user", "completed"])(
    "does not replace an unprovable boundary (%s) at headers time", async (kind) => {
      const f = await setup()
      try {
        const prepared = await f.prepare("openai")
        const raw = f.fixture.sessions.get(sessionID)!
        const valid = structuredClone(raw)
        const boundary = raw.at(-1)
        if (kind === "missing-part") boundary.parts = []
        if (kind === "foreign-part") boundary.parts[0].messageID = "msg_foreign"
        if (kind === "missing-id") delete boundary.info.id
        if (kind === "invalid-time") boundary.info.time.created = NaN
        if (kind === "duplicate") raw.push(structuredClone(boundary))
        if (kind === "foreign-session") boundary.info.sessionID = "ses_foreign"
        if (kind === "newer-user") raw.push({ info: f.user, parts: [{ type: "text", text: "new question" }] })
        if (kind === "completed") {
          f.fixture.addSummary(prepared)
          raw.at(-1).info.finish = "stop"
        }
        await f.hooks["experimental.session.compacting"]?.({ sessionID }, { context: [] })
        f.fixture.sessions.set(sessionID, valid)
        await f.hooks["experimental.chat.messages.transform"]?.({}, { messages: prepared.history } as any)
        f.fixture.addSummary(prepared)
        const output = await f.headers(prepared.request)
        expect(output[f.config.headers.compact]).toBeUndefined()
        await f.ordinary()
        const explicit = { [f.config.headers.session]: sessionID, [f.config.headers.compact]: "1" }
        expect((await f.send(explicit)).status).toBe(400)
        expect(f.network).toHaveBeenCalledTimes(1)
      } finally { f.store.close() }
    },
  )

  test.each(["absent", "old", "multiple", "wrong-parent", "wrong-agent", "wrong-provider", "wrong-model", "read-failure"])(
    "requires a new matching summary (%s)", async (kind) => {
      const f = await setup()
      try {
        const prepared = await f.prepare("openai")
        if (kind === "old") f.fixture.addSummary(prepared)
        await f.hooks["experimental.session.compacting"]?.({ sessionID }, { context: [] })
        await f.hooks["experimental.chat.messages.transform"]?.({}, { messages: prepared.history } as any)
        if (kind !== "old" && kind !== "absent") f.fixture.addSummary(prepared)
        const raw = f.fixture.sessions.get(sessionID)!
        const summary = raw.at(-1).info
        if (kind === "multiple") raw.push({ ...raw.at(-1), info: { ...summary, id: "msg_duplicate_summary" } })
        if (kind === "wrong-parent") summary.parentID = "msg_other"
        if (kind === "wrong-agent") summary.agent = "title"
        if (kind === "wrong-provider") summary.providerID = "other"
        if (kind === "wrong-model") summary.modelID = "other-model"
        if (kind === "read-failure") f.source.read = async () => { throw new Error("unavailable") }
        expect((await f.headers(prepared.request))[f.config.headers.compact]).toBeUndefined()
        await f.ordinary()
      } finally { f.store.close() }
    },
  )

  test.each(["openai", "other"])("history provider does not determine the target %s provider", async (providerID) => {
    const f = await setup()
    try {
      const prepared = await f.begin(providerID)
      const headers = await f.headers(prepared.request)
      expect(headers[operationHeader]).toBeTruthy()
      expect((await f.send(headers, providerID)).status).toBe(200)
      expect(f.store.loadAll()[0].checkpoint.providerID).toBe(providerID)
      expect(f.store.loadAll()[0].checkpoint.afterMessageID).toBe(prepared.request.message.id)
      const request = JSON.parse(f.sent[0].init!.body as string)
      expect(request.input[0].content[0].text).toBe("private captured history")
      expect(new Headers(f.sent[0].init!.headers).has(operationHeader)).toBe(false)
    } finally { f.store.close() }
  })

  test("a different continuation provider closes only its matching capture", async () => {
    const f = await setup()
    try {
      const prepared = await f.begin("openai")
      await f.hooks["experimental.compaction.autocontinue"]?.({
        ...prepared.request, model: targetModel("anthropic"), agent: "build",
      } as any, { enabled: true })
      expect((await f.headers(prepared.request))[f.config.headers.compact]).toBeUndefined()
      await f.ordinary()
    } finally { f.store.close() }
  })

  test.each(["headers", "autocontinue"])("late unsupported %s cannot end a new operation", async (kind) => {
    const f = await setup()
    try {
      const old = await f.begin()
      await f.fixture.finish(f.hooks, sessionID)
      const next = await f.begin("openai")
      if (kind === "headers") expect(await f.headers(old.request)).toEqual({})
      else await f.hooks["experimental.compaction.autocontinue"]?.(old.request as any, { enabled: true })
      const output = await f.headers(next.request)
      expect(output[operationHeader]).toBeTruthy()
      expect((await f.send(output)).status).toBe(200)
      expect(f.store.loadAll()[0].checkpoint.afterMessageID).toBe(next.request.message.id)
    } finally { f.store.close() }
  })

  test.each(["empty", "mixed", "newer-user"])("ambiguous %s transform cannot publish history", async (kind) => {
    const f = await setup()
    try {
      const prepared = await f.prepare("openai")
      await f.hooks["experimental.session.compacting"]?.({ sessionID }, { context: [] })
      const messages = kind === "empty" ? [] : kind === "mixed"
        ? [...prepared.history, { info: { ...f.user, sessionID: "ses_foreign" }, parts: [] }]
        : [{ info: f.user, parts: [{ type: "text", text: "new question" }] }]
      await f.hooks["experimental.chat.messages.transform"]?.({}, { messages } as any)
      f.fixture.addSummary(prepared)
      const output = await f.headers(prepared.request)
      if (kind === "newer-user") expect(output[f.config.headers.compact]).toBeUndefined()
      else expect((await f.send(output)).status).toBe(502)
      expect(f.network).not.toHaveBeenCalled()
      await f.ordinary()
    } finally { f.store.close() }
  })

  test.each(["user", "removed", "deleted", "cancel-and-restart", "dispose"])(
    "delayed headers verification cannot publish after %s", async (action) => {
      const f = await setup()
      let disposed = false
      try {
        const old = await f.begin("openai")
        const gate = deferred<unknown>()
        const started = deferred<void>()
        const oldRaw = f.fixture.sessions.get(sessionID)
        f.source.read = async () => { started.resolve(); return gate.promise }
        const pending = f.headers(old.request)
        await started.promise
        f.source.read = f.fixture.getSessionMessages
        let next: Awaited<ReturnType<typeof f.begin>> | undefined
        if (action === "user") await f.hooks["chat.message"]?.(
          { sessionID, model: f.user.model } as any, { message: f.user, parts: [] } as any,
        )
        if (action === "removed") await f.hooks.event?.({ event: {
          type: "message.removed", properties: { sessionID, messageID: old.request.message.id },
        } as any })
        if (action === "deleted") await f.hooks.event?.({ event: { type: "session.deleted", properties: { sessionID } } as any })
        if (action === "cancel-and-restart") {
          await f.fixture.cancel(f.hooks, sessionID)
          next = await f.begin("openai")
        }
        if (action === "dispose") { await f.hooks.dispose?.(); disposed = true }
        gate.resolve(oldRaw)
        expect((await pending)[operationHeader]).toBeUndefined()
        expect(f.network).not.toHaveBeenCalled()
        if (next) expect((await f.send(await f.headers(next.request))).status).toBe(200)
        else if (!disposed) await f.ordinary()
      } finally { if (!disposed) f.store.close() }
    },
  )

  test("concurrent matching headers share one identity and completed result", async () => {
    const f = await setup()
    try {
      const prepared = await f.begin("openai")
      const [first, second] = await Promise.all([f.headers(prepared.request), f.headers(prepared.request)])
      expect(first[operationHeader]).toBeTruthy()
      expect(second).toEqual(first)
      await Promise.all([f.send(first), f.send(second)])
      expect(f.network).toHaveBeenCalledTimes(1)
      expect(f.store.count()).toBe(1)
    } finally { f.store.close() }
  })

  test("unsupported-to-configured switching cannot claim the same old boundary", async () => {
    const f = await setup()
    try {
      const old = await f.begin()
      expect(await f.headers(old.request)).toEqual({})
      const changed = { ...old.request, model: targetModel("openai") }
      expect((await f.headers(changed))[operationHeader]).toBeUndefined()
      await f.ordinary()
    } finally { f.store.close() }
  })

  test("unsupported capture does not alter an existing valid OpenAI checkpoint", async () => {
    const f = await setup()
    try {
      const previous = await f.begin("openai")
      await f.send(await f.headers(previous.request))
      await f.fixture.finish(f.hooks, sessionID)
      const checkpoint = structuredClone(f.store.loadAll())
      const unsupported = await f.begin()
      await f.headers(unsupported.request)
      // Autocontinue reports the original conversation provider, even when the
      // compaction agent used an unsupported provider. It must not arm OpenAI state.
      await f.hooks["experimental.compaction.autocontinue"]?.({
        ...unsupported.request, model: targetModel("openai"), agent: "build",
      } as any, { enabled: true })
      const output = await f.headers({ sessionID, model: targetModel("openai"), agent: "build", message: f.user })
      expect(output[operationHeader]).toBeUndefined()
      expect(output[f.config.headers.compact]).toBeUndefined()
      const response = await f.cfg.provider.openai.options.fetch(url, {
        method: "POST", headers: output, body: JSON.stringify({ model: "openai-model", input: [{ role: "user", content: "new question" }] }),
      })
      expect(await response.text()).toBe("ordinary answer")
      const body = JSON.parse(f.sent.at(-1)!.init!.body as string)
      expect(body.input).toEqual([...checkpoint[0].checkpoint.items, { role: "user", content: "new question" }])
      expect(f.store.loadAll()).toEqual(checkpoint)
    } finally { f.store.close() }
  })

  test("raw chronological order, not transform order, identifies the boundary", async () => {
    const f = await setup()
    try {
      const prepared = await f.prepare("openai")
      const raw = f.fixture.sessions.get(sessionID)!
      raw.unshift({
        info: { ...prepared.request.message, id: "msg_old_boundary", time: { created: f.history[0].info.time.created - 1 } },
        parts: [{ type: "compaction", sessionID, messageID: "msg_old_boundary" }],
      })
      raw.reverse()
      await f.hooks["experimental.session.compacting"]?.({ sessionID }, { context: [] })
      await f.hooks["experimental.chat.messages.transform"]?.({}, { messages: prepared.history.toReversed() } as any)
      f.fixture.addSummary(prepared)
      const headers = await f.headers(prepared.request)
      expect((await f.send(headers)).status).toBe(200)
      expect(f.store.loadAll()[0].checkpoint.afterMessageID).toBe(prepared.request.message.id)
    } finally { f.store.close() }
  })

  test("headers cannot select a newer boundary after capture", async () => {
    const f = await setup()
    try {
      const old = await f.begin("openai")
      const next = await f.prepare("openai")
      f.fixture.addSummary(next)
      expect((await f.headers(next.request))[operationHeader]).toBeUndefined()
      expect((await f.headers(old.request))[operationHeader]).toBeUndefined()
      await f.ordinary()
    } finally { f.store.close() }
  })

  test("continuation provider cannot move a completed checkpoint", async () => {
    const f = await setup()
    try {
      const prepared = await f.begin("openai")
      const headers = await f.headers(prepared.request)
      await f.send(headers)
      const checkpoint = f.store.loadAll()
      await f.hooks["experimental.compaction.autocontinue"]?.({
        ...prepared.request, agent: "build", model: targetModel("other"),
      } as any, { enabled: true })
      expect(f.store.loadAll()).toEqual(checkpoint)
      expect((await f.send(headers)).status).toBe(200)
      expect(f.network).toHaveBeenCalledTimes(1)
    } finally { f.store.close() }
  })

  test("unsupported capture cannot create native fallback from an invalid OpenAI checkpoint", async () => {
    const f = await setup(true)
    try {
      const prepared = await f.begin()
      await f.headers(prepared.request)
      const output = await f.headers({ sessionID, agent: "build", message: f.user, model: targetModel("openai") })
      expect(output[f.config.headers.compact]).toBeUndefined()
      expect(output[operationHeader]).toBeUndefined()
      await f.hooks.event?.({ event: { type: "session.compacted", properties: { sessionID } } as any })
      expect(f.store.count()).toBe(1)
      expect(f.network).not.toHaveBeenCalled()
    } finally { f.store.close() }
  })

  test.each(["headerless", "old-operation", "other-provider"])("native fallback rejects %s requests", async (kind) => {
    const f = await setup(true)
    try {
      const old = await f.begin("openai")
      const oldHeaders = await f.headers(old.request)
      expect(oldHeaders[f.config.headers.compact]).toBe("native")
      expect(oldHeaders[operationHeader]).toBeTruthy()
      await f.fixture.cancel(f.hooks, sessionID)
      const next = await f.begin("openai")
      const headers = await f.headers(next.request)
      const invalid = { ...headers }
      if (kind === "headerless") delete invalid[operationHeader]
      if (kind === "old-operation") invalid[operationHeader] = oldHeaders[operationHeader]
      expect((await f.send(invalid, kind === "other-provider" ? "other" : "openai")).status).toBe(400)
      expect(f.network).not.toHaveBeenCalled()
      expect((await f.send(headers)).status).toBe(200)
      expect(new Headers(f.sent[0].init?.headers).has(operationHeader)).toBe(false)
      expect(f.sent[0].init?.body).not.toContain("compaction_trigger")
      expect(f.store.count()).toBe(1)
    } finally { f.store.close() }
  })

  test.each(["message", "session"])("native completion only clears its verified summary (%s event)", async (kind) => {
    const f = await setup(true)
    try {
      const prepared = await f.begin("openai")
      const headers = await f.headers(prepared.request)
      expect((await f.send(headers)).status).toBe(200)
      const raw = f.fixture.sessions.get(sessionID)!
      const summary = raw.at(-1).info
      const completedSummary = { ...summary, finish: "stop", time: { ...summary.time, completed: Date.now() } }
      await f.hooks.event?.({ event: {
        type: "message.updated", properties: { sessionID, info: { ...completedSummary, id: "msg_old_summary" } },
      } as any })
      await f.hooks.event?.({ event: { type: "session.compacted", properties: { sessionID } } as any })
      expect(f.store.count()).toBe(1)
      if (kind === "message") await f.hooks.event?.({ event: {
        type: "message.updated", properties: { sessionID, info: completedSummary },
      } as any })
      else {
        raw.at(-1).info = completedSummary
        await f.hooks.event?.({ event: { type: "session.compacted", properties: { sessionID } } as any })
      }
      expect(f.store.count()).toBe(0)
    } finally { f.store.close() }
  })

  test("late native response and completion cannot clear a newer capture", async () => {
    const f = await setup(true)
    try {
      const old = await f.begin("openai")
      const headers = await f.headers(old.request)
      const gate = deferred<Response>()
      const started = deferred<void>()
      f.network.mockImplementationOnce(async () => { started.resolve(); return gate.promise })
      const pending = f.send(headers)
      await started.promise
      await f.fixture.cancel(f.hooks, sessionID)
      const next = await f.begin("openai")
      const nextHeaders = await f.headers(next.request)
      gate.resolve(new Response("late native summary"))
      expect((await pending).status).toBe(400)
      await f.hooks.event?.({ event: { type: "session.compacted", properties: { sessionID } } as any })
      expect(f.store.count()).toBe(1)
      expect((await f.send(nextHeaders)).status).toBe(200)
      expect(f.store.count()).toBe(1)
    } finally { f.store.close() }
  })
})
