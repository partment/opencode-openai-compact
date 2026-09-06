import { describe, expect, test, vi } from "vitest"
import { createCompactHooks } from "../src/compact.js"
import { defaultConfig, OpenAICompactConfigSchema } from "../src/schema.js"
import { CheckpointStore } from "../src/state.js"
import { compactionFixture } from "./compaction-fixture.js"

const sessionID = "ses_control_identity"
const now = Date.now()
const question = "What did we do so far?"
const continuation = "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
const prefix = [
  { role: "developer", content: "Keep every real constraint." },
  { role: "system", content: "Stable system instructions." },
]
const checkpointItems = [
  { role: "user", content: "original history" },
  { type: "compaction", encrypted_content: "checkpoint" },
]

function message(id: string, text: string, role = "user", session = sessionID): any {
  return {
    info: {
      id, sessionID: session, role, agent: "build", time: { created: now + 10 },
      ...(role === "user" ? { model: { providerID: "openai", modelID: "gpt" } } : { providerID: "openai", modelID: "gpt" }),
    },
    parts: [{ type: "text", text, messageID: id, sessionID: session }],
  }
}

function control(id = "control", session = sessionID) {
  const result = message(id, continuation, "user", session)
  Object.assign(result.parts[0], { synthetic: true, metadata: { compaction_continue: true } })
  return result
}

function boundary(session = sessionID) {
  const result = message("boundary", question, "user", session)
  result.info.time.created = now
  result.parts = [{ type: "compaction", messageID: result.info.id, sessionID: session }]
  return result
}

function summary(session = sessionID) {
  const result = message("summary", defaultConfig.summary, "assistant", session)
  Object.assign(result.info, { summary: true, parentID: "boundary", time: { created: now + 1 } })
  return result
}

// Fixed wire items test selection across the point where OpenCode IDs disappear.
// This is not an SDK serializer: tool/attachment items below are supplied explicitly.
function textItems(history: any[], format: "string" | "array" = "string") {
  return new Map<string, any[]>(history.map(({ info, parts }) => [info.id, [{
    role: info.role,
    content: format === "string" ? parts[0].text : [{
      type: info.role === "assistant" ? "output_text" : "input_text", text: parts[0].text,
    }],
  }]]))
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

async function setup(options: {
  summary?: string; checkpoint?: boolean; storedIDs?: string[]; lookup?: boolean; deleteOnSessionDeleted?: boolean;
} = {}) {
  const config = OpenAICompactConfigSchema.parse({
    summary: options.summary, providers: { openai: {}, other: {} },
    state: { deleteOnSessionDeleted: options.deleteOnSessionDeleted ?? true },
  })
  const store = CheckpointStore.openMemory()
  if (options.checkpoint !== false) store.upsert(sessionID, {
    providerID: "openai", responseID: "resp_checkpoint", afterMessageID: "boundary",
    afterCreatedAt: now, createdAt: now, items: checkpointItems,
  })
  for (const id of options.storedIDs ?? []) store.upsertControlMessage({
    providerID: "openai", sessionID, messageID: id, createdAt: now + 10, contentText: continuation,
  })
  const raw = { value: undefined as unknown }
  const read = vi.fn(async (_id: string) => raw.value)
  const network = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response("ok"))
  const hooks = createCompactHooks(config, store, network as typeof fetch,
    options.lookup === false ? {} : { getSessionMessages: read })
  const cfg: any = {}
  await hooks.config?.(cfg)
  const transform = async (history: any[]) => {
    const messages = structuredClone(history)
    await hooks["experimental.chat.messages.transform"]?.({}, { messages } as any)
    return messages
  }
  const event = (type: string, properties: Record<string, unknown> = {}) =>
    hooks.event!({ event: { type, properties: { sessionID, ...properties } } as any })
  const request = async (history: any[], wire: Map<string, any[]>, providerID = "openai") => {
    const messages = await transform(history)
    const output = { headers: {} as Record<string, string> }
    await hooks["chat.headers"]?.({
      sessionID, model: { providerID, id: "gpt" }, agent: "build",
      message: history.findLast((entry) => entry.info.role === "user")?.info,
    } as any, output)
    const body = {
      model: "gpt", instructions: "Stable instructions.", temperature: 0.25,
      input: [...prefix, ...messages.flatMap((entry) => {
        expect(wire.has(entry.info.id)).toBe(true)
        return wire.get(entry.info.id)!
      })],
    }
    await cfg.provider[providerID].options.fetch("https://proxy.test/v1/responses", {
      method: "POST", headers: output.headers, body: JSON.stringify(body),
    })
    const sent = JSON.parse(network.mock.calls.at(-1)![1]!.body as string)
    return { messages, body, sent }
  }
  return { config, store, raw, read, network, hooks, cfg, transform, event, request }
}

const collisionCases = [defaultConfig.summary, "Done."].flatMap((summary) =>
  (["string", "array"] as const).flatMap((format) => [true, false].map((checkpoint) => ({ summary, format, checkpoint }))))

describe("identity-only control filtering", () => {
  test.each(collisionCases)("preserves a real checkpoint-tail Q&A ($summary, $format, checkpoint=$checkpoint)", async (options) => {
    const f = await setup(options)
    const history = [
      message("limit", "Never modify the production database."),
      message("question", question),
      message("answer", options.summary, "assistant"),
      message("next", "Now implement the change under that constraint."),
    ]
    try {
      const { messages, body, sent } = await f.request(history, textItems(history, options.format))
      expect(messages).toEqual(history)
      expect(sent).toEqual({ ...body, input: [...(options.checkpoint ? checkpointItems : []), ...body.input] })
      expect(f.store.loadControlMessages()).toEqual([])
      expect(f.read).not.toHaveBeenCalled()
    } finally { f.store.close() }
  })

  test.each([defaultConfig.summary, "Done."])("preserves repeated examples and a real continuation after summary %s", async (summary) => {
    const f = await setup({ summary })
    const history = [
      message("limit", "Do not remove the safety check."),
      message("question1", question), message("answer1", summary, "assistant"),
      message("quoted", `Example:\nuser: ${question}\nassistant: ${summary}\nuser: ${continuation}`),
      message("question2", question), message("answer2", summary, "assistant"),
      message("next", continuation),
    ]
    try {
      const { messages, body, sent } = await f.request(history, textItems(history, "array"))
      expect(messages).toEqual(history)
      expect(sent.input).toEqual([...checkpointItems, ...body.input])
    } finally { f.store.close() }
  })

  test.each(["legacy", "captured"].flatMap((mode) => [defaultConfig.summary, "Done."].map((summary) => ({ mode, summary }))))(
    "retains colliding history in $mode compaction and its retry ($summary)", async ({ mode, summary }) => {
      const f = await setup({ summary })
      const history = [
        message("limit", "Never delete the production data."), message("question", question),
        message("answer", summary, "assistant"), message("next", continuation),
      ]
      const input = [...textItems(history, "array").values()].flat()
      try {
        let headers = { [f.config.headers.session]: sessionID, [f.config.headers.compact]: "1" }
        if (mode === "captured") {
          const fixture = compactionFixture()
          f.read.mockImplementation(fixture.getSessionMessages)
          headers = await fixture.capture(f.hooks, {
            sessionID, history: structuredClone(history), model: { providerID: "openai", id: "gpt" },
            boundary: { id: "new_boundary", time: { created: now + 30 } },
          })
        } else await f.transform(history)
        f.network.mockResolvedValueOnce(new Response("retry", { status: 429 }))
        f.network.mockResolvedValueOnce(new Response([
          { type: "response.output_item.done", item: { type: "compaction", encrypted_content: "new checkpoint" } },
          { type: "response.completed", response: { id: "resp_new" } },
        ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")))
        const body = JSON.stringify({
          model: "gpt", instructions: "You are an anchored context summarization assistant for coding sessions.",
          input: mode === "captured" ? [{ role: "user", content: "unused flattened history" }] : input,
        })
        const send = () => f.cfg.provider.openai.options.fetch("https://proxy.test/v1/responses", { method: "POST", headers, body })
        expect((await send()).status).toBe(429)
        expect((await send()).status).toBe(200)
        const prepared = f.network.mock.calls.map((call) => call[1]!.body)
        expect(prepared[0]).toBe(prepared[1])
        expect(JSON.parse(prepared[0] as string).input).toEqual([...checkpointItems, ...input, { type: "compaction_trigger" }])
      } finally { f.store.close() }
    },
  )

  test.each(["continuation", "summary", "synthetic reminder"])("does not infer a control from a real %s after autocontinue", async (kind) => {
    const f = await setup()
    const history = [message("real", kind === "continuation" ? continuation : "Keep the compatibility layer.")]
    if (kind === "summary") history.push(message("answer", defaultConfig.summary, "assistant"))
    if (kind === "synthetic reminder") history[0].parts.push({
      type: "text", synthetic: true, text: "<system-reminder>Plan mode</system-reminder>",
    })
    const wire = textItems(history)
    if (kind === "synthetic reminder") wire.set("real", [{ role: "user", content: [
      { type: "input_text", text: "Keep the compatibility layer." },
      { type: "input_text", text: "<system-reminder>Plan mode</system-reminder>" },
    ] }])
    try {
      // Deliberately omit chat.message: a missing notification is not control evidence.
      await f.hooks["experimental.compaction.autocontinue"]?.({
        sessionID, agent: "build", model: { providerID: "openai" }, message: boundary().info,
      } as any, { enabled: true })
      const { messages, body, sent } = await f.request(history, wire)
      expect(messages).toEqual(history)
      expect(sent.input).toEqual([...checkpointItems, ...body.input])
      expect(f.store.loadControlMessages()).toEqual([])
    } finally { f.store.close() }
  })

  test("retains instructions, reasoning, full tool outputs and attachments around colliding text", async () => {
    const f = await setup({ summary: "Done." })
    const history = [
      message("limit", "Do not truncate this evidence."), message("question", question),
      message("answer", "Done.", "assistant"), message("next", continuation),
      message("tools", "", "assistant"),
    ]
    history[0].parts.push({ type: "file", mime: "image/png", url: "data:image/png;base64,AA==" })
    const output = "complete output\n".repeat(10000)
    history[4].parts = [
      { type: "reasoning", text: "investigating", metadata: { openai: { reasoningEncryptedContent: "reasoning" } } },
      { type: "tool", tool: "read", callID: "call", state: { status: "completed", input: {}, output } },
    ]
    const wire = textItems(history.slice(0, 4), "array")
    wire.set("limit", [{ role: "user", content: [
      { type: "input_text", text: "Do not truncate this evidence." },
      { type: "input_image", image_url: "data:image/png;base64,AA==" },
    ] }])
    wire.set("tools", [
      { type: "reasoning", encrypted_content: "reasoning", summary: [] },
      { type: "function_call", name: "read", call_id: "call", arguments: "{}" },
      { type: "function_call_output", call_id: "call", output },
    ])
    try {
      const { messages, body, sent } = await f.request(history, wire)
      expect(messages).toEqual(history)
      expect(sent).toEqual({ ...body, input: [...checkpointItems, ...body.input] })
    } finally { f.store.close() }
  })

  test("removes only the linked summary and verified control after a checkpoint boundary", async () => {
    const f = await setup()
    const history = [
      message("old", "already compacted"), boundary(),
      message("limit", "New constraint between the boundary and summary."), summary(),
      message("ordinary-answer", defaultConfig.summary, "assistant"), control(), message("real", continuation),
    ]
    try {
      const expected = [history[2], history[4], history[6]]
      const { messages, sent } = await f.request(history, textItems(history))
      expect(messages).toEqual(expected)
      expect(sent.input).toEqual([...checkpointItems, ...prefix, ...expected.flatMap((item) => textItems(expected).get(item.info.id)!)])
      expect(f.store.loadControlMessages().map((item) => item.messageID)).toEqual(["control"])
    } finally { f.store.close() }
  })

  test.each(["ordinary assistant", "other parent", "missing boundary"])(
    "retains a summary-shaped message with %s evidence", async (kind) => {
      const f = await setup()
      const candidate = summary()
      if (kind === "ordinary assistant") delete candidate.info.summary
      if (kind === "other parent") candidate.info.parentID = "other_boundary"
      const tail = [message("limit", "Keep this new constraint."), candidate, message("real", "new question")]
      const history = kind === "missing boundary" ? tail : [boundary(), ...tail]
      try {
        const { messages, body, sent } = await f.request(history, textItems(history))
        expect(messages).toEqual(tail)
        expect(sent.input).toEqual([...checkpointItems, ...body.input])
      } finally { f.store.close() }
    },
  )

  test("selects a persisted checkpoint before removing the only session-bearing control", async () => {
    const f = await setup()
    try {
      const { messages, sent } = await f.request([control()], textItems([control()]))
      expect(messages).toEqual([])
      expect(sent.input).toEqual([...checkpointItems, ...prefix])
    } finally { f.store.close() }
  })

  test.each(["no ID", "no session", "wrong part ID", "wrong part session", "assistant", "synthetic only", "metadata only", "duplicate ID"])(
    "does not establish control identity from %s", async (kind) => {
      const f = await setup()
      const candidate = control()
      if (kind === "no ID") delete candidate.info.id
      if (kind === "no session") delete candidate.info.sessionID
      if (kind === "wrong part ID") candidate.parts[0].messageID = "different"
      if (kind === "wrong part session") candidate.parts[0].sessionID = "different"
      if (kind === "assistant") candidate.info.role = "assistant"
      if (kind === "synthetic only") delete candidate.parts[0].metadata
      if (kind === "metadata only") delete candidate.parts[0].synthetic
      const history = [candidate, ...(kind === "duplicate ID" ? [message("control", "real constraint")] : []), message("real", "new question")]
      try {
        expect(await f.transform(history)).toEqual(history)
        expect(f.store.loadControlMessages()).toEqual([])
      } finally { f.store.close() }
    },
  )

  test.each(["unmarked", "missing", "wrong session", "duplicate", "conflicting part", "malformed", "throw", "sync throw", "no lookup"])(
    "retains an old control ID when raw verification is %s", async (kind) => {
      const f = await setup({ storedIDs: ["control"], lookup: kind !== "no lookup" })
      const history = [message("control", continuation), message("real", "new constraint")]
      f.raw.value = [control()]
      if (kind === "unmarked") f.raw.value = [history[0]]
      if (kind === "missing") f.raw.value = []
      if (kind === "wrong session") f.raw.value = [control("control", "other-session")]
      if (kind === "duplicate") f.raw.value = [control(), history[0]]
      if (kind === "conflicting part") (f.raw.value as any[])[0].parts[0].messageID = "other"
      if (kind === "malformed") f.raw.value = { messages: [control()] }
      if (kind === "throw") f.read.mockRejectedValue(new Error("offline"))
      if (kind === "sync throw") f.read.mockImplementation(() => { throw new Error("offline") })
      try {
        const before = f.store.loadControlMessages()
        const { messages, body, sent } = await f.request(history, textItems(history))
        expect(messages).toEqual(history)
        expect(sent.input).toEqual([...checkpointItems, ...body.input])
        expect(f.store.loadControlMessages()).toEqual(before)
      } finally { f.store.close() }
    },
  )

  test.each(["persisted", "revoked"])("does not restore %s identity from stale transformed markers", async (kind) => {
    const f = await setup({ storedIDs: kind === "persisted" ? ["control"] : [] })
    const history = [control(), message("real", "new question")]
    try {
      if (kind === "revoked") {
        expect(await f.transform(history)).toEqual([history[1]])
        await f.event("message.part.updated", { part: message("control", "edited real text").parts[0] })
      }
      f.raw.value = [message("control", "edited real text"), history[1]]
      expect(await f.transform(history)).toEqual(history)
      expect(f.read).toHaveBeenCalledTimes(1)
    } finally { f.store.close() }
  })

  test("batches old-ID verification, caches positive proof and never compares stored text", async () => {
    const f = await setup({ storedIDs: ["a", "b"] })
    const history = [message("a", "changed a"), message("b", "changed b"), message("real", continuation)]
    f.raw.value = [control("a"), control("b"), history[2]]
    try {
      expect(await f.transform(history)).toEqual([history[2]])
      expect(f.read).toHaveBeenCalledTimes(1)
      f.read.mockRejectedValue(new Error("offline after verification"))
      expect(await f.transform(history)).toEqual([history[2]])
      expect(f.read).toHaveBeenCalledTimes(1)
      const restarted = createCompactHooks(f.config, f.store, f.network as typeof fetch)
      const afterRestart = structuredClone(history)
      await restarted["experimental.chat.messages.transform"]?.({}, { messages: afterRestart } as any)
      expect(afterRestart).toEqual(history)
    } finally { f.store.close() }
  })

  test("retries unknown proof later without enabling another provider's control IDs", async () => {
    const f = await setup({ storedIDs: ["control"] })
    const history = [message("control", continuation), message("real", "new constraint")]
    try {
      expect(await f.transform(history)).toEqual(history)
      f.raw.value = [control()]
      expect(await f.transform(history)).toEqual([history[1]])
      const switched = structuredClone(history)
      switched[1].info.model.providerID = "other"
      const { messages, body, sent } = await f.request(switched, textItems(switched), "other")
      expect(messages).toEqual(switched)
      expect(sent).toEqual(body)
      expect(f.read).toHaveBeenCalledTimes(2)
    } finally { f.store.close() }
  })

  test.each(["message.updated", "message.part.updated", "message.part.removed", "message.removed", "chat.message"])(
    "revokes verified identity after %s", async (type) => {
      const f = await setup()
      const history = [message("control", "now a genuine constraint"), message("real", "new question")]
      try {
        expect(await f.transform([control(), history[1]])).toEqual([history[1]])
        f.raw.value = history
        if (type === "chat.message") await f.hooks["chat.message"]?.({
          sessionID, messageID: "control", model: { providerID: "openai", modelID: "gpt" },
        } as any, { message: history[0].info, parts: history[0].parts } as any)
        else await f.event(type, { info: history[0].info, part: history[0].parts[0], messageID: "control", partID: "part" })
        const { messages, body, sent } = await f.request(history, textItems(history))
        expect(messages).toEqual(history)
        expect(sent.input).toEqual([...checkpointItems, ...body.input])
      } finally { f.store.close() }
    },
  )

  test.each(["delete", "delete-retain", "dispose", "new user", "new capture", "revert", "part update"])(
    "does not publish a late raw verification after %s", async (action) => {
      const f = await setup({ storedIDs: ["control"], deleteOnSessionDeleted: action !== "delete-retain" })
      const history = [message("control", continuation), message("real", "new question")]
      const gate = deferred<unknown>()
      const started = deferred<void>()
      const upsert = vi.spyOn(f.store, "commitControlMessages")
      const checkpoint = vi.spyOn(f.store, "commitForkState")
      f.read.mockImplementationOnce(async () => { started.resolve(); return gate.promise })
      let disposed = false
      try {
        const pending = f.transform(history)
        await started.promise
        if (action.startsWith("delete")) await f.event("session.deleted")
        if (action === "dispose") { await f.hooks.dispose?.(); disposed = true }
        if (action === "new user") await f.hooks["chat.message"]?.({ sessionID, messageID: "new" } as any,
          { message: message("new", "new generation").info, parts: [] } as any)
        if (action === "new capture") await f.hooks["experimental.session.compacting"]?.({ sessionID }, { context: [] })
        if (action === "revert") await f.event("session.updated", { info: { id: sessionID, revert: { messageID: "boundary" } } })
        if (action === "part update") await f.event("message.part.updated", { part: history[0].parts[0] })
        gate.resolve([control()])
        expect(await pending).toEqual(history)
        expect(upsert).not.toHaveBeenCalled()
        expect(checkpoint).not.toHaveBeenCalled()
        if (["part update", "new user", "new capture", "revert"].includes(action)) {
          f.raw.value = history
          expect(await f.transform(history)).toEqual(history)
        }
        if (action === "delete") expect(f.store.loadControlMessages()).toEqual([])
        if (action === "delete-retain") expect(f.store.loadControlMessages()).toHaveLength(1)
      } finally { if (!disposed) f.store.close() }
    },
  )

  test("another session's part updates do not invalidate pending verification", async () => {
    const f = await setup({ storedIDs: ["control"] })
    const history = [message("control", continuation), message("real", "new question")]
    const gate = deferred<unknown>()
    const started = deferred<void>()
    f.read.mockImplementationOnce(async () => { started.resolve(); return gate.promise })
    try {
      const pending = f.transform(history)
      await started.promise
      await f.event("message.part.updated", { sessionID: "other", part: message("control", "unrelated", "user", "other").parts[0] })
      gate.resolve([control()])
      expect(await pending).toEqual([history[1]])
      expect(await f.transform(history)).toEqual([history[1]])
      expect(f.read).toHaveBeenCalledTimes(1)
    } finally { f.store.close() }
  })

  test("rejects stale fork proof even when that source finished reading before another source", async () => {
    const f = await setup({ checkpoint: false })
    const histories = new Map<string, any[]>()
    for (const id of ["one", "two", sessionID]) {
      const history = [boundary(id), summary(id), control("control", id)]
      for (const entry of history) {
        entry.info.id = `${id}_${entry.info.id}`
        if (entry.info.parentID) entry.info.parentID = `${id}_${entry.info.parentID}`
        for (const part of entry.parts) part.messageID = entry.info.id
      }
      histories.set(id, history)
      if (id === sessionID) continue
      f.store.upsert(id, {
        providerID: "openai", responseID: "resp_parent", afterMessageID: `${id}_boundary`,
        afterCreatedAt: now, createdAt: now, items: checkpointItems,
      })
      f.store.upsertControlMessage({ providerID: "openai", sessionID: id, messageID: `${id}_control`, createdAt: now + 10, contentText: continuation })
    }
    const first = deferred<unknown>()
    const second = deferred<unknown>()
    const started = deferred<void>()
    const hooks = createCompactHooks(f.config, f.store, f.network as typeof fetch, {
      async getSessionMessages(id) {
        if (id === "one") return first.promise
        if (id === "two") { started.resolve(); return second.promise }
        return histories.get(id)
      },
    })
    try {
      const messages = structuredClone(histories.get(sessionID)!)
      const upsert = vi.spyOn(f.store, "commitForkState")
      const controlWrite = vi.spyOn(f.store, "commitControlMessages")
      const pending = hooks["experimental.chat.messages.transform"]!({}, { messages } as any)
      await started.promise
      first.resolve(histories.get("one"))
      // Drain the resolved source's microtasks while the other source remains blocked.
      await new Promise<void>((resolve) => setImmediate(resolve))
      await hooks.event?.({ event: { type: "message.part.updated", properties: {
        sessionID: "one", part: { messageID: "one_control", sessionID: "one", type: "text", text: "now real" },
      } } as any })
      second.resolve(histories.get("two"))
      await pending
      expect(messages).toEqual(histories.get(sessionID))
      expect(upsert).not.toHaveBeenCalled()
      expect(controlWrite).not.toHaveBeenCalled()
    } finally { f.store.close() }
  })

  test.each([false, true])("only inherits fork controls with raw identity evidence (verified=%s)", async (verified) => {
    const f = await setup({ checkpoint: false })
    const parent = "parent"
    const parentHistory = [boundary(parent), summary(parent), verified ? control("control", parent) : message("control", continuation, "user", parent)]
    const childHistory = [boundary(), summary(), verified ? control() : message("control", continuation)]
    f.store.upsert(parent, {
      providerID: "openai", responseID: "resp_parent", afterMessageID: "boundary",
      afterCreatedAt: now, createdAt: now, items: checkpointItems,
    })
    f.store.upsertControlMessage({ providerID: "openai", sessionID: parent, messageID: "control", createdAt: now + 10, contentText: continuation })
    const raw = new Map([[parent, parentHistory], [sessionID, childHistory]])
    const hooks = createCompactHooks(f.config, f.store, f.network as typeof fetch, {
      getSessionMessages: async (id) => raw.get(id),
    })
    try {
      // Markerless transform must rely on verified raw ancestry, not a copied DB row.
      const messages = structuredClone(childHistory)
      delete messages[2].parts[0].metadata
      delete messages[2].parts[0].synthetic
      await hooks["experimental.chat.messages.transform"]?.({}, { messages } as any)
      expect(f.store.loadAll().some((item) => item.sessionID === sessionID)).toBe(true)
      expect(f.store.loadControlMessages().some((item) => item.sessionID === sessionID)).toBe(verified)
      expect(messages).toEqual(verified ? [] : [childHistory[2]])
    } finally { f.store.close() }
  })
})
