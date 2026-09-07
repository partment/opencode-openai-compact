import { describe, expect, test, vi } from "vitest"
import { createCompactHooks } from "../src/compact.js"
import { openAIOAuthDummyKey } from "../src/oauth.js"
import { OpenAICompactConfigSchema } from "../src/schema.js"
import { CheckpointStore } from "../src/state.js"
import { compactionFixture } from "./compaction-fixture.js"

const operationHeader = "x-opencode-openai-compact-operation"
const url = "https://proxy.test/v1/responses"
const model = { providerID: "openai", id: "gpt-conversation", modelID: "gpt-conversation", variant: "xhigh" }
const embedded = "Here is the conversation so far:\n<conversation>\n[User]: flattened history\n</conversation>"

function completed(id = "resp_retry") {
  return new Response([
    { type: "response.output_item.done", item: { type: "compaction", encrypted_content: "new-checkpoint" } },
    { type: "response.completed", response: { id, created_at: 123 } },
  ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
    headers: { "content-type": "text/event-stream" },
  })
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

async function retrySession(
  fakeFetch: typeof fetch,
  options: { unbound?: boolean; attachments?: boolean; invalid?: "clone" | "convert"; capture?: boolean } = {},
) {
  const config = OpenAICompactConfigSchema.parse({ providers: { openai: {}, other: {} } })
  const store = CheckpointStore.openMemory()
  const now = Date.now()
  const sessionID = "ses_retry"
  const boundary = { id: "msg_boundary", sessionID, role: "user", agent: "plan", model, time: { created: now } }
  const oldItems = [
    { role: "user", content: "previous history" },
    { type: "compaction", encrypted_content: "previous-checkpoint" },
  ]
  store.upsert(sessionID, {
    providerID: "openai", responseID: "resp_previous", afterMessageID: "msg_previous",
    afterCreatedAt: now - 100, createdAt: now - 100, items: oldItems,
  })
  const history = [
    {
      info: { id: "msg_user", sessionID, role: "user", agent: "plan", model, time: { created: now - 10 } },
      parts: [
        { type: "text", text: "structured request" },
        ...(options.attachments ? [{ type: "file", mime: "image/png", url: "data:image/png;base64,AA==" }] : []),
      ],
    },
    {
      info: {
        id: "msg_tool", sessionID, role: "assistant", providerID: "openai", modelID: model.modelID,
        variant: model.variant, time: { created: now - 5 },
      },
      parts: [
        {
          type: "reasoning", text: "tool investigation",
          metadata: { openai: { itemId: "rs_original", reasoningEncryptedContent: "encrypted-reasoning" } },
        },
        {
          type: "tool", tool: "read", callID: "call_read", state: {
            status: "completed", input: { filePath: "source.ts" }, output: "完整工具輸出😀".repeat(1000),
          },
        },
        { type: "text", text: "assistant response" },
      ],
    },
  ]
  if (options.invalid === "clone") (history[0].parts[0] as any).metadata = { uncloneable: () => undefined }
  if (options.invalid === "convert") (history[1].parts[1] as any).state.output = undefined
  const fixture = compactionFixture()
  const hooks = createCompactHooks(config, store, fakeFetch, {
    getSessionMessages: fixture.getSessionMessages, getSessionStatus: fixture.getSessionStatus,
  })
  const cfg: any = {}
  try {
    await hooks.config?.(cfg)
    await hooks["experimental.chat.system.transform"]?.(
      { sessionID, model } as any, { system: ["stable instructions"] },
    )
    await hooks["chat.headers"]?.(
      { sessionID, model, agent: "plan", message: history[0].info } as any, { headers: {} },
    )
    let headers: Record<string, string> = { [config.headers.session]: sessionID, [config.headers.compact]: "1" }
    if (options.capture !== false) {
      if (options.unbound) {
        const prepared = await fixture.prepare({ sessionID, boundary, history, model })
        await hooks["experimental.session.compacting"]?.({ sessionID }, { context: [] })
        await hooks["experimental.chat.messages.transform"]?.({}, { messages: prepared.history } as any)
        fixture.addSummary(prepared)
      } else {
        headers = await fixture.capture(hooks, { sessionID, boundary, history, model })
      }
    }
    const init: RequestInit = {
      method: "POST", headers,
      body: JSON.stringify({
        model: "sdk-summary-model", reasoning: { effort: "low" },
        instructions: "You are an anchored context summarization assistant for coding sessions.",
        input: [{ role: "user", content: embedded }],
        tools: [{ type: "function", name: "read", parameters: { type: "object" } }],
      }),
    }
    return { config, store, hooks, cfg, fixture, sessionID, boundary, history, oldItems, init, fetch: cfg.provider.openai.options.fetch as typeof fetch }
  } catch (error) {
    store.close()
    throw error
  }
}

describe("immutable compaction retries", () => {
  test("replays frozen input after 429 despite changed session state", async () => {
    const sent: RequestInit[] = []
    const f = await retrySession((async (_input, init) => {
      sent.push(init!)
      return sent.length === 1 ? new Response("rate limited", { status: 429, headers: { "retry-after": "12" } }) : completed()
    }) as typeof fetch)
    try {
      const first = await f.fetch(url, f.init)
      expect(first.status).toBe(429)
      expect(first.headers.get("retry-after")).toBe("12")
      expect(await first.text()).toBe("rate limited")
      expect(f.store.count()).toBe(1)
      const initial = JSON.parse(sent[0].body as string)
      expect(initial.model).toBe(model.modelID)
      expect(initial.reasoning.effort).toBe("xhigh")
      expect(initial.instructions).toBe("stable instructions")
      expect(initial.input.slice(0, 2)).toEqual(f.oldItems)
      expect(initial.input.some((item: any) => item.type === "function_call_output")).toBe(true)
      expect(initial.input.find((item: any) => item.type === "reasoning").encrypted_content).toBe("encrypted-reasoning")
      expect(initial.input.find((item: any) => item.role === "assistant").content[0].text).toBe("assistant response")
      expect(JSON.stringify(initial)).not.toContain("<conversation>")

      f.config.providers.openai.compactModel = "changed-model"
      f.config.providers.openai.compactReasoningEffort = "minimal"
      await f.hooks["experimental.chat.system.transform"]?.(
        { sessionID: f.sessionID, model } as any, { system: ["changed instructions"] },
      )
      await f.hooks["chat.headers"]?.(
        { sessionID: f.sessionID, agent: "plan", model, message: f.history[0].info } as any, { headers: {} },
      )
      // Select a history before the checkpoint, clearing the active checkpoint without starting another operation.
      await f.hooks["experimental.chat.messages.transform"]?.({}, { messages: [{
        info: { id: "msg_older", sessionID: f.sessionID, role: "user", model, time: { created: 0 } },
        parts: [{ type: "text", text: "different history" }],
      }] } as any)
      const headers = new Headers(f.init.headers)
      headers.set("authorization", "Bearer refreshed")
      headers.set("x-request-id", "retry-attempt")
      const second = await f.fetch(url, { ...f.init, headers, signal: new AbortController().signal })
      expect(second.status).toBe(200)
      expect(sent[1].body).toBe(sent[0].body)
      expect(new Headers(sent[1].headers).get("authorization")).toBe("Bearer refreshed")
      for (const request of sent) {
        const outbound = new Headers(request.headers)
        expect(outbound.has(operationHeader)).toBe(false)
        expect(outbound.has(f.config.headers.session)).toBe(false)
        expect(outbound.has(f.config.headers.compact)).toBe(false)
      }
      const checkpoint = f.store.loadAll().find((row) => row.checkpoint.responseID === "resp_retry")!.checkpoint
      expect(checkpoint.afterMessageID).toBe(f.boundary.id)
      expect(checkpoint.afterCreatedAt).toBe(f.boundary.time.created)
      const replay = await f.fetch(url, f.init)
      expect(await replay.text()).toBe(await second.text())
      expect(sent).toHaveLength(2)
      expect(f.store.count()).toBe(2)
    } finally { f.store.close() }
  })

  test.each(["400", "401", "408", "500", "503", "network", "incomplete", "missing-id", "stream-error"])(
    "retains the prepared request after %s", async (failure) => {
      const bodies: unknown[] = []
      const f = await retrySession((async (_input, init) => {
        bodies.push(init?.body)
        if (bodies.length > 1) return completed()
        if (failure === "network") throw new TypeError("connection reset")
        if (failure === "incomplete") return new Response("data: [DONE]\n\n")
        if (failure === "missing-id") return completed("").text().then((text) => new Response(text.replace('"id":"",', "")))
        if (failure === "stream-error") return new Response(new ReadableStream({ start(controller) { controller.error(new Error("stream failed")) } }))
        return new Response("API error", { status: Number(failure) })
      }) as typeof fetch)
      try {
        if (failure === "network") await expect(f.fetch(url, f.init)).rejects.toThrow("connection reset")
        else {
          const response = await f.fetch(url, f.init)
          expect(response.status).toBe(Number(failure) || 502)
        }
        expect(f.store.count()).toBe(1)
        expect((await f.fetch(url, f.init)).status).toBe(200)
        expect(bodies).toHaveLength(2)
        expect(bodies[1]).toBe(bodies[0])
        expect(f.store.count()).toBe(2)
      } finally { f.store.close() }
    },
  )

  test("pins attachment fallback across a 429 and a later network error", async () => {
    const bodies: string[] = []
    const f = await retrySession((async (_input, init) => {
      bodies.push(init!.body as string)
      if (bodies.length === 1) return Response.json({ error: { code: "invalid_image" } }, { status: 400 })
      if (bodies.length === 2) return new Response("limited", { status: 429 })
      if (bodies.length === 3) throw new TypeError("disconnected")
      return completed()
    }) as typeof fetch, { attachments: true })
    try {
      expect((await f.fetch(url, f.init)).status).toBe(429)
      expect(f.store.count()).toBe(1)
      await expect(f.fetch(url, f.init)).rejects.toThrow("disconnected")
      expect(f.store.count()).toBe(1)
      expect((await f.fetch(url, f.init)).status).toBe(200)
      expect(bodies).toHaveLength(4)
      expect(bodies[0]).toContain("input_image")
      expect(bodies[1]).toContain("attachment rejected by API")
      expect(bodies.slice(1).every((body) => body === bodies[1])).toBe(true)
      for (const body of bodies) {
        const parsed = JSON.parse(body)
        expect(parsed.input.slice(0, 2)).toEqual(f.oldItems)
        expect(parsed.input.find((item: any) => item.type === "function_call_output").output).toBe("完整工具輸出😀".repeat(1000))
      }
    } finally { f.store.close() }
  })

  test.each(["clone", "convert"] as const)("fails closed on every retry after %s failure", async (invalid) => {
    const network = vi.fn(async () => completed())
    const f = await retrySession(network as typeof fetch, { invalid })
    try {
      for (let attempt = 0; attempt < 2; attempt++) {
        const output = { headers: {} as Record<string, string> }
        await f.hooks["chat.headers"]?.(
          { sessionID: f.sessionID, agent: "compaction", model, message: f.boundary } as any, output,
        )
        expect(output.headers).toEqual(f.init.headers)
        expect((await f.fetch(url, { ...f.init, headers: output.headers })).status).toBe(502)
      }
      expect(network).not.toHaveBeenCalled()
      expect(f.store.count()).toBe(1)
    } finally { f.store.close() }
  })

  test.each(["body", "model", "url", "endpoint", "method", "session", "provider", "id", "missing-id", "markers"])(
    "rejects a retry with changed %s without consuming another snapshot", async (change) => {
      const network = vi.fn(async () => new Response("limited", { status: 429 }))
      const f = await retrySession(network as typeof fetch)
      try {
        expect((await f.fetch(url, f.init)).status).toBe(429)
        const headers = new Headers(f.init.headers)
        const init = { ...f.init, headers }
        let target = url
        let fetch = f.fetch
        if (change === "body") init.body = `${init.body} `
        if (change === "model") init.body = JSON.stringify({ ...JSON.parse(init.body as string), model: "different" })
        if (change === "url") target += "?different=1"
        if (change === "endpoint") target = "https://proxy.test/v1/chat/completions"
        if (change === "method") init.method = "PUT"
        if (change === "session") headers.set(f.config.headers.session, "ses_other")
        if (change === "provider") fetch = f.cfg.provider.other.options.fetch
        if (change === "id") headers.set(operationHeader, "unknown-operation")
        if (change === "missing-id") headers.delete(operationHeader)
        if (change === "markers") headers.delete(f.config.headers.compact)
        expect((await fetch(target, init)).status).toBe(400)
        expect(network).toHaveBeenCalledTimes(1)
        expect(f.store.count()).toBe(1)
        expect((await f.fetch(url, f.init)).status).toBe(429)
        expect(network).toHaveBeenCalledTimes(2)
      } finally { f.store.close() }
    },
  )

  test("rejects an operation ID after restart instead of using flattened history", async () => {
    const network = vi.fn(async () => new Response("limited", { status: 429 }))
    const f = await retrySession(network as typeof fetch)
    try {
      await f.fetch(url, f.init)
      const hooks = createCompactHooks(f.config, f.store, network as typeof fetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      expect((await cfg.provider.openai.options.fetch(url, f.init)).status).toBe(400)
      expect(network).toHaveBeenCalledTimes(1)
      expect(f.store.count()).toBe(1)
    } finally { f.store.close() }
  })

  test("shares an in-flight attempt and replays successful responses without a second commit", async () => {
    const gate = deferred<Response>()
    const started = deferred<void>()
    const network = vi.fn(async () => { started.resolve(); return gate.promise })
    const f = await retrySession(network as typeof fetch)
    try {
      const upsert = vi.spyOn(f.store, "commitCheckpoint")
      const first = f.fetch(url, f.init)
      await started.promise
      const second = f.fetch(url, { ...f.init, signal: new AbortController().signal })
      gate.resolve(completed())
      const responses = await Promise.all([first, second])
      const texts = await Promise.all(responses.map((response) => response.text()))
      expect(texts[1]).toBe(texts[0])
      expect(responses.map((response) => response.status)).toEqual([200, 200])
      expect(await (await f.fetch(url, f.init)).text()).toBe(texts[0])
      expect(network).toHaveBeenCalledTimes(1)
      expect(upsert).toHaveBeenCalledTimes(1)
    } finally { f.store.close() }
  })

  test.each(["user", "removed", "deleted", "cancel-and-restart", "dispose"])(
    "does not commit an in-flight result after %s", async (event) => {
      const gate = deferred<Response>()
      const started = deferred<void>()
      const network = vi.fn(async () => {
        if (network.mock.calls.length > 1) return completed("resp_new_operation")
        started.resolve()
        return gate.promise
      })
      const f = await retrySession(network as typeof fetch)
      let closed = false
      try {
        const upsert = vi.spyOn(f.store, "commitCheckpoint")
        const pending = f.fetch(url, f.init)
        await started.promise
        let nextHeaders: Record<string, string> | undefined
        if (event === "user") await f.hooks["chat.message"]?.(
          { sessionID: f.sessionID, messageID: "msg_new_user", model } as any,
          { message: { id: "msg_new_user" }, parts: [] } as any,
        )
        if (event === "removed") await f.hooks.event?.({ event: {
          type: "message.removed", properties: { sessionID: f.sessionID, messageID: f.history[0].info.id },
        } as any })
        if (event === "deleted") await f.hooks.event?.({ event: {
          type: "session.deleted", properties: { sessionID: f.sessionID },
        } as any })
        if (event === "cancel-and-restart") {
          await f.fixture.cancel(f.hooks, f.sessionID)
          nextHeaders = await f.fixture.capture(f.hooks, {
            sessionID: f.sessionID, boundary: { ...f.boundary, id: "msg_new_boundary" }, history: f.history, model,
          })
        }
        if (event === "dispose") { await f.hooks.dispose?.(); closed = true }
        gate.resolve(completed())
        expect((await pending).status).toBe(400)
        expect((await f.fetch(url, f.init)).status).toBe(400)
        expect(upsert).not.toHaveBeenCalled()
        expect(network).toHaveBeenCalledTimes(1)
        if (nextHeaders) {
          expect(nextHeaders[operationHeader]).not.toBe(new Headers(f.init.headers).get(operationHeader))
          expect((await f.fetch(url, { ...f.init, headers: nextHeaders })).status).toBe(200)
          expect(f.store.loadAll().at(-1)?.checkpoint.afterMessageID).toBe("msg_new_boundary")
        }
      } finally { if (!closed) f.store.close() }
    },
  )

  test.each(["request", "bytes", "arraybuffer"])("retries %s with a fresh signal after an aborted attempt", async (kind) => {
    const started = deferred<void>()
    const sent: RequestInit[] = []
    const f = await retrySession((async (_input, init) => {
      sent.push(init!)
      if (sent.length > 1) return completed()
      started.resolve()
      return new Promise<Response>((_resolve, reject) => {
        init!.signal!.addEventListener("abort", () => reject(init!.signal!.reason), { once: true })
      })
    }) as typeof fetch)
    try {
      const controller = new AbortController()
      const request = new Request(url, { ...f.init, signal: controller.signal })
      const bytes = new TextEncoder().encode(f.init.body as string)
      const init = { ...f.init, body: kind === "arraybuffer" ? bytes.buffer : bytes }
      const pending = kind === "request"
        ? f.fetch(request)
        : f.fetch(url, { ...init, signal: controller.signal })
      const rejected = expect(pending).rejects.toThrow("cancelled attempt")
      await started.promise
      controller.abort(new Error("cancelled attempt"))
      await rejected
      expect(f.store.count()).toBe(1)
      const retryAbort = new AbortController()
      const signal = retryAbort.signal
      const response = kind === "request" ? await f.fetch(request, { signal }) : await f.fetch(url, { ...init, signal })
      expect(response.status).toBe(200)
      expect(sent[1].signal?.aborted).toBe(false)
      retryAbort.abort(new Error("fresh signal"))
      expect(sent[1].signal?.reason).toBe(signal.reason)
      expect(sent[1].body).toBe(sent[0].body)
      expect(request.bodyUsed).toBe(false)
    } finally { f.store.close() }
  })

  test("refreshes OAuth credentials without changing the prepared body or target", async () => {
    const sent: Array<{ target: string; init: RequestInit }> = []
    const f = await retrySession((async (input, init) => {
      sent.push({ target: String(input), init: init! })
      return sent.length === 1 ? new Response("limited", { status: 429 }) : completed()
    }) as typeof fetch)
    try {
      const headers = new Headers(f.init.headers)
      headers.set("authorization", `Bearer ${openAIOAuthDummyKey}`)
      for (const access of ["old-token", "refreshed-token"]) {
        await f.hooks.auth?.loader?.(async () => ({
          type: "oauth", access, refresh: "refresh-token", expires: Date.now() + 120_000, accountId: "account",
        }), {} as any)
        const response = await f.fetch(url, { ...f.init, headers })
        expect(response.status).toBe(access === "old-token" ? 429 : 200)
      }
      expect(sent.map((call) => call.target)).toEqual(Array(2).fill("https://chatgpt.com/backend-api/codex/responses"))
      expect(sent.map((call) => new Headers(call.init.headers).get("authorization"))).toEqual([
        "Bearer old-token", "Bearer refreshed-token",
      ])
      expect(sent[1].init.body).toBe(sent[0].init.body)
      expect(sent.every((call) => !new Headers(call.init.headers).has(operationHeader))).toBe(true)
    } finally { f.store.close() }
  })

  test("an aborted response stream cannot commit, and a fresh signal can retry the same body", async () => {
    const started = deferred<void>()
    let controller!: ReadableStreamDefaultController<Uint8Array>
    const bodies: unknown[] = []
    const f = await retrySession((async (_input, init) => {
      bodies.push(init?.body)
      if (bodies.length > 1) return completed()
      return new Response(new ReadableStream<Uint8Array>({
        start(value) { controller = value; started.resolve() },
      }))
    }) as typeof fetch)
    try {
      const abort = new AbortController()
      const pending = f.fetch(url, { ...f.init, signal: abort.signal })
      const rejected = expect(pending).rejects.toThrow("cancelled stream")
      await started.promise
      abort.abort(new Error("cancelled stream"))
      controller.enqueue(new TextEncoder().encode(await completed().text()))
      controller.close()
      await rejected
      expect(f.store.count()).toBe(1)
      expect((await f.fetch(url, { ...f.init, signal: new AbortController().signal })).status).toBe(200)
      expect(bodies[1]).toBe(bodies[0])
    } finally { f.store.close() }
  })

  test("a delayed capture cannot replace a newer operation's snapshot", async () => {
    const sent: string[] = []
    const network = (async (_input: RequestInfo | URL, init?: RequestInit) => {
      sent.push(init!.body as string)
      return completed()
    }) as typeof fetch
    const f = await retrySession(network, { capture: false })
    try {
      const started = deferred<void>()
      const gate = deferred<unknown>()
      let reads = 0
      const hooks = createCompactHooks(f.config, f.store, network, {
        getSessionStatus: f.fixture.getSessionStatus,
        async getSessionMessages(sessionID) {
          if (++reads === 1) { started.resolve(); return gate.promise }
          return f.fixture.getSessionMessages(sessionID)
        },
      })
      const cfg: any = {}
      await hooks.config?.(cfg)
      await f.fixture.prepare({ sessionID: f.sessionID, boundary: f.boundary, history: f.history, model })
      const oldRaw = f.fixture.sessions.get(f.sessionID)
      const first = hooks["experimental.session.compacting"]?.({ sessionID: f.sessionID }, { context: [] })
      await started.promise
      await f.fixture.cancel(hooks, f.sessionID)
      const newer = structuredClone(f.history)
      newer[0].parts[0].text = "new capture"
      const headers = await f.fixture.capture(hooks, {
        sessionID: f.sessionID, boundary: { ...f.boundary, id: "msg_new_capture" }, history: newer, model,
      })
      gate.resolve(oldRaw)
      await first
      expect((await cfg.provider.openai.options.fetch(url, { ...f.init, headers })).status).toBe(200)
      expect(sent).toHaveLength(1)
      expect(sent[0]).toContain("new capture")
      expect(sent[0]).not.toContain("structured request")
      expect(f.store.loadAll().at(-1)?.checkpoint.afterMessageID).toBe("msg_new_capture")
    } finally { f.store.close() }
  })

  test("an empty operation ID cannot consume an unclaimed snapshot", async () => {
    const network = vi.fn(async () => completed())
    const f = await retrySession(network as typeof fetch, { unbound: true })
    try {
      const headers = new Headers(f.init.headers)
      headers.set(operationHeader, "")
      expect((await f.fetch(url, { ...f.init, headers })).status).toBe(400)
      expect((await f.fetch(url, f.init)).status).toBe(400)
      expect(network).not.toHaveBeenCalled()
      const output = { headers: {} as Record<string, string> }
      await f.hooks["chat.headers"]?.(
        { sessionID: f.sessionID, agent: "compaction", model, message: f.boundary } as any, output,
      )
      expect((await f.fetch(url, { ...f.init, headers: output.headers })).status).toBe(200)
      expect(network).toHaveBeenCalledTimes(1)
    } finally { f.store.close() }
  })

  test("fingerprints a Blob body override instead of using the Request's original body", async () => {
    const network = vi.fn(async () => new Response("limited", { status: 429 }))
    const f = await retrySession(network as typeof fetch)
    try {
      const request = new Request(url, f.init)
      expect((await f.fetch(request)).status).toBe(429)
      const overridden = await f.fetch(request, { body: new Blob(["different body"]) })
      expect(overridden.status).toBe(400)
      expect(await overridden.text()).toContain("cannot change its original request")
      expect(network).toHaveBeenCalledTimes(1)
      expect((await f.fetch(request)).status).toBe(429)
    } finally { f.store.close() }
  })

  test("refuses an OAuth routing change during retries without losing the original operation", async () => {
    const network = vi.fn(async () => new Response("limited", { status: 429 }))
    const f = await retrySession(network as typeof fetch)
    try {
      expect((await f.fetch(url, f.init)).status).toBe(429)
      const headers = new Headers(f.init.headers)
      headers.set("authorization", `Bearer ${openAIOAuthDummyKey}`)
      expect((await f.fetch(url, { ...f.init, headers })).status).toBe(400)
      expect(network).toHaveBeenCalledTimes(1)
      expect((await f.fetch(url, f.init)).status).toBe(429)
      expect(network).toHaveBeenCalledTimes(2)
    } finally { f.store.close() }
  })

  test("repeated headers for a utility agent do not reuse a compaction operation", async () => {
    const f = await retrySession((async () => completed()) as typeof fetch)
    try {
      const output = { headers: {} as Record<string, string> }
      await f.hooks["chat.headers"]?.(
        { sessionID: f.sessionID, agent: "title", model, message: f.boundary } as any, output,
      )
      expect(output.headers).toEqual({})
      expect((await f.fetch(url, f.init)).status).toBe(200)
    } finally { f.store.close() }
  })

  test("keeps the first-time headerless text compatibility path immutable", async () => {
    const bodies: unknown[] = []
    const f = await retrySession((async (_input, init) => {
      bodies.push(init?.body)
      return bodies.length === 1 ? new Response("limited", { status: 429 }) : completed()
    }) as typeof fetch, { capture: false })
    try {
      expect((await f.fetch(url, f.init)).status).toBe(429)
      expect((await f.fetch(url, { ...f.init, body: `${f.init.body} ` })).status).toBe(400)
      expect((await f.fetch(url, f.init)).status).toBe(200)
      expect(bodies).toHaveLength(2)
      expect(bodies[1]).toBe(bodies[0])
      expect(bodies[0]).toContain("<conversation>")
    } finally { f.store.close() }
  })
})
