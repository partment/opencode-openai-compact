import { describe, expect, test, vi } from "vitest"
import { createCompactHooks } from "../src/compact.js"
import { defaultConfig } from "../src/schema.js"
import { CheckpointStore } from "../src/state.js"

const url = "https://proxy.test/openai/v1/responses"
const sessionID = "ses_fetch_normalized"
const summaryPrompt = "You are an anchored context summarization assistant for coding sessions."

function responseWithCompaction() {
  const events = [
    { type: "response.output_item.done", item: { type: "compaction", encrypted_content: "fetch-checkpoint" } },
    { type: "response.completed", response: { id: "resp_fetch", model: "gpt", created_at: 1 } },
  ]
  return new Response(`${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  })
}

function requestBody() {
  return {
    model: "gpt",
    instructions: summaryPrompt,
    input: [{ role: "user", content: "same JSON body" }],
  }
}

async function bodyJSON(value: unknown) {
  return new Response(value as BodyInit).json()
}

async function setup() {
  const store = CheckpointStore.openMemory()
  const sent: Array<{ input: RequestInfo | URL; init?: RequestInit }> = []
  const network = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    sent.push({ input, init })
    return responseWithCompaction()
  })
  const hooks = createCompactHooks(defaultConfig, store, network as typeof fetch)
  const config: any = {}
  await hooks.config?.(config)
  return { store, hooks, network, sent, fetch: config.provider.openai.options.fetch as typeof fetch }
}

describe("normalized fetch wrapper input", () => {
  test.each(["string", "request", "uint8array", "arraybuffer", "blob"])(
    "compacts the same JSON body from %s input",
    async (kind) => {
      const f = await setup()
      const body = JSON.stringify(requestBody())
      const headers = {
        [defaultConfig.headers.compact]: "1",
        [defaultConfig.headers.session]: sessionID,
      }
      try {
        let response: Response
        if (kind === "request") {
          response = await f.fetch(new Request(url, { method: "POST", headers, body }))
        } else {
          const value = kind === "uint8array"
            ? new TextEncoder().encode(body)
            : kind === "arraybuffer"
              ? new TextEncoder().encode(body).buffer
              : kind === "blob"
                ? new Blob([body], { type: "application/json" })
                : body
          response = await f.fetch(url, { method: "POST", headers, body: value as BodyInit })
        }
        expect(response.status).toBe(200)
        expect(f.network).toHaveBeenCalledOnce()
        const sentBody = await bodyJSON(f.sent[0].init?.body)
        expect(sentBody.input.filter((item: any) => item.type === "compaction_trigger")).toHaveLength(1)
        expect(f.store.loadAll().map(({ checkpoint }) => checkpoint.responseID)).toEqual(["resp_fetch"])
      } finally {
        f.store.close()
      }
    },
  )

  test("lets init.body override a Request body", async () => {
    const f = await setup()
    const original = JSON.stringify({ ...requestBody(), input: [{ role: "user", content: "old" }] })
    const replacement = JSON.stringify(requestBody())
    try {
      const response = await f.fetch(
        new Request(url, { method: "POST", headers: { "x-source": "remove-me" }, body: original }),
        {
          body: replacement,
          headers: {
            [defaultConfig.headers.compact]: "1",
            [defaultConfig.headers.session]: sessionID,
          },
        },
      )
      expect(response.status).toBe(200)
      const sentBody = await bodyJSON(f.sent[0].init?.body)
      expect(sentBody.input.find((item: any) => item.role === "user").content).toBe("same JSON body")
      expect(new Headers(f.sent[0].init?.headers).has("x-source")).toBe(false)
    } finally {
      f.store.close()
    }
  })

  test("removes stale Content-Length when checkpoint input is injected", async () => {
    const f = await setup()
    f.store.upsert(sessionID, {
      providerID: "openai",
      responseID: "resp_previous",
      afterMessageID: "msg_previous",
      afterCreatedAt: 1,
      createdAt: Date.now(),
      items: [{ role: "user", content: "checkpoint" }, { type: "compaction", encrypted_content: "old" }],
    })
    try {
      await f.hooks["experimental.chat.messages.transform"]?.({}, { messages: [{
        info: { id: "msg_previous", sessionID, role: "user", time: { created: 1 },
          model: { providerID: "openai", modelID: "gpt" } },
        parts: [{ type: "compaction", messageID: "msg_previous", sessionID }],
      }] } as any)
      const response = await f.fetch(url, {
        method: "POST",
        headers: {
          [defaultConfig.headers.session]: sessionID,
          "content-type": "application/json",
          "content-length": "3",
        },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "tail" }] }),
      })
      expect(response.status).toBe(200)
      const headers = new Headers(f.sent[0].init?.headers)
      expect(headers.has("content-length")).toBe(false)
      expect(headers.get("content-type")).toBe("application/json")
      const sentBody = await bodyJSON(f.sent[0].init?.body)
      expect(sentBody.input[0].content).toBe("checkpoint")
    } finally {
      f.store.close()
    }
  })

  test("rejects marked non-POST Responses requests without calling upstream", async () => {
    const f = await setup()
    try {
      const response = await f.fetch(url, {
        method: "GET",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: sessionID,
        },
      })
      expect(response.status).toBe(400)
      expect(await response.text()).toBe("OpenAI compact requests must use POST")
      expect(f.network).not.toHaveBeenCalled()
    } finally {
      f.store.close()
    }
  })

  test("passes an unmarked non-POST Responses request through unchanged", async () => {
    const f = await setup()
    try {
      const response = await f.fetch(url, {
        method: "GET",
        headers: { "x-custom": "keep", [defaultConfig.headers.session]: sessionID },
      })
      expect(response.status).toBe(200)
      expect(f.network).toHaveBeenCalledOnce()
      expect(f.sent[0].input).toBe(url)
      expect(f.sent[0].init?.method).toBe("GET")
      expect(new Headers(f.sent[0].init?.headers).get("x-custom")).toBe("keep")
    } finally {
      f.store.close()
    }
  })

  test("does not duplicate checkpoint injection when the body already has the checkpoint", async () => {
    const f = await setup()
    f.store.upsert(sessionID, {
      providerID: "openai", responseID: "resp_previous", afterMessageID: "msg_previous",
      afterCreatedAt: 1, createdAt: Date.now(),
      items: [{ role: "user", content: "checkpoint" }, { type: "compaction", encrypted_content: "old" }],
    })
    try {
      await f.hooks["experimental.chat.messages.transform"]?.({}, { messages: [{
        info: { id: "msg_previous", sessionID, role: "user", time: { created: 1 },
          model: { providerID: "openai", modelID: "gpt" } },
        parts: [{ type: "compaction", messageID: "msg_previous", sessionID }],
      }] } as any)
      await f.fetch(url, {
        method: "POST",
        headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({ model: "gpt", input: [
          { role: "user", content: "checkpoint" },
          { type: "compaction", encrypted_content: "old" },
          { role: "user", content: "tail" },
        ] }),
      })
      const sentBody = await bodyJSON(f.sent[0].init?.body)
      expect(sentBody.input.filter((item: any) => item.type === "compaction")).toHaveLength(1)
      expect(sentBody.input.filter((item: any) => item.content === "checkpoint")).toHaveLength(1)
    } finally {
      f.store.close()
    }
  })

  test("keeps custom fetch composition idempotent across config hook calls", async () => {
    const store = CheckpointStore.openMemory()
    const upstream = vi.fn(async () => new Response("ok"))
    const custom = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => upstream(input, init))
    const hooks = createCompactHooks(defaultConfig, store, upstream as typeof fetch)
    const config: any = { provider: { openai: { options: { fetch: custom } } } }
    try {
      await hooks.config?.(config)
      const first = config.provider.openai.options.fetch
      await hooks.config?.(config)
      expect(config.provider.openai.options.fetch).toBe(first)
      await config.provider.openai.options.fetch(url, {
        method: "POST",
        headers: { "x-custom": "keep" },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "ordinary" }] }),
      })
      expect(custom).toHaveBeenCalledOnce()
      expect(upstream).toHaveBeenCalledOnce()
    } finally {
      store.close()
    }
  })

  test("passes non-Responses requests through while removing only plugin headers", async () => {
    const f = await setup()
    try {
      await f.fetch("https://proxy.test/v1/embeddings", {
        method: "POST",
        headers: {
          "x-custom": "keep",
          [defaultConfig.headers.session]: sessionID,
          [defaultConfig.headers.compact]: "1",
        },
        body: "not changed",
      })
      expect(f.network).toHaveBeenCalledOnce()
      expect(f.sent[0].init?.body).toBe("not changed")
      const headers = new Headers(f.sent[0].init?.headers)
      expect(headers.get("x-custom")).toBe("keep")
      expect(headers.has(defaultConfig.headers.session)).toBe(false)
      expect(headers.has(defaultConfig.headers.compact)).toBe(false)
    } finally {
      f.store.close()
    }
  })
})
