import { describe, expect, test } from "vitest"
import { compactBody, compactUrl, createCompactHooks } from "../src/compact.js"
import { defaultConfig, OpenAICompactConfigSchema } from "../src/schema.js"
import { CheckpointStore } from "../src/state.js"

function jsonBody(init: RequestInit | undefined) {
  return JSON.parse(typeof init?.body === "string" ? init.body : "{}")
}

const defaultCompactModel = defaultConfig.providers.openai.compactModel

describe("OpenAI compact hooks", () => {
  test("wraps the configured provider fetch", async () => {
    const store = CheckpointStore.openMemory()
    try {
      const hooks = createCompactHooks(defaultConfig, store)
      const cfg: any = {}

      await hooks.config?.(cfg)

      expect(typeof cfg.provider.openai.options.fetch).toBe("function")
    } finally {
      store.close()
    }
  })

  test("builds compact URL and compact request body", () => {
    expect(compactUrl(new URL("https://api.openai.com/v1/responses?x=1"))).toBe(
      "https://api.openai.com/v1/responses/compact?x=1",
    )
    expect(compactUrl(new URL("https://proxy.test/openai/v1/responses"))).toBe(
      "https://proxy.test/openai/v1/responses/compact",
    )

    const body = compactBody({ model: "ignored", input: [], stream: true, tools: [] })
    expect(body).toEqual({ model: defaultCompactModel, input: [] })
  })

  test("builds standard compact input without OpenCode summarizer prompts", () => {
    const body = compactBody({
      model: "ignored",
      instructions: "You are an anchored context summarization assistant for coding sessions.\n\nSummarize only...",
      previous_response_id: "resp_previous",
      input: [
        { role: "developer", content: "Keep the user's coding preferences." },
        {
          role: "developer",
          content: "You are an anchored context summarization assistant for coding sessions.\n\nSummarize only...",
        },
        { role: "user", content: "Create a new anchored summary from the conversation history. This is quoted." },
        { role: "assistant", content: [{ type: "output_text", text: "quoted response" }] },
        { role: "user", content: "real request" },
        {
          role: "user",
          content: [
            { type: "input_text", text: "Create a new anchored summary from the conversation history.\n\nOutput exactly..." },
          ],
        },
      ],
    })

    expect(body).toEqual({
      model: defaultCompactModel,
      previous_response_id: "resp_previous",
      input: [
        { role: "developer", content: "Keep the user's coding preferences." },
        { role: "user", content: "Create a new anchored summary from the conversation history. This is quoted." },
        { role: "assistant", content: [{ type: "output_text", text: "quoted response" }] },
        { role: "user", content: "real request" },
      ],
    })
  })

  test("wraps multiple providers with their own compact models", async () => {
    const config = OpenAICompactConfigSchema.parse({
      providers: {
        openai: { compactModel: "openai-compact" },
        "custom-openai": { compactModel: "custom-compact" },
      },
    })
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return new Response(
        JSON.stringify({
          id: "resp_compacted",
          created_at: 1,
          output: [{ type: "compaction_summary", encrypted_content: "compacted" }],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      )
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(config, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)

      await cfg.provider["custom-openai"].options.fetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: {
          [config.headers.compact]: "1",
          [config.headers.session]: "ses_custom",
        },
        body: JSON.stringify({ model: "ignored", input: [] }),
      })

      expect(typeof cfg.provider.openai.options.fetch).toBe("function")
      expect(typeof cfg.provider["custom-openai"].options.fetch).toBe("function")
      expect(jsonBody(calls[0]?.init).model).toBe("custom-compact")
    } finally {
      store.close()
    }
  })

  test("routes compaction fetch and prepends stored checkpoint on the next request", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return new Response(
        JSON.stringify({
          id: "resp_compacted",
          model: defaultCompactModel,
          created_at: 1,
          output: [{ type: "compaction_summary", encrypted_content: "compacted" }],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      )
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: "ses_request",
        },
        body: JSON.stringify({ model: "ignored", input: [{ role: "user", content: "hello" }], stream: true }),
      })

      expect(calls[0]?.url).toBe("https://proxy.test/openai/v1/responses/compact")
      expect(jsonBody(calls[0]?.init)).toEqual({
        model: defaultCompactModel,
        input: [{ role: "user", content: "hello" }],
      })
      expect(new Headers(calls[0]?.init?.headers).has(defaultConfig.headers.compact)).toBe(false)
      expect(new Headers(calls[0]?.init?.headers).has(defaultConfig.headers.session)).toBe(false)

      await hooks.event?.({
        event: {
          type: "message.part.updated",
          properties: {
            sessionID: "ses_request",
            part: { messageID: "msg_checkpoint", type: "text", text: defaultConfig.summary },
            time: 2,
          },
        } as any,
      })

      calls.length = 0
      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: "ses_request" },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "after compact" }] }),
      })

      const followupBody = jsonBody(calls[0]?.init)
      expect(calls[0]?.url).toBe("https://proxy.test/openai/v1/responses")
      expect(followupBody.input[0]).toEqual({ type: "compaction", encrypted_content: "compacted" })

      const unknownProviderMessages = [
        { info: { id: "msg_checkpoint", sessionID: "ses_request" } },
        { info: { id: "msg_after", sessionID: "ses_request" } },
      ]
      await hooks["experimental.chat.messages.transform"]?.({}, { messages: unknownProviderMessages } as any)
      expect(unknownProviderMessages.map((message) => message.info.id)).toEqual(["msg_checkpoint", "msg_after"])

      await hooks["chat.message"]?.(
        { model: { providerID: "openai" }, sessionID: "ses_request", messageID: "msg_after" } as any,
        { message: { id: "msg_after" }, parts: [] } as any,
      )
      const inferredProviderMessages = [
        { info: { id: "msg_checkpoint", sessionID: "ses_request" } },
        { info: { id: "msg_after", sessionID: "ses_request" } },
      ]
      await hooks["experimental.chat.messages.transform"]?.({}, { messages: inferredProviderMessages } as any)
      expect(inferredProviderMessages.map((message) => message.info.id)).toEqual(["msg_after"])
    } finally {
      store.close()
    }
  })

  test("adds compaction headers only for OpenAI compaction agent", async () => {
    const store = CheckpointStore.openMemory()
    try {
      const hooks = createCompactHooks(defaultConfig, store)
      const output = { headers: {} as Record<string, string> }

      await hooks["chat.headers"]?.(
        { model: { providerID: "openai" }, sessionID: "ses", agent: "compaction" } as any,
        output,
      )

      expect(output.headers[defaultConfig.headers.session]).toBe("ses")
      expect(output.headers[defaultConfig.headers.compact]).toBe("1")

      const unsupported = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        { model: { providerID: "anthropic" }, sessionID: "ses", agent: "compaction" } as any,
        unsupported,
      )
      expect(unsupported.headers).toEqual({})
    } finally {
      store.close()
    }
  })
})
