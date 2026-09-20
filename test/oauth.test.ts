import { describe, expect, test } from "vitest"
import { createCompactHooks } from "../src/compact.js"
import {
  createOpenAIOAuth,
  disposeOpenAIOAuth,
  openAIOAuthDummyKey,
  openAIAuthMethods,
} from "../src/oauth.js"
import { defaultConfig } from "../src/schema.js"
import { CheckpointStore } from "../src/state.js"

function jsonBody(init: RequestInit | undefined) {
  return JSON.parse(typeof init?.body === "string" ? init.body : "{}")
}

function compactResponse(payload: any) {
  const events = [
    ...(payload.output ?? []).map((item: any) => ({ type: "response.output_item.done", item })),
    { type: "response.completed", response: { ...payload, output: undefined } },
  ]
  return new Response(`${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  })
}

const compactionInstructions = "You are an anchored context summarization assistant for coding sessions."

describe("OpenAI OAuth hooks", () => {
  test("keeps OpenAI connect methods available", async () => {
    const store = CheckpointStore.openMemory()
    try {
      const hooks = createCompactHooks(defaultConfig, store)

      expect(hooks.auth?.methods.map((method) => method.label)).toEqual([
        "ChatGPT Pro/Plus (browser)",
        "ChatGPT Pro/Plus (headless)",
        "Manually enter API Key",
      ])
      expect(hooks.auth?.methods.map((method) => method.type)).toEqual(["oauth", "oauth", "api"])
    } finally {
      store.close()
    }
  })

  test("returns the matching apiKey for OpenAI auth type", async () => {
    const store = CheckpointStore.openMemory()
    try {
      const hooks = createCompactHooks(defaultConfig, store)

      const oauthOptions = await hooks.auth?.loader?.(
        async () => ({
          type: "oauth",
          refresh: "refresh-token",
          access: "real-access-token",
          expires: Date.now() + 120_000,
        }),
        {} as any,
      )
      expect(oauthOptions?.apiKey).toBe(openAIOAuthDummyKey)
      expect(oauthOptions?.fetch).toBeTypeOf("function")

      const apiOptions = await hooks.auth?.loader?.(async () => ({ type: "api", key: "sk-test" }), {} as any)
      expect(apiOptions?.apiKey).toBe("sk-test")
      expect(apiOptions?.fetch).toBe(oauthOptions?.fetch)
    } finally {
      store.close()
    }
  })

  test("keeps the authenticated fetch when provider config is reapplied after auth loaders", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return new Response("ok")
    }) as typeof fetch
    let authReads = 0

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const configuredFetch = cfg.provider.openai.options.fetch as typeof fetch

      const oauthOptions = await hooks.auth?.loader?.(
        async () => {
          authReads++
          if (authReads > 1) throw new Error("OAuth auth was read after provider initialization")
          return {
            type: "oauth",
            refresh: "refresh-token",
            access: "real-access-token",
            expires: Date.now() + 120_000,
            accountId: "acct_test",
          }
        },
        {} as any,
      )
      expect(oauthOptions?.fetch).toBe(configuredFetch)

      const nativeFetch = (async () => new Response("native")) as typeof fetch
      const providerOptions = {
        fetch: nativeFetch,
        ...(oauthOptions ?? {}),
        ...cfg.provider.openai.options,
      }
      expect(providerOptions.fetch).toBe(configuredFetch)

      await providerOptions.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: { authorization: `Bearer ${openAIOAuthDummyKey}` },
        body: JSON.stringify({ model: "gpt", input: [] }),
      })

      expect(authReads).toBe(1)
      expect(calls[0]?.url).toBe("https://chatgpt.com/backend-api/codex/responses")
      expect(new Headers(calls[0]?.init?.headers).get("authorization")).toBe("Bearer real-access-token")
    } finally {
      store.close()
    }
  })

  test("never sends the OAuth dummy key when credentials are unavailable", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return new Response("unexpected")
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      await expect(
        wrappedFetch("https://api.openai.com/v1/responses", {
          method: "POST",
          headers: { authorization: `Bearer ${openAIOAuthDummyKey}` },
          body: JSON.stringify({ model: "gpt", input: [] }),
        }),
      ).rejects.toThrow("OpenAI OAuth credentials are unavailable")
      expect(calls).toEqual([])
    } finally {
      store.close()
    }
  })

  test("keeps API key compaction on OpenAI and prepends the checkpoint", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      const headers = new Headers(init?.headers)
      if (headers.get("authorization") !== "Bearer sk-test") return new Response("unauthorized", { status: 401 })
      return compactResponse({
        id: "resp_compacted",
        model: "ignored",
        created_at: 1,
        output: [{ type: "compaction", encrypted_content: "compacted" }],
      })
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const apiOptions = await hooks.auth?.loader?.(async () => ({ type: "api", key: "sk-test" }), {} as any)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      await wrappedFetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          authorization: `Bearer ${apiOptions?.apiKey}`,
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: "ses_api",
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: compactionInstructions,
          input: [{ role: "user", content: "hello" }],
          service_tier: "priority",
          prompt_cache_key: "cache-key",
        }),
      })

      expect(calls[0]?.url).toBe("https://api.openai.com/v1/responses")
      expect(jsonBody(calls[0]?.init)).toMatchObject({ prompt_cache_key: "cache-key" })
      expect(jsonBody(calls[0]?.init)).toMatchObject({
        service_tier: "priority",
        stream: true,
        store: false,
        tool_choice: "auto",
        include: ["reasoning.encrypted_content"],
      })
      expect(jsonBody(calls[0]?.init).input.at(-1)).toEqual({ type: "compaction_trigger" })
      expect(store.count()).toBe(1)

      await hooks.event?.({
        event: {
          type: "message.part.updated",
          properties: {
            sessionID: "ses_api",
            part: { messageID: "msg_checkpoint", type: "text", text: defaultConfig.summary },
            time: 2,
          },
        } as any,
      })
      expect(store.count()).toBe(1)

      calls.length = 0
      await wrappedFetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: { authorization: `Bearer ${apiOptions?.apiKey}`, [defaultConfig.headers.session]: "ses_api" },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "after compact" }] }),
      })

      const followupBody = jsonBody(calls[0]?.init)
      expect(calls[0]?.url).toBe("https://api.openai.com/v1/responses")
      expect(followupBody.input.slice(0, 2)).toEqual([
        { role: "user", content: "hello" },
        { type: "compaction", encrypted_content: "compacted" },
      ])
    } finally {
      store.close()
    }
  })

  test("uses an API key checkpoint after switching to OpenAI OAuth", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      if (jsonBody(init).input?.at(-1)?.type === "compaction_trigger") {
        return compactResponse({
          id: "resp_api_compacted",
          model: "ignored",
          created_at: 1,
          output: [{ type: "compaction", encrypted_content: "api-compacted" }],
        })
      }
      return new Response("ok")
    }) as typeof fetch

    try {
      const apiHooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const apiOptions = await apiHooks.auth?.loader?.(async () => ({ type: "api", key: "sk-test" }), {} as any)
      const apiCfg: any = {}
      await apiHooks.config?.(apiCfg)
      const apiFetch = apiCfg.provider.openai.options.fetch as typeof fetch

      await apiFetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          authorization: `Bearer ${apiOptions?.apiKey}`,
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: "ses_switch",
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: compactionInstructions,
          input: [{ role: "user", content: "hello" }],
        }),
      })

      expect(calls[0]?.url).toBe("https://api.openai.com/v1/responses")
      expect(store.count()).toBe(1)


      const oauthHooks = createCompactHooks(defaultConfig, store, fakeFetch)
      await oauthHooks.auth?.loader?.(
        async () => ({
          type: "oauth",
          refresh: "refresh-token",
          access: "real-access-token",
          expires: Date.now() + 120_000,
          accountId: "acct_test",
        }),
        {} as any,
      )
      const oauthCfg: any = {}
      await oauthHooks.config?.(oauthCfg)
      await oauthHooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: [{ info: {
          id: "msg_after", sessionID: "ses_switch", time: { created: store.loadAll()[0].checkpoint.afterCreatedAt + 1 },
        } }] } as any,
      )

      calls.length = 0
      const oauthFetch = oauthCfg.provider.openai.options.fetch as typeof fetch
      await oauthFetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: { authorization: `Bearer ${openAIOAuthDummyKey}`, [defaultConfig.headers.session]: "ses_switch" },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "after compact" }] }),
      })

      const followupBody = jsonBody(calls[0]?.init)
      expect(calls[0]?.url).toBe("https://chatgpt.com/backend-api/codex/responses")
      expect(new Headers(calls[0]?.init?.headers).get("authorization")).toBe("Bearer real-access-token")
      expect(followupBody.input.slice(0, 2)).toEqual([
        { role: "user", content: "hello" },
        { type: "compaction", encrypted_content: "api-compacted" },
      ])
    } finally {
      store.close()
    }
  })

  test("replaces a disposed API key fetch wrapper when switching to OpenAI OAuth", async () => {
    const apiStore = CheckpointStore.openMemory()
    const oauthStore = CheckpointStore.openMemory()
    let apiStoreDisposed = false
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return compactResponse({
        id: "resp_after_switch",
        model: "ignored",
        created_at: 1,
        output: [{ type: "compaction", encrypted_content: "after-switch" }],
      })
    }) as typeof fetch

    try {
      const cfg: any = {}
      const apiHooks = createCompactHooks(defaultConfig, apiStore, fakeFetch)
      const apiOptions = await apiHooks.auth?.loader?.(async () => ({ type: "api", key: "sk-test" }), {} as any)
      await apiHooks.config?.(cfg)
      const apiFetch = cfg.provider.openai.options.fetch as typeof fetch

      await apiFetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          authorization: `Bearer ${apiOptions?.apiKey}`,
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: "ses_dispose_switch",
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: compactionInstructions,
          input: [{ role: "user", content: "hello" }],
        }),
      })
      expect(apiStore.count()).toBe(1)

      await apiHooks.dispose?.()
      apiStoreDisposed = true

      const oauthHooks = createCompactHooks(defaultConfig, oauthStore, fakeFetch)
      await oauthHooks.auth?.loader?.(
        async () => ({
          type: "oauth",
          refresh: "refresh-token",
          access: "real-access-token",
          expires: Date.now() + 120_000,
          accountId: "acct_test",
        }),
        {} as any,
      )
      await oauthHooks.config?.(cfg)
      const oauthFetch = cfg.provider.openai.options.fetch as typeof fetch
      expect(oauthFetch).not.toBe(apiFetch)

      calls.length = 0
      const response = await oauthFetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          authorization: `Bearer ${openAIOAuthDummyKey}`,
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: "ses_dispose_switch",
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: compactionInstructions,
          input: [{ role: "user", content: "after switch" }],
        }),
      })

      expect(response.ok).toBe(true)
      expect(calls[0]?.url).toBe("https://chatgpt.com/backend-api/codex/responses")
      expect(oauthStore.count()).toBe(1)
    } finally {
      if (!apiStoreDisposed) apiStore.close()
      oauthStore.close()
    }
  })

  test("routes OpenAI OAuth compaction to ChatGPT Codex responses endpoint", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return compactResponse({
        id: "resp_compacted",
        created_at: 1,
        output: [{ type: "compaction", encrypted_content: "compacted" }],
      })
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      await hooks.auth?.loader?.(
        async () => ({
          type: "oauth",
          refresh: "refresh-token",
          access: "real-access-token",
          expires: Date.now() + 120_000,
          accountId: "acct_test",
        }),
        {} as any,
      )
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      await wrappedFetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          authorization: `Bearer ${openAIOAuthDummyKey}`,
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: "ses_oauth",
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: compactionInstructions,
          input: [{ role: "user", content: "hello" }],
          service_tier: "priority",
          prompt_cache_key: "cache-key",
        }),
      })

      expect(calls[0]?.url).toBe("https://chatgpt.com/backend-api/codex/responses")
      const headers = new Headers(calls[0]?.init?.headers)
      expect(headers.get("authorization")).toBe("Bearer real-access-token")
      expect(headers.get("chatgpt-account-id")).toBe("acct_test")
      expect(headers.has(defaultConfig.headers.compact)).toBe(false)
      expect(headers.has(defaultConfig.headers.session)).toBe(false)
      expect(jsonBody(calls[0]?.init)).toMatchObject({
        service_tier: "priority",
        prompt_cache_key: "cache-key",
      })
    } finally {
      store.close()
    }
  })

  test("routes OpenAI OAuth responses to ChatGPT Codex endpoint", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return new Response("ok")
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      await hooks.auth?.loader?.(
        async () => ({
          type: "oauth",
          refresh: "refresh-token",
          access: "real-access-token",
          expires: Date.now() + 120_000,
          accountId: "acct_test",
        }),
        {} as any,
      )
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      await wrappedFetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          authorization: `Bearer ${openAIOAuthDummyKey}`,
          [defaultConfig.headers.session]: "ses_oauth",
        },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "hello" }] }),
      })

      expect(calls[0]?.url).toBe("https://chatgpt.com/backend-api/codex/responses")
      const headers = new Headers(calls[0]?.init?.headers)
      expect(headers.get("authorization")).toBe("Bearer real-access-token")
      expect(headers.get("chatgpt-account-id")).toBe("acct_test")
      expect(headers.has(defaultConfig.headers.session)).toBe(false)
    } finally {
      store.close()
    }
  })

  test("routes OpenAI OAuth utility agents without session headers to ChatGPT Codex endpoint", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return new Response("ok")
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      await hooks.auth?.loader?.(
        async () => ({
          type: "oauth",
          refresh: "refresh-token",
          access: "real-access-token",
          expires: Date.now() + 120_000,
          accountId: "acct_test",
        }),
        {} as any,
      )
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      for (const agent of ["title", "summary"]) {
        const output = { headers: {} as Record<string, string> }
        await hooks["chat.headers"]?.(
          { sessionID: `ses_${agent}`, agent, model: { providerID: "openai" } } as any,
          output,
        )
        expect(output.headers).toEqual({})

        await wrappedFetch("https://api.openai.com/v1/responses", {
          method: "POST",
          headers: { authorization: `Bearer ${openAIOAuthDummyKey}`, ...output.headers },
          body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: `${agent} request` }] }),
        })
      }

      expect(calls.map((call) => call.url)).toEqual([
        "https://chatgpt.com/backend-api/codex/responses",
        "https://chatgpt.com/backend-api/codex/responses",
      ])
      for (const call of calls) {
        const headers = new Headers(call.init?.headers)
        expect(headers.get("authorization")).toBe("Bearer real-access-token")
        expect(headers.get("chatgpt-account-id")).toBe("acct_test")
        expect(headers.has(defaultConfig.headers.session)).toBe(false)
        expect(headers.has(defaultConfig.headers.compact)).toBe(false)
      }
    } finally {
      store.close()
    }
  })

  test("preserves Request method and body when routing no-session OpenAI OAuth responses", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return new Response("ok")
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      await hooks.auth?.loader?.(
        async () => ({
          type: "oauth",
          refresh: "refresh-token",
          access: "real-access-token",
          expires: Date.now() + 120_000,
          accountId: "acct_test",
        }),
        {} as any,
      )
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch
      const body = { model: "gpt", input: [{ role: "user", content: "title request" }] }

      await wrappedFetch(
        new Request("https://api.openai.com/v1/responses", {
          method: "POST",
          headers: { authorization: `Bearer ${openAIOAuthDummyKey}` },
          body: JSON.stringify(body),
        }),
      )

      expect(calls[0]?.url).toBe("https://chatgpt.com/backend-api/codex/responses")
      expect(calls[0]?.init?.method).toBe("POST")
      expect(await new Response(calls[0]?.init?.body).json()).toEqual(body)
      const headers = new Headers(calls[0]?.init?.headers)
      expect(headers.get("authorization")).toBe("Bearer real-access-token")
      expect(headers.get("chatgpt-account-id")).toBe("acct_test")
    } finally {
      store.close()
    }
  })

  test("keeps no-session API key responses on the original endpoint", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return new Response("ok")
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const apiOptions = await hooks.auth?.loader?.(async () => ({ type: "api", key: "sk-test" }), {} as any)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: { authorization: `Bearer ${apiOptions?.apiKey}` },
        body: JSON.stringify({ model: "gpt", input: [] }),
      })

      expect(calls[0]?.url).toBe("https://proxy.test/openai/v1/responses")
      const headers = new Headers(calls[0]?.init?.headers)
      expect(headers.get("authorization")).toBe("Bearer sk-test")
      expect(headers.has(defaultConfig.headers.session)).toBe(false)
      expect(headers.has(defaultConfig.headers.compact)).toBe(false)
    } finally {
      store.close()
    }
  })

  test("rejects compact requests missing session headers before OAuth routing", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const tokenCalls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return new Response("ok")
    }) as typeof fetch
    const tokenFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      tokenCalls.push({ url: String(requestInput), init })
      throw new Error("token endpoint should not be called")
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch, { tokenFetch })
      await hooks.auth?.loader?.(
        async () => ({
          type: "oauth",
          refresh: "refresh-token",
          access: "expired-access-token",
          expires: Date.now() - 1,
          accountId: "acct_test",
        }),
        {} as any,
      )
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      const response = await wrappedFetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          authorization: `Bearer ${openAIOAuthDummyKey}`,
          [defaultConfig.headers.compact]: "1",
        },
        body: JSON.stringify({ model: "ignored", input: [{ role: "user", content: "compact without session" }] }),
      })

      expect(response.status).toBe(400)
      expect(await response.text()).toBe("OpenAI compact request is missing session header")
      expect(calls).toEqual([])
      expect(tokenCalls).toEqual([])
    } finally {
      store.close()
    }
  })

  test("binds the browser callback to localhost and keeps invalid callbacks isolated", async () => {
    const method = openAIAuthMethods[0] as any
    let flow: any
    try {
      flow = await method.authorize()
      const callbackResult = flow.callback()
      void callbackResult.catch(() => {})
      const authorize = new URL(flow.url)
      const wrong = await fetch("http://localhost:1455/auth/callback?state=wrong&error=ignored")
      expect(wrong.status).toBe(400)
      expect(wrong.headers.get("content-type")).toContain("text/plain")
      expect(wrong.headers.get("cache-control")).toBe("no-store")

      const error = encodeURIComponent("<script>alert(1)</script>")
      const callback = await fetch(`http://localhost:1455/auth/callback?state=${encodeURIComponent(authorize.searchParams.get("state")!)}&error_description=${error}`)
      expect(callback.status).toBe(400)
      expect(callback.headers.get("content-security-policy")).toContain("default-src 'none'")
      expect(await callback.text()).toBe("<script>alert(1)</script>")
      await expect(callbackResult).rejects.toThrow("<script>alert(1)</script>")
    } finally {
      disposeOpenAIOAuth()
    }
  })

  test("requires the matching state to cancel a browser flow", async () => {
    const method = openAIAuthMethods[0] as any
    let flow: any
    try {
      flow = await method.authorize()
      const callbackResult = flow.callback()
      void callbackResult.catch(() => {})
      const state = new URL(flow.url).searchParams.get("state")!
      const wrong = await fetch("http://localhost:1455/cancel?state=wrong")
      expect(wrong.status).toBe(400)
      const cancelled = await fetch(`http://localhost:1455/cancel?state=${encodeURIComponent(state)}`)
      expect(cancelled.status).toBe(200)
      expect(await cancelled.text()).toBe("Login cancelled")
      await expect(callbackResult).rejects.toThrow("Login cancelled")
    } finally {
      disposeOpenAIOAuth()
    }
  })

  test("refreshes inside the safety margin and validates token lifetime", async () => {
    const calls: RequestInit[] = []
    let saved: any
    const source = { type: "oauth" as const, refresh: "refresh-safety", access: "old", expires: Date.now() + 30_000 }
    const oauth = createOpenAIOAuth({
      getAuth: async () => source,
      tokenFetch: (async (_input, init) => {
        calls.push(init!)
        return Response.json({ access_token: "new", refresh_token: "rotated", expires_in: 3600 })
      }) as typeof fetch,
      async setAuth(auth) { saved = auth },
    })
    const request = await oauth.requestInit({ headers: {} })
    expect(new Headers(request.headers).get("authorization")).toBe("Bearer new")
    expect(calls).toHaveLength(1)
    expect(saved.refresh).toBe("rotated")

    const invalid = createOpenAIOAuth({
      getAuth: async () => ({ type: "oauth", refresh: "refresh-invalid", access: "old", expires: 0 }),
      tokenFetch: (async () => Response.json({ access_token: "new", expires_in: Infinity })) as typeof fetch,
      async setAuth() { throw new Error("must not persist") },
    })
    await expect(invalid.requestInit({ headers: {} })).rejects.toThrow("invalid expires_in")
  })

  test("retries only an explicit invalid_token response once", async () => {
    let current = { type: "oauth" as const, refresh: "refresh-retry", access: "old", expires: Date.now() + 120_000 }
    let refreshes = 0
    const oauth = createOpenAIOAuth({
      getAuth: async () => current,
      tokenFetch: (async () => {
        refreshes++
        return Response.json({ access_token: "new", refresh_token: "rotated", expires_in: 3600 })
      }) as typeof fetch,
      async setAuth(auth) { current = auth },
    })
    const calls: string[] = []
    const response = await oauth.request({ method: "POST", body: "stable" }, async (init) => {
      calls.push(new Headers(init.headers).get("authorization") ?? "")
      return calls.length === 1
        ? Response.json({ error: "invalid_token" }, { status: 401 })
        : new Response("ok")
    })
    expect(response.status).toBe(200)
    expect(calls).toEqual(["Bearer old", "Bearer new"])
    expect(refreshes).toBe(1)

    const ordinary = createOpenAIOAuth({
      getAuth: async () => ({ type: "oauth", refresh: "refresh-ordinary", access: "old", expires: Date.now() + 120_000 }),
      tokenFetch: (async () => { throw new Error("must not refresh") }) as typeof fetch,
    })
    const unchanged = await ordinary.request({ headers: {} }, async () => new Response("unauthorized", { status: 401 }))
    expect(unchanged.status).toBe(401)
  })

  test("shares one refresh flight between OAuth instances", async () => {
    let current = { type: "oauth" as const, refresh: "refresh-shared", access: "old", expires: 0 }
    let refreshes = 0
    const tokenFetch = (async () => {
      refreshes++
      await new Promise((resolve) => setTimeout(resolve, 5))
      return Response.json({ access_token: "new", refresh_token: "rotated", expires_in: 3600 })
    }) as typeof fetch
    const make = () => createOpenAIOAuth({
      getAuth: async () => current,
      tokenFetch,
      async setAuth(auth) { current = auth },
    })
    const [first, second] = await Promise.all([make().requestInit({ headers: {} }), make().requestInit({ headers: {} })])
    expect(refreshes).toBe(1)
    expect(new Headers(first.headers).get("authorization")).toBe("Bearer new")
    expect(new Headers(second.headers).get("authorization")).toBe("Bearer new")
  })

  test("shares one token refresh across concurrent OpenAI OAuth responses", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const tokenCalls: Array<{ url: string; init?: RequestInit }> = []
    const savedAuth: any[] = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return new Response("ok")
    }) as typeof fetch
    const tokenFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      tokenCalls.push({ url: String(requestInput), init })
      return new Response(JSON.stringify({ access_token: "new", expires_in: 3600 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch, {
        tokenFetch,
        async setOpenAIAuth(auth) {
          savedAuth.push(auth)
        },
      })
      await hooks.auth?.loader?.(
        async () => ({
          type: "oauth",
          refresh: "refresh-token",
          access: "expired-access-token",
          expires: Date.now() - 1,
          accountId: "acct_test",
        }),
        {} as any,
      )
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      await Promise.all(
        ["ses_oauth_1", "ses_oauth_2"].map((sessionID) =>
          wrappedFetch("https://api.openai.com/v1/responses", {
            method: "POST",
            headers: {
              authorization: `Bearer ${openAIOAuthDummyKey}`,
              [defaultConfig.headers.session]: sessionID,
            },
            body: JSON.stringify({ model: "gpt", input: [] }),
          }),
        ),
      )

      expect(tokenCalls).toHaveLength(1)
      expect(tokenCalls[0]?.url).toBe("https://auth.openai.com/oauth/token")
      expect(String(tokenCalls[0]?.init?.body)).toContain("refresh_token=refresh-token")
      expect(calls).toHaveLength(2)
      for (const call of calls) {
        expect(new Headers(call.init?.headers).get("authorization")).toBe("Bearer new")
      }
      expect(savedAuth).toHaveLength(1)
      expect(savedAuth[0]?.access).toBe("new")
    } finally {
      store.close()
    }
  })
})
