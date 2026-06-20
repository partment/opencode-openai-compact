import { describe, expect, test } from "vitest"
import { createCompactHooks } from "../src/compact.js"
import { openAIOAuthDummyKey } from "../src/oauth.js"
import { defaultConfig } from "../src/schema.js"
import { CheckpointStore } from "../src/state.js"

function jsonBody(init: RequestInit | undefined) {
  return JSON.parse(typeof init?.body === "string" ? init.body : "{}")
}

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
          expires: Date.now() + 60_000,
        }),
        {} as any,
      )
      expect(oauthOptions).toEqual({ apiKey: openAIOAuthDummyKey })

      const apiOptions = await hooks.auth?.loader?.(async () => ({ type: "api", key: "sk-test" }), {} as any)
      expect(apiOptions).toEqual({ apiKey: "sk-test" })
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
      return new Response(
        JSON.stringify({
          id: "resp_compacted",
          model: defaultConfig.providers.openai.compactModel,
          created_at: 1,
          output: [{ type: "compaction_summary", encrypted_content: "compacted" }],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      )
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
        body: JSON.stringify({ model: "ignored", input: [{ role: "user", content: "hello" }] }),
      })

      expect(calls[0]?.url).toBe("https://api.openai.com/v1/responses/compact")
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
      expect(followupBody.input[0]).toEqual({ type: "compaction", encrypted_content: "compacted" })
    } finally {
      store.close()
    }
  })

  test("uses an API key checkpoint after switching to OpenAI OAuth", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      if (String(requestInput).endsWith("/responses/compact")) {
        return new Response(
          JSON.stringify({
            id: "resp_api_compacted",
            model: defaultConfig.providers.openai.compactModel,
            created_at: 1,
            output: [{ type: "compaction_summary", encrypted_content: "api-compacted" }],
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        )
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
        body: JSON.stringify({ model: "ignored", input: [{ role: "user", content: "hello" }] }),
      })

      expect(calls[0]?.url).toBe("https://api.openai.com/v1/responses/compact")
      expect(store.count()).toBe(1)

      await apiHooks.event?.({
        event: {
          type: "message.part.updated",
          properties: {
            sessionID: "ses_switch",
            part: { messageID: "msg_api_checkpoint", type: "text", text: defaultConfig.summary },
            time: 2,
          },
        } as any,
      })
      expect(store.count()).toBe(1)

      const oauthHooks = createCompactHooks(defaultConfig, store, fakeFetch)
      await oauthHooks.auth?.loader?.(
        async () => ({
          type: "oauth",
          refresh: "refresh-token",
          access: "real-access-token",
          expires: Date.now() + 60_000,
          accountId: "acct_test",
        }),
        {} as any,
      )
      const oauthCfg: any = {}
      await oauthHooks.config?.(oauthCfg)
      await oauthHooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: [{ info: { id: "msg_after", sessionID: "ses_switch", time: { created: 3 } } }] } as any,
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
      expect(followupBody.input[0]).toEqual({ type: "compaction", encrypted_content: "api-compacted" })
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
      return new Response(
        JSON.stringify({
          id: "resp_after_switch",
          model: defaultConfig.providers.openai.compactModel,
          created_at: 1,
          output: [{ type: "compaction_summary", encrypted_content: "after-switch" }],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      )
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
        body: JSON.stringify({ model: "ignored", input: [{ role: "user", content: "hello" }] }),
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
          expires: Date.now() + 60_000,
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
        body: JSON.stringify({ model: "ignored", input: [{ role: "user", content: "after switch" }] }),
      })

      expect(response.ok).toBe(true)
      expect(calls[0]?.url).toBe("https://chatgpt.com/backend-api/codex/responses/compact")
      expect(oauthStore.count()).toBe(1)
    } finally {
      if (!apiStoreDisposed) apiStore.close()
      oauthStore.close()
    }
  })

  test("routes OpenAI OAuth compaction to ChatGPT Codex compact endpoint", async () => {
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
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      await hooks.auth?.loader?.(
        async () => ({
          type: "oauth",
          refresh: "refresh-token",
          access: "real-access-token",
          expires: Date.now() + 60_000,
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
        body: JSON.stringify({ model: "ignored", input: [{ role: "user", content: "hello" }] }),
      })

      expect(calls[0]?.url).toBe("https://chatgpt.com/backend-api/codex/responses/compact")
      const headers = new Headers(calls[0]?.init?.headers)
      expect(headers.get("authorization")).toBe("Bearer real-access-token")
      expect(headers.get("chatgpt-account-id")).toBe("acct_test")
      expect(headers.has(defaultConfig.headers.compact)).toBe(false)
      expect(headers.has(defaultConfig.headers.session)).toBe(false)
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
          expires: Date.now() + 60_000,
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

  test("refreshes expired OpenAI OAuth token before routing responses", async () => {
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
      return new Response(JSON.stringify({ access_token: "refreshed-access-token", expires_in: 3600 }), {
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

      await wrappedFetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          authorization: `Bearer ${openAIOAuthDummyKey}`,
          [defaultConfig.headers.session]: "ses_oauth",
        },
        body: JSON.stringify({ model: "gpt", input: [] }),
      })

      expect(tokenCalls[0]?.url).toBe("https://auth.openai.com/oauth/token")
      expect(String(tokenCalls[0]?.init?.body)).toContain("refresh_token=refresh-token")
      expect(new Headers(calls[0]?.init?.headers).get("authorization")).toBe("Bearer refreshed-access-token")
      expect(savedAuth[0]?.access).toBe("refreshed-access-token")
    } finally {
      store.close()
    }
  })
})
