import type { Hooks } from "@opencode-ai/plugin"
import { createServer } from "node:http"
import { setTimeout as sleep } from "node:timers/promises"

type OAuthRecord = Record<string, unknown>
type PkceCodes = { verifier: string; challenge: string }
type TokenResponse = {
  id_token?: string
  access_token: string
  refresh_token?: string
  expires_in?: number
}
type OpenAIOAuthResult = Omit<OpenAIOAuthAuth, "type">
type PendingOAuth = {
  pkce: PkceCodes
  state: string
  resolve: (tokens: TokenResponse) => void
  reject: (error: Error) => void
}

export type OAuthFetchLike = typeof fetch
export type OpenAIAuthMethods = NonNullable<Hooks["auth"]>["methods"]
export type OpenAIOAuthAuth = {
  type: "oauth"
  refresh: string
  access: string
  expires: number
  accountId?: string
}
export type OpenAIOAuthOptions = {
  getAuth: () => Promise<OpenAIOAuthAuth | undefined>
  setAuth?: (auth: OpenAIOAuthAuth) => Promise<void>
  tokenFetch?: OAuthFetchLike
}

export const openAIOAuthDummyKey = "opencode-oauth-dummy-key"
// Public OpenCode/Codex OAuth client id, not a secret.
const openAICodexOAuthClientID = "app_EMoamEEZ73f0CkXaXp7hrann"
const openAIIssuer = "https://auth.openai.com"
const openAITokenEndpoint = "https://auth.openai.com/oauth/token"
const openAIOAuthPort = 1455
const openAIOAuthPollingSafetyMarginMs = 3000

let oauthServer: ReturnType<typeof createServer> | undefined
let pendingOAuth: PendingOAuth | undefined

function asRecord(value: unknown): OAuthRecord | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as OAuthRecord) : undefined
}

function base64Url(value: ArrayBuffer | Uint8Array) {
  return Buffer.from(value instanceof Uint8Array ? value : new Uint8Array(value)).toString("base64url")
}

async function generatePKCE(): Promise<PkceCodes> {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
  const verifier = Array.from(crypto.getRandomValues(new Uint8Array(43)))
    .map((byte) => chars[byte % chars.length])
    .join("")
  const challenge = base64Url(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)))
  return { verifier, challenge }
}

function randomState() {
  return base64Url(crypto.getRandomValues(new Uint8Array(32)))
}

function parseJwtClaims(token: string): OAuthRecord | undefined {
  const parts = token.split(".")
  if (parts.length !== 3) return undefined
  try {
    return asRecord(JSON.parse(Buffer.from(parts[1], "base64url").toString("utf8")))
  } catch {
    return undefined
  }
}

function extractAccountIdFromClaims(claims: OAuthRecord) {
  if (typeof claims.chatgpt_account_id === "string") return claims.chatgpt_account_id
  const auth = asRecord(claims["https://api.openai.com/auth"])
  if (typeof auth?.chatgpt_account_id === "string") return auth.chatgpt_account_id
  if (!Array.isArray(claims.organizations)) return undefined
  const organization = asRecord(claims.organizations[0])
  return typeof organization?.id === "string" ? organization.id : undefined
}

function extractAccountId(tokens: TokenResponse) {
  const idClaims = tokens.id_token ? parseJwtClaims(tokens.id_token) : undefined
  const idAccount = idClaims ? extractAccountIdFromClaims(idClaims) : undefined
  if (idAccount) return idAccount
  const accessClaims = parseJwtClaims(tokens.access_token)
  return accessClaims ? extractAccountIdFromClaims(accessClaims) : undefined
}

function buildAuthorizeUrl(redirectUri: string, pkce: PkceCodes, state: string) {
  const params = new URLSearchParams({
    response_type: "code",
    client_id: openAICodexOAuthClientID,
    redirect_uri: redirectUri,
    scope: "openid profile email offline_access",
    code_challenge: pkce.challenge,
    code_challenge_method: "S256",
    id_token_add_organizations: "true",
    codex_cli_simplified_flow: "true",
    state,
    originator: "opencode",
  })
  return `${openAIIssuer}/oauth/authorize?${params.toString()}`
}

async function exchangeCodeForTokens(code: string, redirectUri: string, codeVerifier: string) {
  const response = await fetch(openAITokenEndpoint, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      code,
      redirect_uri: redirectUri,
      client_id: openAICodexOAuthClientID,
      code_verifier: codeVerifier,
    }).toString(),
  })
  if (!response.ok) throw new Error(`Token exchange failed: ${response.status}`)
  return (await response.json()) as TokenResponse
}

function tokenResponseToAuth(tokens: TokenResponse): OpenAIOAuthResult | undefined {
  if (!tokens.refresh_token) return undefined
  const accountId = extractAccountId(tokens)
  return {
    refresh: tokens.refresh_token,
    access: tokens.access_token,
    expires: Date.now() + (tokens.expires_in ?? 3600) * 1000,
    ...(accountId ? { accountId } : {}),
  }
}

async function startOAuthServer() {
  const redirectUri = `http://localhost:${openAIOAuthPort}/auth/callback`
  if (oauthServer) return redirectUri

  oauthServer = createServer((req, res) => {
    const url = new URL(req.url || "/", `http://localhost:${openAIOAuthPort}`)

    if (url.pathname === "/auth/callback") {
      const code = url.searchParams.get("code")
      const state = url.searchParams.get("state")
      const error = url.searchParams.get("error_description") ?? url.searchParams.get("error")

      if (error || !code) {
        pendingOAuth?.reject(new Error(error ?? "Missing authorization code"))
        pendingOAuth = undefined
        res.writeHead(error ? 200 : 400, { "content-type": "text/html; charset=utf-8" })
        res.end(error ?? "Missing authorization code")
        return
      }

      if (!pendingOAuth || state !== pendingOAuth.state) {
        pendingOAuth?.reject(new Error("Invalid state"))
        pendingOAuth = undefined
        res.writeHead(400, { "content-type": "text/html; charset=utf-8" })
        res.end("Invalid state")
        return
      }

      const current = pendingOAuth
      pendingOAuth = undefined
      exchangeCodeForTokens(code, redirectUri, current.pkce.verifier).then(current.resolve, current.reject)
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" })
      res.end("Authorization successful. You can close this window and return to OpenCode.")
      return
    }

    if (url.pathname === "/cancel") {
      pendingOAuth?.reject(new Error("Login cancelled"))
      pendingOAuth = undefined
      res.writeHead(200)
      res.end("Login cancelled")
      return
    }

    res.writeHead(404)
    res.end("Not found")
  })

  await new Promise<void>((resolve, reject) => {
    const onError = (error: Error) => {
      oauthServer = undefined
      reject(error)
    }
    oauthServer!.once("error", onError)
    oauthServer!.listen(openAIOAuthPort, () => {
      oauthServer!.removeListener("error", onError)
      resolve()
    })
  })
  return redirectUri
}

function stopOAuthServer() {
  oauthServer?.close(() => {})
  oauthServer = undefined
}

function waitForOAuthCallback(pkce: PkceCodes, state: string) {
  pendingOAuth?.reject(new Error("Superseded by a newer OpenAI authorize request"))
  return new Promise<TokenResponse>((resolve, reject) => {
    pendingOAuth = { pkce, state, resolve, reject }
  })
}

async function requestDeviceCode() {
  const response = await fetch(`${openAIIssuer}/api/accounts/deviceauth/usercode`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ client_id: openAICodexOAuthClientID }),
  })
  if (!response.ok) throw new Error("Failed to initiate device authorization")
  return (await response.json()) as { device_auth_id: string; user_code: string; interval: string }
}

async function pollDeviceCodeToken(device: { device_auth_id: string; user_code: string; interval: string }) {
  const interval = Math.max(Number.parseInt(device.interval, 10) || 5, 1) * 1000

  while (true) {
    const response = await fetch(`${openAIIssuer}/api/accounts/deviceauth/token`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ device_auth_id: device.device_auth_id, user_code: device.user_code }),
    })

    if (response.ok) {
      const data = (await response.json()) as { authorization_code: string; code_verifier: string }
      return exchangeCodeForTokens(data.authorization_code, `${openAIIssuer}/deviceauth/callback`, data.code_verifier)
    }
    if (response.status !== 403 && response.status !== 404) return undefined
    await sleep(interval + openAIOAuthPollingSafetyMarginMs)
  }
}

export const openAIAuthMethods: OpenAIAuthMethods = [
  {
    label: "ChatGPT Pro/Plus (browser)",
    type: "oauth",
    authorize: async () => {
      const redirectUri = await startOAuthServer()
      const pkce = await generatePKCE()
      const state = randomState()
      const callbackPromise = waitForOAuthCallback(pkce, state)

      return {
        url: buildAuthorizeUrl(redirectUri, pkce, state),
        instructions: "Complete authorization in your browser. This window will close automatically.",
        method: "auto" as const,
        callback: async () => {
          try {
            const auth = tokenResponseToAuth(await callbackPromise)
            return auth ? { type: "success" as const, ...auth } : { type: "failed" as const }
          } finally {
            stopOAuthServer()
          }
        },
      }
    },
  },
  {
    label: "ChatGPT Pro/Plus (headless)",
    type: "oauth",
    authorize: async () => {
      const device = await requestDeviceCode()
      return {
        url: `${openAIIssuer}/codex/device`,
        instructions: `Enter code: ${device.user_code}`,
        method: "auto" as const,
        callback: async () => {
          const tokens = await pollDeviceCodeToken(device)
          const auth = tokens ? tokenResponseToAuth(tokens) : undefined
          return auth ? { type: "success" as const, ...auth } : { type: "failed" as const }
        },
      }
    },
  },
  {
    label: "Manually enter API Key",
    type: "api",
  },
]

export function usesOpenAIOAuth(providerID: string, headers: Headers) {
  if (providerID !== "openai") return false
  if (headers.get("authorization")?.trim() === `Bearer ${openAIOAuthDummyKey}`) return true
  return headers.has("chatgpt-account-id")
}

export function asOpenAIOAuth(value: unknown): OpenAIOAuthAuth | undefined {
  const auth = asRecord(value)
  if (auth?.type !== "oauth") return undefined
  if (typeof auth.access !== "string" || typeof auth.refresh !== "string" || typeof auth.expires !== "number") {
    return undefined
  }
  const result: OpenAIOAuthAuth = {
    type: "oauth",
    access: auth.access,
    refresh: auth.refresh,
    expires: auth.expires,
  }
  if (typeof auth.accountId === "string") result.accountId = auth.accountId
  return result
}

export function createOpenAIOAuth(options: OpenAIOAuthOptions) {
  let refreshAuth: Promise<OpenAIOAuthAuth> | undefined

  async function freshAuth() {
    const auth = await options.getAuth()
    if (!auth) return undefined
    if (auth.expires > Date.now()) return auth

    refreshAuth ??= (options.tokenFetch ?? fetch)(openAITokenEndpoint, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "refresh_token",
        refresh_token: auth.refresh,
        client_id: openAICodexOAuthClientID,
      }).toString(),
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Token refresh failed: ${response.status}`)
        const payload = asRecord(await response.json())
        if (typeof payload?.access_token !== "string") throw new Error("Token refresh response is missing access_token")
        const next: OpenAIOAuthAuth = {
          type: "oauth",
          refresh: typeof payload.refresh_token === "string" ? payload.refresh_token : auth.refresh,
          access: payload.access_token,
          expires: Date.now() + Number(payload.expires_in ?? 3600) * 1000,
          ...(auth.accountId ? { accountId: auth.accountId } : {}),
        }
        await options.setAuth?.(next)
        return next
      })
      .finally(() => {
        refreshAuth = undefined
      })

    return refreshAuth
  }

  return {
    async requestInit(requestInit: RequestInit): Promise<RequestInit | undefined> {
      const auth = await freshAuth()
      if (!auth) return undefined

      const headers = new Headers(requestInit.headers)
      headers.set("authorization", `Bearer ${auth.access}`)
      if (auth.accountId) headers.set("ChatGPT-Account-Id", auth.accountId)
      return { ...requestInit, headers }
    },
  }
}
