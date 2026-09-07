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
  id: string
  pkce: PkceCodes
  state: string
  timeout: ReturnType<typeof setTimeout>
  resolve: (tokens: TokenResponse) => void
  reject: (error: Error) => void
}
type DeviceCode = {
  device_auth_id: string
  user_code: string
  intervalMs: number
  expiresAt: number
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
  setAuth?: (auth: OpenAIOAuthAuth) => void | Promise<void>
  tokenFetch?: OAuthFetchLike
}

export const openAIOAuthDummyKey = "opencode-oauth-dummy-key"
// Public OpenCode/Codex OAuth client id, not a secret.
const openAICodexOAuthClientID = "app_EMoamEEZ73f0CkXaXp7hrann"
const openAIIssuer = "https://auth.openai.com"
const openAITokenEndpoint = "https://auth.openai.com/oauth/token"
const openAIOAuthPort = 1455
const openAIOAuthRequestTimeoutMs = 15_000
const openAIOAuthLoginTimeoutMs = 10 * 60_000
const openAIOAuthRefreshSafetyMarginMs = 60_000
const openAIOAuthDefaultLifetimeSeconds = 3600
const openAIOAuthMaxLifetimeSeconds = 30 * 24 * 60 * 60
const openAIOAuthPollingSafetyMarginMs = 3000
const openAIOAuthRefreshCacheMs = 30_000

let oauthServer: ReturnType<typeof createServer> | undefined
let pendingOAuth: PendingOAuth | undefined
let oauthRuntimeReferences = 0
const activeDeviceControllers = new Set<AbortController>()
const refreshFlights = new Map<string, Promise<OpenAIOAuthAuth>>()
const refreshCache = new Map<string, { auth: OpenAIOAuthAuth; expiresAt: number }>()

function asRecord(value: unknown): OAuthRecord | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as OAuthRecord) : undefined
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

function tokenLifetimeSeconds(value: unknown) {
  if (value === undefined) return openAIOAuthDefaultLifetimeSeconds
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 || value > openAIOAuthMaxLifetimeSeconds) {
    throw new Error("OAuth token response contains an invalid expires_in")
  }
  return value
}

function parseTokenResponse(value: unknown): TokenResponse {
  const payload = asRecord(value)
  if (typeof payload?.access_token !== "string" || !payload.access_token) {
    throw new Error("OAuth token response is missing access_token")
  }
  if (payload.refresh_token !== undefined &&
    (typeof payload.refresh_token !== "string" || !payload.refresh_token)) {
    throw new Error("OAuth token response contains an invalid refresh_token")
  }
  if (payload.id_token !== undefined && typeof payload.id_token !== "string") {
    throw new Error("OAuth token response contains an invalid id_token")
  }
  const expiresIn = tokenLifetimeSeconds(payload.expires_in)
  return {
    access_token: payload.access_token,
    ...(typeof payload.refresh_token === "string" ? { refresh_token: payload.refresh_token } : {}),
    ...(typeof payload.id_token === "string" ? { id_token: payload.id_token } : {}),
    expires_in: expiresIn,
  }
}

function authLifetime(tokens: TokenResponse) {
  return Date.now() + tokenLifetimeSeconds(tokens.expires_in) * 1000
}

function sameAuth(a: OpenAIOAuthAuth | undefined, b: OpenAIOAuthAuth | undefined) {
  return !!a && !!b && a.refresh === b.refresh && a.access === b.access && a.expires === b.expires &&
    a.accountId === b.accountId
}

function refreshKey(auth: OpenAIOAuthAuth) {
  return `${auth.refresh}\0${auth.accountId ?? ""}`
}

function callbackHeaders(contentType = "text/plain; charset=utf-8") {
  return {
    "content-type": contentType,
    "cache-control": "no-store",
    pragma: "no-cache",
    "x-content-type-options": "nosniff",
    "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
    "referrer-policy": "no-referrer",
  }
}

function callbackResponse(res: import("node:http").ServerResponse, status: number, text: string) {
  res.writeHead(status, callbackHeaders())
  res.end(text)
}

function abortError(signal: AbortSignal) {
  return signal.reason instanceof Error ? signal.reason : new Error("OAuth request was aborted")
}

async function fetchWithTimeout(
  fetcher: OAuthFetchLike,
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs = openAIOAuthRequestTimeoutMs,
  parentSignal?: AbortSignal,
) {
  const timeout = new AbortController()
  const signals = [timeout.signal, init.signal, parentSignal].filter((signal): signal is AbortSignal => !!signal)
  const signal = signals.length === 1 ? signals[0] : AbortSignal.any(signals)
  const timer = setTimeout(() => timeout.abort(new Error("OAuth HTTP request timed out")), timeoutMs)
  try {
    return await fetcher(input, { ...init, signal })
  } finally {
    clearTimeout(timer)
  }
}

function sleepWithSignal(ms: number, signal: AbortSignal) {
  return sleep(ms, undefined, { signal }).catch(() => { throw abortError(signal) })
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
  const response = await fetchWithTimeout(fetch, openAITokenEndpoint, {
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
  return parseTokenResponse(await response.json())
}

function tokenResponseToAuth(tokens: TokenResponse): OpenAIOAuthResult | undefined {
  if (!tokens.refresh_token) return undefined
  const accountId = extractAccountId(tokens)
  return {
    refresh: tokens.refresh_token,
    access: tokens.access_token,
    expires: authLifetime(tokens),
    ...(accountId ? { accountId } : {}),
  }
}

function settlePendingOAuth(current: PendingOAuth, error?: Error, tokens?: TokenResponse) {
  clearTimeout(current.timeout)
  if (pendingOAuth === current) pendingOAuth = undefined
  if (error) current.reject(error)
  else if (tokens) current.resolve(tokens)
}

async function startOAuthServer() {
  const redirectUri = `http://localhost:${openAIOAuthPort}/auth/callback`
  if (oauthServer) return redirectUri

  oauthServer = createServer((req, res) => {
    const url = new URL(req.url || "/", `http://localhost:${openAIOAuthPort}`)
    const current = pendingOAuth

    if (url.pathname === "/auth/callback") {
      const state = url.searchParams.get("state")
      if (!current || state !== current.state) {
        callbackResponse(res, 400, "Invalid state")
        return
      }

      const code = url.searchParams.get("code")
      const error = url.searchParams.get("error_description") ?? url.searchParams.get("error")
      if (error || !code) {
        settlePendingOAuth(current, new Error(error ?? "Missing authorization code"))
        callbackResponse(res, 400, error ?? "Missing authorization code")
        return
      }

      settlePendingOAuth(current)
      exchangeCodeForTokens(code, redirectUri, current.pkce.verifier).then(current.resolve, current.reject)
      callbackResponse(res, 200, "Authorization successful. You can close this window and return to OpenCode.")
      return
    }

    if (url.pathname === "/cancel") {
      const state = url.searchParams.get("state")
      if (!current || state !== current.state) {
        callbackResponse(res, 400, "Invalid cancellation target")
        return
      }
      settlePendingOAuth(current, new Error("Login cancelled"))
      callbackResponse(res, 200, "Login cancelled")
      return
    }

    callbackResponse(res, 404, "Not found")
  })

  await new Promise<void>((resolve, reject) => {
    const onError = (error: NodeJS.ErrnoException) => {
      oauthServer = undefined
      const message = error.code === "EADDRINUSE"
        ? `OpenAI OAuth callback port ${openAIOAuthPort} on localhost is already in use; close the conflicting process and try again`
        : `OpenAI OAuth callback server could not listen on localhost:${openAIOAuthPort}: ${errorMessage(error)}`
      reject(new Error(message, { cause: error }))
    }
    oauthServer!.once("error", onError)
    oauthServer!.listen(openAIOAuthPort, "localhost", () => {
      oauthServer!.removeListener("error", onError)
      resolve()
    })
  })
  return redirectUri
}

function stopOAuthServer(flowID?: string) {
  if (flowID && pendingOAuth && pendingOAuth.id !== flowID) return
  oauthServer?.close(() => {})
  oauthServer = undefined
}

function waitForOAuthCallback(pkce: PkceCodes, state: string) {
  if (pendingOAuth) settlePendingOAuth(pendingOAuth, new Error("Superseded by a newer OpenAI authorize request"))
  return new Promise<TokenResponse>((resolve, reject) => {
    const id = randomState()
    const timeout = setTimeout(() => {
      const current = pendingOAuth
      if (!current || current.id !== id) return
      settlePendingOAuth(current, new Error("OpenAI OAuth authorization timed out"))
      stopOAuthServer(id)
    }, openAIOAuthLoginTimeoutMs)
    pendingOAuth = { id, pkce, state, timeout, resolve, reject }
  })
}

async function requestDeviceCode(): Promise<DeviceCode> {
  const response = await fetchWithTimeout(fetch, `${openAIIssuer}/api/accounts/deviceauth/usercode`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ client_id: openAICodexOAuthClientID }),
  })
  if (!response.ok) throw new Error(`Failed to initiate device authorization: ${response.status}`)
  const payload = asRecord(await response.json())
  if (typeof payload?.device_auth_id !== "string" || !payload.device_auth_id ||
    typeof payload.user_code !== "string" || !payload.user_code) {
    throw new Error("Device authorization response is missing device code fields")
  }
  const intervalSeconds = typeof payload.interval === "string"
    ? Number.parseInt(payload.interval, 10)
    : typeof payload.interval === "number" ? payload.interval : undefined
  const intervalMs = intervalSeconds !== undefined && Number.isFinite(intervalSeconds) && intervalSeconds > 0
    ? Math.min(intervalSeconds * 1000, 60_000)
    : 5000
  const lifetime = tokenLifetimeSeconds(payload.expires_in)
  return {
    device_auth_id: payload.device_auth_id,
    user_code: payload.user_code,
    intervalMs,
    expiresAt: Date.now() + Math.min(lifetime * 1000, openAIOAuthLoginTimeoutMs),
  }
}

async function pollDeviceCodeToken(device: DeviceCode, signal: AbortSignal) {
  while (Date.now() < device.expiresAt) {
    const response = await fetchWithTimeout(fetch, `${openAIIssuer}/api/accounts/deviceauth/token`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ device_auth_id: device.device_auth_id, user_code: device.user_code }),
    }, openAIOAuthRequestTimeoutMs, signal)

    if (response.ok) {
      const data = asRecord(await response.json())
      if (typeof data?.authorization_code !== "string" || typeof data.code_verifier !== "string") {
        throw new Error("Device authorization response is missing token exchange fields")
      }
      return exchangeCodeForTokens(data.authorization_code, `${openAIIssuer}/deviceauth/callback`, data.code_verifier)
    }
    if (response.status !== 403 && response.status !== 404) return undefined

    const remaining = device.expiresAt - Date.now()
    if (remaining <= 0) return undefined
    await sleepWithSignal(Math.min(device.intervalMs + openAIOAuthPollingSafetyMarginMs, remaining), signal)
  }
  return undefined
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
      const flowID = pendingOAuth?.id

      return {
        url: buildAuthorizeUrl(redirectUri, pkce, state),
        instructions: "Complete authorization in your browser. This window will close automatically.",
        method: "auto" as const,
        callback: async () => {
          try {
            const auth = tokenResponseToAuth(await callbackPromise)
            return auth ? { type: "success" as const, ...auth } : { type: "failed" as const }
          } finally {
            stopOAuthServer(flowID)
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
      const controller = new AbortController()
      activeDeviceControllers.add(controller)
      return {
        url: `${openAIIssuer}/codex/device`,
        instructions: `Enter code: ${device.user_code}`,
        method: "auto" as const,
        callback: async () => {
          try {
            const tokens = await pollDeviceCodeToken(device, controller.signal)
            const auth = tokens ? tokenResponseToAuth(tokens) : undefined
            return auth ? { type: "success" as const, ...auth } : { type: "failed" as const }
          } catch (error) {
            if (controller.signal.aborted) return { type: "failed" as const }
            throw error
          } finally {
            activeDeviceControllers.delete(controller)
          }
        },
      }
    },
  },
  {
    label: "Manually enter API Key",
    type: "api",
  },
]

export function retainOpenAIOAuthRuntime() {
  oauthRuntimeReferences++
}

export function releaseOpenAIOAuthRuntime() {
  if (oauthRuntimeReferences > 0) oauthRuntimeReferences--
  if (oauthRuntimeReferences === 0) disposeOpenAIOAuth()
}

export function disposeOpenAIOAuth() {
  if (pendingOAuth) {
    settlePendingOAuth(pendingOAuth, new Error("OpenAI OAuth runtime was disposed"))
    pendingOAuth = undefined
  }
  for (const controller of activeDeviceControllers) controller.abort(new Error("OpenAI OAuth runtime was disposed"))
  activeDeviceControllers.clear()
  refreshCache.clear()
  stopOAuthServer()
}

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

function authWithHeaders(requestInit: RequestInit, auth: OpenAIOAuthAuth): RequestInit {
  const headers = new Headers(requestInit.headers)
  headers.set("authorization", `Bearer ${auth.access}`)
  if (auth.accountId) headers.set("ChatGPT-Account-Id", auth.accountId)
  else headers.delete("ChatGPT-Account-Id")
  return { ...requestInit, headers }
}

async function isExplicitInvalidToken(response: Response) {
  if (response.status !== 401 && response.status !== 403) return false
  if (/\\binvalid_token\\b/i.test(response.headers.get("www-authenticate") ?? "")) return true
  try {
    const payload = asRecord(JSON.parse(await response.clone().text()))
    const error = payload?.error
    if (typeof error === "string") return error.toLowerCase() === "invalid_token"
    const record = asRecord(error)
    return [record?.code, record?.type, record?.error].some((value) =>
      typeof value === "string" && value.toLowerCase() === "invalid_token")
  } catch {
    return false
  }
}

export function createOpenAIOAuth(options: OpenAIOAuthOptions) {
  async function refreshFrom(auth: OpenAIOAuthAuth): Promise<OpenAIOAuthAuth> {
    const key = refreshKey(auth)
    const cached = refreshCache.get(key)
    if (cached && cached.expiresAt > Date.now()) return cached.auth
    const existing = refreshFlights.get(key)
    if (existing) return existing

    const flight = (async () => {
      const response = await fetchWithTimeout(options.tokenFetch ?? fetch, openAITokenEndpoint, {
        method: "POST",
        headers: { "content-type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          grant_type: "refresh_token",
          refresh_token: auth.refresh,
          client_id: openAICodexOAuthClientID,
        }).toString(),
      })
      if (!response.ok) throw new Error(`Token refresh failed: ${response.status}`)
      const payload = parseTokenResponse(await response.json())
      const next: OpenAIOAuthAuth = {
        type: "oauth",
        refresh: payload.refresh_token ?? auth.refresh,
        access: payload.access_token,
        expires: authLifetime(payload),
        ...(auth.accountId ? { accountId: auth.accountId } : {}),
      }

      if (options.setAuth) {
        const latest = await options.getAuth()
        if (!latest || sameAuth(latest, auth)) {
          await options.setAuth(next)
        } else if (latest.expires > Date.now() + openAIOAuthRefreshSafetyMarginMs) {
          return latest
        }
      }
      refreshCache.set(key, { auth: next, expiresAt: Date.now() + openAIOAuthRefreshCacheMs })
      return next
    })()
    refreshFlights.set(key, flight)
    try {
      return await flight
    } finally {
      if (refreshFlights.get(key) === flight) refreshFlights.delete(key)
    }
  }

  async function freshAuth(forceRefresh = false) {
    const auth = await options.getAuth()
    if (!auth) return undefined
    if (!forceRefresh && auth.expires > Date.now() + openAIOAuthRefreshSafetyMarginMs) return auth
    return refreshFrom(auth)
  }

  async function requestInit(requestInit: RequestInit, forceRefresh = false): Promise<RequestInit> {
    const auth = await freshAuth(forceRefresh)
    if (!auth) throw new Error("OpenAI OAuth credentials are unavailable; refusing to send the OAuth dummy key")
    return authWithHeaders(requestInit, auth)
  }

  return {
    requestInit,
    async request(
      requestInitValue: RequestInit,
      send: (requestInit: RequestInit) => Promise<Response>,
    ): Promise<Response> {
      const auth = await freshAuth()
      if (!auth) throw new Error("OpenAI OAuth credentials are unavailable; refusing to send the OAuth dummy key")
      const response = await send(authWithHeaders(requestInitValue, auth))
      if (!await isExplicitInvalidToken(response)) return response

      const latest = await options.getAuth()
      const retryAuth = latest && latest.access !== auth.access &&
        latest.expires > Date.now() + openAIOAuthRefreshSafetyMarginMs
        ? latest
        : await freshAuth(true)
      if (!retryAuth) throw new Error("OpenAI OAuth credentials are unavailable; refusing to retry with the OAuth dummy key")
      return send(authWithHeaders(requestInitValue, retryAuth))
    },
  }
}
