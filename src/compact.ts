import type { Hooks } from "@opencode-ai/plugin"
import { defaultConfig, type OpenAICompactConfig } from "./schema.js"
import {
  asOpenAIOAuth,
  createOpenAIOAuth,
  openAIAuthMethods,
  openAIOAuthDummyKey,
  usesOpenAIOAuth,
  type OpenAIOAuthAuth,
  type OAuthFetchLike,
} from "./oauth.js"
import { CheckpointStore, type AnyRecord, type Checkpoint } from "./state.js"

type FetchLike = typeof fetch
type MessageEntry = {
  info?: { id?: string; sessionID?: string; time?: { created?: number } }
  parts?: Array<{ type?: string; text?: string; messageID?: string; time?: { start?: number } }>
}
type MessageBoundary = { messageID: string; createdAt: number }
type PendingCompactResult = { providerID: string; responseID: string; items: AnyRecord[] }
type ProviderConfig = OpenAICompactConfig["providers"][string]
type StableInstructions = { instructions?: unknown; inputPrefix: unknown[] }
type CompactHookOptions = {
  setOpenAIAuth?: (auth: OpenAIOAuthAuth) => Promise<void>
  tokenFetch?: OAuthFetchLike
}

const wrappedFetch = "__opencodeOpenAICompactFetch"
const wrappedBaseFetch = "__opencodeOpenAICompactBaseFetch"
const chatGPTCodexResponsesEndpoint = "https://chatgpt.com/backend-api/codex/responses"
const chatGPTCodexCompactEndpoint = "https://chatgpt.com/backend-api/codex/responses/compact"
const openCodeCompactionDeveloperPrompt = "You are an anchored context summarization assistant for coding sessions."
const utilityAgents = new Set(["compaction", "title", "summary"])
const openCodeCompactionUserPromptStarts = [
  "Create a new anchored summary from the conversation history.",
  "Update the anchored summary below using the conversation history above.",
] as const

function asRecord(value: unknown): AnyRecord | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as AnyRecord) : undefined
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function urlOf(input: RequestInfo | URL): URL | undefined {
  try {
    if (input instanceof URL) return new URL(input.href)
    if (input instanceof Request) return new URL(input.url)
    return new URL(String(input))
  } catch {
    return undefined
  }
}

function pathWithoutTrailingSlash(value: string) {
  return value.length > 1 ? value.replace(/\/+$/, "") : value
}

export function isResponsesUrl(url: URL, config: OpenAICompactConfig) {
  return pathWithoutTrailingSlash(url.pathname).endsWith(config.responses.endpointPath)
}

export function compactUrl(url: URL, config: OpenAICompactConfig = defaultConfig): string {
  const next = new URL(url.href)
  const path = pathWithoutTrailingSlash(next.pathname)
  const prefix = path.slice(0, path.length - config.responses.endpointPath.length)
  next.pathname = `${prefix}${config.responses.compactEndpointPath}`
  return next.href
}

function requestHeaders(input: RequestInfo | URL, init?: RequestInit): Headers {
  const headers = new Headers(input instanceof Request ? input.headers : undefined)
  new Headers(init?.headers).forEach((value, key) => headers.set(key, value))
  return headers
}

function cleanedHeaders(headers: Headers, config: OpenAICompactConfig): Headers {
  const result = new Headers(headers)
  result.delete(config.headers.compact)
  result.delete(config.headers.session)
  return result
}

function fetchInit(init: RequestInit | undefined, headers: Headers): RequestInit {
  return init ? { ...init, headers } : { headers }
}

function compactMarkers(headers: Headers, config: OpenAICompactConfig) {
  const sessionID = headers.get(config.headers.session) ?? undefined
  const shouldCompact = headers.get(config.headers.compact) === "1"
  headers.delete(config.headers.compact)
  headers.delete(config.headers.session)
  return { sessionID, shouldCompact }
}

async function bodyText(input: RequestInfo | URL, init?: RequestInit): Promise<string | undefined> {
  if (typeof init?.body === "string") return init.body
  if (init?.body instanceof Uint8Array) return new TextDecoder().decode(init.body)
  if (input instanceof Request) return input.clone().text()
  return undefined
}

function contentText(value: unknown): string {
  if (typeof value === "string") return value
  if (!Array.isArray(value)) return ""
  return value
    .map((item) => {
      if (typeof item === "string") return item
      const record = asRecord(item)
      return typeof record?.text === "string" ? record.text : ""
    })
    .filter(Boolean)
    .join("\n")
}

function isOpenCodeCompactionDeveloperPrompt(value: unknown) {
  return contentText(value).trimStart().startsWith(openCodeCompactionDeveloperPrompt)
}

function isOpenCodeCompactionUserPrompt(value: unknown) {
  const text = contentText(value).trimStart()
  return openCodeCompactionUserPromptStarts.some((start) => text.startsWith(start))
}

function compactInput(value: unknown) {
  if (!Array.isArray(value)) return value
  return value.filter((item, index) => {
    const record = asRecord(item)
    if (!record) return true
    if (record.role === "developer" && isOpenCodeCompactionDeveloperPrompt(record.content)) return false
    if (index === value.length - 1 && record.role === "user" && isOpenCodeCompactionUserPrompt(record.content)) {
      return false
    }
    return true
  })
}

function compactBodyValue(key: string, value: unknown) {
  if (key === "input") return compactInput(value)
  if (key === "instructions" && isOpenCodeCompactionDeveloperPrompt(value)) return undefined
  return value
}

function leadingInstructionCount(input: unknown[]) {
  let index = 0
  while (true) {
    const role = asRecord(input[index])?.role
    if (role !== "developer" && role !== "system") return index
    index++
  }
}

function stableInstructionsFrom(body: AnyRecord | undefined): StableInstructions | undefined {
  if (!body) return undefined

  const inputPrefix = Array.isArray(body.input) ? body.input.slice(0, leadingInstructionCount(body.input)) : []
  const instructions = isOpenCodeCompactionDeveloperPrompt(body.instructions) ? undefined : body.instructions
  if (instructions === undefined && !inputPrefix.length) return undefined
  return { instructions, inputPrefix: structuredClone(inputPrefix) }
}

function instructionsFromSystem(system: unknown) {
  if (!Array.isArray(system)) return undefined
  if (!system.every((item): item is string => typeof item === "string")) return undefined
  const instructions = system.join("\n")
  if (!instructions || isOpenCodeCompactionDeveloperPrompt(instructions)) return undefined
  return instructions
}

function withStableInstructions(body: AnyRecord, stable: StableInstructions | undefined, allowInstructions: boolean): AnyRecord {
  if (!stable) return body

  const next = { ...body }
  if (allowInstructions && next.instructions === undefined && stable.instructions !== undefined) {
    next.instructions = structuredClone(stable.instructions)
  }
  if (stable.inputPrefix.length && Array.isArray(next.input)) {
    next.input = [...structuredClone(stable.inputPrefix), ...next.input.slice(leadingInstructionCount(next.input))]
  }
  return next
}

export function compactBody(
  body: AnyRecord,
  compactModel = defaultConfig.providers.openai.compactModel,
  config: OpenAICompactConfig = defaultConfig,
): AnyRecord {
  const result: AnyRecord = { model: compactModel }
  for (const key of config.compactBodyKeys) {
    const value = compactBodyValue(key, body[key])
    if (value !== undefined) result[key] = value
  }
  return result
}

export function compactedItemsFrom(value: unknown): AnyRecord[] | undefined {
  if (!Array.isArray(value)) return undefined
  const items = value
    .map(asRecord)
    .filter((item): item is AnyRecord => item !== undefined)
    .map((item) =>
      item.type === "compaction_summary" && typeof item.encrypted_content === "string"
        ? { type: "compaction", encrypted_content: item.encrypted_content }
        : item,
    )
  return items.length ? items : undefined
}

function parseJsonRecord(text: string | undefined): AnyRecord | undefined {
  if (!text) return undefined
  try {
    return asRecord(JSON.parse(text))
  } catch {
    return undefined
  }
}

function usageFrom(value: AnyRecord | undefined): AnyRecord {
  return {
    input_tokens: value?.input_tokens ?? 0,
    input_tokens_details: {
      cached_tokens: asRecord(value?.input_tokens_details)?.cached_tokens ?? 0,
    },
    output_tokens: value?.output_tokens ?? 0,
    output_tokens_details: {
      reasoning_tokens: asRecord(value?.output_tokens_details)?.reasoning_tokens ?? 0,
    },
    total_tokens: value?.total_tokens ?? Number(value?.input_tokens ?? 0) + Number(value?.output_tokens ?? 0),
  }
}

function responseMessageID(responseID: string) {
  return `msg_${responseID.replace(/[^a-zA-Z0-9]/g, "_")}`
}

function sseResponse(input: {
  responseID: string
  model: string
  createdAt: number
  summary: string
  usage?: AnyRecord
}): Response {
  const messageID = responseMessageID(input.responseID)
  const usage = usageFrom(input.usage)
  const message = {
    id: messageID,
    type: "message",
    status: "completed",
    role: "assistant",
    content: [
      {
        type: "output_text",
        text: input.summary,
        annotations: [],
        logprobs: [],
      },
    ],
  }
  const response = {
    id: input.responseID,
    object: "response",
    created_at: input.createdAt,
    model: input.model,
    status: "completed",
    output: [message],
    incomplete_details: null,
    service_tier: null,
    usage,
  }
  const events = [
    { type: "response.created", response: { ...response, status: "in_progress", output: [] } },
    { type: "response.output_item.added", output_index: 0, item: { ...message, status: "in_progress", content: [] } },
    {
      type: "response.output_text.delta",
      item_id: messageID,
      output_index: 0,
      content_index: 0,
      delta: input.summary,
      logprobs: [],
    },
    { type: "response.output_item.done", output_index: 0, item: message },
    { type: "response.completed", response },
  ]
  const stream = `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`
  return new Response(stream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
    },
  })
}

function messageCreatedAt(entry: MessageEntry) {
  const createdAt = entry.info?.time?.created
  return finiteNumber(createdAt) ? createdAt : undefined
}

function selectCheckpoint(
  checkpoints: Checkpoint[],
  entries: MessageEntry[],
): { checkpoint?: Checkpoint; clearActive: boolean } {
  const messageIDs = new Set(
    entries
      .map((entry) => entry.info?.id)
      .filter((id): id is string => typeof id === "string"),
  )
  for (let index = checkpoints.length - 1; index >= 0; index--) {
    const checkpoint = checkpoints[index]
    if (messageIDs.has(checkpoint.afterMessageID)) return { checkpoint, clearActive: false }
  }

  const createdAts = entries.map(messageCreatedAt).filter((createdAt): createdAt is number => createdAt !== undefined)
  if (!createdAts.length || createdAts.length !== entries.length) {
    return { clearActive: false }
  }

  const minCreatedAt = Math.min(...createdAts)
  for (let index = checkpoints.length - 1; index >= 0; index--) {
    const checkpoint = checkpoints[index]
    if (minCreatedAt >= checkpoint.afterCreatedAt) return { checkpoint, clearActive: false }
  }

  return { clearActive: true }
}

function sessionIDFromMessages(messages: MessageEntry[]): string | undefined {
  for (const message of messages) {
    const sessionID = (message.info as AnyRecord | undefined)?.sessionID
    if (typeof sessionID === "string") return sessionID
  }
  return undefined
}

function sortCheckpoints(checkpoints: Checkpoint[]) {
  return checkpoints.sort((a, b) => a.afterCreatedAt - b.afterCreatedAt || a.createdAt - b.createdAt)
}

function getProviderSessionMap<T>(map: Map<string, Map<string, T>>, providerID: string) {
  const existing = map.get(providerID)
  if (existing) return existing

  const created = new Map<string, T>()
  map.set(providerID, created)
  return created
}

function getProviderID(input: unknown) {
  const record = asRecord(input)
  const model = asRecord(record?.model)
  if (typeof model?.providerID === "string") return model.providerID

  const provider = asRecord(record?.provider)
  if (typeof provider?.providerID === "string") return provider.providerID
  if (typeof provider?.id === "string") return provider.id
  return undefined
}

function messageProviderKey(sessionID: string, messageID: string) {
  return `${sessionID}\0${messageID}`
}

export function createCompactHooks(
  config: OpenAICompactConfig,
  store: CheckpointStore,
  baseFetch: FetchLike = fetch,
  options: CompactHookOptions = {},
): Hooks {
  store.prune(config.state.retentionDays)

  const configuredProviders = new Set(Object.keys(config.providers))
  const checkpointsByProvider = new Map<string, Map<string, Checkpoint[]>>()
  for (const { sessionID, checkpoint } of store.loadAll()) {
    const sessions = getProviderSessionMap(checkpointsByProvider, checkpoint.providerID)
    const checkpoints = sessions.get(sessionID) ?? []
    checkpoints.push(checkpoint)
    sessions.set(sessionID, sortCheckpoints(checkpoints))
  }
  const pendingCompactResults = new Map<string, PendingCompactResult>()
  const activeCheckpointByProvider = new Map<string, Map<string, Checkpoint>>()
  const stableInstructionsByProvider = new Map<string, Map<string, StableInstructions>>()
  const pendingSystemByProvider = new Map<string, Map<string, string>>()
  const providerByMessage = new Map<string, string>()
  let getOpenAIAuth: (() => Promise<OpenAIOAuthAuth | undefined>) | undefined
  const openAIOAuth = createOpenAIOAuth({
    getAuth: async () => getOpenAIAuth?.(),
    setAuth: options.setOpenAIAuth,
    tokenFetch: options.tokenFetch,
  })

  function rememberMessageProvider(input: unknown, output: unknown) {
    const providerID = getProviderID(input)
    if (!providerID || !configuredProviders.has(providerID)) return

    const inputRecord = asRecord(input)
    const sessionID = inputRecord?.sessionID
    if (typeof sessionID !== "string") return

    const outputRecord = asRecord(output)
    const message = asRecord(outputRecord?.message)
    const messageID = typeof message?.id === "string" ? message.id : inputRecord?.messageID
    if (typeof messageID !== "string") return

    providerByMessage.set(messageProviderKey(sessionID, messageID), providerID)
  }

  function providerIDFromMessages(messages: MessageEntry[]) {
    for (const message of messages) {
      const sessionID = (message.info as AnyRecord | undefined)?.sessionID
      const messageID = message.info?.id
      if (typeof sessionID !== "string" || typeof messageID !== "string") continue
      const providerID = providerByMessage.get(messageProviderKey(sessionID, messageID))
      if (providerID) return providerID
    }
    return undefined
  }

  function providerIDFromTrimmedSessionCheckpoint(messages: MessageEntry[]) {
    const sessionID = sessionIDFromMessages(messages)
    if (!sessionID) return undefined

    const messageIDs = new Set(
      messages
        .map((message) => message.info?.id)
        .filter((id): id is string => typeof id === "string"),
    )
    let result: string | undefined
    for (const [providerID, sessions] of checkpointsByProvider) {
      const checkpoints = sessions.get(sessionID)
      if (!checkpoints?.length) continue
      if (checkpoints.some((checkpoint) => messageIDs.has(checkpoint.afterMessageID))) return undefined
      if (result) return undefined
      result = providerID
    }
    return result
  }

  function transformProviderID(input: unknown, messages: MessageEntry[]) {
    return getProviderID(input) ?? providerIDFromMessages(messages) ?? providerIDFromTrimmedSessionCheckpoint(messages)
  }

  function storeStableInstructions(providerID: string, sessionID: string, stable: StableInstructions) {
    const sessions = getProviderSessionMap(stableInstructionsByProvider, providerID)
    const previous = sessions.get(sessionID)
    sessions.set(sessionID, {
      instructions: stable.instructions !== undefined ? stable.instructions : previous?.instructions,
      inputPrefix: stable.inputPrefix.length ? stable.inputPrefix : (previous?.inputPrefix ?? []),
    })
  }

  function rememberStableInstructions(providerID: string, sessionID: string, body: AnyRecord | undefined) {
    const stable = stableInstructionsFrom(body)
    if (stable) storeStableInstructions(providerID, sessionID, stable)
  }

  function rememberPendingSystem(providerID: string, sessionID: string, system: unknown) {
    const sessions = getProviderSessionMap(pendingSystemByProvider, providerID)
    const instructions = instructionsFromSystem(system)
    if (instructions) sessions.set(sessionID, instructions)
    else sessions.delete(sessionID)
  }

  function promotePendingSystem(providerID: string, sessionID: string) {
    const sessions = pendingSystemByProvider.get(providerID)
    const instructions = sessions?.get(sessionID)
    if (!instructions) return
    storeStableInstructions(providerID, sessionID, { instructions, inputPrefix: [] })
    sessions?.delete(sessionID)
  }

  function addCheckpoint(
    providerID: string,
    sessionID: string,
    responseID: string,
    boundary: MessageBoundary,
    items: AnyRecord[],
  ): Checkpoint {
    const checkpoint: Checkpoint = {
      providerID,
      responseID,
      afterMessageID: boundary.messageID,
      afterCreatedAt: boundary.createdAt,
      createdAt: Date.now(),
      items,
    }
    const sessions = getProviderSessionMap(checkpointsByProvider, providerID)
    const checkpoints = sessions.get(sessionID) ?? []
    sessions.set(
      sessionID,
      sortCheckpoints([...checkpoints.filter((checkpoint) => checkpoint.responseID !== responseID), checkpoint]),
    )
    store.upsert(sessionID, checkpoint)
    store.prune(config.state.retentionDays)
    return checkpoint
  }

  function compactCheckpointFromEvent(event: AnyRecord) {
    if (event.type !== "message.part.updated") return undefined
    const properties = asRecord(event.properties)
    const sessionID = properties?.sessionID
    const pending = typeof sessionID === "string" ? pendingCompactResults.get(sessionID) : undefined
    const part = asRecord(properties?.part)
    const messageID = part?.messageID
    const createdAt = properties?.time
    if (!pending || part?.text !== config.summary) return undefined
    if (typeof sessionID !== "string" || typeof messageID !== "string" || !finiteNumber(createdAt)) return undefined
    return { providerID: pending.providerID, sessionID, responseID: pending.responseID, boundary: { messageID, createdAt } }
  }

  function trimMessagesAfterCheckpoint(providerID: string, messages: MessageEntry[]) {
    const sessionID = sessionIDFromMessages(messages)
    const checkpoints = sessionID ? checkpointsByProvider.get(providerID)?.get(sessionID) : undefined
    if (!sessionID || !checkpoints) return

    const { checkpoint, clearActive } = selectCheckpoint(checkpoints, messages)
    const activeCheckpoints = getProviderSessionMap(activeCheckpointByProvider, providerID)
    if (checkpoint) {
      activeCheckpoints.set(sessionID, checkpoint)
    } else if (clearActive) {
      activeCheckpoints.delete(sessionID)
    }
    if (!checkpoint) return

    const index = messages.findIndex((message) => message.info?.id === checkpoint.afterMessageID)
    if (index === -1) return

    const trimmed = messages.slice(index + 1)
    if (trimmed.length) {
      messages.splice(0, messages.length, ...trimmed)
    }
  }

  async function initWithCompactedInput(
    providerID: string,
    requestInput: RequestInfo | URL,
    init: RequestInit | undefined,
    headers: Headers,
    sessionID: string,
  ): Promise<RequestInit> {
    const body = parseJsonRecord(await bodyText(requestInput, init))
    if (!body || !Array.isArray(body.input)) {
      return fetchInit(init, headers)
    }

    const checkpoint = activeCheckpointByProvider.get(providerID)?.get(sessionID)
    if (!checkpoint) return fetchInit(init, headers)

    headers.set("content-type", "application/json")
    const instructionCount = leadingInstructionCount(body.input)
    const next = {
      ...body,
      input: [
        ...body.input.slice(0, instructionCount),
        ...structuredClone(checkpoint.items),
        ...body.input.slice(instructionCount),
      ],
    }
    return { ...init, headers, body: JSON.stringify(next) }
  }

  function wrapFetch(base: FetchLike, providerID: string, provider: ProviderConfig): FetchLike {
    const previousBase = (base as unknown as AnyRecord)[wrappedBaseFetch]
    const baseFetch = typeof previousBase === "function" ? (previousBase as FetchLike) : base

    const wrapped = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      const url = urlOf(requestInput)
      const headers = requestHeaders(requestInput, init)
      const { sessionID: headerSessionID, shouldCompact } = compactMarkers(headers, config)
      const isResponsesRequest = url ? isResponsesUrl(url, config) : false
      const outboundHeaders = cleanedHeaders(headers, config)

      if (!isResponsesRequest) {
        return baseFetch(requestInput, fetchInit(init, outboundHeaders))
      }

      const sessionID = headerSessionID
      if (!sessionID) {
        return baseFetch(requestInput, fetchInit(init, outboundHeaders))
      }

      const requestInit = await initWithCompactedInput(providerID, requestInput, init, outboundHeaders, sessionID)
      const openAIOAuthRequestInit = usesOpenAIOAuth(providerID, new Headers(requestInit.headers))
        ? await openAIOAuth.requestInit(requestInit)
        : undefined
      const routedRequestInput = openAIOAuthRequestInit ? chatGPTCodexResponsesEndpoint : requestInput
      const routedRequestInit = openAIOAuthRequestInit ?? requestInit
      const body = parseJsonRecord(typeof routedRequestInit.body === "string" ? routedRequestInit.body : undefined)
      if (!shouldCompact) {
        rememberStableInstructions(providerID, sessionID, body)
        return baseFetch(routedRequestInput, routedRequestInit)
      }

      if (!body?.model || !body.input || !url) {
        return baseFetch(routedRequestInput, routedRequestInit)
      }

      const outboundCompactHeaders = new Headers(routedRequestInit.headers)
      outboundCompactHeaders.set("content-type", "application/json")
      const compacted = await baseFetch(openAIOAuthRequestInit ? chatGPTCodexCompactEndpoint : compactUrl(url, config), {
        ...routedRequestInit,
        method: "POST",
        headers: outboundCompactHeaders,
        body: JSON.stringify(
          withStableInstructions(
            compactBody(body, provider.compactModel, config),
            stableInstructionsByProvider.get(providerID)?.get(sessionID),
            config.compactBodyKeys.includes("instructions"),
          ),
        ),
      })
      if (!compacted.ok) {
        return compacted
      }

      const payload = asRecord(await compacted.clone().json().catch(() => undefined))
      const responseID = typeof payload?.id === "string" ? payload.id : undefined
      const items = compactedItemsFrom(payload?.output)
      if (!responseID || !items) {
        return compacted
      }

      const checkpoint = addCheckpoint(
        providerID,
        sessionID,
        responseID,
        { messageID: responseMessageID(responseID), createdAt: Date.now() },
        items,
      )
      pendingCompactResults.set(sessionID, { providerID, responseID, items })
      getProviderSessionMap(activeCheckpointByProvider, providerID).set(sessionID, checkpoint)
      return sseResponse({
        responseID,
        model: typeof payload?.model === "string" ? payload.model : provider.compactModel,
        createdAt: typeof payload?.created_at === "number" ? payload.created_at : Math.floor(Date.now() / 1000),
        summary: config.summary,
        usage: asRecord(payload?.usage),
      })
    }) as FetchLike

    Object.defineProperty(wrapped, wrappedFetch, { value: true })
    Object.defineProperty(wrapped, wrappedBaseFetch, { value: baseFetch })
    return wrapped
  }

  async function handleEvent(event: AnyRecord) {
    if (event.type === "session.deleted") {
      const sessionID = asRecord(event.properties)?.sessionID
      if (typeof sessionID !== "string") return
      for (const sessions of checkpointsByProvider.values()) sessions.delete(sessionID)
      pendingCompactResults.delete(sessionID)
      for (const sessions of activeCheckpointByProvider.values()) sessions.delete(sessionID)
      for (const sessions of stableInstructionsByProvider.values()) sessions.delete(sessionID)
      for (const sessions of pendingSystemByProvider.values()) sessions.delete(sessionID)
      for (const key of providerByMessage.keys()) {
        if (key.startsWith(`${sessionID}\0`)) providerByMessage.delete(key)
      }
      if (config.state.deleteOnSessionDeleted) store.deleteSession(sessionID)
      return
    }

    const compact = compactCheckpointFromEvent(event)
    if (!compact) return
    const pending = pendingCompactResults.get(compact.sessionID)
    if (pending?.responseID !== compact.responseID) return
    pendingCompactResults.delete(compact.sessionID)
    const checkpoint = addCheckpoint(compact.providerID, compact.sessionID, compact.responseID, compact.boundary, pending.items)
    getProviderSessionMap(activeCheckpointByProvider, compact.providerID).set(compact.sessionID, checkpoint)
  }

  const hooks: Hooks = {
    auth: {
      provider: "openai",
      methods: openAIAuthMethods,
      async loader(getAuth) {
        getOpenAIAuth = async () => asOpenAIOAuth(await getAuth())
        const auth = await getAuth()
        const apiAuth = asRecord(auth)
        if (asOpenAIOAuth(auth)) return { apiKey: openAIOAuthDummyKey }
        if (apiAuth?.type === "api" && typeof apiAuth.key === "string") return { apiKey: apiAuth.key }
        return {}
      },
    },

    async dispose() {
      store.close()
    },

    async config(cfg) {
      const root = cfg as AnyRecord
      root.provider ??= {}
      const providers = root.provider as AnyRecord
      for (const [providerID, compactProvider] of Object.entries(config.providers)) {
        providers[providerID] ??= {}
        const provider = providers[providerID] as AnyRecord
        provider.options ??= {}
        const options = provider.options as AnyRecord
        options.fetch = wrapFetch((options.fetch as FetchLike | undefined) ?? baseFetch, providerID, compactProvider)
      }
    },

    async event(input) {
      await handleEvent(input.event as AnyRecord)
    },

    "chat.message": async (input, output) => {
      rememberMessageProvider(input, output)
    },

    "chat.headers": async (input, output) => {
      const providerID = getProviderID(input)
      if (!providerID || !configuredProviders.has(providerID)) return
      if (typeof input.sessionID !== "string") return

      if (input.agent === "compaction") {
        pendingSystemByProvider.get(providerID)?.delete(input.sessionID)
        output.headers[config.headers.session] = input.sessionID
        output.headers[config.headers.compact] = "1"
        return
      }

      if (utilityAgents.has(input.agent)) {
        pendingSystemByProvider.get(providerID)?.delete(input.sessionID)
        return
      }

      promotePendingSystem(providerID, input.sessionID)
      output.headers[config.headers.session] = input.sessionID
    },

    "experimental.chat.messages.transform": async (input, output) => {
      const messages = output.messages as unknown as MessageEntry[]
      const providerID = transformProviderID(input, messages)
      if (!providerID || !configuredProviders.has(providerID)) return
      trimMessagesAfterCheckpoint(providerID, messages)
    },

    "experimental.chat.system.transform": async (input, output) => {
      const providerID = getProviderID(input)
      if (!providerID || !configuredProviders.has(providerID)) return
      if (typeof input.sessionID !== "string") return
      rememberPendingSystem(providerID, input.sessionID, output.system)
    },
  }

  return hooks
}
