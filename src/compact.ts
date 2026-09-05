import type { Hooks } from "@opencode-ai/plugin"
import { createHash, randomUUID } from "node:crypto"
import {
  compactReasoningEfforts,
  defaultConfig,
  type CompactReasoningEffort,
  type OpenAICompactConfig,
} from "./schema.js"
import {
  asOpenAIOAuth,
  createOpenAIOAuth,
  openAIAuthMethods,
  openAIOAuthDummyKey,
  usesOpenAIOAuth,
  type OpenAIOAuthAuth,
  type OAuthFetchLike,
} from "./oauth.js"
import {
  CheckpointStore,
  compactedItemsFrom,
  type AnyRecord,
  type Checkpoint,
  type ControlMessage,
} from "./state.js"

export { compactedItemsFrom } from "./state.js"

type FetchLike = typeof fetch
type MessageEntry = {
  info?: {
    id?: string
    sessionID?: string
    role?: string
    providerID?: string
    modelID?: string
    variant?: string
    agent?: string
    parentID?: string
    summary?: boolean
    finish?: string
    model?: {
      providerID?: string
      modelID?: string
      variant?: string
    }
    error?: unknown
    time?: { created?: number; completed?: number }
  }
  parts?: Array<{
    type?: string
    text?: string
    messageID?: string
    sessionID?: string
    synthetic?: boolean
    ignored?: boolean
    mime?: string
    filename?: string
    url?: string
    tool?: string
    callID?: string
    state?: unknown
    metadata?: AnyRecord
    time?: { start?: number }
  }>
}
type MessageBoundary = { messageID: string; createdAt: number }
type SessionStatus = "idle" | "busy" | "retry"
type SessionGeneration = { deleted: boolean; activity: number; revert?: string }
type PendingAutoContinue = {
  providerID: string
  agent?: string
  compactionMessageID?: string
  compactionCreatedAt?: number
}
type PendingNativeCompaction = {
  operationID: string
  summaryID: string
  providerID: string
  checkpoint: Checkpoint
  compactionMessageID: string
  completed: boolean
}
type ConversationSettings = {
  providerID: string
  modelID: string
  reasoningEffort?: CompactReasoningEffort
}
type StructuredCompactionSnapshot = {
  messages?: MessageEntry[]
  conversation?: ConversationSettings
  nativeSummary?: string
}
type CompactionCapture = {
  id: string
  sessionID: string
  createdAt: number
  generation: SessionGeneration
  controller: AbortController
  phase: "pending" | "ready" | "bound" | "native" | "ignored" | "invalidated"
  boundary?: MessageBoundary
  rawMessages?: MessageEntry[]
  priorSummaryIDs?: Set<string>
  snapshot?: StructuredCompactionSnapshot
  binding?: { providerID: string; modelID: string; agent: string; summaryID: string }
}
type PreparedCompactRequest = {
  body: string
  fallbackBody?: string
  target: string
  model: string
  summary: string
  passthrough: boolean
}
type CompactOperation = {
  id: string
  requiresID: boolean
  providerID: string
  sessionID: string
  createdAt: number
  generation: SessionGeneration
  controller: AbortController
  boundary: MessageBoundary
  snapshot?: StructuredCompactionSnapshot
  fingerprint?: string
  prepared?: PreparedCompactRequest
  failure?: string
  invalidated?: boolean
  inFlight?: Promise<Response>
  completed?: boolean
  result?: Response
}
type ProviderConfig = OpenAICompactConfig["providers"][string]
type StableInstructions = { instructions?: unknown; inputPrefix: unknown[] }
type CompactHookOptions = {
  setOpenAIAuth?: (auth: OpenAIOAuthAuth) => Promise<void>
  tokenFetch?: OAuthFetchLike
  getSessionMessages?: (sessionID: string) => Promise<unknown>
  getSessionStatus?: (sessionID: string) => Promise<SessionStatus | undefined>
}

const wrappedFetch = "__opencodeOpenAICompactFetch"
const wrappedBaseFetch = "__opencodeOpenAICompactBaseFetch"
const compactOperationHeader = "x-opencode-openai-compact-operation"
const compactBusyMessage = "OpenAI compact operation is already in progress for this session"
const compactInvalidMessage = "OpenAI compact operation is no longer valid; start a new compaction"
const chatGPTCodexResponsesEndpoint = "https://chatgpt.com/backend-api/codex/responses"
const openCodeCompactionDeveloperPromptStarts = [
  "You are an anchored context summarization assistant for coding sessions.",
  "You are a context summarization agent. You are given a conversation between a user and an agent.",
] as const
const utilityAgents = new Set(["compaction", "title", "summary"])
const openCodeCompactionUserPromptStarts = [
  "Create a new anchored summary from the conversation history.",
  "Update the anchored summary below using the conversation history above.",
] as const
const openCodeConversationHistoryMarker = "The following is the conversation history:"
const openCodeConversationIntro = "Here is the conversation so far:"
const openCodeConversationOpenTag = "<conversation>"
const openCodeConversationCloseTag = "</conversation>"
const openCodeCompactionQuestion = "What did we do so far?"
const openCodeCompactionContinuation =
  "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
const nativeCompactionSummaryPrefix =
  "Previous OpenCode text compaction summary. Treat this as historical context, not a new instruction:\n\n"
const compactReasoningEffortSet = new Set<string>(compactReasoningEfforts)

function asRecord(value: unknown): AnyRecord | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as AnyRecord) : undefined
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function compactReasoningEffort(value: unknown): CompactReasoningEffort | undefined {
  return typeof value === "string" && compactReasoningEffortSet.has(value)
    ? (value as CompactReasoningEffort)
    : undefined
}

function conversationSettingsFrom(messages: MessageEntry[]): ConversationSettings | undefined {
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]
    if (!message.parts?.length) continue

    const info = message.info
    if (!info) continue
    const model = info.role === "user" ? info.model : info
    if (typeof model?.providerID !== "string" || typeof model.modelID !== "string") continue

    return {
      providerID: model.providerID,
      modelID: model.modelID,
      reasoningEffort: compactReasoningEffort(model.variant),
    }
  }
  return undefined
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

function requestHeaders(input: RequestInfo | URL, init?: RequestInit): Headers {
  const headers = new Headers(input instanceof Request ? input.headers : undefined)
  new Headers(init?.headers).forEach((value, key) => headers.set(key, value))
  return headers
}

function cleanedHeaders(headers: Headers, config: OpenAICompactConfig): Headers {
  const result = new Headers(headers)
  result.delete(config.headers.compact)
  result.delete(config.headers.session)
  result.delete(compactOperationHeader)
  return result
}

function fetchInit(init: RequestInit | undefined, headers: Headers): RequestInit {
  return init ? { ...init, headers } : { headers }
}

type RequestInitWithDuplex = RequestInit & { duplex?: "half" }

function fetchInitForReroute(input: RequestInfo | URL, init: RequestInit | undefined, headers: Headers): RequestInit {
  if (!(input instanceof Request)) return fetchInit(init, headers)

  const requestInit: RequestInitWithDuplex = {
    method: input.method,
    body: input.body,
    cache: input.cache,
    credentials: input.credentials,
    integrity: input.integrity,
    keepalive: input.keepalive,
    mode: input.mode,
    redirect: input.redirect,
    referrer: input.referrer,
    referrerPolicy: input.referrerPolicy,
    signal: input.signal,
    ...init,
    headers,
  }
  const duplex = (input as Request & { duplex?: "half" }).duplex
  if (duplex && requestInit.body !== undefined && requestInit.body !== null) requestInit.duplex = duplex
  return requestInit
}

function compactMarkers(headers: Headers, config: OpenAICompactConfig) {
  const sessionID = headers.get(config.headers.session) ?? undefined
  const compact = headers.get(config.headers.compact)
  const shouldCompact = compact === "1"
  const shouldNativeCompact = compact === "native"
  headers.delete(config.headers.compact)
  headers.delete(config.headers.session)
  return { sessionID, shouldCompact, shouldNativeCompact }
}

async function bodyText(input: RequestInfo | URL, init?: RequestInit): Promise<string | undefined> {
  if (typeof init?.body === "string") return init.body
  if (init?.body instanceof ArrayBuffer || ArrayBuffer.isView(init?.body)) return new TextDecoder().decode(init.body)
  // An unsupported override must not silently fall back to the Request's different original body.
  if (init?.body != null) return undefined
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

function messageText(entry: MessageEntry) {
  return (entry.parts ?? [])
    .filter((part) => part.type === "text" && !part.ignored && typeof part.text === "string")
    .map((part) => part.text)
    .filter(Boolean)
    .join("\n")
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue)
  const record = asRecord(value)
  if (!record) return value
  return Object.fromEntries(Object.keys(record).sort().map((key) => [key, canonicalValue(record[key])]))
}

function forkMessageFingerprint(entry: MessageEntry) {
  const info = { ...(asRecord(entry.info) ?? {}) }
  delete info.id
  delete info.sessionID
  delete info.parentID

  const parts = (entry.parts ?? []).map((part) => {
    const result = { ...(asRecord(part) ?? {}) }
    delete result.id
    delete result.messageID
    delete result.sessionID
    delete result.tail_start_id
    return result
  })
  return JSON.stringify(canonicalValue({ info, parts }))
}

function isOpenCodeCompactionDeveloperPrompt(value: unknown) {
  const text = contentText(value).trimStart()
  return openCodeCompactionDeveloperPromptStarts.some((start) => text.startsWith(start))
}

function isTaggedOpenCodeConversation(value: unknown) {
  const text = contentText(value).trimStart()
  if (!text.startsWith(openCodeConversationIntro)) return false
  const open = text.indexOf(openCodeConversationOpenTag, openCodeConversationIntro.length)
  if (open === -1) return false
  return text.indexOf(openCodeConversationCloseTag, open + openCodeConversationOpenTag.length) !== -1
}

function isOpenCodeCompactionUserPrompt(value: unknown) {
  const text = contentText(value).trimStart()
  return openCodeCompactionUserPromptStarts.some((start) => text.startsWith(start)) || isTaggedOpenCodeConversation(value)
}

function hasEmbeddedOpenCodeConversation(value: unknown) {
  if (isTaggedOpenCodeConversation(value)) return true
  return isOpenCodeCompactionUserPrompt(value) && contentText(value).includes(openCodeConversationHistoryMarker)
}

function hasInvalidCheckpointHistory(checkpoint: Checkpoint) {
  return checkpoint.items.some((item) => item.role === "user" && hasEmbeddedOpenCodeConversation(item.content))
}

function compactInput(value: unknown) {
  if (!Array.isArray(value)) return value
  return value.filter((item, index) => {
    const record = asRecord(item)
    if (!record) return true
    if (record.role === "developer" && isOpenCodeCompactionDeveloperPrompt(record.content)) return false
    if (
      index === value.length - 1 &&
      record.role === "user" &&
      isOpenCodeCompactionUserPrompt(record.content) &&
      !hasEmbeddedOpenCodeConversation(record.content)
    ) {
      return false
    }
    return true
  })
}

function isKnownOpenCodeCompactionBody(body: AnyRecord) {
  if (isOpenCodeCompactionDeveloperPrompt(body.instructions)) return true
  if (!Array.isArray(body.input)) return false
  const last = asRecord(body.input.at(-1))
  return last?.role === "user" && isOpenCodeCompactionUserPrompt(last.content)
}

function attachmentContent(value: unknown, fallbacks: Map<AnyRecord, AnyRecord>): AnyRecord {
  const file = asRecord(value)
  const mime = typeof file?.mime === "string" ? file.mime : "unknown MIME"
  const filename = typeof file?.filename === "string" ? file.filename : "file"
  const marker = (reason: string) => ({
    type: "input_text",
    text: `[Attached ${mime}: ${filename}; content unavailable: ${reason}]`,
  })
  if (!file || typeof file.mime !== "string" || !file.mime) return marker("missing MIME")
  if (mime === "application/x-directory") return marker("directory attachment")
  if (typeof file.url !== "string" || !file.url.trim()) return marker("missing content source")
  const url = file.url
  const data = /^data:[^,]+,([\s\S]+)$/.test(url)
  let remote = false
  try {
    const parsed = new URL(url)
    remote = (parsed.protocol === "https:" || parsed.protocol === "http:") && !!parsed.hostname
  } catch {
    // Local paths and malformed URLs are not read or uploaded by this plugin.
  }
  if (!data && !remote) return marker("unsupported or empty content source")
  const content = mime.startsWith("image/")
    ? { type: "input_image", image_url: url }
    : data
      ? { type: "input_file", filename, file_data: url }
      : { type: "input_file", file_url: url }
  fallbacks.set(content, marker("attachment rejected by API"))
  return content
}

async function isAttachmentRejection(response: Response) {
  if (![400, 413, 415, 422].includes(response.status)) return false
  const payload = asRecord(await response.clone().json().catch(() => undefined))
  const error = asRecord(payload?.error)
  if (!error) return false
  const code = typeof error.code === "string" ? error.code : ""
  const param = typeof error.param === "string" ? error.param : ""
  const message = typeof error.message === "string" ? error.message : ""
  if (/context|token/i.test(`${code} ${message}`)) return false
  const attachment = /(?:^|[^a-z])(?:images?|files?|attachments?)(?:$|[^a-z])/i
  return attachment.test(`${code} ${param}`) ||
    (attachment.test(message) && /unsupported|invalid|format|access|fetch|download|size|large|limit|exceed/i.test(message))
}

function attachmentFallbackInput(input: unknown, fallbacks: Map<AnyRecord, AnyRecord>) {
  if (!Array.isArray(input)) return input
  return input.map((value) => {
    const item = asRecord(value)
    if (!item) return value
    const next = { ...item }
    for (const key of ["content", "output"]) {
      if (Array.isArray(item[key])) next[key] = item[key].map((part: AnyRecord) => fallbacks.get(part) ?? part)
    }
    return next
  })
}

function structuredOpenAIInput(
  messages: MessageEntry[],
  providerID: string,
  sourceModel: string,
  attachmentFallbacks: Map<AnyRecord, AnyRecord>,
): AnyRecord[] | undefined {
  const input: AnyRecord[] = []

  for (const message of messages) {
    const info = message.info
    const parts = message.parts ?? []
    if (!info?.role) return undefined
    if (!parts.length) continue

    if (info.role === "user") {
      const content: AnyRecord[] = []
      for (const part of parts) {
        if (part.type === "text" && !part.ignored && part.text) {
          content.push({ type: "input_text", text: part.text })
          continue
        }
        if (part.type === "file") {
          content.push(attachmentContent(part, attachmentFallbacks))
          continue
        }
        if (part.type === "compaction") {
          content.push({ type: "input_text", text: openCodeCompactionQuestion })
          continue
        }
        if (part.type === "subtask") {
          content.push({ type: "input_text", text: "The following tool was executed by the user" })
        }
      }
      if (content.length) input.push({ role: "user", content })
      continue
    }

    if (info.role !== "assistant") return undefined
    const differentModel = `${providerID}/${sourceModel}` !== `${info.providerID}/${info.modelID}`
    const assistantItems: AnyRecord[] = []
    const toolOutputs: AnyRecord[] = []
    const reasoningByID = new Map<string, AnyRecord>()
    const hasSignedReasoning = parts.some(
      (part) => part.type === "reasoning" && asRecord(asRecord(part.metadata)?.anthropic)?.signature != null,
    )

    for (const part of parts) {
      if (part.type === "text") {
        const text = part.text === "" && hasSignedReasoning ? " " : part.text
        if (typeof text !== "string") return undefined
        const metadata = differentModel ? undefined : asRecord(asRecord(part.metadata)?.openai)
        const item: AnyRecord = {
          role: "assistant",
          content: [{ type: "output_text", text }],
        }
        if (typeof metadata?.itemId === "string") item.id = metadata.itemId
        if (metadata?.phase === "commentary" || metadata?.phase === "final_answer") item.phase = metadata.phase
        assistantItems.push(item)
        continue
      }

      if (part.type === "reasoning") {
        if (typeof part.text !== "string") return undefined
        if (differentModel) {
          if (part.text.trim()) {
            assistantItems.push({ role: "assistant", content: [{ type: "output_text", text: part.text }] })
          }
          continue
        }

        const metadata = asRecord(asRecord(part.metadata)?.openai)
        const encryptedContent = metadata?.reasoningEncryptedContent
        if (typeof encryptedContent !== "string") continue
        const summary = part.text ? [{ type: "summary_text", text: part.text }] : []
        const itemID = metadata?.itemId
        if (typeof itemID !== "string") {
          assistantItems.push({ type: "reasoning", encrypted_content: encryptedContent, summary })
          continue
        }

        const existing = reasoningByID.get(itemID)
        if (existing) {
          const existingSummary = existing.summary as AnyRecord[]
          existingSummary.push(...summary)
          existing.encrypted_content = encryptedContent
          continue
        }
        const item: AnyRecord = {
          type: "reasoning",
          id: itemID,
          encrypted_content: encryptedContent,
          summary,
        }
        reasoningByID.set(itemID, item)
        assistantItems.push(item)
        continue
      }

      if (part.type !== "tool") continue
      if (part.metadata?.providerExecuted === true) return undefined
      if (typeof part.tool !== "string" || typeof part.callID !== "string") return undefined
      const state = asRecord(part.state)
      if (!state || typeof state.status !== "string") return undefined

      let argumentsText: string
      try {
        argumentsText = JSON.stringify(state.input === undefined ? {} : state.input)
      } catch {
        return undefined
      }
      assistantItems.push({
        type: "function_call",
        call_id: part.callID,
        name: part.tool,
        arguments: argumentsText,
      })

      let output: string
      if (state.status === "completed") {
        const time = asRecord(state.time)
        if (time?.compacted) output = "[Old tool result content cleared]"
        else if (typeof state.output === "string") output = state.output
        else return undefined
      } else if (state.status === "error") {
        const metadata = asRecord(state.metadata)
        if (metadata?.interrupted === true && typeof metadata.output === "string") output = metadata.output
        else if (typeof state.error === "string") output = state.error
        else return undefined
      } else if (state.status === "pending" || state.status === "running") {
        output = "[Tool execution was interrupted]"
      } else {
        return undefined
      }
      const attachments = state.status === "completed" && !asRecord(state.time)?.compacted && Array.isArray(state.attachments)
        ? state.attachments
        : []
      toolOutputs.push({
        type: "function_call_output",
        call_id: part.callID,
        output: attachments.length
          ? [{ type: "input_text", text: output }, ...attachments.map((file) => attachmentContent(file, attachmentFallbacks))]
          : output,
      })
    }

    input.push(...assistantItems, ...toolOutputs)
  }

  return input.length ? input : undefined
}

function cloneMessages(messages: MessageEntry[]) {
  try {
    return structuredClone(messages)
  } catch {
    return undefined
  }
}

function messageHasText(value: unknown, role: "assistant" | "user", text: string) {
  const message = asRecord(value)
  return message?.role === role && contentText(message.content) === text
}

function postCompactionInput(input: unknown[], summary: string) {
  const start = input.findIndex(
    (item, index) =>
      messageHasText(item, "user", openCodeCompactionQuestion) &&
      messageHasText(input[index + 1], "assistant", summary),
  )
  if (start === -1) return input

  const result = [...input.slice(0, leadingInstructionCount(input)), ...input.slice(start + 2)]
  if (messageHasText(result.at(-1), "user", openCodeCompactionContinuation)) result.pop()
  return result
}

function withoutLatestUserInput(input: unknown[]) {
  for (let index = input.length - 1; index >= 0; index--) {
    if (asRecord(input[index])?.role === "user") {
      return [...input.slice(0, index), ...input.slice(index + 1)]
    }
  }
  return input
}

function compactBodyValue(key: string, value: unknown) {
  if (key === "input") {
    const input = compactInput(value)
    if (!Array.isArray(input)) return input
    return [
      ...input.filter((item) => asRecord(item)?.type !== "compaction_trigger"),
      { type: "compaction_trigger" },
    ]
  }
  if (key === "instructions" && (value === "" || isOpenCodeCompactionDeveloperPrompt(value))) return undefined
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

function latestInstructionPrefix(input: unknown[]) {
  let start = 0
  for (let index = input.length - 1; index >= 0; index--) {
    const type = asRecord(input[index])?.type
    if (type === "compaction" || type === "compaction_summary") {
      start = index + 1
      break
    }
  }

  let end = start
  while (true) {
    const role = asRecord(input[end])?.role
    if (role !== "developer" && role !== "system") return input.slice(start, end)
    end++
  }
}

function stableInstructionsFrom(body: AnyRecord | undefined): StableInstructions | undefined {
  if (!body) return undefined

  const inputPrefix = Array.isArray(body.input) ? latestInstructionPrefix(body.input) : []
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
  reasoningEffort = defaultConfig.providers.openai.compactReasoningEffort,
): AnyRecord {
  const model = compactModel ?? (typeof body.model === "string" ? body.model : undefined)
  const result: AnyRecord = model ? { model } : {}
  for (const key of config.compactBodyKeys) {
    if (key === "model") continue
    const value = compactBodyValue(key, body[key])
    if (value !== undefined) result[key] = value
  }
  if (!config.compactBodyKeys.includes("input")) {
    const input = compactBodyValue("input", body.input)
    if (input !== undefined) result.input = input
  }
  const effort = reasoningEffort ?? compactReasoningEffort(asRecord(body.reasoning)?.effort)
  if (effort) result.reasoning = { ...(asRecord(result.reasoning) ?? {}), effort }
  result.tool_choice = "auto"
  result.store = false
  result.stream = true
  result.include = ["reasoning.encrypted_content"]
  return result
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

function compactedItemsForV2(input: unknown, compaction: AnyRecord): AnyRecord[] | undefined {
  if (!Array.isArray(input)) return undefined
  const retained = input.filter((item) => asRecord(item)?.role === "user" || isNativeCompactionSummaryItem(item))
  return compactedItemsFrom([...retained, compaction])
}

async function compactV2Payload(response: Response): Promise<AnyRecord | undefined> {
  const text = await response.text()
  const data: string[] = []
  let lines: string[] = []
  const flush = () => {
    if (lines.length) data.push(lines.join("\n"))
    lines = []
  }

  for (const line of text.replace(/^\uFEFF/, "").split(/\r?\n/)) {
    if (!line) {
      flush()
      continue
    }
    if (line.startsWith(":")) continue
    const separator = line.indexOf(":")
    const field = separator === -1 ? line : line.slice(0, separator)
    if (field !== "data") continue
    const value = separator === -1 ? "" : line.slice(separator + 1)
    lines.push(value.startsWith(" ") ? value.slice(1) : value)
  }
  flush()

  let completed: AnyRecord | undefined
  let compaction: AnyRecord | undefined
  let compactionCount = 0
  for (const value of data) {
    if (value === "[DONE]") continue
    let event: AnyRecord | undefined
    try {
      event = asRecord(JSON.parse(value))
    } catch {
      return undefined
    }
    if (
      !event ||
      event.type === "error" ||
      event.type === "response.failed" ||
      event.type === "response.incomplete"
    ) {
      return undefined
    }
    if (event.type === "response.output_item.done") {
      const item = asRecord(event.item)
      if (item?.type === "compaction") {
        compactionCount++
        compaction ??= item
      }
    }
    if (event.type === "response.completed") {
      if (completed) return undefined
      completed = asRecord(event.response)
    }
  }

  if (
    !completed ||
    (completed.status !== undefined && completed.status !== "completed") ||
    compactionCount !== 1 ||
    !compaction ||
    typeof compaction.encrypted_content !== "string"
  ) {
    return undefined
  }
  return { ...completed, compaction }
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

function isAfterBoundary(info: MessageEntry["info"], boundary: MessageBoundary) {
  const createdAt = info?.time?.created
  return !finiteNumber(createdAt) || typeof info?.id !== "string" ||
    createdAt > boundary.createdAt || (createdAt === boundary.createdAt && info.id > boundary.messageID)
}

function isCompactionUser(message: MessageEntry) {
  const info = message?.info
  return info?.role === "user" && Array.isArray(message.parts) && message.parts.some((part) =>
    part?.type === "compaction" && part.messageID === info.id && part.sessionID === info.sessionID)
}

function compactionBoundaryFrom(
  messages: MessageEntry[], sessionID: string, fixed?: MessageBoundary,
): MessageBoundary | undefined {
  let latest: MessageEntry | undefined
  const ids = new Set<string>()
  for (const message of messages) {
    const info = message?.info
    if (info?.sessionID !== sessionID || typeof info.id !== "string" || !info.id || ids.has(info.id)) return undefined
    ids.add(info.id)
    if (info.role !== "user") continue
    if (!finiteNumber(info.time?.created)) return undefined
    if (fixed) {
      if (info.id === fixed.messageID && info.time.created === fixed.createdAt) latest = message
      if (isAfterBoundary(info, fixed) && !isCompactionUser(message)) return undefined
      continue
    }
    if (!latest || isAfterBoundary(info, { messageID: latest.info!.id!, createdAt: latest.info!.time!.created! })) {
      latest = message
    }
  }
  const info = latest?.info
  if (!info || !latest || !isCompactionUser(latest)) return undefined
  if (messages.some((message) => message.info?.role === "assistant" && message.info.summary === true &&
    message.info.parentID === info.id && message.info.finish && !message.info.error)) return undefined
  return { messageID: info.id!, createdAt: info.time!.created! }
}

function matchesCaptureMessage(capture: CompactionCapture, value: unknown) {
  const message = asRecord(value)
  return !!capture.boundary && message?.role === "user" && message.sessionID === capture.sessionID &&
    message.id === capture.boundary.messageID && asRecord(message.time)?.created === capture.boundary.createdAt
}

function isCompletedCompactionSummary(value: unknown) {
  const info = asRecord(value)
  const time = asRecord(info?.time)
  return (
    info?.role === "assistant" &&
    info.summary === true &&
    info.error == null &&
    typeof info.finish === "string" &&
    finiteNumber(time?.completed)
  )
}

function isNativeCompactionSummaryItem(value: unknown) {
  const item = asRecord(value)
  return item?.role === "assistant" && contentText(item.content).startsWith(nativeCompactionSummaryPrefix)
}

function latestNativeCompactionSummary(messages: MessageEntry[], pluginSummary: string) {
  const compactionParents = new Set(
    messages
      .filter(
        (message) =>
          message.info?.role === "user" &&
          typeof message.info.id === "string" &&
          message.parts?.some((part) => part.type === "compaction"),
      )
      .map((message) => message.info!.id!),
  )

  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]
    const parentID = message.info?.parentID
    if (!isCompletedCompactionSummary(message.info) || typeof parentID !== "string" || !compactionParents.has(parentID)) {
      continue
    }
    const text = messageText(message)
    if (text && text !== pluginSummary) return text
  }
  return undefined
}

function hasCompletedCompactionAfterCheckpoint(checkpoint: Checkpoint, messages: MessageEntry[]) {
  const compactionParents = new Set(
    messages
      .filter((message) => {
        const info = message.info
        const createdAt = messageCreatedAt(message)
        return (
          info?.role === "user" &&
          typeof info.id === "string" &&
          info.id !== checkpoint.afterMessageID &&
          createdAt !== undefined &&
          createdAt >= checkpoint.afterCreatedAt &&
          message.parts?.some((part) => part.type === "compaction")
        )
      })
      .map((message) => message.info!.id!),
  )
  return messages.some(
    (message) =>
      isCompletedCompactionSummary(message.info) &&
      typeof message.info?.parentID === "string" &&
      compactionParents.has(message.info.parentID),
  )
}

function isOpenCodeCompactionContinuation(entry: MessageEntry) {
  return entry.parts?.some(
    (part) => part.type === "text" && part.synthetic === true && part.metadata?.compaction_continue === true,
  )
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
  const controlMessagesByProvider = new Map<string, Map<string, Map<string, ControlMessage>>>()
  for (const control of store.loadControlMessages()) {
    const sessions = getProviderSessionMap(controlMessagesByProvider, control.providerID)
    const messages = sessions.get(control.sessionID) ?? new Map<string, ControlMessage>()
    messages.set(control.messageID, control)
    sessions.set(control.sessionID, messages)
  }
  const pendingAutoContinues = new Map<string, PendingAutoContinue>()
  const pendingAutoContinueRequests = new Map<string, string>()
  const activeCheckpointByProvider = new Map<string, Map<string, Checkpoint>>()
  const stableInstructionsByProvider = new Map<string, Map<string, StableInstructions>>()
  const pendingSystemByProvider = new Map<string, Map<string, string>>()
  const providerByMessage = new Map<string, string>()
  const sessionGenerations = new Map<string, SessionGeneration>()
  const compactionOwners = new Map<string, string>()
  const compactionCaptures = new Map<string, CompactionCapture>()
  const compactOperationsByProvider = new Map<string, Map<string, CompactOperation>>()
  const pendingNativeCompactions = new Map<string, PendingNativeCompaction>()
  let disposed = false
  let openAIAuth: OpenAIOAuthAuth | undefined
  let openAIWrappedFetch: FetchLike | undefined
  const openAIOAuth = createOpenAIOAuth({
    getAuth: async () => openAIAuth,
    async setAuth(auth) {
      openAIAuth = auth
      await options.setOpenAIAuth?.(auth)
    },
    tokenFetch: options.tokenFetch,
  })

  function rememberMessageProvider(input: unknown, output: unknown) {
    const providerID = getProviderID(input)
    if (!providerID) return

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
    for (let index = messages.length - 1; index >= 0; index--) {
      const info = messages[index].info
      if (info?.role !== "user") continue
      if (typeof info.model?.providerID === "string") return info.model.providerID

      // Only the latest user identifies this turn; older users may belong to another provider.
      return typeof info.sessionID === "string" && typeof info.id === "string"
        ? providerByMessage.get(messageProviderKey(info.sessionID, info.id))
        : undefined
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

  function controlsFor(providerID: string, sessionID: string) {
    return controlMessagesByProvider.get(providerID)?.get(sessionID)
  }

  function rememberControlIdentity(
    providerID: string,
    sessionID: string,
    messageID: string,
    createdAt: number,
    contentText: string,
  ) {
    const sessions = getProviderSessionMap(controlMessagesByProvider, providerID)
    const messages = sessions.get(sessionID) ?? new Map<string, ControlMessage>()
    const existing = messages.get(messageID)
    if (existing) {
      // Text can match genuine user input, so message IDs remain the only removal identity.
      if (existing.contentText || !contentText) return
      const updated = { ...existing, contentText }
      messages.set(messageID, updated)
      store.upsertControlMessage(updated)
      return
    }

    const control: ControlMessage = {
      providerID,
      sessionID,
      messageID,
      createdAt,
      contentText,
    }
    messages.set(messageID, control)
    sessions.set(sessionID, messages)
    store.upsertControlMessage(control)
  }

  function rememberControlMessage(providerID: string, sessionID: string, message: MessageEntry) {
    const messageID = message.info?.id
    if (typeof messageID !== "string") return
    rememberControlIdentity(
      providerID,
      sessionID,
      messageID,
      messageCreatedAt(message) ?? Date.now(),
      messageText(message),
    )
  }

  function forgetControlMessage(sessionID: string, messageID: string) {
    for (const sessions of controlMessagesByProvider.values()) {
      const controls = sessions.get(sessionID)
      if (!controls?.delete(messageID)) continue
      if (!controls.size) sessions.delete(sessionID)
    }
    store.deleteControlMessage(sessionID, messageID)
  }

  function isPendingAutoContinueCandidate(message: MessageEntry, pending: PendingAutoContinue) {
    const info = message.info
    if (info?.role !== "user" || typeof info.id !== "string" || info.id === pending.compactionMessageID) return false
    if (pending.agent && info.agent && pending.agent !== info.agent) return false
    const createdAt = messageCreatedAt(message)
    if (
      createdAt !== undefined &&
      pending.compactionCreatedAt !== undefined &&
      createdAt < pending.compactionCreatedAt
    ) {
      return false
    }
    return message.parts?.some((part) => part.type === "text" && part.synthetic === true) === true
  }

  function captureControlMessages(providerID: string, sessionID: string, messages: MessageEntry[]) {
    for (const message of messages) {
      if (isOpenCodeCompactionContinuation(message)) rememberControlMessage(providerID, sessionID, message)
    }

    const pending = pendingAutoContinues.get(sessionID)
    if (pending?.providerID !== providerID) return
    const continuation = messages.find((message) => isPendingAutoContinueCandidate(message, pending))
    if (!continuation) return
    rememberControlMessage(providerID, sessionID, continuation)
    pendingAutoContinues.delete(sessionID)
  }

  function removeControlMessages(providerID: string, sessionID: string, messages: MessageEntry[]) {
    const controls = controlsFor(providerID, sessionID)
    if (!controls?.size) return
    const filtered = messages.filter((message) => {
      const messageID = message.info?.id
      return typeof messageID !== "string" || !controls.has(messageID)
    })
    if (filtered.length !== messages.length) messages.splice(0, messages.length, ...filtered)
  }

  // Transform history is compaction-filtered, so verify forks using raw source and child messages.
  async function inheritForkState(
    providerID: string, sessionID: string, messages: MessageEntry[], capture?: CompactionCapture,
  ) {
    const getSessionMessages = options.getSessionMessages
    const capturePhase = capture?.phase
    const generation = sessionGeneration(sessionID)
    if (!getSessionMessages || !currentSession(sessionID, generation)) return
    const providerSessions = checkpointsByProvider.get(providerID)
    if (!providerSessions || providerSessions.has(sessionID)) return

    const boundaryTimes = new Set<number>()
    for (const message of messages) {
      const createdAt = messageCreatedAt(message)
      if (createdAt === undefined || !message.parts?.some((part) => part.type === "compaction")) continue
      boundaryTimes.add(createdAt)
    }
    if (!boundaryTimes.size) return

    const childValue = await readSessionMessages(sessionID)
    if (!Array.isArray(childValue) || !currentSession(sessionID, generation) || compactionCaptures.get(sessionID) !== capture ||
      capture?.phase !== capturePhase) return
    const childMessages = childValue as MessageEntry[]
    const childFingerprints = childMessages.map(forkMessageFingerprint)
    const candidates = (
      await Promise.all(
        [...providerSessions.entries()].map(async ([sourceSessionID, checkpoints]) => {
          if (sourceSessionID === sessionID || !checkpoints.some((item) => boundaryTimes.has(item.afterCreatedAt))) {
            return undefined
          }

          const sourceGeneration = sessionGeneration(sourceSessionID)
          if (!currentSession(sourceSessionID, sourceGeneration)) return undefined
          const value = await readSessionMessages(sourceSessionID)
          if (!Array.isArray(value) || !currentSession(sourceSessionID, sourceGeneration) ||
            providerSessions.get(sourceSessionID) !== checkpoints) return undefined
          const sourceMessages = value as MessageEntry[]
          const limit = Math.min(sourceMessages.length, childMessages.length)
          let prefixLength = 0
          while (
            prefixLength < limit &&
            forkMessageFingerprint(sourceMessages[prefixLength]) === childFingerprints[prefixLength]
          ) {
            prefixLength++
          }
          if (!prefixLength) return undefined

          const messageIDs = new Map<string, string>()
          for (let index = 0; index < prefixLength; index++) {
            const sourceMessageID = sourceMessages[index]?.info?.id
            const childMessageID = childMessages[index]?.info?.id
            if (typeof sourceMessageID === "string" && typeof childMessageID === "string") {
              messageIDs.set(sourceMessageID, childMessageID)
            }
          }
          const inherited = checkpoints.filter((checkpoint) => messageIDs.has(checkpoint.afterMessageID))
          if (!inherited.length) return undefined

          return {
            sourceSessionID,
            sourceGeneration,
            checkpoints,
            prefixLength,
            messageIDs,
            inherited,
            signature: inherited
              .map((checkpoint) => `${checkpoint.afterCreatedAt}\0${checkpoint.responseID}`)
              .sort()
              .join("\x01"),
          }
        }),
      )
    ).filter((candidate): candidate is NonNullable<typeof candidate> => candidate !== undefined &&
      currentSession(candidate.sourceSessionID, candidate.sourceGeneration) &&
      providerSessions.get(candidate.sourceSessionID) === candidate.checkpoints)
    if (!candidates.length || providerSessions.has(sessionID) || !currentSession(sessionID, generation) || compactionCaptures.get(sessionID) !== capture ||
      capture?.phase !== capturePhase) return

    const prefixLength = Math.max(...candidates.map((candidate) => candidate.prefixLength))
    const finalists = candidates.filter((candidate) => candidate.prefixLength === prefixLength)
    if (new Set(finalists.map((candidate) => candidate.signature)).size !== 1) return

    const selected = finalists[0]
    const inherited = selected.inherited.map((checkpoint) => ({
      ...checkpoint,
      afterMessageID: selected.messageIDs.get(checkpoint.afterMessageID)!,
      items: structuredClone(checkpoint.items),
    }))
    providerSessions.set(sessionID, sortCheckpoints(inherited))
    for (const checkpoint of inherited) store.upsert(sessionID, checkpoint)

    const inheritedControls = new Map<string, ControlMessage>()
    const conflictingControls = new Set<string>()
    for (const candidate of finalists) {
      for (const control of controlsFor(providerID, candidate.sourceSessionID)?.values() ?? []) {
        const messageID = candidate.messageIDs.get(control.messageID)
        if (!messageID || conflictingControls.has(messageID)) continue
        const next = { ...control, sessionID, messageID }
        const existing = inheritedControls.get(messageID)
        if (
          existing &&
          (existing.createdAt !== next.createdAt || existing.contentText !== next.contentText)
        ) {
          inheritedControls.delete(messageID)
          conflictingControls.add(messageID)
          continue
        }
        inheritedControls.set(messageID, next)
      }
    }
    for (const control of inheritedControls.values()) {
      rememberControlIdentity(
        control.providerID,
        control.sessionID,
        control.messageID,
        control.createdAt,
        control.contentText,
      )
    }
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

    let start = index + 1
    const boundary = messages[index]
    if (boundary?.info?.role === "user" && boundary.parts?.some((part) => part.type === "compaction")) {
      const summaryIndex = messages.findIndex(
        (message, messageIndex) =>
          messageIndex > index &&
          message.info?.role === "assistant" &&
          message.info.summary === true &&
          message.info.parentID === boundary.info?.id,
      )
      if (summaryIndex !== -1) start = summaryIndex + 1
    }

    const trimmed = messages.slice(start).filter((message) => !isOpenCodeCompactionContinuation(message))
    messages.splice(0, messages.length, ...trimmed)
  }

  function releaseCapture(capture: CompactionCapture, phase: CompactionCapture["phase"]) {
    capture.phase = phase
    capture.rawMessages = undefined
    capture.priorSummaryIDs = undefined
    capture.snapshot = undefined
  }

  function sessionGeneration(sessionID: string) {
    let generation = sessionGenerations.get(sessionID)
    if (!generation) sessionGenerations.set(sessionID, generation = { deleted: false, activity: 0 })
    return generation
  }

  function currentSession(sessionID: string, generation: SessionGeneration) {
    return !disposed && !generation.deleted && sessionGenerations.get(sessionID) === generation
  }

  async function readSessionMessages(sessionID: string) {
    try { return await options.getSessionMessages?.(sessionID) }
    catch { return undefined }
  }

  function clearStructuredCapture(sessionID: string) {
    const capture = compactionCaptures.get(sessionID)
    // Retain identity after releasing history: late headerless requests are not fresh compactions.
    if (capture) {
      releaseCapture(capture, "invalidated")
      capture.controller.abort(new Error(compactInvalidMessage))
    }
  }

  function currentCapture(capture: CompactionCapture) {
    return currentSession(capture.sessionID, capture.generation) &&
      compactionCaptures.get(capture.sessionID) === capture && capture.phase !== "invalidated"
  }

  function currentOperation(operation: CompactOperation) {
    return currentSession(operation.sessionID, operation.generation) && !operation.invalidated &&
      compactOperationsByProvider.get(operation.providerID)?.get(operation.sessionID) === operation &&
      (operation.completed || compactionOwners.get(operation.sessionID) === operation.id)
  }

  function invalidateSession(sessionID: string, deleted = false) {
    const previous = sessionGeneration(sessionID)
    const generation = { deleted: deleted || previous.deleted, activity: 0, revert: previous.revert }
    sessionGenerations.set(sessionID, generation)
    invalidateCompactOperations(sessionID)
    clearStructuredCapture(sessionID)
    compactionOwners.delete(sessionID)
    pendingNativeCompactions.delete(sessionID)
    pendingAutoContinues.delete(sessionID)
    pendingAutoContinueRequests.delete(sessionID)
    return generation
  }

  function invalidateCompactOperations(sessionID: string) {
    for (const sessions of compactOperationsByProvider.values()) {
      const operation = sessions.get(sessionID)
      if (!operation) continue
      // Keep a small tombstone so a headerless late retry cannot become a new text compaction.
      operation.invalidated = true
      operation.snapshot = undefined
      operation.prepared = undefined
      operation.result = undefined
      operation.controller.abort(new Error(compactInvalidMessage))
    }
  }

  function terminalCaptureSummary(capture: CompactionCapture, value: unknown) {
    const info = asRecord(value)
    return !!capture.boundary && info?.role === "assistant" && info.summary === true &&
      info.sessionID === capture.sessionID && info.parentID === capture.boundary.messageID &&
      typeof info.id === "string" && finiteNumber(asRecord(info.time)?.completed) &&
      (capture.binding ? info.id === capture.binding.summaryID && info.providerID === capture.binding.providerID &&
        info.modelID === capture.binding.modelID && info.agent === capture.binding.agent
        : !capture.priorSummaryIDs?.has(info.id))
  }

  function finishCapture(capture: CompactionCapture, info: unknown) {
    if (!currentCapture(capture) || compactionOwners.get(capture.sessionID) !== capture.id ||
      !terminalCaptureSummary(capture, info)) return
    const native = pendingNativeCompactions.get(capture.sessionID)
    if (currentNativeCompaction(capture.sessionID, native) && native?.completed && isCompletedCompactionSummary(info)) {
      clearNativeFallbackSession(capture.sessionID)
      return
    }
    const operation = capture.binding && compactOperationsByProvider.get(capture.binding.providerID)?.get(capture.sessionID)
    if (operation?.id === capture.id && operation.completed && !asRecord(info)?.error) {
      compactionOwners.delete(capture.sessionID)
      return
    }
    invalidateSession(capture.sessionID)
  }

  // Session-only notifications are hints, not ownership. Recheck the authoritative state,
  // and ensure neither the session generation nor its owner changed while reading it.
  async function refreshCompactionOwner(sessionID: string): Promise<SessionGeneration | undefined> {
    const generation = sessionGeneration(sessionID)
    const owner = compactionOwners.get(sessionID)
    const activity = generation.activity
    const current = () => currentSession(sessionID, generation) && compactionOwners.get(sessionID) === owner &&
      generation.activity === activity
    if (!owner) return current() ? generation : undefined
    const capture = compactionCaptures.get(sessionID)
    if (capture?.id === owner && capture.boundary && options.getSessionMessages) {
      const raw = await readSessionMessages(sessionID)
      if (!current()) return undefined
      const summaries = Array.isArray(raw) ? raw.filter((message) => {
        const info = asRecord(message?.info)
        return info?.sessionID === sessionID && info.role === "assistant" && info.summary === true &&
          info.parentID === capture.boundary!.messageID && typeof info.id === "string" &&
          (capture.binding ? info.id === capture.binding.summaryID : !capture.priorSummaryIDs?.has(info.id))
      }) : []
      if (summaries.length === 1 && terminalCaptureSummary(capture, summaries[0].info)) {
        finishCapture(capture, summaries[0].info)
        return sessionGeneration(sessionID)
      }
    }
    let status: SessionStatus | undefined
    try { status = await options.getSessionStatus?.(sessionID) }
    catch { status = undefined }
    if (!current()) return undefined
    return status === "idle" ? invalidateSession(sessionID) : generation
  }

  function createCompactOperation(
    providerID: string,
    sessionID: string,
    capture?: CompactionCapture,
  ): CompactOperation {
    const generation = capture?.generation ?? invalidateSession(sessionID)
    const sessions = getProviderSessionMap(compactOperationsByProvider, providerID)
    const id = capture?.id ?? randomUUID()
    const createdAt = capture?.createdAt ?? Date.now()
    const operation: CompactOperation = {
      id, providerID, sessionID, requiresID: !!capture, createdAt, generation,
      controller: capture?.controller ?? new AbortController(),
      boundary: capture?.boundary ?? { messageID: `msg_compact_${id}`, createdAt },
      snapshot: capture?.snapshot,
      failure: capture && !capture.snapshot
        ? "OpenAI compact structured history could not be captured safely"
        : undefined,
    }
    if (capture) releaseCapture(capture, "bound")
    sessions.set(sessionID, operation)
    compactionOwners.set(sessionID, operation.id)
    return operation
  }

  function currentNativeCompaction(sessionID: string, pending: PendingNativeCompaction | undefined) {
    const capture = compactionCaptures.get(sessionID)
    return !!pending && !!capture && currentCapture(capture) && pendingNativeCompactions.get(sessionID) === pending &&
      capture.phase === "native" && capture.id === pending.operationID && compactionOwners.get(sessionID) === capture.id
  }

  function clearNativeFallbackSession(sessionID: string, preserveCapture?: CompactionCapture) {
    if (!preserveCapture) invalidateSession(sessionID)
    else invalidateCompactOperations(sessionID)
    store.deleteSession(sessionID)
    pendingNativeCompactions.delete(sessionID)
    for (const sessions of checkpointsByProvider.values()) sessions.delete(sessionID)
    for (const sessions of activeCheckpointByProvider.values()) sessions.delete(sessionID)
    for (const sessions of controlMessagesByProvider.values()) sessions.delete(sessionID)
  }

  function structuredInputFor(
    providerID: string,
    sessionID: string,
    sourceModel: string,
    snapshot: StructuredCompactionSnapshot | undefined,
    attachmentFallbacks: Map<AnyRecord, AnyRecord>,
  ) {
    const messages = snapshot?.messages ? cloneMessages(snapshot.messages) : undefined
    if (!messages) return undefined

    try {
      trimMessagesAfterCheckpoint(providerID, messages)
      const history = structuredOpenAIInput(messages, providerID, sourceModel, attachmentFallbacks)
      if (!history) return undefined

      const checkpoint = activeCheckpointByProvider.get(providerID)?.get(sessionID)
      const checkpointItems = checkpoint ? structuredClone(checkpoint.items) : []
      const nativeSummaryText = snapshot?.nativeSummary
      if (!nativeSummaryText) return [...checkpointItems, ...history]

      const nativeSummary = {
        role: "assistant",
        content: [{ type: "output_text", text: `${nativeCompactionSummaryPrefix}${nativeSummaryText}` }],
      }
      return [...checkpointItems.filter((item) => !isNativeCompactionSummaryItem(item)), nativeSummary, ...history]
    } catch {
      return undefined
    }
  }

  async function initWithCompactedInput(
    providerID: string,
    requestInput: RequestInfo | URL,
    init: RequestInit | undefined,
    headers: Headers,
    sessionID: string,
    removeLatestUser: boolean,
    selectedCheckpoint?: Checkpoint,
  ): Promise<RequestInit> {
    const checkpoint = selectedCheckpoint ?? activeCheckpointByProvider.get(providerID)?.get(sessionID)
    const body = parseJsonRecord(await bodyText(requestInput, init))
    if (!body || !Array.isArray(body.input)) {
      return fetchInitForReroute(requestInput, init, headers)
    }

    if (!checkpoint) return fetchInitForReroute(requestInput, init, headers)

    headers.set("content-type", "application/json")
    const postCompaction = postCompactionInput(body.input, config.summary)
    const input = removeLatestUser ? withoutLatestUserInput(postCompaction) : postCompaction
    const next = {
      ...body,
      input: [...structuredClone(checkpoint.items), ...input],
    }
    return { ...fetchInitForReroute(requestInput, init, headers), body: JSON.stringify(next) }
  }

  function prepareCompactOperation(
    operation: CompactOperation,
    originalBody: string,
    url: URL,
    headers: Headers,
    provider: ProviderConfig,
  ): PreparedCompactRequest | undefined {
    const { providerID, sessionID, snapshot } = operation
    let body = parseJsonRecord(originalBody)
    const checkpoint = activeCheckpointByProvider.get(providerID)?.get(sessionID)
    if (body && Array.isArray(body.input) && checkpoint) {
      body = { ...body, input: [...structuredClone(checkpoint.items), ...postCompactionInput(body.input, config.summary)] }
    }
    const target = usesOpenAIOAuth(providerID, headers) ? chatGPTCodexResponsesEndpoint : url.href
    if (!body || typeof body.model !== "string" || !Array.isArray(body.input)) {
      if (snapshot) {
        operation.failure = "OpenAI compact structured request could not be prepared safely"
        return undefined
      }
      return { body: originalBody, target, model: "", summary: config.summary, passthrough: true }
    }

    const conversation = snapshot?.conversation?.providerID === providerID ? snapshot.conversation : undefined
    const model = provider.compactModel ?? conversation?.modelID ?? body.model
    const reasoningEffort = provider.compactReasoningEffort ?? conversation?.reasoningEffort ??
      compactReasoningEffort(asRecord(body.reasoning)?.effort) ?? null
    const attachmentFallbacks = new Map<AnyRecord, AnyRecord>()
    const input = structuredInputFor(providerID, sessionID, model, snapshot, attachmentFallbacks)
    if (snapshot && !input) {
      operation.failure = "OpenAI compact structured history could not be converted safely"
      return undefined
    }
    if (!input && !isKnownOpenCodeCompactionBody(body)) {
      return { body: JSON.stringify(body), target, model, summary: config.summary, passthrough: true }
    }

    const compact = compactBody(body, model, config, reasoningEffort)
    delete compact.instructions
    if (input) compact.input = [...input, { type: "compaction_trigger" }]
    const request = withStableInstructions(
      compact,
      stableInstructionsByProvider.get(providerID)?.get(sessionID),
      config.compactBodyKeys.includes("instructions"),
    )
    return {
      body: JSON.stringify(request),
      fallbackBody: attachmentFallbacks.size
        ? JSON.stringify({ ...request, input: attachmentFallbackInput(request.input, attachmentFallbacks) })
        : undefined,
      target, model, summary: config.summary, passthrough: false,
    }
  }

  async function runCompactOperation(
    operation: CompactOperation,
    requestInput: RequestInfo | URL,
    init: RequestInit | undefined,
    headers: Headers,
    baseFetch: FetchLike,
  ): Promise<Response> {
    const prepared = operation.prepared!
    const signal = init?.signal !== undefined ? init.signal : (requestInput instanceof Request ? requestInput.signal : undefined)
    const combinedSignal = signal ? AbortSignal.any([signal, operation.controller.signal]) : operation.controller.signal
    const send = async () => {
      if (!currentOperation(operation)) return new Response(compactInvalidMessage, { status: 400 })
      const oauth = usesOpenAIOAuth(operation.providerID, headers)
      const target = oauth ? chatGPTCodexResponsesEndpoint : urlOf(requestInput)!.href
      if (target !== prepared.target) {
        return new Response("OpenAI compact operation cannot change its target endpoint", { status: 400 })
      }
      const outboundHeaders = new Headers(headers)
      outboundHeaders.set("content-type", "application/json")
      let request: RequestInit = {
        ...fetchInitForReroute(requestInput, init, outboundHeaders),
        method: "POST",
        body: prepared.body,
        signal: combinedSignal,
      }
      if (oauth) request = await openAIOAuth.requestInit(request)
      if (!currentOperation(operation)) return new Response(compactInvalidMessage, { status: 400 })
      request.signal?.throwIfAborted()
      return baseFetch(prepared.target, request)
    }
    let response = await send()
    if (prepared.fallbackBody && await isAttachmentRejection(response)) {
      if (!currentOperation(operation)) return new Response(compactInvalidMessage, { status: 400 })
      // Promote before sending: a failed fallback attempt must never restore the original attachments.
      prepared.body = prepared.fallbackBody
      prepared.fallbackBody = undefined
      response = await send()
    }
    if (!currentOperation(operation)) return new Response(compactInvalidMessage, { status: 400 })
    signal?.throwIfAborted()
    if (!response.ok) return response

    if (!prepared.passthrough) {
      const payload = await compactV2Payload(response).catch(() => undefined)
      if (!currentOperation(operation)) return new Response(compactInvalidMessage, { status: 400 })
      signal?.throwIfAborted()
      const compaction = asRecord(payload?.compaction)
      const items = compaction ? compactedItemsForV2(JSON.parse(prepared.body).input, compaction) : undefined
      if (!items) {
        return new Response("OpenAI compact response stream must complete with exactly one valid compaction item", {
          status: 502,
        })
      }
      const responseID = typeof payload?.id === "string" ? payload.id : undefined
      if (!responseID) {
        return new Response("OpenAI compact response.completed event must contain a response id", { status: 502 })
      }
      const checkpoint = addCheckpoint(
        operation.providerID,
        operation.sessionID,
        responseID,
        operation.boundary,
        items,
      )
      getProviderSessionMap(activeCheckpointByProvider, operation.providerID).set(operation.sessionID, checkpoint)
      response = sseResponse({
        responseID,
        model: typeof payload?.model === "string" ? payload.model : prepared.model,
        createdAt: typeof payload?.created_at === "number" ? payload.created_at : Math.floor(operation.createdAt / 1000),
        summary: prepared.summary,
        usage: asRecord(payload?.usage),
      })
    }
    operation.result = response
    operation.completed = true
    operation.prepared = undefined
    if (!operation.requiresID) compactionOwners.delete(operation.sessionID)
    return response
  }

  async function fetchCompactOperation(
    providerID: string,
    provider: ProviderConfig,
    sessionID: string,
    operationID: string | null,
    requestInput: RequestInfo | URL,
    init: RequestInit | undefined,
    headers: Headers,
    baseFetch: FetchLike,
  ): Promise<Response> {
    let operation = compactOperationsByProvider.get(providerID)?.get(sessionID)
    if (operationID !== null) {
      if (!operation || !operation.requiresID || operation.id !== operationID) {
        return new Response("OpenAI compact operation is unavailable; start a new compaction", { status: 400 })
      }
    } else {
      if (compactionCaptures.has(sessionID) || operation?.requiresID) {
        return new Response("OpenAI compact operation ID is required", { status: 400 })
      }
      if (!operation) {
        if (compactionOwners.has(sessionID)) return new Response(compactBusyMessage, { status: 409 })
        operation = createCompactOperation(providerID, sessionID)
      }
    }
    // Completed legacy operations may be replaced by a different request after a context change,
    // but never after deletion, or while another provider owns this session.
    const generation = sessionGeneration(sessionID)
    if (!currentSession(sessionID, generation) || (operation.invalidated && (operationID !== null || !operation.completed))) {
      return new Response(compactInvalidMessage, { status: 400 })
    }

    const text = await bodyText(requestInput, init)
    if (!currentSession(sessionID, generation) || compactOperationsByProvider.get(providerID)?.get(sessionID) !== operation ||
      (compactionCaptures.has(sessionID) && compactionCaptures.get(sessionID)?.id !== operation.id)) {
      return new Response("OpenAI compact operation is no longer valid; start a new compaction", { status: 400 })
    }
    if (text === undefined) {
      return new Response("OpenAI compact request body could not be read safely", { status: 400 })
    }
    const url = urlOf(requestInput)!
    const method = (init?.method ?? (requestInput instanceof Request ? requestInput.method : "GET")).toUpperCase()
    const fingerprint = createHash("sha256").update(JSON.stringify([url.href, method, text])).digest("hex")
    if (operation.fingerprint && operation.fingerprint !== fingerprint) {
      if (operationID === null && operation.completed) {
        if (compactionOwners.has(sessionID)) return new Response(compactBusyMessage, { status: 409 })
        operation = createCompactOperation(providerID, sessionID)
      } else return new Response("OpenAI compact operation cannot change its original request", { status: 400 })
    }
    if (!currentOperation(operation)) return new Response(compactInvalidMessage, { status: 400 })
    operation.fingerprint = fingerprint
    const signal = init?.signal !== undefined ? init.signal : (requestInput instanceof Request ? requestInput.signal : undefined)
    signal?.throwIfAborted()
    if (operation.failure) return new Response(operation.failure, { status: 502 })
    if (operation.result) return operation.result.clone()
    if (!operation.prepared) {
      try {
        operation.prepared = prepareCompactOperation(operation, text, url, headers, provider)
      } catch {
        operation.failure = "OpenAI compact structured request could not be prepared safely"
      }
      operation.snapshot = undefined
      if (!operation.prepared) return new Response(operation.failure, { status: 502 })
    }
    if (!operation.inFlight) {
      operation.generation.activity++
      const current = operation
      current.inFlight = runCompactOperation(current, requestInput, init, headers, baseFetch)
        .catch((error) => {
          if (!currentOperation(current)) return new Response(compactInvalidMessage, { status: 400 })
          throw error
        })
        .finally(() => { current.inFlight = undefined })
    }
    const response = await operation.inFlight!
    if (!currentOperation(operation)) return new Response(compactInvalidMessage, { status: 400 })
    signal?.throwIfAborted()
    return response.clone()
  }

  function wrapFetch(base: FetchLike, providerID: string, provider: ProviderConfig): FetchLike {
    const previousBase = (base as unknown as AnyRecord)[wrappedBaseFetch]
    const baseFetch = typeof previousBase === "function" ? (previousBase as FetchLike) : base

    const wrapped = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      const url = urlOf(requestInput)
      const headers = requestHeaders(requestInput, init)
      const operationID = headers.get(compactOperationHeader)
      const { sessionID: headerSessionID, shouldCompact, shouldNativeCompact } = compactMarkers(headers, config)
      const isResponsesRequest = url ? isResponsesUrl(url, config) : false
      const outboundHeaders = cleanedHeaders(headers, config)

      if (disposed) return new Response("OpenAI compact plugin has been disposed", { status: 400 })
      if (operationID !== null && (!isResponsesRequest || (!shouldCompact && !shouldNativeCompact))) {
        return new Response("OpenAI compact operation requires its original compaction endpoint and markers", { status: 400 })
      }
      if (!isResponsesRequest) {
        return baseFetch(requestInput, fetchInit(init, outboundHeaders))
      }

      const sessionID = headerSessionID
      const generation = sessionID ? sessionGeneration(sessionID) : undefined
      if (generation?.deleted) return new Response(compactInvalidMessage, { status: 400 })
      if ((shouldCompact || shouldNativeCompact) && !sessionID) {
        return new Response("OpenAI compact request is missing session header", { status: 400 })
      }
      if (shouldCompact && sessionID) {
        return fetchCompactOperation(providerID, provider, sessionID, operationID, requestInput, init, outboundHeaders, baseFetch)
      }

      const removeLatestUser =
        sessionID !== undefined && pendingAutoContinueRequests.get(sessionID) === providerID
      if (removeLatestUser) pendingAutoContinueRequests.delete(sessionID)
      const nativeCompaction = sessionID ? pendingNativeCompactions.get(sessionID) : undefined
      const matchingNativeCompaction = nativeCompaction?.providerID === providerID ? nativeCompaction : undefined
      if (shouldNativeCompact && (operationID !== matchingNativeCompaction?.operationID ||
        !currentNativeCompaction(sessionID!, matchingNativeCompaction))) {
        return new Response("OpenAI native compact operation is unavailable; start a new compaction", { status: 400 })
      }
      let originalRequestInit = fetchInitForReroute(requestInput, init, outboundHeaders)
      if (shouldNativeCompact) {
        const originalBody = await bodyText(requestInput, init)
        if (originalBody !== undefined) originalRequestInit = { ...originalRequestInit, body: originalBody }
      }
      const suppressInvalidCheckpoint = !!matchingNativeCompaction && !shouldNativeCompact
      const requestInit =
        sessionID && !suppressInvalidCheckpoint && (!shouldNativeCompact || matchingNativeCompaction)
          ? await initWithCompactedInput(
              providerID,
              requestInput,
              init,
              outboundHeaders,
              sessionID,
              removeLatestUser,
              shouldNativeCompact ? matchingNativeCompaction?.checkpoint : undefined,
            )
          : originalRequestInit
      if (shouldNativeCompact) {
        const controller = compactionCaptures.get(sessionID!)?.controller
        if (!controller || !currentNativeCompaction(sessionID!, matchingNativeCompaction)) {
          return new Response(compactInvalidMessage, { status: 400 })
        }
        requestInit.signal = requestInit.signal ? AbortSignal.any([requestInit.signal, controller.signal]) : controller.signal
        originalRequestInit.signal = requestInit.signal
      }
      const route = async (request: RequestInit) => {
        const authRequest = usesOpenAIOAuth(providerID, new Headers(request.headers))
          ? await openAIOAuth.requestInit(request)
          : undefined
        return {
          input: authRequest ? chatGPTCodexResponsesEndpoint : requestInput,
          init: authRequest ?? request,
        }
      }
      const routed = await route(requestInit)
      const routedRequestInput = routed.input
      const routedRequestInit = routed.init
      if (!sessionID) {
        return baseFetch(routedRequestInput, routedRequestInit)
      }

      if (disposed || sessionGeneration(sessionID).deleted) return new Response(compactInvalidMessage, { status: 400 })
      const body = parseJsonRecord(typeof routedRequestInit.body === "string" ? routedRequestInit.body : undefined)
      if (shouldNativeCompact) {
        const unavailable = () => new Response("OpenAI native compact operation is no longer valid", { status: 400 })
        if (!currentNativeCompaction(sessionID, matchingNativeCompaction)) return unavailable()
        const send = async (input: RequestInfo | URL, request: RequestInit) => {
          sessionGeneration(sessionID).activity++
          try {
            request.signal?.throwIfAborted()
            return await baseFetch(input, request)
          } catch (error) {
            if (!currentNativeCompaction(sessionID, matchingNativeCompaction)) return unavailable()
            throw error
          }
        }
        const response = await send(routedRequestInput, routedRequestInit)
        if (!currentNativeCompaction(sessionID, matchingNativeCompaction)) return unavailable()
        routedRequestInit.signal?.throwIfAborted()
        if (response.ok) {
          if (matchingNativeCompaction) matchingNativeCompaction.completed = true
          return response
        }
        if (
          !matchingNativeCompaction ||
          (response.status !== 400 && response.status !== 413 && response.status !== 422)
        ) {
          return response
        }

        const retry = await route(originalRequestInit)
        if (!currentNativeCompaction(sessionID, matchingNativeCompaction)) return unavailable()
        const retriedResponse = await send(retry.input, retry.init)
        if (!currentNativeCompaction(sessionID, matchingNativeCompaction)) return unavailable()
        retry.init.signal?.throwIfAborted()
        if (!retriedResponse.ok) return response
        matchingNativeCompaction.completed = true
        return retriedResponse
      }
      if (generation && currentSession(sessionID, generation)) rememberStableInstructions(providerID, sessionID, body)
      return baseFetch(routedRequestInput, routedRequestInit)
    }) as FetchLike

    Object.defineProperty(wrapped, wrappedFetch, { value: true })
    Object.defineProperty(wrapped, wrappedBaseFetch, { value: baseFetch })
    return wrapped
  }

  function getOpenAIWrappedFetch(base: FetchLike = baseFetch) {
    const provider = config.providers.openai
    if (!provider) return undefined
    openAIWrappedFetch ??= wrapFetch(base, "openai", provider)
    return openAIWrappedFetch
  }

  async function handleEvent(event: AnyRecord) {
    if (disposed) return
    const properties = asRecord(event.properties)
    const sessionID = properties?.sessionID
    if (typeof sessionID !== "string" || sessionGeneration(sessionID).deleted) return
    const statusType = asRecord(properties?.status)?.type
    if (event.type === "session.status" && (statusType === "busy" || statusType === "retry")) {
      sessionGeneration(sessionID).activity++
      return
    }
    if (event.type === "session.compacted" || event.type === "session.idle" || event.type === "session.error" ||
      (event.type === "session.status" && statusType === "idle")) {
      await refreshCompactionOwner(sessionID)
      return
    }

    if (event.type === "message.updated") {
      const capture = compactionCaptures.get(sessionID)
      if (capture?.binding) finishCapture(capture, properties?.info)
      else if (capture && terminalCaptureSummary(capture, properties?.info)) await refreshCompactionOwner(sessionID)
      return
    }

    if (event.type === "session.updated") {
      const info = asRecord(properties?.info)
      if (info?.id !== sessionID) return
      const revert = asRecord(info.revert)
      const key = typeof revert?.messageID === "string" ? JSON.stringify([revert.messageID, revert.partID]) : undefined
      if (sessionGeneration(sessionID).revert !== key) invalidateSession(sessionID).revert = key
      return
    }

    if (event.type === "session.deleted") {
      invalidateSession(sessionID, true)
      for (const sessions of checkpointsByProvider.values()) sessions.delete(sessionID)
      for (const sessions of activeCheckpointByProvider.values()) sessions.delete(sessionID)
      for (const sessions of controlMessagesByProvider.values()) sessions.delete(sessionID)
      for (const sessions of stableInstructionsByProvider.values()) sessions.delete(sessionID)
      for (const sessions of pendingSystemByProvider.values()) sessions.delete(sessionID)
      for (const key of providerByMessage.keys()) {
        if (key.startsWith(`${sessionID}\0`)) providerByMessage.delete(key)
      }
      if (config.state.deleteOnSessionDeleted) store.deleteSession(sessionID)
      return
    }

    if (event.type === "message.removed") {
      const messageID = properties?.messageID
      if (typeof messageID !== "string") return

      providerByMessage.delete(messageProviderKey(sessionID, messageID))
      invalidateSession(sessionID)
      forgetControlMessage(sessionID, messageID)
      for (const [providerID, sessions] of checkpointsByProvider) {
        const checkpoints = sessions.get(sessionID)
        if (!checkpoints) continue

        const removed = checkpoints.filter((checkpoint) => checkpoint.afterMessageID === messageID)
        if (!removed.length) continue

        const remaining = checkpoints.filter((checkpoint) => checkpoint.afterMessageID !== messageID)
        if (remaining.length) sessions.set(sessionID, remaining)
        else sessions.delete(sessionID)

        const activeCheckpoints = activeCheckpointByProvider.get(providerID)
        if (activeCheckpoints?.get(sessionID)?.afterMessageID === messageID) activeCheckpoints.delete(sessionID)
        for (const checkpoint of removed) store.deleteCheckpoint(sessionID, providerID, checkpoint.responseID)
      }
      return
    }
  }

  const hooks: Hooks = {
    auth: {
      provider: "openai",
      methods: openAIAuthMethods,
      async loader(getAuth) {
        const auth = await getAuth()
        openAIAuth = asOpenAIOAuth(auth)
        const apiAuth = asRecord(auth)
        const fetch = getOpenAIWrappedFetch()
        if (openAIAuth) return { apiKey: openAIOAuthDummyKey, ...(fetch ? { fetch } : {}) }
        if (apiAuth?.type === "api" && typeof apiAuth.key === "string") {
          return { apiKey: apiAuth.key, ...(fetch ? { fetch } : {}) }
        }
        return {}
      },
    },

    async dispose() {
      disposed = true
      for (const sessionID of sessionGenerations.keys()) invalidateSession(sessionID)
      compactOperationsByProvider.clear()
      compactionCaptures.clear()
      compactionOwners.clear()
      sessionGenerations.clear()
      for (const map of [checkpointsByProvider, activeCheckpointByProvider, controlMessagesByProvider,
        stableInstructionsByProvider, pendingSystemByProvider]) map.clear()
      pendingNativeCompactions.clear()
      pendingAutoContinues.clear()
      pendingAutoContinueRequests.clear()
      providerByMessage.clear()
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
        const currentFetch = (options.fetch as FetchLike | undefined) ?? baseFetch
        options.fetch =
          providerID === "openai"
            ? getOpenAIWrappedFetch(currentFetch)
            : wrapFetch(currentFetch, providerID, compactProvider)
      }
    },

    async event(input) {
      await handleEvent(input.event as AnyRecord)
    },

    "chat.message": async (input, output) => {
      if (disposed) return
      if (typeof input.sessionID === "string") {
        if (sessionGeneration(input.sessionID).deleted) return
        invalidateSession(input.sessionID)
        const message = asRecord(output.message)
        const messageID = typeof input.messageID === "string" ? input.messageID : message?.id
        if (typeof messageID === "string") forgetControlMessage(input.sessionID, messageID)
      }
      rememberMessageProvider(input, output)
    },

    "chat.headers": async (input, output) => {
      const providerID = getProviderID(input)
      if (!providerID || typeof input.sessionID !== "string" || disposed) return
      const sessionID = input.sessionID
      const generation = sessionGeneration(sessionID)
      if (!currentSession(sessionID, generation)) return
      const capture = compactionCaptures.get(sessionID)
      const modelID = asRecord(input.model)?.id
      if (capture && typeof modelID === "string" && modelID && typeof input.agent === "string" && input.agent &&
        currentCapture(capture) && matchesCaptureMessage(capture, input.message)) {
        if ((capture.phase === "pending" || capture.phase === "ready") && options.getSessionMessages) {
          const raw = await readSessionMessages(sessionID)
          if (!currentCapture(capture)) return
          if (capture.phase === "pending" || capture.phase === "ready") {
            // The summary is persisted AFTER messages.transform. Only verify the frozen boundary here;
            // never adopt a newer boundary from this second read (including utility requests).
            const boundary = Array.isArray(raw) ? compactionBoundaryFrom(raw, sessionID, capture.boundary) : undefined
            const summaries = boundary?.messageID === capture.boundary!.messageID &&
              boundary.createdAt === capture.boundary!.createdAt
              ? (raw as MessageEntry[]).filter(({ info }) => info?.role === "assistant" && info.summary === true &&
                  info.parentID === boundary.messageID && !info.finish && !info.error &&
                  typeof info.id === "string" && !capture.priorSummaryIDs?.has(info.id))
              : []
            const summary = summaries.length === 1 ? summaries[0].info : undefined
            if (summary?.providerID === providerID && summary.modelID === modelID && summary.agent === input.agent) {
              capture.generation.activity++
              capture.binding = { providerID, modelID, agent: input.agent, summaryID: summary.id! }
              if (!configuredProviders.has(providerID)) {
                releaseCapture(capture, "ignored")
                compactionOwners.delete(sessionID)
              } else {
                // Choose the target provider's checkpoint, not the history provider's checkpoint.
                const checkpoint = activeCheckpointByProvider.get(providerID)?.get(sessionID) ??
                  checkpointsByProvider.get(providerID)?.get(sessionID)?.at(-1)
                if (checkpoint) getProviderSessionMap(activeCheckpointByProvider, providerID).set(sessionID, checkpoint)
                if (checkpoint && hasInvalidCheckpointHistory(checkpoint)) {
                  pendingNativeCompactions.set(sessionID, {
                    operationID: capture.id, summaryID: summary.id!, providerID, checkpoint,
                    compactionMessageID: capture.boundary!.messageID, completed: false,
                  })
                  releaseCapture(capture, "native")
                } else createCompactOperation(providerID, sessionID, capture)
              }
            }
          }
        }
        const binding = capture.binding
        if (binding?.providerID === providerID && binding.modelID === modelID && binding.agent === input.agent) {
          if (capture.phase === "ignored") return
          const operation = compactOperationsByProvider.get(providerID)?.get(sessionID)
          const native = pendingNativeCompactions.get(sessionID)
          if ((capture.phase === "bound" && operation?.id === capture.id && !operation.invalidated) ||
            (capture.phase === "native" && native?.operationID === capture.id)) {
            pendingSystemByProvider.get(providerID)?.delete(sessionID)
            output.headers[config.headers.session] = sessionID
            output.headers[config.headers.compact] = capture.phase === "native" ? "native" : "1"
            output.headers[compactOperationHeader] = capture.id
            return
          }
        }
      }
      if (!configuredProviders.has(providerID)) return

      const pendingAutoContinue = pendingAutoContinues.get(input.sessionID)
      if (pendingAutoContinue?.providerID === providerID) {
        const message = asRecord(input.message)
        const messageID = message?.id
        const createdAt = asRecord(message?.time)?.created
        const createdAfterCompaction =
          !finiteNumber(createdAt) ||
          pendingAutoContinue.compactionCreatedAt === undefined ||
          createdAt >= pendingAutoContinue.compactionCreatedAt
        const matchingAgent = !pendingAutoContinue.agent || pendingAutoContinue.agent === input.agent
        if (
          createdAfterCompaction &&
          matchingAgent &&
          typeof messageID === "string" &&
          messageID !== pendingAutoContinue.compactionMessageID
        ) {
          rememberControlIdentity(
            providerID,
            input.sessionID,
            messageID,
            finiteNumber(createdAt) ? createdAt : Date.now(),
            "",
          )
          pendingAutoContinueRequests.set(input.sessionID, providerID)
          pendingAutoContinues.delete(input.sessionID)
        }
      }

      const messageAgent = asRecord(input.message)?.agent
      if (
        (typeof messageAgent === "string" && messageAgent !== input.agent) ||
        (messageAgent === undefined && utilityAgents.has(input.agent))
      ) {
        pendingSystemByProvider.get(providerID)?.delete(input.sessionID)
        return
      }

      promotePendingSystem(providerID, input.sessionID)
      output.headers[config.headers.session] = input.sessionID
    },

    "experimental.chat.messages.transform": async (_input, output) => {
      const messages = output.messages as unknown as MessageEntry[]
      const sessionID = sessionIDFromMessages(messages)
      if (!sessionID || disposed) return
      let generation = sessionGeneration(sessionID)
      if (!currentSession(sessionID, generation)) return
      const capture = compactionCaptures.get(sessionID)
      if (messages.some((message) => message.info?.sessionID !== sessionID)) {
        // Do not let an ambiguous transform modify either session's history.
        return
      }
      if (capture?.boundary && currentCapture(capture) && messages.some((message) =>
        message.info?.role === "user" && isAfterBoundary(message.info, capture.boundary!) && !isCompactionUser(message))) {
        generation = invalidateSession(sessionID)
      }
      const capturing = capture?.phase === "pending" && currentCapture(capture)
      const capturePhase = capture?.phase
      const providerID = providerIDFromMessages(messages) ?? providerIDFromTrimmedSessionCheckpoint(messages)
      if (providerID && configuredProviders.has(providerID)) {
        await inheritForkState(providerID, sessionID, messages, capture)
        if (!currentSession(sessionID, generation) || compactionCaptures.get(sessionID) !== capture || capture?.phase !== capturePhase) return
        if (capturing && !activeCheckpointByProvider.get(providerID)?.has(sessionID)) {
          const checkpoint = checkpointsByProvider.get(providerID)?.get(sessionID)?.at(-1)
          if (checkpoint) getProviderSessionMap(activeCheckpointByProvider, providerID).set(sessionID, checkpoint)
        }
        captureControlMessages(providerID, sessionID, messages)
        removeControlMessages(providerID, sessionID, messages)
        const checkpoint = activeCheckpointByProvider.get(providerID)?.get(sessionID)
        if (capturing && capture.boundary && checkpoint && capture.rawMessages &&
          hasCompletedCompactionAfterCheckpoint(checkpoint, capture.rawMessages)) {
          clearNativeFallbackSession(sessionID, capture)
        }
      }
      if (capturing && currentCapture(capture)) {
        const clonedMessages = capture.boundary ? cloneMessages(messages) : undefined
        capture.snapshot = clonedMessages ? {
          messages: clonedMessages,
          conversation: conversationSettingsFrom(messages),
          nativeSummary: capture.rawMessages ? latestNativeCompactionSummary(capture.rawMessages, config.summary) : undefined,
        } : undefined
        capture.rawMessages = undefined
        capture.phase = "ready"
      }
      // An unbound compaction must retain OpenCode's native history, not a plugin-trimmed prompt.
      if (providerID && configuredProviders.has(providerID) && !capturing) trimMessagesAfterCheckpoint(providerID, messages)
    },

    "experimental.chat.system.transform": async (input, output) => {
      const providerID = getProviderID(input)
      if (!providerID || !configuredProviders.has(providerID)) return
      if (typeof input.sessionID !== "string" || disposed || sessionGeneration(input.sessionID).deleted) return
      rememberPendingSystem(providerID, input.sessionID, output.system)
    },

    "experimental.session.compacting": async (input) => {
      if (typeof input.sessionID !== "string" || disposed) return
      const sessionID = input.sessionID
      const observed = sessionGeneration(sessionID)
      if (!currentSession(sessionID, observed)) throw new Error(compactInvalidMessage)
      if (compactionOwners.has(sessionID)) {
        const checked = await refreshCompactionOwner(sessionID)
        if (!checked && currentSession(sessionID, observed) && compactionOwners.has(sessionID)) throw new Error(compactBusyMessage)
        if (!checked || !currentSession(sessionID, checked)) throw new Error(compactInvalidMessage)
        if (compactionOwners.has(sessionID)) throw new Error(compactBusyMessage)
      }
      const generation = invalidateSession(sessionID)
      const capture: CompactionCapture = {
        id: randomUUID(), sessionID, createdAt: Date.now(), phase: "pending", generation,
        controller: new AbortController(),
      }
      compactionCaptures.set(sessionID, capture)
      compactionOwners.set(sessionID, capture.id)
      const raw = await readSessionMessages(sessionID)
      if (!currentCapture(capture)) return
      capture.boundary = Array.isArray(raw) ? compactionBoundaryFrom(raw, sessionID) : undefined
      if (!capture.boundary) {
        // Release the lease, but keep the pending transform marker so an unbound
        // native compaction is not trimmed using a plugin checkpoint.
        compactionOwners.delete(sessionID)
        return
      }
      capture.rawMessages = raw as MessageEntry[]
      capture.priorSummaryIDs = new Set((raw as MessageEntry[]).flatMap((message: MessageEntry) =>
        message.info?.summary && typeof message.info.id === "string" ? [message.info.id] : []))
    },

    "experimental.compaction.autocontinue": async (input) => {
      if (typeof input.sessionID !== "string" || disposed) return
      if (sessionGeneration(input.sessionID).deleted) return
      const capture = compactionCaptures.get(input.sessionID)
      if (capture) {
        if (!currentCapture(capture) || !matchesCaptureMessage(capture, input.message)) return
        // This hook's provider is the continuation provider, not necessarily the compaction provider.
        releaseCapture(capture, capture.phase === "pending" || capture.phase === "ready" ? "ignored" : capture.phase)
        // Native compaction on an unsupported target must not arm continuation filtering
        // for a configured source provider's next ordinary request.
        if (capture.phase === "ignored") {
          if (compactionOwners.get(input.sessionID) === capture.id) compactionOwners.delete(input.sessionID)
          return
        }
      }
      // Legacy responses have no verifiable OpenCode parent. Their late callbacks must
      // neither move a boundary nor arm filtering for a later, genuine user request.
      if (!capture && [...compactOperationsByProvider.values()].some((sessions) => sessions.has(input.sessionID))) return
      const providerID = getProviderID(input)
      if (!providerID || !configuredProviders.has(providerID)) return
      const message = asRecord(input.message)
      const messageID = message?.id
      const createdAt = asRecord(message?.time)?.created
      pendingAutoContinueRequests.delete(input.sessionID)
      pendingAutoContinues.set(input.sessionID, {
        providerID,
        agent: typeof input.agent === "string" ? input.agent : undefined,
        compactionMessageID: typeof messageID === "string" ? messageID : undefined,
        compactionCreatedAt: finiteNumber(createdAt) ? createdAt : undefined,
      })
    },
  }

  return hooks
}
