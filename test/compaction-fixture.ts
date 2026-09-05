import type { Hooks } from "@opencode-ai/plugin"

type CaptureInput = {
  sessionID: string
  history: any[]
  boundary?: any
  model?: any
  agent?: string
  summaryID?: string
}

// v1.18.23 persists the compaction user before compacting, excludes it from the
// transform, and only then persists a summary assistant before chat.headers.
export function compactionFixture(rawHistory?: (sessionID: string) => Promise<unknown>) {
  const sessions = new Map<string, any[]>()
  const getSessionMessages = async (sessionID: string) => sessions.get(sessionID) ?? await rawHistory?.(sessionID)

  async function prepare(input: CaptureInput) {
    const { sessionID } = input
    const model = { providerID: "openai", id: "gpt-current", ...input.model }
    model.id = input.model?.id ?? input.model?.modelID ?? model.id
    const latestTime = Math.max(Date.now(), ...input.history.map((message) => message.info?.time?.created ?? 0))
    const boundary = {
      sessionID, role: "user", agent: "build", model: { providerID: model.providerID, modelID: model.id },
      id: `msg_fixture_boundary_${sessions.size}_${latestTime}`, time: { created: latestTime + 1 },
      ...input.boundary,
    }
    const history = input.history.map((message) => ({
      ...message,
      info: {
        ...message.info,
        time: message.info.time ?? { created: boundary.time.created - 1 },
      },
    }))
    const prior = await rawHistory?.(sessionID)
    const raw = (Array.isArray(prior) ? prior : history).map((message) => ({
      ...message,
      info: { ...message.info, time: message.info.time ?? { created: boundary.time.created - 1 } },
      parts: message.parts.map((part: any) => ({ messageID: message.info.id, sessionID, ...part })),
    }))
    sessions.set(sessionID, [...raw, {
      info: boundary, parts: [{ type: "compaction", messageID: boundary.id, sessionID, auto: true }],
    }])
    const request = { sessionID, message: boundary, model, agent: input.agent ?? "compaction" }
    return { history, request, summaryID: input.summaryID ?? `${boundary.id}_summary` }
  }

  function addSummary(prepared: Awaited<ReturnType<typeof prepare>>) {
    const { request, summaryID } = prepared
    sessions.get(request.sessionID)!.push({
      info: {
        id: summaryID, sessionID: request.sessionID, role: "assistant", summary: true,
        parentID: request.message.id, providerID: request.model.providerID,
        modelID: request.model.id, agent: request.agent,
        time: { created: request.message.time.created + 1 },
      },
      parts: [],
    })
  }

  async function capture(hooks: Hooks, input: CaptureInput) {
    const prepared = await prepare(input)
    await hooks["experimental.session.compacting"]?.({ sessionID: input.sessionID }, { context: [] })
    await hooks["experimental.chat.messages.transform"]?.({}, { messages: prepared.history } as any)
    addSummary(prepared)
    const output = { headers: {} as Record<string, string> }
    await hooks["chat.headers"]?.(prepared.request as any, output)
    return output.headers
  }

  return { getSessionMessages, prepare, addSummary, capture, sessions }
}
