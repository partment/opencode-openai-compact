import { describe, expect, test } from "vitest"
import { compactBody, compactedItemsFrom, createCompactHooks } from "../src/compact.js"
import { defaultConfig, OpenAICompactConfigSchema } from "../src/schema.js"
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

async function attachmentRequest(
  files: any[],
  text: string,
  firstResponse?: () => Response,
  secondResponse?: () => Response,
  cleared = false,
  checkpointItems?: any[],
) {
  const store = CheckpointStore.openMemory()
  if (checkpointItems) {
    store.upsert("ses_attachments", {
      providerID: "openai", responseID: "resp_previous", afterMessageID: "msg_previous",
      afterCreatedAt: 1, createdAt: Date.now(), items: checkpointItems,
    })
  }
  const bodies: any[] = []
  const fakeFetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    bodies.push(jsonBody(init))
    const response = bodies.length === 1 ? firstResponse : secondResponse
    return response ? response() : compactResponse({
      id: "resp_attachments", created_at: 1,
      output: [{ type: "compaction", encrypted_content: "compacted" }],
    })
  }) as typeof fetch
  try {
    const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
    const cfg: any = {}
    await hooks.config?.(cfg)
    const sessionID = "ses_attachments"
    await hooks["experimental.session.compacting"]?.({ sessionID } as any, { context: [], prompt: undefined })
    await hooks["experimental.chat.messages.transform"]?.({}, { messages: [
      { info: { id: "msg_user", sessionID, role: "user" }, parts: files.length
        ? files.map((file) => ({ type: "file", ...file }))
        : [{ type: "text", text: "request" }] },
      { info: { id: "msg_assistant", sessionID, role: "assistant", providerID: "openai", modelID: currentModel },
        parts: [{ type: "tool", tool: "read", callID: "call_attachment", state: {
          status: "completed", input: {}, output: text, attachments: files,
          time: { start: 1, end: 2, ...(cleared ? { compacted: 3 } : {}) },
        } }] },
    ] } as any)
    const response = await cfg.provider.openai.options.fetch("https://proxy.test/openai/v1/responses", {
      method: "POST",
      headers: { [defaultConfig.headers.compact]: "1", [defaultConfig.headers.session]: sessionID },
      body: JSON.stringify({ model: currentModel, input: [{ role: "user", content: "compact" }] }),
    })
    return { bodies, response, checkpoints: store.loadAll() }
  } finally {
    store.close()
  }
}

const compactionInstructions = "You are an anchored context summarization assistant for coding sessions."
const currentModel = "gpt-current"
const embeddedOpenCodeHistory = [
  "Here is the conversation so far:",
  "<conversation>",
  "[User]: preserved history",
  "[Assistant]: preserved response",
  "</conversation>",
  "Here is the summary of the conversation before the <conversation> above:",
].join("\n\n")

function invalidCheckpointItems() {
  return [
    { role: "user", content: embeddedOpenCodeHistory },
    { type: "compaction", encrypted_content: "invalid-checkpoint" },
  ]
}

function forkHistory(sessionID: string, prefix: string, now: number) {
  return [
    {
      info: { id: `${prefix}_start`, sessionID, role: "user", time: { created: now } },
      parts: [{ id: `${prefix}_start_part`, messageID: `${prefix}_start`, sessionID, type: "text", text: "start" }],
    },
    {
      info: {
        id: `${prefix}_answer`,
        sessionID,
        role: "assistant",
        parentID: `${prefix}_start`,
        time: { created: now + 1 },
      },
      parts: [
        { id: `${prefix}_answer_part`, messageID: `${prefix}_answer`, sessionID, type: "text", text: "answer" },
      ],
    },
    {
      info: { id: `${prefix}_checkpoint`, sessionID, role: "user", time: { created: now + 2 } },
      parts: [
        {
          id: `${prefix}_checkpoint_part`,
          messageID: `${prefix}_checkpoint`,
          sessionID,
          type: "compaction",
          tail_start_id: `${prefix}_start`,
        },
      ],
    },
    {
      info: {
        id: `${prefix}_summary`,
        sessionID,
        role: "assistant",
        parentID: `${prefix}_checkpoint`,
        summary: true,
        time: { created: now + 3 },
      },
      parts: [
        {
          id: `${prefix}_summary_part`,
          messageID: `${prefix}_summary`,
          sessionID,
          type: "text",
          text: defaultConfig.summary,
        },
      ],
    },
    {
      info: { id: `${prefix}_control`, sessionID, role: "user", time: { created: now + 4 } },
      parts: [
        {
          id: `${prefix}_control_part`,
          messageID: `${prefix}_control`,
          sessionID,
          type: "text",
          text: "markerless continuation",
          synthetic: true,
        },
      ],
    },
    {
      info: {
        id: `${prefix}_continued`,
        sessionID,
        role: "assistant",
        parentID: `${prefix}_control`,
        time: { created: now + 5 },
      },
      parts: [
        {
          id: `${prefix}_continued_part`,
          messageID: `${prefix}_continued`,
          sessionID,
          type: "text",
          text: "continued work",
        },
      ],
    },
    {
      info: {
        id: `${prefix}_target`,
        sessionID,
        role: "user",
        model: { providerID: "openai", modelID: currentModel },
        time: { created: now + 6 },
      },
      parts: [{ id: `${prefix}_target_part`, messageID: `${prefix}_target`, sessionID, type: "text", text: "target" }],
    },
    {
      info: {
        id: `${prefix}_second_answer`,
        sessionID,
        role: "assistant",
        parentID: `${prefix}_target`,
        time: { created: now + 7 },
      },
      parts: [
        {
          id: `${prefix}_second_answer_part`,
          messageID: `${prefix}_second_answer`,
          sessionID,
          type: "text",
          text: "second answer",
        },
      ],
    },
    {
      info: { id: `${prefix}_second_checkpoint`, sessionID, role: "user", time: { created: now + 8 } },
      parts: [
        {
          id: `${prefix}_second_checkpoint_part`,
          messageID: `${prefix}_second_checkpoint`,
          sessionID,
          type: "compaction",
          tail_start_id: `${prefix}_target`,
        },
      ],
    },
    {
      info: {
        id: `${prefix}_second_summary`,
        sessionID,
        role: "assistant",
        parentID: `${prefix}_second_checkpoint`,
        summary: true,
        time: { created: now + 9 },
      },
      parts: [
        {
          id: `${prefix}_second_summary_part`,
          messageID: `${prefix}_second_summary`,
          sessionID,
          type: "text",
          text: defaultConfig.summary,
        },
      ],
    },
  ]
}

describe("OpenAI compact hooks", () => {
  test("defaults to following the conversation model and reasoning effort", () => {
    expect(defaultConfig.providers.openai).toEqual({
      compactModel: null,
      compactReasoningEffort: null,
    })
    for (const effort of ["none", "minimal", "low", "medium", "high", "xhigh", "max"] as const) {
      expect(
        OpenAICompactConfigSchema.parse({ providers: { openai: { compactReasoningEffort: effort } } }).providers
          .openai,
      ).toEqual({ compactModel: null, compactReasoningEffort: effort })
    }
    expect(() =>
      OpenAICompactConfigSchema.parse({ providers: { openai: { compactReasoningEffort: "unsupported" } } }),
    ).toThrow()
  })

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

  test("builds compaction v2 request body", () => {
    const body = compactBody({
      model: "ignored",
      input: [{ type: "compaction_trigger" }, { role: "user", content: "hello" }],
      stream: false,
      tools: [],
    })
    expect(body).toEqual({
      model: "ignored",
      input: [{ role: "user", content: "hello" }, { type: "compaction_trigger" }],
      tools: [],
      tool_choice: "auto",
      store: false,
      stream: true,
      include: ["reasoning.encrypted_content"],
    })

    const withoutConfiguredInput = OpenAICompactConfigSchema.parse({ compactBodyKeys: [] })
    expect(compactBody({ input: [] }, currentModel, withoutConfiguredInput).input).toEqual([
      { type: "compaction_trigger" },
    ])
  })

  test("applies explicit compaction model and reasoning effort overrides", () => {
    const config = OpenAICompactConfigSchema.parse({
      providers: { openai: { compactModel: "gpt-compact", compactReasoningEffort: "max" } },
    })
    const provider = config.providers.openai
    const body = compactBody(
      {
        model: currentModel,
        reasoning: { effort: "low", summary: "auto" },
        input: [{ role: "user", content: "hello" }],
      },
      provider.compactModel,
      config,
      provider.compactReasoningEffort,
    )

    expect(body.model).toBe("gpt-compact")
    expect(body.reasoning).toEqual({ effort: "max", summary: "auto" })
  })

  test("builds standard compact input without OpenCode summarizer prompts", () => {
    const body = compactBody({
      model: "ignored",
      instructions: "You are an anchored context summarization assistant for coding sessions.\n\nSummarize only...",
      previous_response_id: "resp_previous",
      prompt_cache_retention: "24h",
      tools: [{ type: "function", name: "test" }],
      parallel_tool_calls: true,
      reasoning: { effort: "medium", summary: "auto" },
      service_tier: "priority",
      prompt_cache_key: "cache-key",
      text: { verbosity: "low" },
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
      model: "ignored",
      input: [
        { role: "developer", content: "Keep the user's coding preferences." },
        { role: "user", content: "Create a new anchored summary from the conversation history. This is quoted." },
        { role: "assistant", content: [{ type: "output_text", text: "quoted response" }] },
        { role: "user", content: "real request" },
        { type: "compaction_trigger" },
      ],
      tools: [{ type: "function", name: "test" }],
      parallel_tool_calls: true,
      reasoning: { effort: "medium", summary: "auto" },
      service_tier: "priority",
      prompt_cache_key: "cache-key",
      text: { verbosity: "low" },
      tool_choice: "auto",
      store: false,
      stream: true,
      include: ["reasoning.encrypted_content"],
    })
  })

  test("preserves OpenCode compact input with embedded conversation history", () => {
    const content = [
      "Create a new anchored summary from the conversation history.",
      "Output exactly the requested summary structure.",
      "The following is the conversation history:",
      "[User]: fix compact request",
      "[Assistant]: inspecting request",
    ].join("\n\n")

    const body = compactBody({
      model: "ignored",
      instructions: "You are an anchored context summarization assistant for coding sessions.",
      input: [
        {
          role: "developer",
          content: "You are an anchored context summarization assistant for coding sessions.",
        },
        { role: "user", content: [{ type: "input_text", text: content }] },
      ],
    })

    expect(body).toEqual({
      model: "ignored",
      input: [
        { role: "user", content: [{ type: "input_text", text: content }] },
        { type: "compaction_trigger" },
      ],
      tool_choice: "auto",
      store: false,
      stream: true,
      include: ["reasoning.encrypted_content"],
    })
  })

  test("preserves OpenCode 1.18.18 tagged conversation input", () => {
    const developerPrompt =
      "You are a context summarization agent. You are given a conversation between a user and an agent."
    const content = [
      "Here is the conversation so far:",
      "<conversation>",
      "[User]: fix compact request",
      "[Assistant]: inspecting request",
      "</conversation>",
      "Here is the summary of the conversation before the <conversation> above:",
      "<prior-summary>",
      "## Objective\n- Preserve the task",
      "</prior-summary>",
    ].join("\n\n")

    expect(
      compactBody({
        model: "ignored",
        instructions: developerPrompt,
        input: [
          { role: "developer", content: developerPrompt },
          { role: "user", content: [{ type: "input_text", text: content }] },
        ],
      }),
    ).toEqual({
      model: "ignored",
      input: [
        { role: "user", content: [{ type: "input_text", text: content }] },
        { type: "compaction_trigger" },
      ],
      tool_choice: "auto",
      store: false,
      stream: true,
      include: ["reasoning.encrypted_content"],
    })
  })

  test("restores structured compact input from an aborted assistant with completed work", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return compactResponse({
        id: "resp_structured",
        created_at: 1,
        output: [{ type: "compaction", encrypted_content: "compacted" }],
      })
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch
      const sessionID = "ses_structured"

      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: { id: "msg_user", sessionID, role: "user" },
              parts: [
                { type: "text", text: "fix compact request" },
                { type: "file", mime: "image/png", filename: "request.png", url: "data:image/png;base64,AA==" },
              ],
            },
            {
              info: {
                id: "msg_assistant",
                sessionID,
                role: "assistant",
                providerID: "openai",
                modelID: currentModel,
                error: { name: "MessageAbortedError", data: { message: "Aborted" } },
              },
              parts: [
                {
                  type: "reasoning",
                  text: "Inspected the request.",
                  metadata: {
                    openai: { itemId: "rs_structured", reasoningEncryptedContent: "encrypted-reasoning" },
                  },
                },
                {
                  type: "text",
                  text: "The request loses history.",
                  metadata: { openai: { itemId: "msg_structured", phase: "final_answer" } },
                },
                {
                  type: "tool",
                  tool: "read_file",
                  callID: "call_structured",
                  state: {
                    status: "completed",
                    input: { path: "src/compact.ts" },
                    output: "file contents",
                    time: { start: 1, end: 2 },
                  },
                },
              ],
            },
          ],
        } as any,
      )

      const embedded = "Unrecognized future compaction format with flattened history that must not be sent"
      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: sessionID,
        },
        body: JSON.stringify({
          model: currentModel,
          instructions: "Unrecognized future compaction agent instructions.",
          input: [{ role: "user", content: [{ type: "input_text", text: embedded }] }],
        }),
      })

      expect(calls).toHaveLength(1)
      expect(calls[0]?.url).toBe("https://proxy.test/openai/v1/responses")
      expect(jsonBody(calls[0]?.init)).toEqual({
        model: currentModel,
        input: [
          {
            role: "user",
            content: [
              { type: "input_text", text: "fix compact request" },
              { type: "input_image", image_url: "data:image/png;base64,AA==" },
            ],
          },
          {
            type: "reasoning",
            id: "rs_structured",
            encrypted_content: "encrypted-reasoning",
            summary: [{ type: "summary_text", text: "Inspected the request." }],
          },
          {
            role: "assistant",
            content: [{ type: "output_text", text: "The request loses history." }],
            id: "msg_structured",
            phase: "final_answer",
          },
          {
            type: "function_call",
            call_id: "call_structured",
            name: "read_file",
            arguments: JSON.stringify({ path: "src/compact.ts" }),
          },
          { type: "function_call_output", call_id: "call_structured", output: "file contents" },
          { type: "compaction_trigger" },
        ],
        tool_choice: "auto",
        store: false,
        stream: true,
        include: ["reasoning.encrypted_content"],
      })
    } finally {
      store.close()
    }
  })

  test.each([1999, 2000, 2001, 12000])("preserves all %i characters of tool output", async (length) => {
    const text = "漢😀" + "x".repeat(length - 3)
    const { bodies } = await attachmentRequest([], text)
    expect(bodies[0].input.find((item: any) => item.type === "function_call_output").output).toBe(text)
  })

  test("preserves user and tool attachment content and order", async () => {
    const files = [
      { mime: "image/png", filename: "image.png", url: "data:image/png;base64,AA==" },
      { mime: "image/jpeg", url: "https://example.test/image.jpg" },
      { mime: "application/pdf", filename: "doc.pdf", url: "data:application/pdf;base64,AA==" },
      { mime: "application/pdf", url: "https://example.test/doc.pdf" },
      { mime: "text/plain", url: "data:text/plain,hello" },
      { mime: "application/json", url: "https://example.test/file.json" },
    ]
    const expected = [
      { type: "input_image", image_url: files[0].url },
      { type: "input_image", image_url: files[1].url },
      { type: "input_file", filename: "doc.pdf", file_data: files[2].url },
      { type: "input_file", file_url: files[3].url },
      { type: "input_file", filename: "file", file_data: files[4].url },
      { type: "input_file", file_url: files[5].url },
    ]
    const { bodies } = await attachmentRequest(files, "full text")
    expect(bodies[0].input[0].content).toEqual(expected)
    expect(bodies[0].input.find((item: any) => item.type === "function_call_output")).toEqual({
      type: "function_call_output", call_id: "call_attachment",
      output: [{ type: "input_text", text: "full text" }, ...expected],
    })
  })

  test("marks unavailable attachments without exposing content sources", async () => {
    const files = [
      { filename: "missing-mime", url: "https://secret.test/?token=secret" },
      { mime: "image/png", filename: "missing-url" },
      { mime: "image/png", url: "file:///private/image.png" },
      { mime: "image/png", url: "C:\\private\\image.png" },
      { mime: "image/png", url: "data:image/png;base64," },
      { mime: "application/x-directory", url: "file:///private" },
      { mime: "image/png", url: "ftp://secret.test/image.png" },
      null,
    ]
    const { bodies } = await attachmentRequest(files, "unchanged")
    const output = bodies[0].input.find((item: any) => item.type === "function_call_output").output
    expect(output[0]).toEqual({ type: "input_text", text: "unchanged" })
    expect(output.slice(1)).toHaveLength(files.length)
    for (const part of output.slice(1)) {
      expect(part.type).toBe("input_text")
      expect(part.text).toContain("content unavailable:")
      expect(part.text).not.toMatch(/secret|private|base64/)
    }
  })

  test.each([
    [400, { code: "invalid_image" }, true],
    [413, { message: "File size exceeds limit" }, true],
    [415, { param: "input[0].content[0].file_data" }, true],
    [422, { message: "Cannot download image" }, true],
    [400, { code: "context_length_exceeded", message: "Image tokens exceed context" }, false],
    [400, { message: "Invalid model" }, false],
    [400, { code: "invalid_profile" }, false],
    [401, { code: "invalid_image" }, false],
    [429, { code: "invalid_image" }, false],
    [500, { code: "invalid_image" }, false],
  ])("handles attachment rejection %i %j", async (status, error, retry) => {
    const text = "完整😀".repeat(3000)
    const { bodies, response } = await attachmentRequest(
      [{ mime: "image/png", url: "data:image/png;base64,AA==" }], text,
      () => Response.json({ error }, { status }),
    )
    expect(bodies).toHaveLength(retry ? 2 : 1)
    expect(response.status).toBe(retry ? 200 : status)
    if (retry) {
      const output = bodies[1].input.find((item: any) => item.type === "function_call_output").output
      expect(output[0].text).toBe(text)
      expect(output[1].text).toContain("attachment rejected by API")
      expect(bodies[1].input[0].content[0].text).toContain("attachment rejected by API")
      expect(bodies[1].model).toBe(bodies[0].model)
    }
  })

  test("does not retry attachment rejection twice or retry without attachments", async () => {
    const reject = () => Response.json({ error: { code: "invalid_image" } }, { status: 400 })
    const failed = await attachmentRequest([{ mime: "image/png", url: "https://example.test/a.png" }], "text", reject, reject)
    expect(failed.bodies).toHaveLength(2)
    expect(failed.response.status).toBe(400)
    const plain = await attachmentRequest([], "text", reject)
    expect(plain.bodies).toHaveLength(1)
    expect(plain.response.status).toBe(400)
  })

  test("preserves existing checkpoints during attachment fallback and failure", async () => {
    const items = [
      { role: "user", content: [{ type: "input_image", image_url: "https://example.test/old.png" }] },
      { type: "compaction", encrypted_content: "old-encrypted-checkpoint" },
    ]
    const reject = () => Response.json({ error: { code: "invalid_image" } }, { status: 400 })
    for (const failRetry of [false, true]) {
      const result = await attachmentRequest(
        [{ mime: "image/png", url: "https://example.test/new.png" }], "text",
        reject, failRetry ? reject : undefined, false, items,
      )
      expect(result.bodies).toHaveLength(2)
      for (const body of result.bodies) expect(body.input.slice(0, 2)).toEqual(items)
      if (failRetry) {
        expect(result.checkpoints[0]?.checkpoint.responseID).toBe("resp_previous")
        expect(result.checkpoints[0]?.checkpoint.items).toEqual(items)
      } else {
        expect(result.response.status).toBe(200)
      }
    }
  })

  test("does not restore attachments of cleared tool results", async () => {
    const { bodies } = await attachmentRequest([{ mime: "image/png", url: "https://example.test/a.png" }], "old", undefined, undefined, true)
    expect(bodies[0].input.find((item: any) => item.type === "function_call_output").output).toBe("[Old tool result content cleared]")
  })

  test("follows the latest conversation settings without inspecting the serialized prompt", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ init?: RequestInit }> = []
    const fakeFetch = (async (_requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ init })
      return compactResponse({
        id: "resp_follow_conversation",
        created_at: 1,
        output: [{ type: "compaction", encrypted_content: "compacted" }],
      })
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const sessionID = "ses_follow_conversation"

      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: {
                id: "msg_old",
                sessionID,
                role: "assistant",
                providerID: "openai",
                modelID: "gpt-old",
                variant: "low",
              },
              parts: [{ type: "text", text: "old response" }],
            },
            {
              info: {
                id: "msg_latest",
                sessionID,
                role: "user",
                model: { providerID: "openai", modelID: "gpt-latest", variant: "xhigh" },
              },
              parts: [{ type: "text", text: "latest request" }],
            },
          ],
        } as any,
      )

      const embedded = "Unrecognized future compaction format with flattened history"
      await cfg.provider.openai.options.fetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: sessionID,
        },
        body: JSON.stringify({
          model: "compaction-agent-model",
          instructions: "Unrecognized future compaction agent instructions.",
          reasoning: { effort: "medium", summary: "auto" },
          input: [{ role: "user", content: embedded }],
        }),
      })

      const body = jsonBody(calls[0]?.init)
      expect(body.model).toBe("gpt-latest")
      expect(body.reasoning).toEqual({ effort: "xhigh", summary: "auto" })
    } finally {
      store.close()
    }
  })

  test("orders stable instructions, checkpoint, and restored structured history", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      const call = calls.length
      return compactResponse({
        id: `resp_${call}`,
        created_at: call,
        output: [{ id: `cmp_${call}`, type: "compaction", encrypted_content: `compacted-${call}` }],
      })
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch
      const sessionID = "ses_structured_checkpoint"

      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({
          model: "gpt",
          instructions: "You are OpenCode.",
          input: [{ role: "developer", content: "Stable developer context." }, { role: "user", content: "hello" }],
        }),
      })
      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: sessionID,
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: "You are an anchored context summarization assistant for coding sessions.",
          input: [
            {
              role: "developer",
              content: "You are an anchored context summarization assistant for coding sessions.",
            },
            { role: "user", content: "old history" },
            { role: "user", content: "Create a new anchored summary from the conversation history." },
          ],
        }),
      })

      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: { id: "msg_tail", sessionID, role: "user", model: { providerID: "openai", modelID: "ignored" } },
              parts: [{ type: "text", text: "structured tail" }],
            },
          ],
        } as any,
      )

      const embedded = [
        "Create a new anchored summary from the conversation history.",
        "The following is the conversation history:",
        "[User]: flattened tail",
      ].join("\n\n")
      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: sessionID,
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: "You are an anchored context summarization assistant for coding sessions.",
          input: [{ role: "user", content: embedded }],
        }),
      })

      expect(jsonBody(calls.at(-1)?.init)).toEqual({
        model: "ignored",
        instructions: "You are OpenCode.",
        input: [
          { role: "developer", content: "Stable developer context." },
          { role: "user", content: "old history" },
          { id: "cmp_2", type: "compaction", encrypted_content: "compacted-2" },
          { role: "user", content: [{ type: "input_text", text: "structured tail" }] },
          { type: "compaction_trigger" },
        ],
        tool_choice: "auto",
        store: false,
        stream: true,
        include: ["reasoning.encrypted_content"],
      })
    } finally {
      store.close()
    }
  })

  test("reuses a persisted checkpoint when repeated compaction omits the prior boundary", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ init?: RequestInit }> = []
    const fakeFetch = (async (_requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ init })
      return compactResponse({
        id: "resp_repeated",
        model: "gpt",
        output: [{ type: "compaction", encrypted_content: "repeated" }],
      })
    }) as typeof fetch
    const sessionID = "ses_repeated_without_boundary"
    const now = Date.now()
    store.upsert(sessionID, {
      providerID: "openai",
      responseID: "resp_previous",
      afterMessageID: "msg_hidden_boundary",
      afterCreatedAt: now,
      createdAt: now,
      items: [
        { role: "user", content: "previous history" },
        { type: "compaction", encrypted_content: "previous" },
      ],
    })
    store.upsertControlMessage({
      providerID: "openai",
      sessionID,
      messageID: "msg_old_continue",
      createdAt: now + 1,
      contentText: "old internal request",
    })

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: { id: "msg_old_continue", sessionID, role: "user", time: { created: now + 1 } },
              parts: [{ type: "text", text: "text and metadata changed" }],
            },
            {
              info: {
                id: "msg_tail",
                sessionID,
                role: "user",
                model: { providerID: "openai", modelID: "gpt" },
                time: { created: now + 2 },
              },
              parts: [{ type: "text", text: "retained tail" }],
            },
          ],
        } as any,
      )
      const headers = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        {
          sessionID,
          agent: "another-internal-name",
          model: { providerID: "openai" },
          message: { id: "msg_new_compaction", time: { created: now + 3 }, agent: "build" },
        } as any,
        headers,
      )
      await cfg.provider.openai.options.fetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: headers.headers,
        body: JSON.stringify({
          model: "ignored",
          instructions: "Completely changed OpenCode compaction prompt.",
          input: [{ role: "user", content: "Unknown flattened format." }],
        }),
      })

      expect(jsonBody(calls[0]?.init).input).toEqual([
        { role: "user", content: "previous history" },
        { type: "compaction", encrypted_content: "previous" },
        { role: "user", content: [{ type: "input_text", text: "retained tail" }] },
        { type: "compaction_trigger" },
      ])
    } finally {
      store.close()
    }
  })

  test("rejects embedded history retries when structured messages cannot be cloned", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return compactResponse({
        id: "resp_fallback",
        created_at: 1,
        output: [{ type: "compaction", encrypted_content: "compacted" }],
      })
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch
      const sessionID = "ses_structured_fallback"

      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: { id: "msg_uncloneable", sessionID, role: "user" },
              parts: [{ type: "text", text: "history", metadata: { uncloneable: () => undefined } }],
            },
          ],
        } as any,
      )

      const embedded = [
        "Here is the conversation so far:",
        "<conversation>",
        "[User]: preserved fallback history",
        "</conversation>",
        "Here is the summary of the conversation before the <conversation> above:",
        "<prior-summary>",
        "## Objective\n- Preserve the fallback",
        "</prior-summary>",
      ].join("\n\n")
      const request = {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: sessionID,
        },
        body: JSON.stringify({
          model: "ignored",
          instructions:
            "You are a context summarization agent. You are given a conversation between a user and an agent.",
          input: [{ role: "user", content: embedded }],
        }),
      }
      for (let attempt = 0; attempt < 2; attempt++) {
        const response = await wrappedFetch("https://proxy.test/openai/v1/responses", request)
        expect(response.status).toBe(502)
        expect(await response.text()).toContain("could not be captured safely")
      }
      expect(calls).toEqual([])
      expect(store.count()).toBe(0)
    } finally {
      store.close()
    }
  })

  test("fails closed when a captured compaction transaction cannot clone its history", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ init?: RequestInit }> = []
    const fakeFetch = (async (_requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ init })
      return new Response("must not be called")
    }) as typeof fetch
    const sessionID = "ses_failed_transaction_capture"

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: { id: "msg_uncloneable", sessionID, role: "user", model: { providerID: "openai", modelID: "gpt" } },
              parts: [{ type: "text", text: "history", metadata: { uncloneable: () => undefined } }],
            },
          ],
        } as any,
      )
      const headers = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        {
          sessionID,
          agent: "changed-compaction-agent",
          model: { providerID: "openai" },
          message: { id: "msg_compaction", time: { created: Date.now() }, agent: "build" },
        } as any,
        headers,
      )
      const response = await cfg.provider.openai.options.fetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: headers.headers,
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "changed prompt" }] }),
      })

      expect(response.status).toBe(502)
      expect(await response.text()).toContain("could not be captured safely")
      expect(calls).toEqual([])
      expect(store.count()).toBe(0)
    } finally {
      store.close()
    }
  })

  test("uses native compaction for an invalid checkpoint, clears on the completed summary, then resumes plugin compaction", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ init?: RequestInit }> = []
    let callCount = 0
    const fakeFetch = (async (_requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ init })
      callCount++
      if (callCount === 2) return new Response("checkpoint rejected", { status: 422 })
      if (callCount === 3) return new Response("native summary", { status: 200 })
      if (jsonBody(init).input?.at(-1)?.type === "compaction_trigger") {
        return compactResponse({
          id: "resp_after_native",
          model: currentModel,
          created_at: 1,
          output: [{ type: "compaction", encrypted_content: "healthy-checkpoint" }],
        })
      }
      return new Response("ok")
    }) as typeof fetch
    const sessionID = "ses_invalid_checkpoint_native_fallback"
    const now = Date.now()
    store.upsert(sessionID, {
      providerID: "openai",
      responseID: "resp_invalid_checkpoint",
      afterMessageID: "msg_invalid_checkpoint",
      afterCreatedAt: now,
      createdAt: now,
      items: invalidCheckpointItems(),
    })
    store.upsertControlMessage({
      providerID: "openai",
      sessionID,
      messageID: "msg_old_control",
      createdAt: now,
      contentText: "old control",
    })

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)

      await cfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({ model: currentModel, input: [{ role: "user", content: "before checkpoint check" }] }),
      })
      expect(jsonBody(calls[0]?.init).input).toEqual([{ role: "user", content: "before checkpoint check" }])

      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: { id: "msg_invalid_checkpoint", sessionID, role: "user", time: { created: now } },
              parts: [{ type: "compaction" }],
            },
            {
              info: {
                id: "msg_invalid_summary",
                sessionID,
                role: "assistant",
                parentID: "msg_invalid_checkpoint",
                summary: true,
                finish: "stop",
                time: { created: now + 1, completed: now + 2 },
              },
              parts: [{ type: "text", text: defaultConfig.summary }],
            },
            {
              info: {
                id: "msg_native_tail",
                sessionID,
                role: "user",
                model: { providerID: "openai", modelID: currentModel },
                time: { created: now + 2 },
              },
              parts: [{ type: "text", text: "tail for native summary" }],
            },
          ],
        } as any,
      )
      const nativeHeaders = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        {
          sessionID,
          agent: "renamed-compaction-agent",
          model: { providerID: "openai" },
          message: { id: "msg_native_compaction", time: { created: now + 3 }, agent: "build" },
        } as any,
        nativeHeaders,
      )
      expect(nativeHeaders.headers[defaultConfig.headers.compact]).toBe("native")

      const nativeBody = {
        model: currentModel,
        instructions: compactionInstructions,
        input: [{ role: "user", content: "OpenCode native compaction request" }],
      }
      const nativeResponse = await cfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: nativeHeaders.headers,
        body: JSON.stringify(nativeBody),
      })

      expect(await nativeResponse.text()).toBe("native summary")
      expect(jsonBody(calls[1]?.init).input).toEqual([
        ...invalidCheckpointItems(),
        { role: "user", content: "OpenCode native compaction request" },
      ])
      expect(jsonBody(calls[2]?.init)).toEqual(nativeBody)
      expect(store.count()).toBe(1)
      expect(store.loadControlMessages()).toHaveLength(1)

      await hooks.event?.({
        event: {
          type: "message.updated",
          properties: {
            sessionID,
            info: {
              id: "msg_unrelated_summary",
              sessionID,
              role: "assistant",
              parentID: "msg_another_compaction",
              summary: true,
              finish: "stop",
              time: { created: now + 4, completed: now + 5 },
            },
          },
        } as any,
      })
      expect(store.count()).toBe(1)

      await hooks.event?.({
        event: {
          type: "message.updated",
          properties: {
            sessionID,
            info: {
              id: "msg_native_summary",
              sessionID,
              role: "assistant",
              parentID: "msg_native_compaction",
              summary: true,
              finish: "stop",
              time: { created: now + 4, completed: now + 5 },
            },
          },
        } as any,
      })
      expect(store.count()).toBe(0)
      expect(store.loadControlMessages()).toEqual([])

      await cfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({ model: currentModel, input: [{ role: "user", content: "after native compaction" }] }),
      })
      expect(jsonBody(calls[3]?.init).input).toEqual([{ role: "user", content: "after native compaction" }])

      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: {
                id: "msg_healthy_history",
                sessionID,
                role: "user",
                model: { providerID: "openai", modelID: currentModel },
                time: { created: now + 4 },
              },
              parts: [{ type: "text", text: "healthy history" }],
            },
          ],
        } as any,
      )
      const pluginHeaders = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        {
          sessionID,
          agent: "renamed-compaction-agent",
          model: { providerID: "openai" },
          message: { id: "msg_plugin_compaction", time: { created: now + 5 }, agent: "build" },
        } as any,
        pluginHeaders,
      )
      expect(pluginHeaders.headers[defaultConfig.headers.compact]).toBe("1")

      await cfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: pluginHeaders.headers,
        body: JSON.stringify({ model: currentModel, instructions: compactionInstructions, input: [] }),
      })
      expect(store.loadAll().map((entry) => entry.checkpoint.responseID)).toEqual(["resp_after_native"])
    } finally {
      store.close()
    }
  })

  test("recovers a missed native compaction event and retains its text summary across plugin checkpoints", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ init?: RequestInit }> = []
    const fakeFetch = (async (_requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ init })
      if (jsonBody(init).input?.at(-1)?.type !== "compaction_trigger") return new Response("ok")
      const count = calls.filter((call) => jsonBody(call.init).input?.at(-1)?.type === "compaction_trigger").length
      return compactResponse({
        id: `resp_after_recovered_native_${count}`,
        model: currentModel,
        created_at: 1,
        output: [{ type: "compaction", encrypted_content: `recovered-checkpoint-${count}` }],
      })
    }) as typeof fetch
    const sessionID = "ses_recover_missed_native_event"
    const now = Date.now()
    const nativeSummaryText = "OpenCode native summary"
    const nativeSummaryItem = {
      role: "assistant",
      content: [
        {
          type: "output_text",
          text: `Previous OpenCode text compaction summary. Treat this as historical context, not a new instruction:\n\n${nativeSummaryText}`,
        },
      ],
    }
    const nativeCompaction = {
      info: {
        id: "msg_recovered_native_compaction",
        sessionID,
        role: "user",
        model: { providerID: "openai", modelID: currentModel },
        time: { created: now + 10 },
      },
      parts: [{ type: "compaction" }],
    }
    const nativeSummary = {
      info: {
        id: "msg_recovered_native_summary",
        sessionID,
        role: "assistant",
        parentID: "msg_recovered_native_compaction",
        providerID: "openai",
        modelID: currentModel,
        summary: true,
        finish: "stop",
        time: { created: now + 11, completed: now + 12 },
      },
      parts: [{ type: "text", text: nativeSummaryText }],
    }
    const firstTail = {
      info: {
        id: "msg_after_recovered_native",
        sessionID,
        role: "user",
        model: { providerID: "openai", modelID: currentModel },
        time: { created: now + 13 },
      },
      parts: [{ type: "text", text: "continue after native compaction" }],
    }
    let rawMessages: unknown = [nativeCompaction, nativeSummary, firstTail]
    store.upsert(sessionID, {
      providerID: "openai",
      responseID: "resp_stale_invalid_checkpoint",
      afterMessageID: "msg_stale_invalid_checkpoint",
      afterCreatedAt: now,
      createdAt: now,
      items: invalidCheckpointItems(),
    })
    store.upsertControlMessage({
      providerID: "openai",
      sessionID,
      messageID: "msg_stale_control",
      createdAt: now,
      contentText: "stale control",
    })

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch, {
        async getSessionMessages() {
          return rawMessages
        },
      })
      const cfg: any = {}
      await hooks.config?.(cfg)
      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          // OpenCode removes completed compaction pairs before this transform.
          messages: [firstTail],
        } as any,
      )

      expect(store.count()).toBe(0)
      expect(store.loadControlMessages()).toEqual([])

      const headers = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        {
          sessionID,
          agent: "compaction",
          model: { providerID: "openai" },
          message: { id: "msg_recovered_plugin_compaction", time: { created: now + 14 }, agent: "build" },
        } as any,
        headers,
      )
      expect(headers.headers[defaultConfig.headers.compact]).toBe("1")

      await cfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: headers.headers,
        body: JSON.stringify({ model: currentModel, instructions: compactionInstructions, input: [] }),
      })

      expect(jsonBody(calls[0]?.init).input).toEqual([
        nativeSummaryItem,
        { role: "user", content: [{ type: "input_text", text: "continue after native compaction" }] },
        { type: "compaction_trigger" },
      ])
      expect(store.loadAll()[0]?.checkpoint.items).toEqual([
        nativeSummaryItem,
        { role: "user", content: [{ type: "input_text", text: "continue after native compaction" }] },
        { type: "compaction", encrypted_content: "recovered-checkpoint-1" },
      ])

      await cfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({ model: currentModel, input: [{ role: "user", content: "after first plugin checkpoint" }] }),
      })
      expect(jsonBody(calls[1]?.init).input).toEqual([
        nativeSummaryItem,
        { role: "user", content: [{ type: "input_text", text: "continue after native compaction" }] },
        { type: "compaction", encrypted_content: "recovered-checkpoint-1" },
        { role: "user", content: "after first plugin checkpoint" },
      ])

      const secondTail = {
        info: {
          id: "msg_after_first_plugin_checkpoint",
          sessionID,
          role: "user",
          model: { providerID: "openai", modelID: currentModel },
          time: { created: now + 16 },
        },
        parts: [{ type: "text", text: "retain summary again" }],
      }
      rawMessages = [
        nativeCompaction,
        nativeSummary,
        firstTail,
        {
          info: {
            id: "msg_recovered_plugin_compaction",
            sessionID,
            role: "user",
            time: { created: now + 14 },
          },
          parts: [{ type: "compaction" }],
        },
        {
          info: {
            id: "msg_recovered_plugin_summary",
            sessionID,
            role: "assistant",
            parentID: "msg_recovered_plugin_compaction",
            summary: true,
            finish: "stop",
            time: { created: now + 15, completed: now + 15 },
          },
          parts: [{ type: "text", text: defaultConfig.summary }],
        },
        secondTail,
      ]
      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: [secondTail] } as any,
      )
      const secondHeaders = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        {
          sessionID,
          agent: "compaction",
          model: { providerID: "openai" },
          message: { id: "msg_second_plugin_compaction", time: { created: now + 17 }, agent: "build" },
        } as any,
        secondHeaders,
      )
      await cfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: secondHeaders.headers,
        body: JSON.stringify({ model: currentModel, instructions: compactionInstructions, input: [] }),
      })

      expect(jsonBody(calls[2]?.init).input).toEqual([
        { role: "user", content: [{ type: "input_text", text: "continue after native compaction" }] },
        { type: "compaction", encrypted_content: "recovered-checkpoint-1" },
        nativeSummaryItem,
        { role: "user", content: [{ type: "input_text", text: "retain summary again" }] },
        { type: "compaction_trigger" },
      ])
      expect(store.loadAll().at(-1)?.checkpoint.items).toEqual([
        { role: "user", content: [{ type: "input_text", text: "continue after native compaction" }] },
        nativeSummaryItem,
        { role: "user", content: [{ type: "input_text", text: "retain summary again" }] },
        { type: "compaction", encrypted_content: "recovered-checkpoint-2" },
      ])
    } finally {
      store.close()
    }
  })

  test("keeps session state when native fallback and its checkpoint-free retry both fail", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ init?: RequestInit }> = []
    const fakeFetch = (async (_requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ init })
      return calls.length === 1
        ? new Response("checkpoint rejected", { status: 400 })
        : new Response("native failed", { status: 500 })
    }) as typeof fetch
    const sessionID = "ses_failed_native_fallback"
    const now = Date.now()
    store.upsert(sessionID, {
      providerID: "openai",
      responseID: "resp_failed_native",
      afterMessageID: "msg_failed_native",
      afterCreatedAt: now,
      createdAt: now,
      items: invalidCheckpointItems(),
    })
    store.upsertControlMessage({
      providerID: "openai",
      sessionID,
      messageID: "msg_failed_control",
      createdAt: now,
      contentText: "failed control",
    })

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: {
                id: "msg_failed_native", sessionID, role: "user",
                model: { providerID: "openai", modelID: currentModel }, time: { created: now },
              },
              parts: [{ type: "compaction" }],
            },
          ],
        } as any,
      )
      const headers = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        {
          sessionID,
          agent: "compaction",
          model: { providerID: "openai" },
          message: { id: "msg_retry_failure", time: { created: now + 1 }, agent: "build" },
        } as any,
        headers,
      )
      const response = await cfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: headers.headers,
        body: JSON.stringify({ model: currentModel, input: [{ role: "user", content: "native request" }] }),
      })

      expect(response.status).toBe(400)
      expect(calls).toHaveLength(2)
      await hooks.event?.({
        event: {
          type: "message.updated",
          properties: {
            sessionID,
            info: {
              id: "msg_failed_native_summary",
              sessionID,
              role: "assistant",
              parentID: "msg_retry_failure",
              summary: true,
              finish: "error",
              error: { name: "APIError" },
              time: { created: now + 2, completed: now + 3 },
            },
          },
        } as any,
      })
      expect(store.count()).toBe(1)
      expect(store.loadControlMessages()).toHaveLength(1)
      await hooks.event?.({ event: { type: "session.compacted", properties: { sessionID } } as any })
      expect(store.count()).toBe(1)
      expect(store.loadControlMessages()).toHaveLength(1)
    } finally {
      store.close()
    }
  })

  test("keeps state when a completed summary arrives before native fallback succeeds", async () => {
    const store = CheckpointStore.openMemory()
    const sessionID = "ses_pending_native_summary"
    const now = Date.now()
    store.upsert(sessionID, {
      providerID: "openai",
      responseID: "resp_pending_native",
      afterMessageID: "msg_invalid_checkpoint",
      afterCreatedAt: now,
      createdAt: now,
      items: invalidCheckpointItems(),
    })

    try {
      const hooks = createCompactHooks(defaultConfig, store)
      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: {
                id: "msg_invalid_checkpoint", sessionID, role: "user",
                model: { providerID: "openai", modelID: currentModel }, time: { created: now },
              },
              parts: [{ type: "compaction" }],
            },
          ],
        } as any,
      )
      const headers = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        {
          sessionID,
          agent: "compaction",
          model: { providerID: "openai" },
          message: { id: "msg_native_attempt", time: { created: now + 1 } },
        } as any,
        headers,
      )
      expect(headers.headers[defaultConfig.headers.compact]).toBe("native")

      await hooks.event?.({
        event: {
          type: "message.updated",
          properties: {
            sessionID,
            info: {
              id: "msg_native_summary",
              sessionID,
              role: "assistant",
              parentID: "msg_native_attempt",
              summary: true,
              finish: "stop",
              time: { created: now + 2, completed: now + 3 },
            },
          },
        } as any,
      })

      expect(store.count()).toBe(1)
    } finally {
      store.close()
    }
  })

  test("uses the previous checkpoint after removing a pending native fallback boundary", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ init?: RequestInit }> = []
    const fakeFetch = (async (_requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ init })
      return new Response("ok")
    }) as typeof fetch
    const sessionID = "ses_native_boundary_removed"
    const now = Date.now()
    const oldItems = [
      { role: "user", content: "older valid history" },
      { type: "compaction", encrypted_content: "older-valid-checkpoint" },
    ]
    store.upsert(sessionID, {
      providerID: "openai",
      responseID: "resp_old_valid",
      afterMessageID: "msg_old_checkpoint",
      afterCreatedAt: now,
      createdAt: now,
      items: oldItems,
    })
    store.upsert(sessionID, {
      providerID: "openai",
      responseID: "resp_new_invalid",
      afterMessageID: "msg_invalid_checkpoint",
      afterCreatedAt: now + 10,
      createdAt: now + 10,
      items: invalidCheckpointItems(),
    })

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: {
                id: "msg_invalid_checkpoint",
                sessionID,
                role: "user",
                model: { providerID: "openai", modelID: currentModel },
                time: { created: now + 10 },
              },
              parts: [{ type: "compaction" }],
            },
          ],
        } as any,
      )
      const headers = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        {
          sessionID,
          agent: "compaction",
          model: { providerID: "openai" },
          message: { id: "msg_native_attempt", time: { created: now + 11 } },
        } as any,
        headers,
      )
      expect(headers.headers[defaultConfig.headers.compact]).toBe("native")

      await hooks.event?.({
        event: {
          type: "message.removed",
          properties: { sessionID, messageID: "msg_invalid_checkpoint" },
        } as any,
      })
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: { id: "msg_old_checkpoint", sessionID, role: "user", time: { created: now } },
              parts: [{ type: "compaction" }],
            },
            {
              info: {
                id: "msg_after_undo", sessionID, role: "user",
                model: { providerID: "openai", modelID: currentModel }, time: { created: now + 12 },
              },
              parts: [{ type: "text", text: "after undo" }],
            },
          ],
        } as any,
      )
      await cfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({ model: currentModel, input: [{ role: "user", content: "after undo" }] }),
      })

      expect(jsonBody(calls[0]?.init).input).toEqual([
        ...oldItems,
        { role: "user", content: "after undo" },
      ])
    } finally {
      store.close()
    }
  })

  test("inherits only pre-fork checkpoints and controls, including across another fork and restart", async () => {
    const store = CheckpointStore.openMemory()
    const parentSessionID = "ses_fork_parent"
    const childSessionID = "ses_fork_child"
    const grandchildSessionID = "ses_fork_grandchild"
    const now = Date.now()
    const parentMessages = [
      ...forkHistory(parentSessionID, "parent", now),
      {
        info: {
          id: "parent_third_target",
          sessionID: parentSessionID,
          role: "user",
          time: { created: now + 10 },
        },
        parts: [{ type: "text", text: "third target" }],
      },
      {
        info: {
          id: "parent_third_answer",
          sessionID: parentSessionID,
          role: "assistant",
          parentID: "parent_third_target",
          time: { created: now + 11 },
        },
        parts: [{ type: "text", text: "third answer" }],
      },
      {
        info: {
          id: "parent_third_checkpoint",
          sessionID: parentSessionID,
          role: "user",
          time: { created: now + 12 },
        },
        parts: [{ type: "compaction" }],
      },
      {
        info: {
          id: "parent_third_summary",
          sessionID: parentSessionID,
          role: "assistant",
          parentID: "parent_third_checkpoint",
          summary: true,
          time: { created: now + 13 },
        },
        parts: [{ type: "text", text: defaultConfig.summary }],
      },
    ]
    const childMessages = [
      ...forkHistory(childSessionID, "child", now),
      {
        info: {
          id: "child_new_request",
          sessionID: childSessionID,
          role: "user",
          model: { providerID: "openai", modelID: currentModel },
          time: { created: now + 20 },
        },
        parts: [{ type: "text", text: "child request" }],
      },
    ]
    const childTransformMessages = childMessages.slice(8)
    const grandchildMessages = [
      ...forkHistory(grandchildSessionID, "grandchild", now),
      {
        info: {
          id: "grandchild_new_request",
          sessionID: grandchildSessionID,
          role: "user",
          model: { providerID: "openai", modelID: currentModel },
          time: { created: now + 30 },
        },
        parts: [{ type: "text", text: "grandchild request" }],
      },
    ]
    const grandchildTransformMessages = grandchildMessages.slice(8)
    store.upsert(parentSessionID, {
      providerID: "openai",
      responseID: "resp_parent_first",
      afterMessageID: "parent_checkpoint",
      afterCreatedAt: now + 2,
      createdAt: now + 10,
      items: [
        { role: "user", content: "first checkpoint history" },
        { type: "compaction", encrypted_content: "first checkpoint" },
      ],
    })
    store.upsert(parentSessionID, {
      providerID: "openai",
      responseID: "resp_parent_second",
      afterMessageID: "parent_second_checkpoint",
      afterCreatedAt: now + 8,
      createdAt: now + 11,
      items: [
        { role: "user", content: "second checkpoint history" },
        { type: "compaction", encrypted_content: "second checkpoint" },
      ],
    })
    store.upsert(parentSessionID, {
      providerID: "openai",
      responseID: "resp_parent_third",
      afterMessageID: "parent_third_checkpoint",
      afterCreatedAt: now + 12,
      createdAt: now + 14,
      items: [
        { role: "user", content: "third checkpoint history" },
        { type: "compaction", encrypted_content: "third checkpoint" },
      ],
    })
    store.upsertControlMessage({
      providerID: "openai",
      sessionID: parentSessionID,
      messageID: "parent_control",
      createdAt: now + 4,
      contentText: "markerless continuation",
    })

    const calls: Array<{ init?: RequestInit }> = []
    const fakeFetch = (async (_requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ init })
      return new Response("ok")
    }) as typeof fetch
    const sourceMessages = new Map<string, unknown>([
      [parentSessionID, parentMessages],
      [childSessionID, childMessages],
      [grandchildSessionID, grandchildMessages],
    ])

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch, {
        async getSessionMessages(sessionID) {
          return sourceMessages.get(sessionID)
        },
      })
      const cfg: any = {}
      await hooks.config?.(cfg)
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: childTransformMessages } as any,
      )

      expect(childTransformMessages.map((message) => message.info.id)).toEqual(["child_new_request"])
      expect(
        store
          .loadAll()
          .filter((entry) => entry.sessionID === childSessionID)
          .map((entry) => ({ responseID: entry.checkpoint.responseID, afterMessageID: entry.checkpoint.afterMessageID })),
      ).toEqual([
        { responseID: "resp_parent_first", afterMessageID: "child_checkpoint" },
        { responseID: "resp_parent_second", afterMessageID: "child_second_checkpoint" },
      ])
      expect(store.loadControlMessages().filter((entry) => entry.sessionID === childSessionID)).toEqual([
        {
          providerID: "openai",
          sessionID: childSessionID,
          messageID: "child_control",
          createdAt: now + 4,
          contentText: "markerless continuation",
        },
      ])

      await cfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: childSessionID },
        body: JSON.stringify({ model: currentModel, input: [{ role: "user", content: "child request" }] }),
      })
      expect(jsonBody(calls[0]?.init).input).toEqual([
        { role: "user", content: "second checkpoint history" },
        { type: "compaction", encrypted_content: "second checkpoint" },
        { role: "user", content: "child request" },
      ])

      await hooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: grandchildTransformMessages } as any,
      )
      expect(
        store
          .loadAll()
          .filter((entry) => entry.sessionID === grandchildSessionID)
          .map((entry) => ({ responseID: entry.checkpoint.responseID, afterMessageID: entry.checkpoint.afterMessageID })),
      ).toEqual([
        { responseID: "resp_parent_first", afterMessageID: "grandchild_checkpoint" },
        { responseID: "resp_parent_second", afterMessageID: "grandchild_second_checkpoint" },
      ])
      expect(store.loadControlMessages().filter((entry) => entry.sessionID === grandchildSessionID)).toEqual([
        {
          providerID: "openai",
          sessionID: grandchildSessionID,
          messageID: "grandchild_control",
          createdAt: now + 4,
          contentText: "markerless continuation",
        },
      ])

      const restartCalls: Array<{ init?: RequestInit }> = []
      const restartedHooks = createCompactHooks(
        defaultConfig,
        store,
        (async (_requestInput: RequestInfo | URL, init?: RequestInit) => {
          restartCalls.push({ init })
          return new Response("ok")
        }) as typeof fetch,
      )
      const restartedCfg: any = {}
      await restartedHooks.config?.(restartedCfg)
      const replayedGrandchildMessages = [
        ...forkHistory(grandchildSessionID, "grandchild", now),
        {
          info: {
            id: "grandchild_after_restart",
            sessionID: grandchildSessionID,
            role: "user",
            model: { providerID: "openai", modelID: currentModel },
            time: { created: now + 40 },
          },
          parts: [{ type: "text", text: "after restart" }],
        },
      ]
      await restartedHooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: replayedGrandchildMessages } as any,
      )
      await restartedCfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: grandchildSessionID },
        body: JSON.stringify({ model: currentModel, input: [{ role: "user", content: "after restart" }] }),
      })
      expect(jsonBody(restartCalls[0]?.init).input).toEqual([
        { role: "user", content: "second checkpoint history" },
        { type: "compaction", encrypted_content: "second checkpoint" },
        { role: "user", content: "after restart" },
      ])
    } finally {
      store.close()
    }
  })

  test("does not inherit a checkpoint at or after the fork point", async () => {
    const store = CheckpointStore.openMemory()
    const parentSessionID = "ses_fork_before_checkpoint_parent"
    const childSessionID = "ses_fork_before_checkpoint_child"
    const now = Date.now()
    const parentMessages = forkHistory(parentSessionID, "before_parent", now)
    let messageReads = 0
    store.upsert(parentSessionID, {
      providerID: "openai",
      responseID: "resp_after_fork",
      afterMessageID: "before_parent_checkpoint",
      afterCreatedAt: now + 2,
      createdAt: now + 10,
      items: [{ type: "compaction", encrypted_content: "after fork" }],
    })

    try {
      const hooks = createCompactHooks(defaultConfig, store, fetch, {
        async getSessionMessages() {
          messageReads++
          return parentMessages
        },
      })
      const childMessages = [
        ...forkHistory(childSessionID, "before_child", now).slice(0, 2),
        {
          info: {
            id: "before_child_new",
            sessionID: childSessionID,
            role: "user",
            model: { providerID: "openai", modelID: currentModel },
            time: { created: now + 20 },
          },
          parts: [{ type: "text", text: "forked before checkpoint" }],
        },
      ]
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: childMessages } as any,
      )

      expect(messageReads).toBe(0)
      expect(store.loadAll().filter((entry) => entry.sessionID === childSessionID)).toEqual([])
    } finally {
      store.close()
    }
  })

  test("does not inherit when matching fork sources disagree on the checkpoint", async () => {
    const store = CheckpointStore.openMemory()
    const firstParentID = "ses_ambiguous_parent_first"
    const secondParentID = "ses_ambiguous_parent_second"
    const childSessionID = "ses_ambiguous_child"
    const now = Date.now()
    const childMessages = [
      ...forkHistory(childSessionID, "ambiguous_child", now).slice(0, 6),
      {
        info: {
          id: "ambiguous_child_new",
          sessionID: childSessionID,
          role: "user",
          model: { providerID: "openai", modelID: currentModel },
          time: { created: now + 20 },
        },
        parts: [{ type: "text", text: "ambiguous fork" }],
      },
    ]
    const sourceMessages = new Map<string, unknown>([
      [firstParentID, forkHistory(firstParentID, "ambiguous_first", now)],
      [secondParentID, forkHistory(secondParentID, "ambiguous_second", now)],
      [childSessionID, childMessages],
    ])
    store.upsert(firstParentID, {
      providerID: "openai",
      responseID: "resp_ambiguous_first",
      afterMessageID: "ambiguous_first_checkpoint",
      afterCreatedAt: now + 2,
      createdAt: now + 10,
      items: [{ type: "compaction", encrypted_content: "first" }],
    })
    store.upsert(secondParentID, {
      providerID: "openai",
      responseID: "resp_ambiguous_second",
      afterMessageID: "ambiguous_second_checkpoint",
      afterCreatedAt: now + 2,
      createdAt: now + 10,
      items: [{ type: "compaction", encrypted_content: "second" }],
    })

    try {
      const hooks = createCompactHooks(defaultConfig, store, fetch, {
        async getSessionMessages(sessionID) {
          return sourceMessages.get(sessionID)
        },
      })
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: childMessages.slice(2) } as any,
      )

      expect(store.loadAll().filter((entry) => entry.sessionID === childSessionID)).toEqual([])
    } finally {
      store.close()
    }
  })

  test("keeps failed capture blocked even when the serialized compaction format is unknown", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return new Response("original summary response", { status: 200 })
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const sessionID = "ses_unknown_compaction"
      await hooks["experimental.session.compacting"]?.(
        { sessionID } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: { id: "msg_uncloneable_unknown", sessionID, role: "user" },
              parts: [{ type: "text", text: "history", metadata: { uncloneable: () => undefined } }],
            },
          ],
        } as any,
      )
      const body = {
        model: currentModel,
        instructions: "Unknown future summarizer instructions.",
        input: [{ role: "user", content: "Unknown future serialized summary request." }],
      }
      const response = await cfg.provider.openai.options.fetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: sessionID,
        },
        body: JSON.stringify(body),
      })

      expect(response.status).toBe(502)
      expect(await response.text()).toContain("could not be captured safely")
      expect(calls).toEqual([])
      expect(store.count()).toBe(0)

      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: { id: "msg_after_failed_capture", sessionID, role: "user" },
              parts: [{ type: "text", text: "must not become a stale snapshot" }],
            },
          ],
        } as any,
      )
      const second = await cfg.provider.openai.options.fetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: sessionID,
        },
        body: JSON.stringify(body),
      })
      expect(second.status).toBe(502)
      expect(await second.text()).toContain("could not be captured safely")
      expect(calls).toEqual([])
      expect(store.count()).toBe(0)
    } finally {
      store.close()
    }
  })

  test("normalizes exactly one compaction while preserving passthrough fields", () => {
    expect(
      compactedItemsFrom([
        { type: "message", role: "developer", content: "stale developer context" },
        { type: "message", role: "system", content: "stale system context" },
        { type: "message", role: "user", content: "retained user" },
        {
          id: "cmp_123",
          type: "compaction_summary",
          encrypted_content: "compacted",
          internal_chat_message_metadata_passthrough: { turn_id: "turn_123" },
          status: "completed",
        },
      ]),
    ).toEqual([
      { type: "message", role: "user", content: "retained user" },
      {
        id: "cmp_123",
        type: "compaction",
        encrypted_content: "compacted",
        internal_chat_message_metadata_passthrough: { turn_id: "turn_123" },
        status: "completed",
      },
    ])
    expect(compactedItemsFrom([{ type: "message", role: "user", content: "no compaction" }])).toBeUndefined()
    expect(
      compactedItemsFrom([
        { type: "compaction", encrypted_content: "first" },
        { type: "compaction_summary", encrypted_content: "second" },
      ]),
    ).toBeUndefined()
  })

  test("keeps session instructions when routing compaction", async () => {
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
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: "ses_instructions" },
        body: JSON.stringify({
          model: "gpt",
          instructions: "You are OpenCode.",
          input: [{ role: "developer", content: "stable instructions" }, { role: "user", content: "hello" }],
        }),
      })

      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: "ses_instructions" },
        body: JSON.stringify({
          model: "gpt",
          input: [{ role: "developer", content: "stable instructions" }, { role: "user", content: "next" }],
        }),
      })

      calls.length = 0
      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: "ses_instructions",
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: "You are an anchored context summarization assistant for coding sessions.\n\nSummarize only...",
          input: [
            {
              role: "developer",
              content: "You are an anchored context summarization assistant for coding sessions.\n\nSummarize only...",
            },
            { role: "user", content: "hello" },
            { role: "assistant", content: [{ type: "output_text", text: "done" }] },
            { role: "user", content: "Create a new anchored summary from the conversation history.\n\nOutput exactly..." },
          ],
        }),
      })

      expect(calls[0]?.url).toBe("https://proxy.test/openai/v1/responses")
      expect(jsonBody(calls[0]?.init)).toEqual({
        model: "ignored",
        instructions: "You are OpenCode.",
        input: [
          { role: "developer", content: "stable instructions" },
          { role: "user", content: "hello" },
          { role: "assistant", content: [{ type: "output_text", text: "done" }] },
          { type: "compaction_trigger" },
        ],
        tool_choice: "auto",
        store: false,
        stream: true,
        include: ["reasoning.encrypted_content"],
      })
    } finally {
      store.close()
    }
  })

  test("keeps rendered system instructions when routing compaction", async () => {
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
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      await hooks["experimental.chat.system.transform"]?.(
        { sessionID: "ses_rendered", model: { providerID: "openai" } } as any,
        { system: ["You are OpenCode.", " ", "AGENTS instructions"] },
      )
      await hooks["chat.headers"]?.(
        { sessionID: "ses_rendered", agent: "build", model: { providerID: "openai" } } as any,
        { headers: {} },
      )

      for (const agent of ["title", "summary"]) {
        const utilityHeaders = { headers: {} as Record<string, string> }
        await hooks["experimental.chat.system.transform"]?.(
          { sessionID: "ses_rendered", model: { providerID: "openai" } } as any,
          { system: [`${agent} prompt`] },
        )
        await hooks["chat.headers"]?.(
          { sessionID: "ses_rendered", agent, model: { providerID: "openai" } } as any,
          utilityHeaders,
        )
        expect(utilityHeaders.headers).toEqual({})
        await wrappedFetch("https://proxy.test/openai/v1/responses", {
          method: "POST",
          headers: utilityHeaders.headers,
          body: JSON.stringify({
            model: "gpt",
            instructions: `${agent} instructions`,
            input: [
              { role: "developer", content: `${agent} developer prompt` },
              { role: "user", content: `${agent} request` },
            ],
          }),
        })
      }
      calls.length = 0

      await hooks["experimental.chat.system.transform"]?.(
        { sessionID: "ses_rendered", model: { providerID: "openai" } } as any,
        { system: ["You are an anchored context summarization assistant for coding sessions.\n\nSummarize only..."] },
      )
      await hooks["experimental.session.compacting"]?.(
        { sessionID: "ses_rendered" } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: {
                id: "msg_rendered_user",
                sessionID: "ses_rendered",
                role: "user",
                model: { providerID: "openai", modelID: "gpt" },
              },
              parts: [{ type: "text", text: "hello" }],
            },
            {
              info: {
                id: "msg_rendered_assistant",
                sessionID: "ses_rendered",
                role: "assistant",
                providerID: "openai",
                modelID: "gpt",
              },
              parts: [{ type: "text", text: "done" }],
            },
          ],
        } as any,
      )
      const compactHeaders = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        {
          sessionID: "ses_rendered",
          agent: "renamed-internal-agent",
          model: { providerID: "openai" },
          message: { id: "msg_compaction", time: { created: 3 }, agent: "build" },
        } as any,
        compactHeaders,
      )

      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: compactHeaders.headers,
        body: JSON.stringify({
          model: "ignored",
          instructions: "You are an anchored context summarization assistant for coding sessions.\n\nSummarize only...",
          input: [
            {
              role: "developer",
              content: "You are an anchored context summarization assistant for coding sessions.\n\nSummarize only...",
            },
            { role: "user", content: "hello" },
            { role: "assistant", content: [{ type: "output_text", text: "done" }] },
            { role: "user", content: "Create a new anchored summary from the conversation history.\n\nOutput exactly..." },
          ],
        }),
      })

      expect(calls[0]?.url).toBe("https://proxy.test/openai/v1/responses")
      expect(jsonBody(calls[0]?.init)).toEqual({
        model: "gpt",
        instructions: "You are OpenCode.\n \nAGENTS instructions",
        input: [
          { role: "user", content: [{ type: "input_text", text: "hello" }] },
          { role: "assistant", content: [{ type: "output_text", text: "done" }] },
          { type: "compaction_trigger" },
        ],
        tool_choice: "auto",
        store: false,
        stream: true,
        include: ["reasoning.encrypted_content"],
      })
    } finally {
      store.close()
    }
  })

  test("does not restore stable instructions when config omits instructions", async () => {
    const config = OpenAICompactConfigSchema.parse({ compactBodyKeys: ["input"] })
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
      const hooks = createCompactHooks(config, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: { [config.headers.session]: "ses_no_instructions" },
        body: JSON.stringify({
          model: "gpt",
          instructions: "You are OpenCode.",
          input: [{ role: "developer", content: "stable instructions" }, { role: "user", content: "hello" }],
        }),
      })

      calls.length = 0
      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: {
          [config.headers.compact]: "1",
          [config.headers.session]: "ses_no_instructions",
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: "You are an anchored context summarization assistant for coding sessions.",
          input: [
            { role: "developer", content: "You are an anchored context summarization assistant for coding sessions." },
            { role: "user", content: "hello" },
            { role: "user", content: "Create a new anchored summary from the conversation history.\n\nOutput exactly..." },
          ],
        }),
      })

      expect(jsonBody(calls[0]?.init)).toEqual({
        model: "ignored",
        input: [
          { role: "developer", content: "stable instructions" },
          { role: "user", content: "hello" },
          { type: "compaction_trigger" },
        ],
        tool_choice: "auto",
        store: false,
        stream: true,
        include: ["reasoning.encrypted_content"],
      })
    } finally {
      store.close()
    }
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
      return compactResponse({
        id: "resp_compacted",
        created_at: 1,
        output: [{ type: "compaction", encrypted_content: "compacted" }],
      })
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
        body: JSON.stringify({ model: "ignored", instructions: compactionInstructions, input: [] }),
      })

      expect(typeof cfg.provider.openai.options.fetch).toBe("function")
      expect(typeof cfg.provider["custom-openai"].options.fetch).toBe("function")
      expect(jsonBody(calls[0]?.init).model).toBe("custom-compact")
      expect(jsonBody(calls[0]?.init).input.at(-1)).toEqual({ type: "compaction_trigger" })
    } finally {
      store.close()
    }
  })

  test("routes compaction fetch and prepends stored checkpoint on the next request", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return compactResponse({
        id: "resp_compacted",
        model: currentModel,
        created_at: 1,
        output: [
          {
            id: "cmp_compacted",
            type: "compaction",
            encrypted_content: "compacted",
            internal_chat_message_metadata_passthrough: { turn_id: "turn_compacted" },
          },
        ],
      })
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
        body: JSON.stringify({
          model: "ignored",
          instructions: compactionInstructions,
          input: [{ role: "user", content: "hello" }],
          stream: true,
        }),
      })

      expect(calls[0]?.url).toBe("https://proxy.test/openai/v1/responses")
      expect(jsonBody(calls[0]?.init)).toEqual({
        model: "ignored",
        input: [{ role: "user", content: "hello" }, { type: "compaction_trigger" }],
        tool_choice: "auto",
        store: false,
        stream: true,
        include: ["reasoning.encrypted_content"],
      })
      expect(new Headers(calls[0]?.init?.headers).has(defaultConfig.headers.compact)).toBe(false)
      expect(new Headers(calls[0]?.init?.headers).has(defaultConfig.headers.session)).toBe(false)

      const messagesBeforeBoundaryEvent = [
        {
          info: {
            id: "msg_original", sessionID: "ses_request", role: "user",
            model: { providerID: "openai", modelID: "gpt" }, time: { created: 1 },
          },
          parts: [],
        },
      ]
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: messagesBeforeBoundaryEvent } as any,
      )
      expect(messagesBeforeBoundaryEvent).toHaveLength(1)

      calls.length = 0
      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: "ses_request" },
        body: JSON.stringify({
          model: "gpt",
          input: [
            { role: "developer", content: "stable instructions" },
            { role: "system", content: "more stable instructions" },
            { role: "user", content: [{ type: "input_text", text: "retained user" }] },
            { role: "user", content: [{ type: "input_text", text: "What did we do so far?" }] },
            { role: "assistant", content: [{ type: "output_text", text: defaultConfig.summary }] },
            { role: "user", content: "after compact" },
            {
              role: "user",
              content: [
                {
                  type: "input_text",
                  text: "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed.",
                },
              ],
            },
          ],
        }),
      })

      const followupBody = jsonBody(calls[0]?.init)
      expect(calls[0]?.url).toBe("https://proxy.test/openai/v1/responses")
      expect(followupBody.input).toEqual([
        { role: "user", content: "hello" },
        {
          id: "cmp_compacted",
          type: "compaction",
          encrypted_content: "compacted",
          internal_chat_message_metadata_passthrough: { turn_id: "turn_compacted" },
        },
        { role: "developer", content: "stable instructions" },
        { role: "system", content: "more stable instructions" },
        { role: "user", content: "after compact" },
      ])

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
        {
          info: { id: "msg_continue", sessionID: "ses_request" },
          parts: [{ type: "text", synthetic: true, metadata: { compaction_continue: true } }],
        },
        { info: { id: "msg_after", sessionID: "ses_request", role: "user" } },
      ]
      await hooks["experimental.chat.messages.transform"]?.({}, { messages: inferredProviderMessages } as any)
      expect(inferredProviderMessages.map((message) => message.info.id)).toEqual(["msg_after"])

      calls.length = 0
      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: "ses_request",
        },
        body: JSON.stringify({
          model: "ignored",
          input: [
            {
              role: "developer",
              content: "You are an anchored context summarization assistant for coding sessions.\n\nSummarize only...",
            },
            { role: "user", content: "after compact" },
            { role: "user", content: "Create a new anchored summary from the conversation history.\n\nOutput exactly..." },
          ],
        }),
      })

      expect(jsonBody(calls[0]?.init).input).toEqual([
        { role: "developer", content: "stable instructions" },
        { role: "system", content: "more stable instructions" },
        { role: "user", content: "hello" },
        {
          id: "cmp_compacted",
          type: "compaction",
          encrypted_content: "compacted",
          internal_chat_message_metadata_passthrough: { turn_id: "turn_compacted" },
        },
        { role: "user", content: "after compact" },
        { type: "compaction_trigger" },
      ])
    } finally {
      store.close()
    }
  })

  test("keeps real user checkpoint content when it matches a persisted control message", () => {
    const store = CheckpointStore.openMemory()
    const sessionID = "ses_control_text_collision"
    const collisionText = "same text from a real user"
    const now = Date.now()

    store.upsertControlMessage({
      providerID: "openai",
      sessionID,
      messageID: "msg_old_control",
      createdAt: now,
      contentText: collisionText,
    })
    store.upsert(sessionID, {
      providerID: "openai",
      responseID: "resp_after_real_user",
      afterMessageID: "msg_checkpoint",
      afterCreatedAt: now + 1,
      createdAt: now + 1,
      items: [
        { role: "user", content: collisionText },
        { type: "compaction", encrypted_content: "checkpoint" },
      ],
    })

    try {
      createCompactHooks(defaultConfig, store)

      expect(store.loadAll()[0]?.checkpoint.items).toEqual([
        { role: "user", content: collisionText },
        { type: "compaction", encrypted_content: "checkpoint" },
      ])
    } finally {
      store.close()
    }
  })

  test("tracks auto-continue by message id and removes it across later turns and plugin restarts", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ init?: RequestInit }> = []
    const fakeFetch = (async (_requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ init })
      return new Response("ok")
    }) as typeof fetch
    const sessionID = "ses_control_message"
    const controlText = "A future OpenCode version may use completely different continuation text."
    const startedAt = Date.now()

    store.upsert(sessionID, {
      providerID: "openai",
      responseID: "resp_control",
      afterMessageID: "msg_compaction_boundary",
      afterCreatedAt: startedAt,
      createdAt: startedAt,
      items: [
        { role: "user", content: "before compaction" },
        { type: "compaction", encrypted_content: "checkpoint" },
      ],
    })

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)

      await hooks["experimental.compaction.autocontinue"]?.(
        {
          sessionID,
          model: { providerID: "openai" },
          message: { id: "msg_compaction_boundary", time: { created: startedAt } },
        } as any,
        { enabled: true },
      )

      const firstContinuation = [
        {
          info: { id: "msg_compaction_boundary", sessionID, role: "user", time: { created: startedAt } },
          parts: [{ type: "compaction" }],
        },
        {
          info: {
            id: "msg_summary",
            sessionID,
            role: "assistant",
            parentID: "msg_compaction_boundary",
            summary: true,
            time: { created: startedAt + 1 },
          },
          parts: [{ type: "text", text: "A changed summary placeholder." }],
        },
        {
          info: {
            id: "msg_internal_continue", sessionID, role: "user", agent: "plan",
            model: { providerID: "openai", modelID: "gpt" }, time: { created: startedAt + 2 },
          },
          parts: [{ type: "text", text: controlText, synthetic: true }],
        },
      ]
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: firstContinuation } as any,
      )
      expect(firstContinuation).toEqual([])
      expect(store.loadControlMessages().map((entry) => entry.messageID)).toEqual(["msg_internal_continue"])
      expect(store.loadAll()[0]?.checkpoint.items).toEqual([
        { role: "user", content: "before compaction" },
        { type: "compaction", encrypted_content: "checkpoint" },
      ])

      const laterMessages = [
        {
          info: { id: "msg_compaction_boundary", sessionID, role: "user", time: { created: startedAt } },
          parts: [{ type: "compaction" }],
        },
        {
          info: {
            id: "msg_summary",
            sessionID,
            role: "assistant",
            parentID: "msg_compaction_boundary",
            summary: true,
            time: { created: startedAt + 1 },
          },
          parts: [{ type: "text", text: "A changed summary placeholder." }],
        },
        {
          info: { id: "msg_internal_continue", sessionID, role: "user", time: { created: startedAt + 2 } },
          parts: [{ type: "text", text: controlText }],
        },
        {
          info: {
            id: "msg_continued_assistant",
            sessionID,
            role: "assistant",
            providerID: "openai",
            modelID: "gpt",
            time: { created: startedAt + 3 },
          },
          parts: [{ type: "text", text: "continued work" }],
        },
        {
          info: {
            id: "msg_real_user",
            sessionID,
            role: "user",
            model: { providerID: "openai", modelID: "gpt" },
            time: { created: startedAt + 4 },
          },
          parts: [{ type: "text", text: controlText }],
        },
      ]
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: laterMessages } as any,
      )
      expect(laterMessages.map((message) => message.info.id)).toEqual(["msg_continued_assistant", "msg_real_user"])

      await cfg.provider.openai.options.fetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: sessionID },
        body: JSON.stringify({
          model: "gpt",
          input: [
            { role: "assistant", content: [{ type: "output_text", text: "continued work" }] },
            { role: "user", content: controlText },
          ],
        }),
      })
      expect(jsonBody(calls[0]?.init).input).toEqual([
        { role: "user", content: "before compaction" },
        { type: "compaction", encrypted_content: "checkpoint" },
        { role: "assistant", content: [{ type: "output_text", text: "continued work" }] },
        { role: "user", content: controlText },
      ])

      const restartedHooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const afterRestart = [
        {
          info: { id: "msg_compaction_boundary", sessionID, role: "user", time: { created: startedAt } },
          parts: [{ type: "compaction" }],
        },
        {
          info: {
            id: "msg_summary",
            sessionID,
            role: "assistant",
            parentID: "msg_compaction_boundary",
            summary: true,
            time: { created: startedAt + 1 },
          },
          parts: [{ type: "text", text: "A changed summary placeholder." }],
        },
        {
          info: { id: "msg_internal_continue", sessionID, role: "user", time: { created: startedAt + 2 } },
          parts: [{ type: "text", text: "metadata and text can both change later" }],
        },
        {
          info: {
            id: "msg_after_restart", sessionID, role: "user", agent: "plan",
            model: { providerID: "openai", modelID: "gpt" }, time: { created: startedAt + 5 },
          },
          parts: [{ type: "text", text: "real request" }],
        },
      ]
      await restartedHooks["experimental.chat.messages.transform"]?.(
        {},
        { messages: afterRestart } as any,
      )
      expect(afterRestart.map((message) => message.info.id)).toEqual(["msg_after_restart"])
    } finally {
      store.close()
    }
  })

  test.each(["marked", "markerless"])(
    "filters %s continuation from consecutive tool requests and after restart with empty transform input",
    async (kind) => {
      const store = CheckpointStore.openMemory()
      const bodies: any[] = []
      const fakeFetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
        bodies.push(jsonBody(init))
        return new Response("ok")
      }) as typeof fetch
      const sessionID = "ses_tool_continuation"
      const now = Date.now()
      const model = { providerID: "openai", modelID: "gpt" }
      const controlText = "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
      const reminder = "<system-reminder>Plan Mode: read-only investigation.</system-reminder>"
      const checkpointItems = [
        { role: "user", content: "original question" },
        { type: "compaction", encrypted_content: "checkpoint" },
      ]
      const boundary = {
        info: { id: "msg_compaction", sessionID, role: "user", agent: "plan", model, time: { created: now } },
        parts: [{ type: "compaction" }],
      }
      const summary = {
        info: {
          id: "msg_summary", sessionID, role: "assistant", parentID: boundary.info.id,
          providerID: "openai", modelID: "gpt", summary: true, time: { created: now + 1 },
        },
        parts: [{ type: "text", text: defaultConfig.summary }],
      }
      const continuation = {
        info: { id: "msg_continue", sessionID, role: "user", agent: "plan", model, time: { created: now + 2 } },
        parts: [
          {
            type: "text", text: controlText,
            ...(kind === "marked" ? { synthetic: true, metadata: { compaction_continue: true } } : {}),
          },
          { type: "text", text: reminder },
        ],
      }
      const continuationItem = {
        role: "user",
        content: continuation.parts.map((part) => ({ type: "input_text", text: part.text })),
      }
      const developer = { role: "developer", content: "stable instructions" }
      // Fixed SDK wire items let this test exercise message selection without reimplementing its serializer.
      const wireItems = new Map<string, any[]>([
        [boundary.info.id, [{ role: "user", content: "What did we do so far?" }]],
        [summary.info.id, [{ role: "assistant", content: defaultConfig.summary }]],
        [continuation.info.id, [continuationItem]],
      ])
      store.upsert(sessionID, {
        providerID: "openai", responseID: "resp_tool_continuation",
        afterMessageID: boundary.info.id, afterCreatedAt: now, createdAt: now, items: checkpointItems,
      })

      try {
        let hooks = createCompactHooks(defaultConfig, store, fakeFetch)
        let cfg: any = {}
        await hooks.config?.(cfg)
        const original = {
          info: { id: "msg_original", sessionID, role: "user", agent: "plan", model, time: { created: now - 1 } },
          parts: [{ type: "text", text: "original question" }],
        }
        await hooks["chat.message"]?.(
          { sessionID, model, messageID: original.info.id } as any,
          { message: original.info, parts: original.parts } as any,
        )
        // Select the checkpoint before OpenCode drops the original, cached user from transform history.
        await hooks["experimental.chat.messages.transform"]?.(
          {}, { messages: structuredClone([original, boundary]) } as any,
        )
        await hooks["experimental.compaction.autocontinue"]?.(
          { sessionID, agent: "plan", model, message: boundary.info } as any, { enabled: true },
        )

        const toolHistory: any[] = []
        const toolItems: any[] = []
        for (let turn = 0; turn < 4; turn++) {
          if (turn === 3) {
            hooks = createCompactHooks(defaultConfig, store, fakeFetch)
            cfg = {}
            await hooks.config?.(cfg)
          }
          const messages = structuredClone([boundary, summary, continuation, ...toolHistory])
          await hooks["experimental.chat.messages.transform"]?.({}, { messages } as any)
          const headers = { headers: {} as Record<string, string> }
          await hooks["chat.headers"]?.(
            { sessionID, agent: "plan", model, message: continuation.info } as any, headers,
          )
          await cfg.provider.openai.options.fetch("https://proxy.test/openai/v1/responses", {
            method: "POST", headers: headers.headers,
            body: JSON.stringify({
              model: "gpt",
              input: [developer, ...messages.flatMap((message) => wireItems.get(message.info.id)!)],
            }),
          })
          expect(bodies.at(-1).input).toEqual([...checkpointItems, developer, ...toolItems])
          expect(store.loadControlMessages().map((message) => message.messageID)).toEqual([continuation.info.id])
          if (turn > 0) expect(messages).toEqual(toolHistory)

          const messageID = `msg_tool_${turn}`
          const callID = `call_${turn}`
          const input = { filePath: `source-${turn}.ts` }
          const output = `complete tool output ${turn}\n${"source line\n".repeat(100)}`
          const items = [
            { type: "reasoning", encrypted_content: `reasoning-${turn}`, summary: [] },
            { type: "function_call", name: "read", call_id: callID, arguments: JSON.stringify(input) },
            { type: "function_call_output", call_id: callID, output },
          ]
          wireItems.set(messageID, items)
          toolItems.push(...items)
          toolHistory.push({
            info: {
              id: messageID, sessionID, role: "assistant", parentID: continuation.info.id,
              providerID: "openai", modelID: "gpt", time: { created: now + 3 + turn },
            },
            parts: [
              { type: "reasoning", text: `investigate ${turn}` },
              { type: "tool", tool: "read", callID, state: { status: "completed", input, output } },
            ],
          })
        }

        const realUser = {
          info: { id: "msg_real", sessionID, role: "user", agent: "plan", model, time: { created: now + 10 } },
          parts: [{ type: "text", text: controlText }, { type: "text", text: reminder, synthetic: true }],
        }
        await hooks["chat.message"]?.(
          { sessionID, model, messageID: realUser.info.id } as any,
          { message: realUser.info, parts: realUser.parts } as any,
        )
        const laterMessages = structuredClone([boundary, summary, continuation, ...toolHistory, realUser])
        await hooks["experimental.chat.messages.transform"]?.({}, { messages: laterMessages } as any)
        expect(laterMessages).toEqual([...toolHistory, realUser])
      } finally {
        store.close()
      }
    },
  )

  test.each([
    ["custom-openai", true],
    ["custom-openai", false],
    ["anthropic", true],
    ["anthropic", false],
  ])("uses the latest user's provider %s (message model: %s), not older cached history", async (providerID, hasModel) => {
    const config = OpenAICompactConfigSchema.parse({ providers: { openai: {}, "custom-openai": {} } })
    const store = CheckpointStore.openMemory()
    const sessionID = "ses_provider_switch"
    const now = Date.now()
    const bodies: any[] = []
    const fakeFetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
      bodies.push(jsonBody(init))
      return new Response("ok")
    }) as typeof fetch
    for (const id of ["openai", "custom-openai"]) {
      store.upsert(sessionID, {
        providerID: id, responseID: `resp_${id}`, afterMessageID: `msg_${id}`,
        afterCreatedAt: now, createdAt: now,
        items: [{ type: "compaction", encrypted_content: id }],
      })
    }

    try {
      const hooks = createCompactHooks(config, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const oldUser = {
        info: {
          id: "msg_original", sessionID, role: "user", model: { providerID: "openai", modelID: "gpt" },
          time: { created: now - 1 },
        },
        parts: [{ type: "text", text: "old request" }],
      }
      const currentUser = {
        info: {
          id: "msg_current", sessionID, role: "user", time: { created: now + 2 },
          ...(hasModel ? { model: { providerID, modelID: "current" } } : {}),
        },
        parts: [{ type: "text", text: "current request" }],
      }
      for (const message of hasModel ? [oldUser] : [oldUser, currentUser]) {
        await hooks["chat.message"]?.(
          {
            sessionID, messageID: message.info.id,
            model: { providerID: message === oldUser ? "openai" : providerID, modelID: "current" },
          } as any,
          { message: message.info, parts: message.parts } as any,
        )
      }
      const assistant = {
        info: { id: "msg_assistant", sessionID, role: "assistant", providerID: "openai", modelID: "gpt", time: { created: now + 3 } },
        parts: [{ type: "text", text: "older provider's assistant metadata" }],
      }
      const history = [
        oldUser,
        ...["openai", "custom-openai"].map((id) => ({
          info: {
            id: `msg_${id}`, sessionID, role: "user", model: { providerID: id, modelID: "gpt" },
            time: { created: now },
          },
          parts: [{ type: "compaction" }],
        })),
        currentUser,
        assistant,
      ]
      const messages = structuredClone(history)
      await hooks["experimental.chat.messages.transform"]?.({}, { messages } as any)
      if (providerID === "anthropic") {
        expect(messages).toEqual(history)
      } else {
        expect(messages).toEqual([currentUser, assistant])
        await cfg.provider[providerID].options.fetch("https://proxy.test/v1/responses", {
          method: "POST", headers: { [config.headers.session]: sessionID },
          body: JSON.stringify({ model: "current", input: [{ role: "user", content: "current request" }] }),
        })
        expect(bodies[0].input).toEqual([
          { type: "compaction", encrypted_content: providerID },
          { role: "user", content: "current request" },
        ])
      }
    } finally {
      store.close()
    }
  })

  test("does not use an older cached user when the latest user's provider is unknown at a checkpoint boundary", async () => {
    const store = CheckpointStore.openMemory()
    const sessionID = "ses_unknown_latest_provider"
    const now = Date.now()
    store.upsert(sessionID, {
      providerID: "openai", responseID: "resp_unknown_latest", afterMessageID: "msg_checkpoint",
      afterCreatedAt: now, createdAt: now, items: [{ type: "compaction", encrypted_content: "checkpoint" }],
    })
    try {
      const hooks = createCompactHooks(defaultConfig, store)
      await hooks["chat.message"]?.(
        { sessionID, messageID: "msg_old", model: { providerID: "openai" } } as any,
        { message: { id: "msg_old" }, parts: [] } as any,
      )
      const history = [
        { info: { id: "msg_old", sessionID, role: "user", model: { providerID: "openai", modelID: "gpt" } }, parts: [] },
        { info: { id: "msg_checkpoint", sessionID, role: "user" }, parts: [{ type: "compaction" }] },
        { info: { id: "msg_unknown", sessionID, role: "user" }, parts: [{ type: "text", text: "unknown provider" }] },
      ]
      const messages = structuredClone(history)
      await hooks["experimental.chat.messages.transform"]?.({}, { messages } as any)
      expect(messages).toEqual(history)
    } finally {
      store.close()
    }
  })

  test("does not classify the newest real user when pending first observes an older marked continuation", async () => {
    const store = CheckpointStore.openMemory()
    const sessionID = "ses_pending_with_real_user"
    const now = Date.now()
    try {
      const hooks = createCompactHooks(defaultConfig, store)
      await hooks["experimental.compaction.autocontinue"]?.(
        {
          sessionID,
          agent: "plan",
          model: { providerID: "openai" },
          message: { id: "msg_compaction", time: { created: now } },
        } as any,
        { enabled: true },
      )

      const messages = [
        {
          info: { id: "msg_internal_continue", sessionID, role: "user", agent: "plan", time: { created: now + 1 } },
          parts: [
            {
              type: "text",
              text: "changed internal continuation",
              synthetic: true,
              metadata: { compaction_continue: true },
            },
          ],
        },
        {
          info: {
            id: "msg_real_user", sessionID, role: "user", agent: "plan",
            model: { providerID: "openai", modelID: "gpt" }, time: { created: now + 2 },
          },
          parts: [{ type: "text", text: "real user request" }],
        },
      ]
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        { messages } as any,
      )

      expect(messages.map((message) => message.info.id)).toEqual(["msg_real_user"])
      expect(store.loadControlMessages().map((message) => message.messageID)).toEqual(["msg_internal_continue"])
    } finally {
      store.close()
    }
  })

  test("uses chat.message as proof that a pending message is real user input", async () => {
    const store = CheckpointStore.openMemory()
    const sessionID = "ses_real_user_proof"
    const now = Date.now()
    try {
      const hooks = createCompactHooks(defaultConfig, store)
      await hooks["experimental.compaction.autocontinue"]?.(
        {
          sessionID,
          agent: "plan",
          model: { providerID: "openai" },
          message: { id: "msg_compaction", time: { created: now } },
        } as any,
        { enabled: true },
      )
      await hooks["chat.message"]?.(
        { sessionID, agent: "plan", model: { providerID: "openai", modelID: "gpt" }, messageID: "msg_real" } as any,
        {
          message: { id: "msg_real", role: "user", agent: "plan" },
          parts: [{ type: "text", text: "real request" }],
        } as any,
      )
      const messages = [
        {
          info: { id: "msg_real", sessionID, role: "user", agent: "plan", time: { created: now + 1 } },
          parts: [{ type: "text", text: "real request" }],
        },
      ]
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        { messages } as any,
      )

      expect(messages.map((message) => message.info.id)).toEqual(["msg_real"])
      expect(store.loadControlMessages()).toEqual([])
    } finally {
      store.close()
    }
  })

  test("captures markerless auto-continue from chat headers and removes its request user item", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ init?: RequestInit }> = []
    const fakeFetch = (async (_requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ init })
      return new Response("ok")
    }) as typeof fetch
    const sessionID = "ses_header_auto_continue"
    const now = Date.now() - 60_000
    const continuationCreatedAt = Date.now()
    store.upsert(sessionID, {
      providerID: "openai",
      responseID: "resp_header_auto_continue",
      afterMessageID: "msg_compaction",
      afterCreatedAt: now,
      createdAt: now,
      items: [
        { role: "user", content: "checkpoint history" },
        { type: "compaction", encrypted_content: "checkpoint" },
      ],
    })

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      await hooks["experimental.compaction.autocontinue"]?.(
        {
          sessionID,
          agent: "plan",
          model: { providerID: "openai" },
          message: { id: "msg_compaction", time: { created: now } },
        } as any,
        { enabled: true },
      )
      const messages = [
        {
          info: {
            id: "msg_compaction", sessionID, role: "user", agent: "plan",
            model: { providerID: "openai", modelID: "gpt" }, time: { created: now },
          },
          parts: [{ type: "compaction" }],
        },
        {
          info: {
            id: "msg_summary",
            sessionID,
            role: "assistant",
            parentID: "msg_compaction",
            summary: true,
            time: { created: now + 1 },
          },
          parts: [{ type: "text", text: "summary" }],
        },
      ]
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        { messages } as any,
      )
      const headers = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.(
        {
          sessionID,
          agent: "plan",
          model: { providerID: "openai" },
          message: { id: "msg_markerless_continue", time: { created: continuationCreatedAt }, agent: "plan" },
        } as any,
        headers,
      )
      await cfg.provider.openai.options.fetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: headers.headers,
        body: JSON.stringify({
          model: "gpt",
          input: [
            { role: "user", content: "retained tail" },
            { role: "user", content: "markerless internal continuation" },
          ],
        }),
      })

      expect(jsonBody(calls[0]?.init).input).toEqual([
        { role: "user", content: "checkpoint history" },
        { type: "compaction", encrypted_content: "checkpoint" },
        { role: "user", content: "retained tail" },
      ])
      expect(store.loadControlMessages()).toEqual([
        {
          providerID: "openai",
          sessionID,
          messageID: "msg_markerless_continue",
          createdAt: continuationCreatedAt,
          contentText: "",
        },
      ])
    } finally {
      store.close()
    }
  })

  test("keeps checkpoints through undo and redo until undo removes the boundary", async () => {
    const store = CheckpointStore.openMemory()
    const calls: Array<{ url: string; init?: RequestInit }> = []
    const fakeFetch = (async (requestInput: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(requestInput), init })
      return compactResponse({
        id: "resp_undo",
        model: currentModel,
        created_at: 1,
        output: [{ type: "compaction", encrypted_content: "undo-compacted" }],
      })
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
          [defaultConfig.headers.session]: "ses_undo",
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: compactionInstructions,
          input: [{ role: "user", content: "before compact" }],
        }),
      })
      await hooks.event?.({
        event: {
          type: "message.part.updated",
          properties: {
            sessionID: "ses_undo",
            part: { messageID: "msg_checkpoint", type: "text", text: defaultConfig.summary },
            time: 2,
          },
        } as any,
      })
      expect(store.count()).toBe(1)

      await hooks.event?.({
        event: {
          type: "session.updated",
          properties: { sessionID: "ses_undo", info: { id: "ses_undo", revert: { messageID: "msg_before" } } },
        } as any,
      })
      expect(store.count()).toBe(1)

      await hooks.event?.({
        event: { type: "session.updated", properties: { sessionID: "ses_undo", info: { id: "ses_undo" } } } as any,
      })

      calls.length = 0
      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: "ses_undo" },
        body: JSON.stringify({ model: "gpt", input: [{ role: "user", content: "after redo" }] }),
      })
      expect(jsonBody(calls[0]?.init).input).toEqual([
        { role: "user", content: "before compact" },
        { type: "compaction", encrypted_content: "undo-compacted" },
        { role: "user", content: "after redo" },
      ])

      await hooks.event?.({
        event: {
          type: "message.removed",
          properties: { sessionID: "ses_undo", messageID: "msg_after_checkpoint" },
        } as any,
      })
      expect(store.count()).toBe(1)

      await hooks.event?.({
        event: { type: "message.removed", properties: { sessionID: "ses_undo", messageID: "msg_checkpoint" } } as any,
      })
      expect(store.count()).toBe(0)

      calls.length = 0
      const afterCommittedUndo = [{ role: "user", content: "new branch" }]
      await wrappedFetch("https://proxy.test/openai/v1/responses", {
        method: "POST",
        headers: { [defaultConfig.headers.session]: "ses_undo" },
        body: JSON.stringify({ model: "gpt", input: afterCommittedUndo }),
      })
      expect(jsonBody(calls[0]?.init).input).toEqual(afterCommittedUndo)
    } finally {
      store.close()
    }
  })

  test.each([
    ["no compaction", []],
    ["a legacy compaction summary", [{ type: "compaction_summary", encrypted_content: "legacy" }]],
    [
      "multiple compactions",
      [
        { type: "compaction", encrypted_content: "first" },
        { type: "compaction", encrypted_content: "second" },
      ],
    ],
  ])("rejects compact output with %s", async (_name, output) => {
    const store = CheckpointStore.openMemory()
    const fakeFetch = (async () => compactResponse({ id: "resp_invalid", output })) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const wrappedFetch = cfg.provider.openai.options.fetch as typeof fetch

      const response = await wrappedFetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: "ses_invalid",
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: compactionInstructions,
          input: [{ role: "user", content: "hello" }],
        }),
      })

      expect(response.status).toBe(502)
      expect(await response.text()).toContain("exactly one valid compaction item")
      expect(store.count()).toBe(0)
    } finally {
      store.close()
    }
  })

  test.each([
    ["a missing completed event", `data: ${JSON.stringify({ type: "response.output_item.done", item: { type: "compaction", encrypted_content: "encrypted" } })}\n\n`],
    [
      "a completed event without an id",
      `${[
        { type: "response.output_item.done", item: { type: "compaction", encrypted_content: "encrypted" } },
        { type: "response.completed", response: {} },
      ]
        .map((event) => `data: ${JSON.stringify(event)}\n\n`)
        .join("")}data: [DONE]\n\n`,
    ],
    [
      "a failed event",
      `data: ${JSON.stringify({ type: "response.failed", response: { id: "resp_failed" } })}\n\ndata: [DONE]\n\n`,
    ],
  ])("rejects compaction v2 stream with %s", async (_name, stream) => {
    const store = CheckpointStore.openMemory()
    const fakeFetch = (async () =>
      new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } })) as typeof fetch

    try {
      const hooks = createCompactHooks(defaultConfig, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      const response = await cfg.provider.openai.options.fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          [defaultConfig.headers.compact]: "1",
          [defaultConfig.headers.session]: "ses_invalid_stream",
        },
        body: JSON.stringify({
          model: "ignored",
          instructions: compactionInstructions,
          input: [{ role: "user", content: "hello" }],
        }),
      })

      expect(response.status).toBe(502)
      expect(store.count()).toBe(0)
    } finally {
      store.close()
    }
  })

  test("ignores deprecated compactEndpointPath", async () => {
    const config = OpenAICompactConfigSchema.parse({
      responses: { endpointPath: "/responses", compactEndpointPath: "/removed/compact" },
    })
    const store = CheckpointStore.openMemory()
    const calls: string[] = []
    const fakeFetch = (async (input: RequestInfo | URL) => {
      calls.push(String(input))
      return compactResponse({
        id: "resp_deprecated_path",
        output: [{ type: "compaction", encrypted_content: "encrypted" }],
      })
    }) as typeof fetch

    try {
      const hooks = createCompactHooks(config, store, fakeFetch)
      const cfg: any = {}
      await hooks.config?.(cfg)
      await cfg.provider.openai.options.fetch("https://proxy.test/v1/responses", {
        method: "POST",
        headers: { [config.headers.compact]: "1", [config.headers.session]: "ses_deprecated_path" },
        body: JSON.stringify({
          model: "ignored",
          instructions: compactionInstructions,
          input: [{ role: "user", content: "hello" }],
        }),
      })

      expect(calls).toEqual(["https://proxy.test/v1/responses"])
      expect(store.count()).toBe(1)
    } finally {
      store.close()
    }
  })

  test("adds compaction headers from the captured transaction without relying on the agent name", async () => {
    const store = CheckpointStore.openMemory()
    try {
      const hooks = createCompactHooks(defaultConfig, store)
      const output = { headers: {} as Record<string, string> }

      await hooks["experimental.session.compacting"]?.(
        { sessionID: "ses" } as any,
        { context: [], prompt: undefined },
      )
      await hooks["experimental.chat.messages.transform"]?.(
        {},
        {
          messages: [
            {
              info: {
                id: "msg_user",
                sessionID: "ses",
                role: "user",
                model: { providerID: "openai", modelID: "gpt" },
              },
              parts: [{ type: "text", text: "history" }],
            },
          ],
        } as any,
      )

      await hooks["chat.headers"]?.(
        {
          model: { providerID: "openai" },
          sessionID: "ses",
          agent: "renamed-internal-agent",
          message: { id: "msg_compaction", time: { created: 2 }, agent: "build" },
        } as any,
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
