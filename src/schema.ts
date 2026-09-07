import { z } from "zod"

export const defaultCompactBodyKeys = [
  "input",
  "instructions",
  "tools",
  "parallel_tool_calls",
  "reasoning",
  "service_tier",
  "prompt_cache_key",
  "text",
] as const

export const defaultCompactSummary = [
  "Context compacted.",
  "Following conversations will continue from this compacted checkpoint.",
].join("\n")

export const compactReasoningEfforts = ["none", "minimal", "low", "medium", "high", "xhigh", "max"] as const
export type CompactReasoningEffort = (typeof compactReasoningEfforts)[number]

const defaultHeaders = {
  compact: "x-opencode-openai-responses-compact",
  session: "x-opencode-openai-responses-compact-session",
}

const defaultResponses = {
  endpointPath: "/responses",
  compactEndpointPath: "/responses/compact",
}

const defaultState = {
  retentionDays: 30,
  deleteOnSessionDeleted: true,
}

const defaultProviders = {
  openai: {
    enabled: true,
    compactModel: null,
    compactReasoningEffort: null,
  },
}

const defaultConfigValues = {
  enabled: true,
  providers: defaultProviders,
  headers: defaultHeaders,
  responses: defaultResponses,
  compactBodyKeys: [...defaultCompactBodyKeys],
  summary: defaultCompactSummary,
  state: defaultState,
}

function endpoint(value: string) {
  const trimmed = value.trim()
  const prefixed = trimmed.startsWith("/") ? trimmed : `/${trimmed}`
  return prefixed.length > 1 ? prefixed.replace(/\/+$/, "") : prefixed
}

const endpointPath = z.string()
  .transform(endpoint)
  .refine((value) => value.length > 1 && value !== "/", "Endpoint path must name a non-root path")

const reservedHeaderNames = new Set([
  "authorization",
  "proxy-authorization",
  "content-type",
  "content-length",
  "host",
  "connection",
  "transfer-encoding",
  "upgrade",
  "cookie",
  "set-cookie",
  "chatgpt-account-id",
  "x-opencode-openai-compact-operation",
])
const headerName = z.string()
  .transform((value) => value.trim().toLowerCase())
  .refine((value) => /^[!#$%&'*+.^_`|~0-9a-z-]+$/.test(value), "Invalid HTTP header name")
  .refine((value) => !reservedHeaderNames.has(value), "Header name is reserved for protocol or authentication use")

const headersSchema = z
  .object({
    compact: headerName.default(defaultHeaders.compact),
    session: headerName.default(defaultHeaders.session),
  })
  .strict()
  .superRefine((headers, context) => {
    if (headers.compact === headers.session) {
      context.addIssue({ code: "custom", path: ["session"], message: "Header names must be different" })
    }
  })

export const OpenAICompactConfigSchema = z
  .object({
    $schema: z.string().optional(),
    enabled: z.boolean().default(true),
    providers: z
      .record(
        z.string().min(1),
        z
          .object({
            enabled: z.boolean().default(true),
            compactModel: z.string().min(1).nullable().default(null),
            compactReasoningEffort: z.enum(compactReasoningEfforts).nullable().default(null),
          })
          .strict(),
      )
      .refine((providers) => Object.keys(providers).length > 0, "At least one provider is required")
      .default(defaultProviders),
    headers: headersSchema.default(defaultHeaders),
    responses: z
      .object({
        endpointPath: endpointPath.default(defaultResponses.endpointPath),
        // Kept so existing config files continue to load; compaction v2 uses endpointPath.
        compactEndpointPath: endpointPath.default(defaultResponses.compactEndpointPath),
      })
      .strict()
      .default(defaultResponses),
    compactBodyKeys: z.array(z.string().min(1)).default([...defaultCompactBodyKeys]),
    summary: z.string().min(1).default(defaultCompactSummary),
    state: z
      .object({
        retentionDays: z.number().int().positive()
          .describe("Global database retention policy; non-global config overrides are ignored")
          .default(defaultState.retentionDays),
        deleteOnSessionDeleted: z.boolean()
          .describe("Retain deleted-session rows when false, but never keep using them")
          .default(defaultState.deleteOnSessionDeleted),
      })
      .strict()
      .default(defaultState),
  })
  .strict()
  .default(defaultConfigValues)

export type OpenAICompactConfig = z.infer<typeof OpenAICompactConfigSchema>
export type OpenAICompactConfigInput = z.input<typeof OpenAICompactConfigSchema>

export const defaultConfig = OpenAICompactConfigSchema.parse({})
