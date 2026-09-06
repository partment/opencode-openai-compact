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
  if (!trimmed) return trimmed
  const prefixed = trimmed.startsWith("/") ? trimmed : `/${trimmed}`
  return prefixed.length > 1 ? prefixed.replace(/\/+$/, "") : prefixed
}

export const OpenAICompactConfigSchema = z
  .object({
    $schema: z.string().optional(),
    enabled: z.boolean().default(true),
    providers: z
      .record(
        z.string().min(1),
        z
          .object({
            compactModel: z.string().min(1).nullable().default(null),
            compactReasoningEffort: z.enum(compactReasoningEfforts).nullable().default(null),
          })
          .strict(),
      )
      .refine((providers) => Object.keys(providers).length > 0, "At least one provider is required")
      .default(defaultProviders),
    headers: z
      .object({
        compact: z.string().min(1).default(defaultHeaders.compact),
        session: z.string().min(1).default(defaultHeaders.session),
      })
      .default(defaultHeaders),
    responses: z
      .object({
        endpointPath: z.string().min(1).transform(endpoint).default(defaultResponses.endpointPath),
        // Kept so existing config files continue to load; compaction v2 uses endpointPath.
        compactEndpointPath: z.string().min(1).transform(endpoint).default(defaultResponses.compactEndpointPath),
      })
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
      .default(defaultState),
  })
  .strict()
  .default(defaultConfigValues)

export type OpenAICompactConfig = z.infer<typeof OpenAICompactConfigSchema>
export type OpenAICompactConfigInput = z.input<typeof OpenAICompactConfigSchema>

export const defaultConfig = OpenAICompactConfigSchema.parse({})
