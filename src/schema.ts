import { z } from "zod"

export const defaultCompactBodyKeys = [
  "input",
  "instructions",
  "previous_response_id",
  "prompt_cache_key",
  "prompt_cache_retention",
  "service_tier",
] as const

export const defaultCompactSummary = [
  "Context compacted.",
  "Following conversations will continue from this compacted checkpoint.",
].join("\n")

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
    compactModel: "gpt-5.4",
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
            compactModel: z.string().min(1),
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
        compactEndpointPath: z.string().min(1).transform(endpoint).default(defaultResponses.compactEndpointPath),
      })
      .default(defaultResponses),
    compactBodyKeys: z.array(z.string().min(1)).default([...defaultCompactBodyKeys]),
    summary: z.string().min(1).default(defaultCompactSummary),
    state: z
      .object({
        retentionDays: z.number().int().positive().default(defaultState.retentionDays),
        deleteOnSessionDeleted: z.boolean().default(defaultState.deleteOnSessionDeleted),
      })
      .default(defaultState),
  })
  .strict()
  .default(defaultConfigValues)

export type OpenAICompactConfig = z.infer<typeof OpenAICompactConfigSchema>
export type OpenAICompactConfigInput = z.input<typeof OpenAICompactConfigSchema>

export const defaultConfig = OpenAICompactConfigSchema.parse({})
