import fs from "node:fs/promises"
import path from "node:path"
import { parse, printParseErrorCode, type ParseError } from "jsonc-parser"
import { mergeDeep } from "./merge.js"
import { defaultConfig, OpenAICompactConfigSchema, type OpenAICompactConfig } from "./schema.js"
import {
  getConfigSources,
  getDatabasePath,
  getDefaultConfigPath,
  getGlobalConfigSources,
  type ConfigContext,
  type ConfigSource,
} from "./paths.js"

const configSchemaUrl = "https://raw.githubusercontent.com/partment/opencode-openai-compact/main/configSchema.json"

async function readOptionalJsonc(source: ConfigSource) {
  let text: string
  try {
    text = await fs.readFile(source.path, "utf8")
  } catch (error) {
    if (source.optional && (error as NodeJS.ErrnoException).code === "ENOENT") return undefined
    throw error
  }

  const errors: ParseError[] = []
  const parsed = parse(text, errors, { allowTrailingComma: true })
  if (errors.length > 0) {
    const first = errors[0]
    const lines = text.slice(0, first.offset).split(/\r\n|\r|\n/)
    const line = lines.length
    const column = lines[lines.length - 1].length + 1
    throw new Error(
      `Invalid JSONC in ${source.path}: ${printParseErrorCode(first.error)} at ` +
      `line ${line}, column ${column} (offset ${first.offset})`,
    )
  }
  return parsed ?? {}
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined
}

function configPathIdentity(file: string) {
  const resolved = path.resolve(file)
  return process.platform === "win32" ? resolved.toLowerCase() : resolved
}

function withoutNonGlobalRetention(
  data: unknown,
  source: ConfigSource,
  globalPaths: Set<string>,
  ignored: string[],
) {
  if (globalPaths.has(configPathIdentity(source.path))) return data
  const root = asRecord(data)
  const state = asRecord(root?.state)
  if (!root || !state || !Object.hasOwn(state, "retentionDays")) return data

  const warning =
    `opencode-openai-compact: ignoring state.retentionDays in ${source.path}; ` +
    `retention is a global database policy. Move it to ${getDefaultConfigPath()}.`
  ignored.push(warning)
  console.warn(warning)
  const { retentionDays: _ignored, ...remainingState } = state
  return { ...root, state: remainingState }
}

async function fileExists(file: string) {
  try {
    await fs.stat(file)
    return true
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false
    throw error
  }
}

async function ensureGlobalConfigFile() {
  for (const source of getGlobalConfigSources()) {
    if (await fileExists(source.path)) return
  }

  const file = getDefaultConfigPath()
  await fs.mkdir(path.dirname(file), { recursive: true })
  await fs
    .writeFile(
      file,
      `{
  "$schema": "${configSchemaUrl}"
}
`,
      { flag: "wx" },
    )
    .catch((error) => {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error
    })
}

export async function loadConfig(context: ConfigContext): Promise<OpenAICompactConfig> {
  await ensureGlobalConfigFile()

  let merged: unknown = defaultConfig
  const sources = getConfigSources(context)
  const globalPaths = new Set(getGlobalConfigSources().map((source) => configPathIdentity(source.path)))
  const sourceStatus: Array<{ path: string; exists: boolean }> = []
  const overrides: Array<{ source: string; path: string }> = []
  const ignored: string[] = []
  const deprecated: Array<{ source: string; path: string }> = []
  for (const source of sources) {
    const data = await readOptionalJsonc(source)
    sourceStatus.push({ path: source.path, exists: data !== undefined })
    if (data === undefined) continue
    const root = asRecord(data)
    const responses = asRecord(root?.responses)
    if (Object.hasOwn(responses ?? {}, "compactEndpointPath")) {
      deprecated.push({ source: source.path, path: "responses.compactEndpointPath" })
    }
    const next = withoutNonGlobalRetention(data, source, globalPaths, ignored)
    merged = mergeDeep(merged, next, (path) => overrides.push({ source: source.path, path }))
  }

  const config = OpenAICompactConfigSchema.parse(merged)
  if (process.env.OPENCODE_OPENAI_COMPACT_DEBUG === "1") {
    console.debug("opencode-openai-compact: effective configuration", JSON.stringify({
      sources: sourceStatus,
      overrides,
      enabled: config.enabled,
      activeProviders: config.enabled
        ? Object.keys(config.providers).filter((id) => config.providers[id].enabled)
        : [],
      providers: config.providers,
      headers: config.headers,
      responses: config.responses,
      databasePath: getDatabasePath(),
      retention: { days: config.state.retentionDays, scope: "global" },
      ignored,
      deprecated,
    }, null, 2))
  }
  return config
}

export { getConfigSources }
