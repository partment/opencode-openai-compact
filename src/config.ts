import fs from "node:fs/promises"
import path from "node:path"
import { parse, printParseErrorCode, type ParseError } from "jsonc-parser"
import { mergeDeep } from "./merge.js"
import { defaultConfig, OpenAICompactConfigSchema, type OpenAICompactConfig } from "./schema.js"
import {
  getConfigSources,
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
    throw new Error(`Invalid JSONC in ${source.path}: ${printParseErrorCode(first.error)} at offset ${first.offset}`)
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

function withoutNonGlobalRetention(data: unknown, source: ConfigSource, globalPaths: Set<string>) {
  if (globalPaths.has(configPathIdentity(source.path))) return data
  const root = asRecord(data)
  const state = asRecord(root?.state)
  if (!root || !state || !Object.hasOwn(state, "retentionDays")) return data

  console.warn(
    `opencode-openai-compact: ignoring state.retentionDays in ${source.path}; ` +
      `retention is a global database policy. Move it to ${getDefaultConfigPath()}.`,
  )
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
  const globalPaths = new Set(getGlobalConfigSources().map((source) => configPathIdentity(source.path)))
  for (const source of getConfigSources(context)) {
    const data = await readOptionalJsonc(source)
    if (data === undefined) continue
    merged = mergeDeep(merged, withoutNonGlobalRetention(data, source, globalPaths))
  }

  return OpenAICompactConfigSchema.parse(merged)
}

export { getConfigSources }
