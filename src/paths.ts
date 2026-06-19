import { statSync } from "node:fs"
import os from "node:os"
import path from "node:path"

export type ConfigContext = {
  directory: string
  worktree: string
}

export type ConfigSource = {
  path: string
  optional: boolean
}

const configFileNames = ["openai-compact.json", "openai-compact.jsonc"] as const

export function getGlobalConfigDir() {
  return process.env.XDG_CONFIG_HOME
    ? path.join(process.env.XDG_CONFIG_HOME, "opencode")
    : path.join(os.homedir(), ".config", "opencode")
}

export function getDefaultConfigPath() {
  return path.join(getGlobalConfigDir(), "openai-compact.jsonc")
}

export function getDatabasePath() {
  return path.join(getGlobalConfigDir(), "openai-compact", "checkpoints.db")
}

export function getGlobalConfigSources(): ConfigSource[] {
  return configFilesIn(getGlobalConfigDir())
}

function configFilesIn(dir: string): ConfigSource[] {
  return configFileNames.map((file) => ({ path: path.join(dir, file), optional: true }))
}

function isDirectory(dir: string) {
  try {
    return statSync(dir).isDirectory()
  } catch {
    return false
  }
}

function findNearestOpencodeDir(start: string) {
  let current = path.resolve(start)

  while (true) {
    const candidate = path.join(current, ".opencode")
    if (isDirectory(candidate)) return candidate

    const parent = path.dirname(current)
    if (parent === current) return undefined
    current = parent
  }
}

export function getConfigSources(context: ConfigContext): ConfigSource[] {
  const sources: ConfigSource[] = [...getGlobalConfigSources()]

  if (process.env.OPENCODE_CONFIG_DIR) sources.push(...configFilesIn(process.env.OPENCODE_CONFIG_DIR))

  const projectConfigDir = findNearestOpencodeDir(context.directory)
  if (projectConfigDir) sources.push(...configFilesIn(projectConfigDir))

  return sources
}
