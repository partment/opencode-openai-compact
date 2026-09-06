import { afterEach, describe, expect, test, vi } from "vitest"
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { loadConfig } from "../src/config.js"

const originalEnv = { ...process.env }
const tempRoots: string[] = []

afterEach(async () => {
  vi.restoreAllMocks()
  process.env = { ...originalEnv }
  await Promise.all(tempRoots.splice(0).map((dir) => rm(dir, { recursive: true, force: true })))
})

async function tempDir() {
  const dir = await mkdtemp(path.join(os.tmpdir(), "opencode-openai-compact-"))
  tempRoots.push(dir)
  return dir
}

async function exists(file: string) {
  try {
    await stat(file)
    return true
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false
    throw error
  }
}

describe("loadConfig", () => {
  test("merges global, OPENCODE_CONFIG_DIR, and nearest .opencode config", async () => {
    const root = await tempDir()
    const xdg = path.join(root, "xdg")
    const global = path.join(xdg, "opencode")
    const customDir = path.join(root, "custom")
    const project = path.join(root, "project")
    const parentOpencode = path.join(project, ".opencode")
    const nearestOpencode = path.join(project, "src", ".opencode")
    const nested = path.join(project, "src", "feature")

    await mkdir(global, { recursive: true })
    await mkdir(customDir, { recursive: true })
    await mkdir(parentOpencode, { recursive: true })
    await mkdir(nearestOpencode, { recursive: true })
    await mkdir(nested, { recursive: true })

    await writeFile(path.join(global, "openai-compact.json"), `{ "providers": { "openai": { "compactModel": "global-model", "compactReasoningEffort": "minimal" } } }`)
    await writeFile(path.join(global, "openai-compact.jsonc"), `{ "state": { "retentionDays": 10 } }`)
    await writeFile(path.join(customDir, "openai-compact.json"), `{ "providers": { "custom-openai": { "compactModel": "custom-model" } } }`)
    await writeFile(
      path.join(customDir, "openai-compact.jsonc"),
      `{ "responses": { "endpointPath": "responses", "compactEndpointPath": "legacy/compact" } }`,
    )
    await writeFile(path.join(parentOpencode, "openai-compact.jsonc"), `{ "summary": "ignored" }`)
    await writeFile(path.join(nearestOpencode, "openai-compact.json"), `{ "summary": "nearest" }`)
    await writeFile(path.join(nearestOpencode, "openai-compact.jsonc"), `{ "summary": "nearest-jsonc" }`)

    process.env.XDG_CONFIG_HOME = xdg
    process.env.OPENCODE_CONFIG_DIR = customDir

    const config = await loadConfig({ directory: nested, worktree: project })

    expect(config.providers.openai?.compactModel).toBe("global-model")
    expect(config.providers.openai?.compactReasoningEffort).toBe("minimal")
    expect(config.providers["custom-openai"]?.compactModel).toBe("custom-model")
    expect(config.providers["custom-openai"]?.compactReasoningEffort).toBeNull()
    expect(config.state.retentionDays).toBe(10)
    expect(config.responses.endpointPath).toBe("/responses")
    expect(config.responses.compactEndpointPath).toBe("/legacy/compact")
    expect(config.summary).toBe("nearest-jsonc")
    expect(config.state.deleteOnSessionDeleted).toBe(true)
  })

  test("uses global retention and warns for ignored non-global overrides", async () => {
    const root = await tempDir()
    const xdg = path.join(root, "xdg")
    const global = path.join(xdg, "opencode")
    const customDir = path.join(root, "custom")
    const project = path.join(root, "project")
    const projectConfig = path.join(project, ".opencode")
    await mkdir(global, { recursive: true })
    await mkdir(customDir, { recursive: true })
    await mkdir(projectConfig, { recursive: true })
    await writeFile(path.join(global, "openai-compact.jsonc"), `{ "state": { "retentionDays": 10 } }`)
    await writeFile(path.join(customDir, "openai-compact.jsonc"),
      `{ "state": { "retentionDays": 1 }, "summary": "custom" }`)
    await writeFile(path.join(projectConfig, "openai-compact.jsonc"),
      `{ "state": { "retentionDays": 2, "deleteOnSessionDeleted": false }, "summary": "project" }`)
    process.env.XDG_CONFIG_HOME = xdg
    process.env.OPENCODE_CONFIG_DIR = customDir
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})

    const config = await loadConfig({ directory: project, worktree: project })

    expect(config.state.retentionDays).toBe(10)
    expect(config.state.deleteOnSessionDeleted).toBe(false)
    expect(config.summary).toBe("project")
    expect(warn).toHaveBeenCalledTimes(2)
    expect(warn.mock.calls.map(([message]) => String(message))).toEqual(expect.arrayContaining([
      expect.stringContaining(path.join(customDir, "openai-compact.jsonc")),
      expect.stringContaining(path.join(projectConfig, "openai-compact.jsonc")),
    ]))
    warn.mockRestore()
  })

  test("uses the 30-day default when global retention is absent", async () => {
    const root = await tempDir()
    process.env.XDG_CONFIG_HOME = path.join(root, "xdg")
    const customDir = path.join(root, "custom")
    await mkdir(customDir, { recursive: true })
    await writeFile(path.join(customDir, "openai-compact.jsonc"), `{ "state": { "retentionDays": 1 } }`)
    process.env.OPENCODE_CONFIG_DIR = customDir
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})

    const config = await loadConfig({ directory: root, worktree: root })

    expect(config.state.retentionDays).toBe(30)
    expect(warn).toHaveBeenCalledOnce()
    warn.mockRestore()
  })

  test("creates a global openai-compact.jsonc when no global config exists", async () => {
    const root = await tempDir()
    const xdg = path.join(root, "xdg")
    process.env.XDG_CONFIG_HOME = xdg

    await loadConfig({ directory: root, worktree: root })

    const configPath = path.join(xdg, "opencode", "openai-compact.jsonc")

    expect(await exists(configPath)).toBe(true)
    expect(JSON.parse(await readFile(configPath, "utf8"))).toEqual({
      $schema: "https://raw.githubusercontent.com/partment/opencode-openai-compact/main/configSchema.json",
    })
  })
})
