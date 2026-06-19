import { createRequire } from "node:module"

export type SQLiteSource = "bun:sqlite" | "node:sqlite" | "better-sqlite3"

export type SQLiteStatement<T = Record<string, unknown>> = {
  get(...params: unknown[]): T | null | undefined
  all(...params: unknown[]): T[]
  run(...params: unknown[]): unknown
}

export type SQLiteDatabase = {
  exec(sql: string): unknown
  query<T = Record<string, unknown>>(sql: string): SQLiteStatement<T>
  close(): void
}

type RawStatement = {
  get(...params: unknown[]): unknown
  all(...params: unknown[]): unknown[]
  run(...params: unknown[]): unknown
}

type RawDatabase = {
  exec(sql: string): unknown
  query?(sql: string): RawStatement
  prepare?(sql: string): RawStatement
  close(): void
}

type RawDatabaseCtor = new (filename?: string, options?: unknown) => RawDatabase

type RequireLike = (id: string) => unknown

type ResolveOptions = {
  isBun?: boolean
  require?: RequireLike
}

export type SQLiteBinding = {
  source: SQLiteSource
  Database: RawDatabaseCtor
}

let cachedBinding: SQLiteBinding | undefined

function isBunRuntime() {
  return typeof (globalThis as { Bun?: unknown }).Bun !== "undefined"
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

export function resolveSQLiteBinding(options: ResolveOptions = {}): SQLiteBinding {
  const useCache = options.isBun === undefined && options.require === undefined
  if (useCache && cachedBinding) return cachedBinding

  const req = options.require ?? createRequire(import.meta.url)
  const failures: string[] = []

  if (options.isBun ?? isBunRuntime()) {
    try {
      const sqlite = req("bun:sqlite") as { Database: RawDatabaseCtor }
      const binding = { source: "bun:sqlite" as const, Database: sqlite.Database }
      if (useCache) cachedBinding = binding
      return binding
    } catch (error) {
      failures.push(`bun:sqlite: ${errorMessage(error)}`)
    }
  }

  try {
    const sqlite = req("node:sqlite") as { DatabaseSync: RawDatabaseCtor }
    const binding = { source: "node:sqlite" as const, Database: sqlite.DatabaseSync }
    if (useCache) cachedBinding = binding
    return binding
  } catch (error) {
    failures.push(`node:sqlite: ${errorMessage(error)}`)
  }

  try {
    const Database = req("better-sqlite3") as RawDatabaseCtor
    const binding = { source: "better-sqlite3" as const, Database }
    if (useCache) cachedBinding = binding
    return binding
  } catch (error) {
    failures.push(`better-sqlite3: ${errorMessage(error)}`)
  }

  throw new Error(
    "opencode-openai-compact: no SQLite binding available. Install better-sqlite3, " +
      "or run on Node with node:sqlite support, or use Bun. " +
      `Attempts: ${failures.join("; ")}`,
  )
}

export function openSQLiteDatabase(filename: string, options: ResolveOptions = {}): SQLiteDatabase {
  const binding = resolveSQLiteBinding(options)
  return adaptSQLiteDatabase(new binding.Database(filename))
}

function adaptSQLiteDatabase(db: RawDatabase): SQLiteDatabase {
  return {
    exec(sql) {
      return db.exec(sql)
    },
    query<T = Record<string, unknown>>(sql: string): SQLiteStatement<T> {
      const statement = db.query ? db.query(sql) : db.prepare?.(sql)
      if (!statement) throw new Error("SQLite binding does not expose query() or prepare()")

      return {
        get(...params) {
          return statement.get(...params) as T | null | undefined
        },
        all(...params) {
          return statement.all(...params) as T[]
        },
        run(...params) {
          return statement.run(...params)
        },
      }
    },
    close() {
      db.close()
    },
  }
}
