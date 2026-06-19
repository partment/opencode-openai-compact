import { describe, expect, test } from "vitest"
import { openSQLiteDatabase, resolveSQLiteBinding } from "../src/sqlite.js"

class FakeDatabase {
  exec() {}
  query() {
    return {
      get: () => null,
      all: () => [],
      run: () => undefined,
    }
  }
  close() {}
}

describe("resolveSQLiteBinding", () => {
  test("uses bun:sqlite first in Bun", () => {
    const calls: string[] = []
    const binding = resolveSQLiteBinding({
      isBun: true,
      require(id) {
        calls.push(id)
        if (id === "bun:sqlite") return { Database: FakeDatabase }
        throw new Error(`unexpected ${id}`)
      },
    })

    expect(binding.source).toBe("bun:sqlite")
    expect(calls).toEqual(["bun:sqlite"])
  })

  test("falls back from bun:sqlite to node:sqlite", () => {
    const calls: string[] = []
    const binding = resolveSQLiteBinding({
      isBun: true,
      require(id) {
        calls.push(id)
        if (id === "bun:sqlite") throw new Error("no bun sqlite")
        if (id === "node:sqlite") return { DatabaseSync: FakeDatabase }
        throw new Error(`unexpected ${id}`)
      },
    })

    expect(binding.source).toBe("node:sqlite")
    expect(calls).toEqual(["bun:sqlite", "node:sqlite"])
  })

  test("uses better-sqlite3 after node:sqlite fails", () => {
    const calls: string[] = []
    const binding = resolveSQLiteBinding({
      isBun: false,
      require(id) {
        calls.push(id)
        if (id === "node:sqlite") throw new Error("no node sqlite")
        if (id === "better-sqlite3") return FakeDatabase
        throw new Error(`unexpected ${id}`)
      },
    })

    expect(binding.source).toBe("better-sqlite3")
    expect(calls).toEqual(["node:sqlite", "better-sqlite3"])
  })

  test("reports all failed binding attempts", () => {
    expect(() =>
      resolveSQLiteBinding({
        isBun: false,
        require(id) {
          throw new Error(`missing ${id}`)
        },
      }),
    ).toThrow("opencode-openai-compact: no SQLite binding available")
  })
})

describe("openSQLiteDatabase", () => {
  test("adapts prepare-based sqlite APIs", () => {
    class PrepareDatabase {
      static last: PrepareDatabase | undefined
      prepared: string[] = []
      execs: string[] = []

      constructor(readonly filename: string) {
        PrepareDatabase.last = this
      }

      exec(sql: string) {
        this.execs.push(sql)
      }

      prepare(sql: string) {
        this.prepared.push(sql)
        return {
          get: (...params: unknown[]) => ({ sql, params }),
          all: (...params: unknown[]) => [{ sql, params }],
          run: (...params: unknown[]) => ({ sql, params }),
        }
      }

      close() {}
    }

    const db = openSQLiteDatabase("example.sqlite", {
      isBun: false,
      require(id) {
        if (id === "node:sqlite") return { DatabaseSync: PrepareDatabase }
        throw new Error(`unexpected ${id}`)
      },
    })

    db.exec("create table t (x text)")
    expect(db.query("select ? as x").get("ok")).toEqual({ sql: "select ? as x", params: ["ok"] })
    expect(PrepareDatabase.last?.filename).toBe("example.sqlite")
    expect(PrepareDatabase.last?.execs).toEqual(["create table t (x text)"])
    expect(PrepareDatabase.last?.prepared).toEqual(["select ? as x"])
  })
})
