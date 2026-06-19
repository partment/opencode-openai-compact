import fs from "node:fs/promises"
import path from "node:path"
import { openSQLiteDatabase, type SQLiteDatabase } from "./sqlite.js"

export type AnyRecord = Record<string, unknown>

export type Checkpoint = {
  providerID: string
  responseID: string
  afterMessageID: string
  afterCreatedAt: number
  createdAt: number
  items: AnyRecord[]
}

type CheckpointRow = {
  provider_id: string
  session_id: string
  response_id: string
  after_message_id: string
  after_created_at: number
  created_at: number
  items_json: string
}

const schemaVersion = 1
const dayMs = 24 * 60 * 60 * 1000

function asRecord(value: unknown): AnyRecord | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as AnyRecord) : undefined
}

function itemsFrom(value: unknown): AnyRecord[] | undefined {
  if (!Array.isArray(value)) return undefined
  const items = value
    .map(asRecord)
    .filter((item): item is AnyRecord => item !== undefined)
    .map((item) =>
      item.type === "compaction_summary" && typeof item.encrypted_content === "string"
        ? { type: "compaction", encrypted_content: item.encrypted_content }
        : item,
    )
  return items.length ? items : undefined
}

function checkpointFromRow(row: CheckpointRow): Checkpoint | undefined {
  const items = itemsFrom(JSON.parse(row.items_json))
  if (!items) return undefined
  return {
    providerID: row.provider_id,
    responseID: row.response_id,
    afterMessageID: row.after_message_id,
    afterCreatedAt: row.after_created_at,
    createdAt: row.created_at,
    items,
  }
}

export class CheckpointStore {
  private constructor(private readonly db: SQLiteDatabase) {}

  static async open(file: string) {
    await fs.mkdir(path.dirname(file), { recursive: true })
    const db = openSQLiteDatabase(file)
    configureFileDatabase(db)
    const store = new CheckpointStore(db)
    store.migrate()
    return store
  }

  static openMemory() {
    const store = new CheckpointStore(openSQLiteDatabase(":memory:"))
    store.migrate()
    return store
  }

  close() {
    this.db.close()
  }

  loadAll() {
    const rows = this.db
      .query<CheckpointRow>(
        `select provider_id, session_id, response_id, after_message_id, after_created_at, created_at, items_json
         from checkpoints
         order by provider_id, session_id, after_created_at, created_at`,
      )
      .all()
    const result: Array<{ sessionID: string; checkpoint: Checkpoint }> = []

    for (const row of rows) {
      try {
        const checkpoint = checkpointFromRow(row)
        if (!checkpoint) continue
        result.push({ sessionID: row.session_id, checkpoint })
      } catch {
        // Malformed rows are ignored; future saves overwrite by response id.
      }
    }

    return result
  }

  upsert(sessionID: string, checkpoint: Checkpoint) {
    this.db
      .query(
        `insert into checkpoints (
           provider_id, session_id, response_id, after_message_id, after_created_at, created_at, items_json
         ) values (?, ?, ?, ?, ?, ?, ?)
         on conflict(provider_id, session_id, response_id) do update set
           after_message_id = excluded.after_message_id,
           after_created_at = excluded.after_created_at,
           created_at = excluded.created_at,
           items_json = excluded.items_json`,
      )
      .run(
        checkpoint.providerID,
        sessionID,
        checkpoint.responseID,
        checkpoint.afterMessageID,
        checkpoint.afterCreatedAt,
        checkpoint.createdAt,
        JSON.stringify(checkpoint.items),
      )
  }

  deleteSession(sessionID: string) {
    this.db.query("delete from checkpoints where session_id = ?").run(sessionID)
  }

  prune(retentionDays: number) {
    const cutoff = Date.now() - retentionDays * dayMs
    this.db.query("delete from checkpoints where created_at < ?").run(cutoff)
  }

  count() {
    const row = this.db.query<{ count: number }>("select count(*) as count from checkpoints").get()
    return row?.count ?? 0
  }

  version() {
    return this.schemaVersion()
  }

  private migrate() {
    const version = this.schemaVersion()
    if (version > schemaVersion) {
      throw new Error(`Unsupported openai-compact database schema version: ${version}`)
    }

    if (version === 0) {
      this.db.exec(`
        create table if not exists checkpoints (
          provider_id text not null,
          session_id text not null,
          response_id text not null,
          after_message_id text not null,
          after_created_at integer not null,
          created_at integer not null,
          items_json text not null,
          primary key (provider_id, session_id, response_id)
        );

        create index if not exists checkpoints_provider_session_boundary_idx
        on checkpoints (provider_id, session_id, after_created_at, created_at);

        PRAGMA user_version = ${schemaVersion};
      `)
    }
  }

  private schemaVersion() {
    const row = this.db.query<{ user_version: number }>("PRAGMA user_version").get()
    return row?.user_version ?? 0
  }
}

export const currentSchemaVersion = schemaVersion

function configureFileDatabase(db: SQLiteDatabase) {
  db.query("PRAGMA journal_mode = WAL").get()
  db.exec(`
    PRAGMA synchronous = NORMAL;
    PRAGMA busy_timeout = 5000;
  `)
}
