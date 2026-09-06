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

export type ControlMessage = {
  providerID: string
  sessionID: string
  messageID: string
  createdAt: number
  contentText: string
}

export type SessionState = {
  sessionID: string
  revision: number
  deleted: boolean
  checkpoints: Checkpoint[]
  controls: ControlMessage[]
}

export type StateCommitResult =
  | { status: "committed"; state: SessionState }
  | { status: "conflict"; state: SessionState }

type CheckpointRow = {
  provider_id: string
  session_id: string
  response_id: string
  after_message_id: string
  after_created_at: number
  created_at: number
  items_json: string
}

type ControlMessageRow = {
  provider_id: string
  session_id: string
  message_id: string
  created_at: number
  content_text: string
}

type SessionStateRow = {
  session_id: string
  revision: number
  deleted: number
  updated_at: number
}

const schemaVersion = 3
const dayMs = 24 * 60 * 60 * 1000

function asRecord(value: unknown): AnyRecord | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as AnyRecord) : undefined
}

export function compactedItemsFrom(value: unknown): AnyRecord[] | undefined {
  if (!Array.isArray(value)) return undefined
  const items: AnyRecord[] = []
  for (const valueItem of value) {
    const record = asRecord(valueItem)
    if (!record) continue

    let item = record
    if (item.type === "compaction_summary") {
      if (typeof item.encrypted_content !== "string") return undefined
      item = { ...item, type: "compaction" }
    }
    if (item.role === "developer" || item.role === "system") continue
    items.push(item)
  }

  const compactions = items.filter((item) => item.type === "compaction")
  if (compactions.length !== 1 || typeof compactions[0].encrypted_content !== "string") return undefined
  return items
}

function checkpointFromRow(row: CheckpointRow): Checkpoint | undefined {
  const items = compactedItemsFrom(JSON.parse(row.items_json))
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

function controlFromRow(row: ControlMessageRow): ControlMessage {
  return {
    providerID: row.provider_id,
    sessionID: row.session_id,
    messageID: row.message_id,
    createdAt: row.created_at,
    contentText: row.content_text,
  }
}

function sameCheckpoint(a: Checkpoint, b: Checkpoint) {
  return a.providerID === b.providerID && a.responseID === b.responseID &&
    a.afterMessageID === b.afterMessageID && a.afterCreatedAt === b.afterCreatedAt &&
    a.createdAt === b.createdAt && JSON.stringify(a.items) === JSON.stringify(b.items)
}

function sameControl(a: ControlMessage, b: ControlMessage) {
  return a.providerID === b.providerID && a.sessionID === b.sessionID && a.messageID === b.messageID &&
    a.createdAt === b.createdAt && a.contentText === b.contentText
}

export class CheckpointStore {
  private constructor(private readonly db: SQLiteDatabase) {}

  static async open(file: string) {
    await fs.mkdir(path.dirname(file), { recursive: true })
    const db = openSQLiteDatabase(file)
    try {
      configureFileDatabase(db)
      const store = new CheckpointStore(db)
      store.migrate()
      return store
    } catch (error) {
      try { db.close() }
      catch { /* Preserve the original open error. */ }
      throw error
    }
  }

  static openMemory() {
    const db = openSQLiteDatabase(":memory:")
    try {
      const store = new CheckpointStore(db)
      store.migrate()
      return store
    } catch (error) {
      try { db.close() }
      catch { /* Preserve the original open error. */ }
      throw error
    }
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

  loadControlMessages() {
    return this.db
      .query<ControlMessageRow>(
        `select provider_id, session_id, message_id, created_at, content_text
         from control_messages
         order by provider_id, session_id, created_at, message_id`,
      )
      .all()
      .map(controlFromRow)
  }

  loadSession(sessionID: string, retentionDays: number): SessionState {
    const cutoff = Date.now() - retentionDays * dayMs
    return this.db.transaction(() => {
      this.ensureSessionState(sessionID)
      const expiredCheckpoint = this.db
        .query<{ value: number }>("select 1 as value from checkpoints where session_id = ? and created_at < ? limit 1")
        .get(sessionID, cutoff)
      if (expiredCheckpoint) {
        this.db.query("delete from checkpoints where session_id = ? and created_at < ?").run(sessionID, cutoff)
      }
      const expiredControl = this.db
        .query<{ value: number }>(
          `select 1 as value from control_messages
           where session_id = ? and created_at < ?
             and not exists (
               select 1 from checkpoints
               where checkpoints.provider_id = control_messages.provider_id
                 and checkpoints.session_id = control_messages.session_id
             )
           limit 1`,
        )
        .get(sessionID, cutoff)
      if (expiredControl) {
        this.db
          .query(
            `delete from control_messages
             where session_id = ? and created_at < ?
               and not exists (
                 select 1 from checkpoints
                 where checkpoints.provider_id = control_messages.provider_id
                   and checkpoints.session_id = control_messages.session_id
               )`,
          )
          .run(sessionID, cutoff)
      }
      if (expiredCheckpoint || expiredControl) this.bumpSessionState(sessionID)
      return this.readSessionState(sessionID)
    })
  }

  upsert(sessionID: string, checkpoint: Checkpoint) {
    const itemsJson = JSON.stringify(checkpoint.items)
    this.db.transaction(() => {
      const state = this.ensureSessionState(sessionID)
      if (state.deleted) throw new Error(`Cannot update deleted openai-compact session: ${sessionID}`)
      const existing = this.checkpoint(sessionID, checkpoint.providerID, checkpoint.responseID)
      if (existing && sameCheckpoint(existing, checkpoint)) return
      this.writeCheckpoint(sessionID, checkpoint, itemsJson)
      this.bumpSessionState(sessionID)
    })
  }

  commitCheckpoint(sessionID: string, expectedRevision: number, checkpoint: Checkpoint): StateCommitResult {
    const itemsJson = JSON.stringify(checkpoint.items)
    return this.db.transaction(() => {
      const current = this.ensureSessionState(sessionID)
      if (current.deleted) return { status: "conflict", state: this.readSessionState(sessionID) }
      const existing = this.checkpoint(sessionID, checkpoint.providerID, checkpoint.responseID)
      if (existing && sameCheckpoint(existing, checkpoint) && current.revision === expectedRevision + 1) {
        return { status: "committed", state: this.readSessionState(sessionID) }
      }
      if (current.revision !== expectedRevision) {
        return { status: "conflict", state: this.readSessionState(sessionID) }
      }
      this.writeCheckpoint(sessionID, checkpoint, itemsJson)
      this.bumpSessionState(sessionID)
      return { status: "committed", state: this.readSessionState(sessionID) }
    })
  }

  upsertControlMessage(message: ControlMessage) {
    this.db.transaction(() => {
      const state = this.ensureSessionState(message.sessionID)
      if (state.deleted) throw new Error(`Cannot update deleted openai-compact session: ${message.sessionID}`)
      const existing = this.control(message.sessionID, message.providerID, message.messageID)
      if (existing && sameControl(existing, message)) return
      this.writeControl(message)
      this.bumpSessionState(message.sessionID)
    })
  }

  commitControlMessages(
    sessionID: string,
    expectedRevision: number,
    messages: ControlMessage[],
  ): StateCommitResult {
    if (messages.some((message) => message.sessionID !== sessionID)) {
      throw new Error(`Control message session does not match commit target: ${sessionID}`)
    }
    return this.db.transaction(() => {
      const current = this.ensureSessionState(sessionID)
      if (current.deleted) return { status: "conflict", state: this.readSessionState(sessionID) }
      const changed = messages.filter((message) => {
        const existing = this.control(sessionID, message.providerID, message.messageID)
        return !existing || !sameControl(existing, message)
      })
      if (!changed.length) return { status: "committed", state: this.readSessionState(sessionID) }
      if (current.revision !== expectedRevision) {
        return { status: "conflict", state: this.readSessionState(sessionID) }
      }
      for (const message of changed) this.writeControl(message)
      this.bumpSessionState(sessionID)
      return { status: "committed", state: this.readSessionState(sessionID) }
    })
  }

  commitForkState(input: {
    sources: Array<{ sessionID: string; revision: number }>
    sessionID: string
    revision: number
    checkpoints: Checkpoint[]
    controls: ControlMessage[]
  }): StateCommitResult {
    if (input.controls.some((message) => message.sessionID !== input.sessionID)) {
      throw new Error(`Control message session does not match fork target: ${input.sessionID}`)
    }
    const checkpoints = input.checkpoints.map((checkpoint) => ({ checkpoint, itemsJson: JSON.stringify(checkpoint.items) }))
    return this.db.transaction(() => {
      const sources = input.sources.map((source) => ({ ...source, current: this.ensureSessionState(source.sessionID) }))
      const target = this.ensureSessionState(input.sessionID)
      if (sources.some((source) => source.current.deleted) || target.deleted) {
        return { status: "conflict", state: this.readSessionState(input.sessionID) }
      }

      const exactCheckpoints = checkpoints.every(({ checkpoint }) => {
        const existing = this.checkpoint(input.sessionID, checkpoint.providerID, checkpoint.responseID)
        return !!existing && sameCheckpoint(existing, checkpoint)
      })
      const exactControls = input.controls.every((message) => {
        const existing = this.control(input.sessionID, message.providerID, message.messageID)
        return !!existing && sameControl(existing, message)
      })
      if (exactCheckpoints && exactControls) {
        return { status: "committed", state: this.readSessionState(input.sessionID) }
      }
      if (sources.some((source) => source.current.revision !== source.revision) || target.revision !== input.revision) {
        return { status: "conflict", state: this.readSessionState(input.sessionID) }
      }
      for (const { checkpoint, itemsJson } of checkpoints) this.writeCheckpoint(input.sessionID, checkpoint, itemsJson)
      for (const control of input.controls) this.writeControl(control)
      this.bumpSessionState(input.sessionID)
      return { status: "committed", state: this.readSessionState(input.sessionID) }
    })
  }

  findCheckpointSessions(providerID: string, boundaryTimes: number[]) {
    if (!boundaryTimes.length) return []
    const placeholders = boundaryTimes.map(() => "?").join(", ")
    return this.db.transaction(() => this.db
      .query<{ session_id: string }>(
        `select distinct session_id from checkpoints
         where provider_id = ? and after_created_at in (${placeholders})`,
      )
      .all(providerID, ...boundaryTimes)
      .map((row) => row.session_id))
  }

  invalidateSessionState(sessionID: string, options: { deleted?: boolean; deleteData?: boolean } = {}) {
    return this.db.transaction(() => {
      const current = this.ensureSessionState(sessionID)
      if (options.deleteData) {
        this.db.query("delete from checkpoints where session_id = ?").run(sessionID)
        this.db.query("delete from control_messages where session_id = ?").run(sessionID)
      }
      this.writeSessionState(sessionID, current.revision + 1, current.deleted || options.deleted === true)
      return this.readSessionState(sessionID)
    })
  }

  clearSessionState(sessionID: string, expectedRevision?: number): StateCommitResult {
    return this.db.transaction(() => {
      const current = this.ensureSessionState(sessionID)
      if (current.deleted) return { status: "conflict", state: this.readSessionState(sessionID) }
      if (expectedRevision !== undefined && current.revision !== expectedRevision) {
        return { status: "conflict", state: this.readSessionState(sessionID) }
      }
      this.db.query("delete from checkpoints where session_id = ?").run(sessionID)
      this.db.query("delete from control_messages where session_id = ?").run(sessionID)
      this.bumpSessionState(sessionID)
      return { status: "committed", state: this.readSessionState(sessionID) }
    })
  }

  removeMessageState(sessionID: string, messageID: string) {
    return this.db.transaction(() => {
      this.ensureSessionState(sessionID)
      this.db.query("delete from checkpoints where session_id = ? and after_message_id = ?").run(sessionID, messageID)
      this.db.query("delete from control_messages where session_id = ? and message_id = ?").run(sessionID, messageID)
      this.bumpSessionState(sessionID)
      return this.readSessionState(sessionID)
    })
  }

  deleteSession(sessionID: string) {
    return this.invalidateSessionState(sessionID, { deleted: true, deleteData: true })
  }

  deleteCheckpoint(sessionID: string, providerID: string, responseID: string) {
    this.db.transaction(() => {
      this.ensureSessionState(sessionID)
      if (!this.checkpoint(sessionID, providerID, responseID)) return
      this.db
        .query("delete from checkpoints where session_id = ? and provider_id = ? and response_id = ?")
        .run(sessionID, providerID, responseID)
      this.bumpSessionState(sessionID)
    })
  }

  deleteControlMessage(sessionID: string, messageID: string) {
    this.db.transaction(() => {
      this.ensureSessionState(sessionID)
      const existing = this.db
        .query<{ value: number }>(
          "select 1 as value from control_messages where session_id = ? and message_id = ? limit 1",
        )
        .get(sessionID, messageID)
      if (!existing) return
      this.db.query("delete from control_messages where session_id = ? and message_id = ?").run(sessionID, messageID)
      this.bumpSessionState(sessionID)
    })
  }

  prune(retentionDays: number) {
    const cutoff = Date.now() - retentionDays * dayMs
    return this.db.transaction(() => {
      const affected = new Set(
        this.db
          .query<{ session_id: string }>("select distinct session_id from checkpoints where created_at < ?")
          .all(cutoff)
          .map((row) => row.session_id),
      )
      this.db.query("delete from checkpoints where created_at < ?").run(cutoff)
      const controls = this.db
        .query<{ session_id: string }>(
          `select distinct session_id from control_messages
           where created_at < ?
             and not exists (
               select 1 from checkpoints
               where checkpoints.provider_id = control_messages.provider_id
                 and checkpoints.session_id = control_messages.session_id
             )`,
        )
        .all(cutoff)
      for (const row of controls) affected.add(row.session_id)
      this.db
        .query(
          `delete from control_messages
           where created_at < ?
             and not exists (
               select 1 from checkpoints
               where checkpoints.provider_id = control_messages.provider_id
                 and checkpoints.session_id = control_messages.session_id
             )`,
        )
        .run(cutoff)
      for (const sessionID of affected) {
        this.ensureSessionState(sessionID)
        this.bumpSessionState(sessionID)
      }
      return [...affected]
    })
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

    if (version === schemaVersion) return
    this.db.transaction(() => {
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
        `)
      }

      if (version <= 1) {
        this.db.exec(`
          create table if not exists control_messages (
            provider_id text not null,
            session_id text not null,
            message_id text not null,
            created_at integer not null,
            content_text text not null,
            primary key (provider_id, session_id, message_id)
          );

          create index if not exists control_messages_session_created_idx
          on control_messages (session_id, created_at);
        `)
      }

      this.db.exec(`
        create table if not exists session_state (
          session_id text primary key,
          revision integer not null,
          deleted integer not null default 0,
          updated_at integer not null
        );

        insert or ignore into session_state (session_id, revision, deleted, updated_at)
        select session_id, 0, 0, max(created_at) from checkpoints group by session_id;

        insert or ignore into session_state (session_id, revision, deleted, updated_at)
        select session_id, 0, 0, max(created_at) from control_messages group by session_id;

        PRAGMA user_version = ${schemaVersion};
      `)
    })
  }

  private schemaVersion() {
    const row = this.db.query<{ user_version: number }>("PRAGMA user_version").get()
    return row?.user_version ?? 0
  }

  private ensureSessionState(sessionID: string) {
    const existing = this.sessionStateRow(sessionID)
    if (existing) return existing
    const now = Date.now()
    this.db
      .query("insert or ignore into session_state (session_id, revision, deleted, updated_at) values (?, 0, 0, ?)")
      .run(sessionID, now)
    return this.sessionStateRow(sessionID) ?? { sessionID, revision: 0, deleted: false, updatedAt: now }
  }

  private sessionStateRow(sessionID: string) {
    const row = this.db
      .query<SessionStateRow>("select session_id, revision, deleted, updated_at from session_state where session_id = ?")
      .get(sessionID)
    return row ? {
      sessionID: row.session_id,
      revision: row.revision,
      deleted: row.deleted !== 0,
      updatedAt: row.updated_at,
    } : undefined
  }

  private writeSessionState(sessionID: string, revision: number, deleted: boolean) {
    this.db
      .query(
        `insert into session_state (session_id, revision, deleted, updated_at) values (?, ?, ?, ?)
         on conflict(session_id) do update set
           revision = excluded.revision,
           deleted = excluded.deleted,
           updated_at = excluded.updated_at`,
      )
      .run(sessionID, revision, deleted ? 1 : 0, Date.now())
  }

  private bumpSessionState(sessionID: string) {
    const current = this.ensureSessionState(sessionID)
    this.writeSessionState(sessionID, current.revision + 1, current.deleted)
  }

  private readSessionState(sessionID: string): SessionState {
    const state = this.ensureSessionState(sessionID)
    const checkpoints: Checkpoint[] = []
    for (const row of this.db
      .query<CheckpointRow>(
        `select provider_id, session_id, response_id, after_message_id, after_created_at, created_at, items_json
         from checkpoints where session_id = ?
         order by provider_id, after_created_at, created_at`,
      )
      .all(sessionID)) {
      try {
        const checkpoint = checkpointFromRow(row)
        if (checkpoint) checkpoints.push(checkpoint)
      } catch {
        // Malformed rows remain inert until overwritten or pruned.
      }
    }
    const controls = this.db
      .query<ControlMessageRow>(
        `select provider_id, session_id, message_id, created_at, content_text
         from control_messages where session_id = ?
         order by provider_id, created_at, message_id`,
      )
      .all(sessionID)
      .map(controlFromRow)
    return { sessionID, revision: state.revision, deleted: state.deleted, checkpoints, controls }
  }

  private checkpoint(sessionID: string, providerID: string, responseID: string) {
    const row = this.db
      .query<CheckpointRow>(
        `select provider_id, session_id, response_id, after_message_id, after_created_at, created_at, items_json
         from checkpoints where session_id = ? and provider_id = ? and response_id = ?`,
      )
      .get(sessionID, providerID, responseID)
    if (!row) return undefined
    try { return checkpointFromRow(row) }
    catch { return undefined }
  }

  private control(sessionID: string, providerID: string, messageID: string) {
    const row = this.db
      .query<ControlMessageRow>(
        `select provider_id, session_id, message_id, created_at, content_text
         from control_messages where session_id = ? and provider_id = ? and message_id = ?`,
      )
      .get(sessionID, providerID, messageID)
    return row ? controlFromRow(row) : undefined
  }

  private writeCheckpoint(sessionID: string, checkpoint: Checkpoint, itemsJson: string) {
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
        itemsJson,
      )
  }

  private writeControl(message: ControlMessage) {
    this.db
      .query(
        `insert into control_messages (provider_id, session_id, message_id, created_at, content_text)
         values (?, ?, ?, ?, ?)
         on conflict(provider_id, session_id, message_id) do update set
           created_at = excluded.created_at,
           content_text = excluded.content_text`,
      )
      .run(message.providerID, message.sessionID, message.messageID, message.createdAt, message.contentText)
  }
}

export const currentSchemaVersion = schemaVersion

function configureFileDatabase(db: SQLiteDatabase) {
  db.exec("PRAGMA busy_timeout = 5000;")
  db.query("PRAGMA journal_mode = WAL").get()
  db.exec("PRAGMA synchronous = NORMAL;")
}
