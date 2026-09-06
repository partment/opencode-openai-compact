import { describe, expect, test } from "vitest"
import { mkdtemp, rm } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { CheckpointStore, currentSchemaVersion, type Checkpoint, type ControlMessage } from "../src/state.js"
import { openSQLiteDatabase } from "../src/sqlite.js"

const dayMs = 24 * 60 * 60 * 1000

function checkpoint(responseID: string, createdAt = Date.now()): Checkpoint {
  return {
    providerID: "openai",
    responseID,
    afterMessageID: `msg_${responseID}`,
    afterCreatedAt: createdAt,
    createdAt,
    items: [{ type: "compaction", encrypted_content: responseID }],
  }
}

function control(sessionID: string, messageID = "msg_control"): ControlMessage {
  return { providerID: "openai", sessionID, messageID, createdAt: Date.now(), contentText: "control" }
}

async function fileStore(prefix: string) {
  const root = await mkdtemp(path.join(os.tmpdir(), prefix))
  const file = path.join(root, "checkpoints.db")
  const store = await CheckpointStore.open(file)
  return { root, file, store }
}

describe("transactional checkpoint state", () => {
  test("commits by revision, accepts exact retries, and tombstones retained rows", () => {
    const store = CheckpointStore.openMemory()
    try {
      const initial = store.loadSession("ses", 30)
      const first = checkpoint("resp_first")
      const committed = store.commitCheckpoint("ses", initial.revision, first)
      expect(committed.status).toBe("committed")
      expect(committed.state.checkpoints).toEqual([first])

      const exact = store.commitCheckpoint("ses", initial.revision, first)
      expect(exact.status).toBe("committed")
      expect(exact.state.revision).toBe(committed.state.revision)

      const conflict = store.commitCheckpoint("ses", initial.revision, checkpoint("resp_stale"))
      expect(conflict.status).toBe("conflict")
      expect(store.loadAll().map(({ checkpoint }) => checkpoint.responseID)).toEqual(["resp_first"])

      store.invalidateSessionState("ses")
      expect(store.commitCheckpoint("ses", initial.revision, first).status).toBe("conflict")
      const deleted = store.invalidateSessionState("ses", { deleted: true, deleteData: false })
      expect(deleted.deleted).toBe(true)
      expect(store.loadAll()).toHaveLength(1)
      expect(store.loadSession("ses", 30).deleted).toBe(true)
      expect(store.commitCheckpoint("ses", deleted.revision, checkpoint("resp_late")).status).toBe("conflict")
    } finally { store.close() }
  })

  test("rolls back checkpoint and control inheritance together", async () => {
    const f = await fileStore("openai-compact-fork-transaction-")
    const db = openSQLiteDatabase(f.file)
    try {
      f.store.upsert("parent", checkpoint("resp_parent"))
      f.store.upsertControlMessage(control("parent"))
      const source = f.store.loadSession("parent", 30)
      const target = f.store.loadSession("child", 30)
      db.exec(`
        create trigger fail_child_control before insert on control_messages
        when new.session_id = 'child'
        begin select raise(abort, 'control write failed'); end;
      `)

      expect(() => f.store.commitForkState({
        sources: [{ sessionID: "parent", revision: source.revision }],
        sessionID: "child",
        revision: target.revision,
        checkpoints: [{ ...checkpoint("resp_parent"), afterMessageID: "child_boundary" }],
        controls: [{ ...control("child"), messageID: "child_control" }],
      })).toThrow("control write failed")

      expect(f.store.loadAll().filter(({ sessionID }) => sessionID === "child")).toEqual([])
      expect(f.store.loadControlMessages().filter(({ sessionID }) => sessionID === "child")).toEqual([])
      expect(f.store.loadSession("child", 30).revision).toBe(target.revision)
    } finally {
      db.close()
      f.store.close()
      await rm(f.root, { recursive: true, force: true })
    }
  })

  test("rolls back session deletion across both data tables", async () => {
    const f = await fileStore("openai-compact-delete-transaction-")
    const db = openSQLiteDatabase(f.file)
    try {
      f.store.upsert("ses", checkpoint("resp_keep"))
      f.store.upsertControlMessage(control("ses"))
      const before = f.store.loadSession("ses", 30)
      db.exec(`
        create trigger fail_control_delete before delete on control_messages
        when old.session_id = 'ses'
        begin select raise(abort, 'control delete failed'); end;
      `)

      expect(() => f.store.invalidateSessionState("ses", { deleted: true, deleteData: true }))
        .toThrow("control delete failed")
      const after = f.store.loadSession("ses", 30)
      expect(after.deleted).toBe(false)
      expect(after.revision).toBe(before.revision)
      expect(after.checkpoints).toHaveLength(1)
      expect(after.controls).toHaveLength(1)
    } finally {
      db.close()
      f.store.close()
      await rm(f.root, { recursive: true, force: true })
    }
  })

  test("prune updates revisions observed by another connection", async () => {
    const f = await fileStore("openai-compact-prune-revision-")
    const second = await CheckpointStore.open(f.file)
    try {
      f.store.upsert("ses", checkpoint("resp_old", Date.now() - 31 * dayMs))
      const before = second.loadSession("ses", 60)
      expect(before.checkpoints).toHaveLength(1)

      expect(f.store.prune(30)).toEqual(["ses"])
      const after = second.loadSession("ses", 60)
      expect(after.revision).toBeGreaterThan(before.revision)
      expect(after.checkpoints).toEqual([])
    } finally {
      second.close()
      f.store.close()
      await rm(f.root, { recursive: true, force: true })
    }
  })

  test("loadSession enforces retention while the process stays open", () => {
    const store = CheckpointStore.openMemory()
    try {
      store.upsert("ses", checkpoint("resp_old", Date.now() - 31 * dayMs))
      store.upsertControlMessage({ ...control("ses"), createdAt: Date.now() - 31 * dayMs })
      const before = store.loadSession("ses", 60)
      const after = store.loadSession("ses", 30)
      expect(after.revision).toBeGreaterThan(before.revision)
      expect(after.checkpoints).toEqual([])
      expect(after.controls).toEqual([])
    } finally { store.close() }
  })

  test("rolls back a failed version 2 migration and leaves the database reopenable", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "openai-compact-v2-rollback-"))
    const file = path.join(root, "checkpoints.db")
    const db = openSQLiteDatabase(file)
    db.exec(`
      create table checkpoints (
        provider_id text not null, session_id text not null, response_id text not null,
        after_message_id text not null, after_created_at integer not null,
        created_at integer not null, items_json text not null,
        primary key (provider_id, session_id, response_id)
      );
      create table control_messages (
        provider_id text not null, session_id text not null, message_id text not null,
        created_at integer not null, content_text text not null,
        primary key (provider_id, session_id, message_id)
      );
      create table session_state (
        session_id text primary key, revision integer not null,
        deleted integer not null default 0, updated_at integer not null
      );
      create trigger fail_state_migration before insert on session_state
      begin select raise(abort, 'migration state failed'); end;
      insert into checkpoints values (
        'openai', 'ses_v2', 'resp_v2', 'msg_v2', 1, ${Date.now()},
        '[{"type":"compaction","encrypted_content":"v2"}]'
      );
      PRAGMA user_version = 2;
    `)
    db.close()

    try {
      await expect(CheckpointStore.open(file)).rejects.toThrow("migration state failed")
      const reopened = openSQLiteDatabase(file)
      try {
        expect(reopened.query<{ user_version: number }>("PRAGMA user_version").get()?.user_version).toBe(2)
        expect(reopened.query<{ count: number }>("select count(*) as count from session_state").get()?.count).toBe(0)
      } finally { reopened.close() }
    } finally { await rm(root, { recursive: true, force: true }) }
  })

  test("migrates version 2 data into revisioned session state", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "openai-compact-v2-"))
    const file = path.join(root, "checkpoints.db")
    const db = openSQLiteDatabase(file)
    const now = Date.now()
    db.exec(`
      create table checkpoints (
        provider_id text not null, session_id text not null, response_id text not null,
        after_message_id text not null, after_created_at integer not null,
        created_at integer not null, items_json text not null,
        primary key (provider_id, session_id, response_id)
      );
      create table control_messages (
        provider_id text not null, session_id text not null, message_id text not null,
        created_at integer not null, content_text text not null,
        primary key (provider_id, session_id, message_id)
      );
      insert into checkpoints values (
        'openai', 'ses_v2', 'resp_v2', 'msg_v2', 1, ${now},
        '[{"type":"compaction","encrypted_content":"v2"}]'
      );
      insert into control_messages values ('openai', 'ses_v2', 'msg_control', ${now}, 'control');
      PRAGMA user_version = 2;
    `)
    db.close()

    const store = await CheckpointStore.open(file)
    try {
      expect(store.version()).toBe(currentSchemaVersion)
      const state = store.loadSession("ses_v2", 30)
      expect(state.revision).toBe(0)
      expect(state.deleted).toBe(false)
      expect(state.checkpoints.map((item) => item.responseID)).toEqual(["resp_v2"])
      expect(state.controls.map((item) => item.messageID)).toEqual(["msg_control"])
    } finally {
      store.close()
      await rm(root, { recursive: true, force: true })
    }
  })
})
