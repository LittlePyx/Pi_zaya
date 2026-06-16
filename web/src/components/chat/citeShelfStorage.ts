import {
  legacyConversationShelfStorageKey,
  normalizeCiteDetail,
  normalizeShelfNote,
  normalizeShelfTags,
  shelfStorageKey,
  toShelfItem,
  type CiteShelfItem,
} from './citationState'
import {
  SHELF_MAX_ITEMS,
  dedupeShelfItems,
  sameShelfItems,
  shelfPaperIdentity,
} from './citeShelfRuntime'

const SHELF_SCHEMA_VERSION = 4
const SHELF_SAVED_SCHEMA_VERSION = 1
export const SHELF_SAVED_MAX_ITEMS = 16
export const SHELF_SAVED_SUFFIX = ':saved_snapshots'

const shelfStorageFallback = new Map<string, string>()
export interface ShelfSnapshot {
  version: number
  revision: number
  updatedAt: number
  open: boolean
  items: CiteShelfItem[]
}
export interface ShelfSavedSnapshot {
  id: string
  name: string
  createdAt: number
  items: CiteShelfItem[]
}
interface ShelfSavedSnapshotPayload {
  version: number
  updatedAt: number
  snapshots: ShelfSavedSnapshot[]
}
const shelfSnapshotMemory = new Map<string, ShelfSnapshot>()
const shelfSavedSnapshotMemory = new Map<string, ShelfSavedSnapshotPayload>()

function cloneShelfSnapshot(snapshot: ShelfSnapshot): ShelfSnapshot {
  return {
    version: Number(snapshot.version || 0),
    revision: Number(snapshot.revision || 0),
    updatedAt: Number(snapshot.updatedAt || 0),
    open: Boolean(snapshot.open),
    items: (snapshot.items || []).map((item) => ({ ...item })),
  }
}

function cloneSavedSnapshot(snapshot: ShelfSavedSnapshot): ShelfSavedSnapshot {
  return {
    id: String(snapshot.id || ''),
    name: String(snapshot.name || ''),
    createdAt: Number(snapshot.createdAt || 0),
    items: (snapshot.items || []).map((item) => ({ ...item })),
  }
}

function cloneSavedSnapshotPayload(payload: ShelfSavedSnapshotPayload): ShelfSavedSnapshotPayload {
  return {
    version: Number(payload.version || 0),
    updatedAt: Number(payload.updatedAt || 0),
    snapshots: (payload.snapshots || []).map((entry) => cloneSavedSnapshot(entry)),
  }
}

function listShelfStorages(): Storage[] {
  const out: Storage[] = []
  try {
    out.push(window.localStorage)
  } catch {
    // ignore
  }
  try {
    if (!out.includes(window.sessionStorage)) out.push(window.sessionStorage)
  } catch {
    // ignore
  }
  return out
}

function readShelfStorage(key: string): string {
  for (const storage of listShelfStorages()) {
    try {
      const raw = storage.getItem(key)
      if (typeof raw === 'string') {
        shelfStorageFallback.set(key, raw)
        return raw
      }
    } catch {
      // ignore
    }
  }
  return shelfStorageFallback.get(key) || ''
}

function writeShelfStorage(key: string, payload: string) {
  // Always keep in-memory raw snapshot first to survive temporary storage failures.
  shelfStorageFallback.set(key, payload)
  let wrote = false
  for (const storage of listShelfStorages()) {
    try {
      storage.setItem(key, payload)
      wrote = true
    } catch {
      // ignore
    }
  }
  if (!wrote) return
}

function removeShelfStorage(key: string) {
  shelfSnapshotMemory.delete(key)
  for (const storage of listShelfStorages()) {
    try {
      storage.removeItem(key)
    } catch {
      // ignore
    }
  }
  shelfStorageFallback.delete(key)
}

export function shelfSavedStorageKey(projectId?: string | null): string {
  return `${shelfStorageKey(projectId)}${SHELF_SAVED_SUFFIX}`
}

export function legacyShelfStorageKeys(convId?: string | null): string[] {
  const out = new Set<string>()
  const cid = String(convId || '').trim()
  if (cid) out.add(legacyConversationShelfStorageKey(cid))
  else out.add(legacyConversationShelfStorageKey(null))
  return Array.from(out)
}

export function migrateLegacyShelfSnapshot(storageKey: string, legacyKeys: string[]): ShelfSnapshot | null {
  const current = readShelfSnapshot(storageKey)
  const legacySnapshots = legacyKeys
    .filter((key) => key && key !== storageKey)
    .map((key) => ({ key, snapshot: readShelfSnapshot(key) }))
    .filter((entry): entry is { key: string; snapshot: ShelfSnapshot } => Boolean(entry.snapshot))
  if (legacySnapshots.length <= 0) return current

  const nextItems = dedupeShelfItems([
    ...(current?.items || []),
    ...legacySnapshots.flatMap((entry) => entry.snapshot.items || []),
  ]).slice(0, SHELF_MAX_ITEMS)
  const nextOpen = Boolean(current?.open || legacySnapshots.some((entry) => entry.snapshot.open))
  const currentRevision = Number(current?.revision || 0)
  persistShelfSnapshot(storageKey, { open: nextOpen, items: nextItems }, currentRevision)
  for (const entry of legacySnapshots) removeShelfStorage(entry.key)
  return readShelfSnapshot(storageKey)
}

function mergeSavedShelfSnapshots(current: ShelfSavedSnapshot[], incoming: ShelfSavedSnapshot[]): ShelfSavedSnapshot[] {
  const seen = new Set<string>()
  const out: ShelfSavedSnapshot[] = []
  for (const entry of [...current, ...incoming]) {
    const id = String(entry.id || '').trim()
    if (!id || seen.has(id)) continue
    out.push(cloneSavedSnapshot(entry))
    seen.add(id)
    if (out.length >= SHELF_SAVED_MAX_ITEMS) break
  }
  return out
}

export function migrateLegacySavedShelfSnapshots(storageKey: string, legacyKeys: string[]): ShelfSavedSnapshot[] {
  const current = readSavedShelfSnapshots(storageKey)
  const legacySnapshots = legacyKeys
    .filter((key) => key && key !== storageKey)
    .flatMap((key) => readSavedShelfSnapshots(key))
  if (legacySnapshots.length <= 0) return current
  const merged = mergeSavedShelfSnapshots(current, legacySnapshots)
  if (merged.length > 0) persistSavedShelfSnapshots(storageKey, merged)
  for (const key of legacyKeys) {
    if (key && key !== storageKey) removeSavedShelfStorage(key)
  }
  return merged
}

export function restoreShelfItems(rawItems: unknown[]): CiteShelfItem[] {
  const seen = new Set<string>()
  const seenIdentity = new Set<string>()
  const out: CiteShelfItem[] = []
  for (const raw of rawItems) {
    if (!raw || typeof raw !== 'object') continue
    const rec = raw as Record<string, unknown>
    const detail = normalizeCiteDetail(rec)
    if (!detail) continue
    const base = toShelfItem(detail)
    const key = String(rec.key || '').trim() || base.key
    const main = String(rec.main || '').trim() || base.main
    if (!key || seen.has(key)) continue
    const identity = shelfPaperIdentity({ ...base, key, main })
    if (seenIdentity.has(identity)) continue
    seen.add(key)
    seenIdentity.add(identity)
    out.push({
      ...base,
      key,
      main,
      tags: normalizeShelfTags(rec.tags),
      note: normalizeShelfNote(rec.note),
    })
    if (out.length >= SHELF_MAX_ITEMS) break
  }
  return out
}

export function readShelfSnapshot(key: string, rawOverride?: string): ShelfSnapshot | null {
  if (typeof rawOverride !== 'string') {
    const mem = shelfSnapshotMemory.get(key)
    if (mem) return cloneShelfSnapshot(mem)
  }
  const raw = typeof rawOverride === 'string' ? rawOverride : readShelfStorage(key)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw)
    const itemsRaw: unknown[] = Array.isArray(parsed?.items) ? parsed.items : []
    const revision0 = Number(parsed?.revision || 0)
    const updatedAt0 = Number(parsed?.updatedAt || 0)
    const snapshot: ShelfSnapshot = {
      version: Number(parsed?.version || 0) || 0,
      revision: Number.isFinite(revision0) && revision0 > 0 ? Math.floor(revision0) : 0,
      updatedAt: Number.isFinite(updatedAt0) && updatedAt0 > 0 ? Math.floor(updatedAt0) : 0,
      open: Boolean(parsed?.open),
      items: restoreShelfItems(itemsRaw),
    }
    shelfSnapshotMemory.set(key, snapshot)
    return cloneShelfSnapshot(snapshot)
  } catch {
    // Corrupted payload: keep running and clear stale bad data.
    shelfSnapshotMemory.delete(key)
    removeShelfStorage(key)
    return null
  }
}

function removeSavedShelfStorage(key: string) {
  shelfSavedSnapshotMemory.delete(key)
  for (const storage of listShelfStorages()) {
    try {
      storage.removeItem(key)
    } catch {
      // ignore
    }
  }
  shelfStorageFallback.delete(key)
}

export function readSavedShelfSnapshots(storageKey: string, rawOverride?: string): ShelfSavedSnapshot[] {
  if (typeof rawOverride !== 'string') {
    const mem = shelfSavedSnapshotMemory.get(storageKey)
    if (mem) return cloneSavedSnapshotPayload(mem).snapshots
  }
  const raw = typeof rawOverride === 'string' ? rawOverride : readShelfStorage(storageKey)
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    const snapshotsRaw: unknown[] = Array.isArray(parsed?.snapshots) ? parsed.snapshots : []
    const seen = new Set<string>()
    const snapshots: ShelfSavedSnapshot[] = []
    for (const rawItem of snapshotsRaw) {
      if (!rawItem || typeof rawItem !== 'object') continue
      const rec = rawItem as Record<string, unknown>
      const id = String(rec.id || '').trim()
      if (!id || seen.has(id)) continue
      const createdAt0 = Number(rec.createdAt || 0)
      const createdAt = Number.isFinite(createdAt0) && createdAt0 > 0 ? Math.floor(createdAt0) : Date.now()
      const name = String(rec.name || '').trim() || 'Untitled snapshot'
      const itemsRaw: unknown[] = Array.isArray(rec.items) ? rec.items : []
      snapshots.push({
        id,
        name,
        createdAt,
        items: restoreShelfItems(itemsRaw),
      })
      seen.add(id)
      if (snapshots.length >= SHELF_SAVED_MAX_ITEMS) break
    }
    const payload: ShelfSavedSnapshotPayload = {
      version: Number(parsed?.version || 0) || 0,
      updatedAt: Number(parsed?.updatedAt || 0) || 0,
      snapshots,
    }
    shelfSavedSnapshotMemory.set(storageKey, payload)
    return cloneSavedSnapshotPayload(payload).snapshots
  } catch {
    removeSavedShelfStorage(storageKey)
    return []
  }
}

export function persistSavedShelfSnapshots(storageKey: string, snapshots: ShelfSavedSnapshot[]) {
  if (!Array.isArray(snapshots) || snapshots.length <= 0) {
    removeSavedShelfStorage(storageKey)
    return
  }
  const normalized = snapshots
    .slice(0, SHELF_SAVED_MAX_ITEMS)
    .map((entry) => ({
      id: String(entry.id || '').trim(),
      name: String(entry.name || '').trim() || 'Untitled snapshot',
      createdAt: Number(entry.createdAt || 0) > 0 ? Number(entry.createdAt) : Date.now(),
      items: dedupeShelfItems(entry.items || []).slice(0, SHELF_MAX_ITEMS).map((item) => ({ ...item })),
    }))
    .filter((entry) => Boolean(entry.id))
  if (normalized.length <= 0) {
    removeSavedShelfStorage(storageKey)
    return
  }
  const payload: ShelfSavedSnapshotPayload = {
    version: SHELF_SAVED_SCHEMA_VERSION,
    updatedAt: Date.now(),
    snapshots: normalized,
  }
  shelfSavedSnapshotMemory.set(storageKey, cloneSavedSnapshotPayload(payload))
  writeShelfStorage(storageKey, JSON.stringify(payload))
}

export function persistShelfSnapshot(
  storageKey: string,
  payload: { open: boolean; items: CiteShelfItem[] },
  currentRevision: number,
): number {
  const normalizedItems = payload.items.slice(0, SHELF_MAX_ITEMS)
  const existing = readShelfSnapshot(storageKey)
  if (existing && existing.open === payload.open && sameShelfItems(existing.items, normalizedItems)) {
    return Math.max(currentRevision, existing.revision)
  }
  const nextRevision = Math.max(currentRevision, existing?.revision || 0) + 1
  const snapshot: ShelfSnapshot = {
    version: SHELF_SCHEMA_VERSION,
    revision: nextRevision,
    updatedAt: Date.now(),
    open: payload.open,
    items: normalizedItems,
  }
  shelfSnapshotMemory.set(storageKey, cloneShelfSnapshot(snapshot))
  const raw = JSON.stringify(snapshot)
  writeShelfStorage(storageKey, raw)
  return nextRevision
}

export function invalidateShelfSnapshotCache(key: string) {
  shelfSnapshotMemory.delete(key)
  shelfStorageFallback.delete(key)
}

export function invalidateSavedShelfSnapshotCache(key: string) {
  shelfSavedSnapshotMemory.delete(key)
  shelfStorageFallback.delete(key)
}
