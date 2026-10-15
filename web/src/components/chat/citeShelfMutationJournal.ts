import { shelfProjectScopeId, shelfStorageKey, type CiteShelfItem } from './citationState'
import { dedupeShelfItems, sameShelfItem, sameShelfItems, shelfPaperIdentity, SHELF_MAX_ITEMS } from './citeShelfRuntime'
import { readShelfStorage, restoreShelfItems, writeShelfStorage } from './citeShelfStorage'
import type { ShelfScopeSnapshot } from './citeShelfPersistence'
import { acknowledgeShelfOperations, readShelfOperations, storeShelfOperation } from './citeShelfJournalEntries'

// Unacknowledged operations, not an empty-cache heuristic or an expiring flag.
export interface ShelfMutationIntent {
  version: 1
  id: string
  projectId: string
  upserts: CiteShelfItem[]
  removedKeys: string[]
  removedIdentities: string[]
  order: string[]
  clear: boolean
  open?: boolean
  createdAt?: number
  operations?: ShelfMutationIntent[]
}

function journalKey(projectId?: string | null): string {
  return `${shelfStorageKey(projectId)}:pending_mutations:v1`
}

function decodeIntent(value: unknown, projectId?: string | null): ShelfMutationIntent | null {
  try {
    const raw = value as ShelfMutationIntent
    if (!raw || raw.version !== 1 || typeof raw.id !== 'string' || !raw.id
      || raw.projectId !== shelfProjectScopeId(projectId)
      || !Array.isArray(raw.upserts) || !Array.isArray(raw.removedKeys) || !Array.isArray(raw.removedIdentities)
      || typeof raw.clear !== 'boolean') return null
    return {
      version: 1, id: raw.id, projectId: raw.projectId,
      upserts: restoreShelfItems(raw.upserts),
      removedKeys: raw.removedKeys.filter((key: unknown): key is string => typeof key === 'string' && !!key),
      removedIdentities: raw.removedIdentities.filter((key: unknown): key is string => typeof key === 'string' && !!key),
      order: Array.isArray(raw.order) ? raw.order.filter((key: unknown): key is string => typeof key === 'string') : [],
      clear: raw.clear, open: typeof raw.open === 'boolean' ? raw.open : undefined,
      createdAt: Number(raw.createdAt || 0),
    }
  } catch {
    // Malformed storage must never authorize clearing backend evidence.
    return null
  }
}

export function readShelfMutationIntent(projectId?: string | null): ShelfMutationIntent | null {
  const key = journalKey(projectId)
  let legacy: ShelfMutationIntent | null = null
  try { legacy = JSON.parse(readShelfStorage(key) || 'null') } catch { /* Ignore corrupt cache. */ }
  const operations = readShelfOperations(key).map((entry) => decodeIntent(entry, projectId))
    .filter((entry): entry is ShelfMutationIntent => !!entry)
    .sort((a, b) => Number(a.createdAt || 0) - Number(b.createdAt || 0) || a.id.localeCompare(b.id))
  if (!operations.length) return legacy?.operations ? null : decodeIntent(legacy, projectId)
  let merged: ShelfMutationIntent = { ...operations[0], upserts: [], removedKeys: [], removedIdentities: [], clear: false }
  for (const op of operations) {
    if (op.clear) merged = { ...merged, upserts: [], removedKeys: [], removedIdentities: [], clear: true }
    merged.upserts = applyShelfMutationIntent({ projectId, open: false, items: merged.upserts }, op).items
    merged.removedKeys = [...new Set([...merged.removedKeys, ...op.removedKeys])].filter((id) => !op.upserts.some((item) => item.key === id))
    merged.removedIdentities = [...new Set([...merged.removedIdentities, ...op.removedIdentities])].filter((id) => !op.upserts.some((item) => shelfPaperIdentity(item) === id))
    merged = { ...merged, id: op.id, order: op.order, open: op.open ?? merged.open }
  }
  return { ...merged, operations }
}

export function recordShelfMutation(previous: ShelfScopeSnapshot, next: ShelfScopeSnapshot, clear = false): void {
  if (shelfStorageKey(previous.projectId) !== shelfStorageKey(next.projectId)) return
  if (!clear && previous.open === next.open && sameShelfItems(previous.items, next.items)) return
  const saved = readShelfMutationIntent(next.projectId)
  if (saved && !saved.operations) storeShelfOperation(journalKey(next.projectId), saved.id, saved)
  const intent: ShelfMutationIntent = {
    version: 1, id: '', projectId: shelfProjectScopeId(next.projectId),
    upserts: [], removedKeys: [], removedIdentities: [], order: [], clear: false,
  }
  if (clear) {
    intent.clear = true
    intent.upserts = []
    intent.removedKeys = []
    intent.removedIdentities = []
  }
  const nextKeys = new Set(next.items.map((item) => item.key))
  for (const item of previous.items) {
    if (nextKeys.has(item.key)) continue
    intent.removedKeys.push(item.key)
    const identity = shelfPaperIdentity(item)
    if (identity) intent.removedIdentities.push(identity)
    intent.upserts = intent.upserts.filter((entry) => entry.key !== item.key)
  }
  for (const item of next.items) {
    const old = previous.items.find((entry) => entry.key === item.key)
    if (!clear && old && sameShelfItem(old, item)) continue
    const identity = shelfPaperIdentity(item)
    intent.upserts = [item, ...intent.upserts.filter((entry) => entry.key !== item.key && shelfPaperIdentity(entry) !== identity)]
    intent.removedKeys = intent.removedKeys.filter((key) => key !== item.key)
    intent.removedIdentities = intent.removedIdentities.filter((key) => key !== identity)
  }
  intent.id = crypto.randomUUID()
  intent.createdAt = Math.max(Date.now(), ...(saved?.operations || []).map((entry) => Number(entry.createdAt || 0) + 1))
  intent.order = next.items.map((item) => item.key)
  intent.removedKeys = [...new Set(intent.removedKeys)]
  intent.removedIdentities = [...new Set(intent.removedIdentities)]
  if (previous.open !== next.open || clear) intent.open = next.open
  storeShelfOperation(journalKey(next.projectId), intent.id, intent)
  writeShelfStorage(journalKey(next.projectId), JSON.stringify({ ...intent, operations: [intent] }))
}

export function applyShelfMutationIntent(state: ShelfScopeSnapshot, intent: ShelfMutationIntent | null): ShelfScopeSnapshot {
  if (!intent || intent.projectId !== shelfProjectScopeId(state.projectId)) return state
  const removedKeys = new Set(intent.removedKeys)
  const removedIdentities = new Set(intent.removedIdentities)
  const upsertKeys = new Set(intent.upserts.map((item) => item.key))
  const upsertIdentities = new Set(intent.upserts.map(shelfPaperIdentity).filter(Boolean))
  const retained = intent.clear ? [] : state.items.filter((item) => (
    !removedKeys.has(item.key) && !removedIdentities.has(shelfPaperIdentity(item))
    && !upsertKeys.has(item.key) && !upsertIdentities.has(shelfPaperIdentity(item))
  ))
  const order = new Map(intent.order.map((key, index) => [key, index]))
  const items = dedupeShelfItems([...intent.upserts, ...retained])
    .sort((a, b) => (order.get(a.key) ?? SHELF_MAX_ITEMS) - (order.get(b.key) ?? SHELF_MAX_ITEMS))
    .slice(0, SHELF_MAX_ITEMS)
  return { ...state, open: intent.open ?? state.open, items }
}

export function acknowledgeShelfMutation(projectId: string | null | undefined, id: string, operations?: ShelfMutationIntent[]): void {
  const current = readShelfMutationIntent(projectId)
  const acknowledged = operations || (current?.id === id ? current.operations : undefined)
  if (acknowledged) acknowledgeShelfOperations(journalKey(projectId), acknowledged.map((entry) => entry.id))
  if (current?.id === id) writeShelfStorage(journalKey(projectId), '')
}

function comparableStorageValue(value: unknown): unknown {
  // The shelf store compacts whitespace and omits empty optional JSON fields.
  // Normalize only for acknowledgement; keep the user's local formatting intact.
  const present = (entry: unknown) => entry !== null && entry !== undefined && entry !== ''
    && !(typeof entry === 'object' && Object.keys(entry).length === 0)
  if (typeof value === 'string') return value.replace(/\0/g, ' ').replace(/\s+/g, ' ').trim()
  if (Array.isArray(value)) return value.map(comparableStorageValue).filter(present)
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
    .map(([key, entry]) => [key, comparableStorageValue(entry)]).filter(([, entry]) => present(entry)))
  return value
}

export function shelfMutationWasApplied(stored: ShelfScopeSnapshot, intent: ShelfMutationIntent): boolean {
  const replay = applyShelfMutationIntent(stored, intent)
  const comparableItems = (items: CiteShelfItem[]) => restoreShelfItems(
    items.map((item) => comparableStorageValue(item) as Record<string, unknown>),
  )
  return stored.open === replay.open && sameShelfItems(comparableItems(stored.items), comparableItems(replay.items))
}
