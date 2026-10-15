import type { CitationShelfRecord, CitationShelfSaveBody } from '../../api/chat'
import { shelfProjectScopeId, shelfStorageKey, type CiteShelfItem } from './citationState'
import { dedupeShelfItems, sameShelfItems, shelfItemsForBackend, shelfPaperIdentity, SHELF_MAX_ITEMS } from './citeShelfRuntime'
import { restoreShelfItems } from './citeShelfStorage'
import { applyShelfMutationIntent, type ShelfMutationIntent } from './citeShelfMutationJournal'

// Project identity travels with the snapshot, never with a later render's props.
export interface ShelfScopeSnapshot {
  convId?: string | null
  projectId?: string | null
  open: boolean
  items: CiteShelfItem[]
}

export interface ShelfHydrationRequest {
  storageKey: string
  sequence: number
  snapshot: ShelfScopeSnapshot
  clearRevision: number
}

export interface ShelfPendingSave {
  storageKey: string
  body: CitationShelfSaveBody
}

export function shelfHydrationIsCurrent(request: ShelfHydrationRequest, sequence: number, current: ShelfScopeSnapshot): boolean {
  return request.sequence === sequence && request.storageKey === shelfStorageKey(current.projectId)
}

export function buildShelfPendingSave(
  state: ShelfScopeSnapshot,
  options: { hydrated: boolean; backendRevision: number; allowEmptyOverwrite: boolean },
): ShelfPendingSave | null {
  if (!options.hydrated) return null
  const items = shelfItemsForBackend(state.items)
  if (!items.length && !options.allowEmptyOverwrite && options.backendRevision > 0) return null
  const projectId = shelfProjectScopeId(state.projectId)
  return {
    storageKey: shelfStorageKey(projectId),
    body: { convId: state.convId || undefined, projectId: projectId === '__default__' ? undefined : projectId,
      scope: 'project', open: state.open, items, allowEmptyOverwrite: options.allowEmptyOverwrite,
      expectedRevision: options.backendRevision },
  }
}

// The request-start snapshot distinguishes an untouched empty cache from a real
// deletion. Wall-clock tolerances cannot reliably distinguish those transitions.
export function reconcileShelfHydration(options: {
  request: ShelfHydrationRequest
  current: ShelfScopeSnapshot
  record: CitationShelfRecord
  localUpdatedAt: number
  clearRevision: number
  intent?: ShelfMutationIntent | null
}): { snapshot: ShelfScopeSnapshot; shouldSaveBackend: boolean; allowEmptyOverwrite: boolean } {
  const { request, current, record, localUpdatedAt } = options
  const backendItems = restoreShelfItems(Array.isArray(record.items) ? record.items : [])
  const currentItems = dedupeShelfItems(current.items).slice(0, SHELF_MAX_ITEMS)
  const cleared = options.clearRevision !== request.clearRevision
  const currentKeys = new Set(currentItems.map((item) => item.key))
  const removed = request.snapshot.items.filter((item) => !currentKeys.has(item.key))
  const removedKeys = new Set(removed.map((item) => item.key))
  const removedIdentities = new Set(removed.map(shelfPaperIdentity).filter(Boolean))
  const changed = cleared || !sameShelfItems(request.snapshot.items, currentItems) || request.snapshot.open !== current.open
  const rawUpdatedAt = Number(record.updated_at || 0)
  const backendUpdatedAt = rawUpdatedAt > 1e12 ? rawUpdatedAt : rawUpdatedAt * 1000
  let items: CiteShelfItem[]
  let open = Boolean(record.open)
  let shouldSaveBackend = false
  if (record.concurrency_guarded && !changed) {
    // Server revisions override cache timestamps; only retained operations replay.
    items = backendItems
  } else if (changed) {
    const retainedBackend = cleared ? [] : backendItems.filter((item) => (
      !removedKeys.has(item.key) && !removedIdentities.has(shelfPaperIdentity(item))
    ))
    items = dedupeShelfItems([...currentItems, ...retainedBackend]).slice(0, SHELF_MAX_ITEMS)
    open = current.open
    shouldSaveBackend = !sameShelfItems(items, backendItems) || open !== Boolean(record.open)
  } else if (Number(record.revision || 0) <= 0) {
    items = currentItems.length ? currentItems : backendItems
    open = current.open || Boolean(record.open)
    shouldSaveBackend = items.length > 0 || open
  } else if (!backendItems.length && currentItems.length) {
    const localNewer = localUpdatedAt > backendUpdatedAt + 500
    items = localNewer ? currentItems : []
    open = localNewer ? current.open : Boolean(record.open)
    shouldSaveBackend = localNewer
  } else if (!currentItems.length || backendUpdatedAt > localUpdatedAt + 500) {
    items = backendItems
  } else {
    items = dedupeShelfItems([...currentItems, ...backendItems]).slice(0, SHELF_MAX_ITEMS)
    open = current.open || Boolean(record.open)
    shouldSaveBackend = !sameShelfItems(items, backendItems) || open !== Boolean(record.open)
  }
  const snapshot = applyShelfMutationIntent({ ...current, open, items }, options.intent || null)
  return { snapshot, shouldSaveBackend: shouldSaveBackend || !!options.intent,
    allowEmptyOverwrite: !snapshot.items.length && (cleared || removed.length > 0
      || !!options.intent?.clear || !!options.intent?.removedKeys.length) }
}
