import { chatApi, type CitationShelfRecord, type CitationShelfRequest, type CitationShelfSaveBody } from '../../api/chat'
import { shelfStorageKey } from './citationState'
import { shelfItemsForBackend } from './citeShelfRuntime'
import { restoreShelfItems } from './citeShelfStorage'
import { acknowledgeShelfMutation, applyShelfMutationIntent, readShelfMutationIntent, shelfMutationWasApplied, type ShelfMutationIntent } from './citeShelfMutationJournal'
import type { ShelfPendingSave, ShelfScopeSnapshot } from './citeShelfPersistence'

interface SaveJob {
  pending: ShelfPendingSave
  intent: ShelfMutationIntent | null
  onSaved: (record: CitationShelfRecord) => void
}

export function shelfOperationPayload(intent: ShelfMutationIntent | null): Array<Record<string, unknown>> | undefined {
  return intent ? (intent.operations || [intent]).map((op) => ({ id: op.id, clear: op.clear,
    upserts: shelfItemsForBackend(op.upserts), removedKeys: op.removedKeys,
    removedIdentities: op.removedIdentities, order: op.order, ...(op.open === undefined ? {} : { open: op.open }) })) : undefined
}

export async function persistShelfWrite(body: CitationShelfSaveBody, intent: ShelfMutationIntent | null): Promise<CitationShelfRecord> {
  const payload = body.mutations || []
  if (payload.length <= 120 && JSON.stringify(payload).length <= 200_000) return chatApi.saveCitationShelf(body)
  // A long offline journal must drain without increasing API body limits.
  let record: CitationShelfRecord | undefined
  for (let start = 0; start < payload.length;) {
    const batch: Array<Record<string, unknown>> = []
    let size = 2
    while (start < payload.length && batch.length < 120) {
      const op = payload[start]
      const length = JSON.stringify(op).length + 1
      if (batch.length && size + length > 200_000) break
      batch.push(op); size += length; start += 1
    }
    const id = String(batch[batch.length - 1].id)
    record = await chatApi.saveCitationShelf({ ...body, items: [], mutations: batch, mutationId: id,
      expectedRevision: record?.revision ?? body.expectedRevision, allowEmptyOverwrite: true })
    if (record.acknowledged_mutation_id !== id) throw new Error('Shelf operation batch not acknowledged')
    acknowledgeShelfMutation(body.projectId, id, intent?.operations?.filter((op) => batch.some((entry) => entry.id === op.id)))
  }
  return record!
}

// Workspace entry points use the same operation protocol; no blind snapshot writes.
export async function saveShelfForWorkspace(state: ShelfScopeSnapshot): Promise<CitationShelfRecord> {
  const intent = readShelfMutationIntent(state.projectId)
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const current = await chatApi.getCitationShelf(state)
    if (!intent) return current
    const applied = applyShelfMutationIntent({ ...state, open: current.open, items: restoreShelfItems(current.items) }, intent)
    try {
      const result = await persistShelfWrite({ ...state, items: shelfItemsForBackend(applied.items),
        open: applied.open, expectedRevision: current.revision, mutationId: intent.id,
        mutations: shelfOperationPayload(intent), allowEmptyOverwrite: true }, intent)
      if (result.acknowledged_mutation_id !== intent.id && !shelfMutationWasApplied({ ...state,
        open: result.open, items: restoreShelfItems(result.items) }, intent)) throw new Error('Shelf save not acknowledged')
      acknowledgeShelfMutation(state.projectId, intent.id, intent.operations)
      return result
    } catch (error) { if (attempt === 3) throw error }
  }
  throw new Error('Shelf save failed')
}
interface SaveLane {
  active?: SaveJob
  next?: SaveJob
  retry?: SaveJob
  failures: number
  timer?: number
}

const lanes = new Map<string, SaveLane>()
const RETRY_DELAYS_MS = [1000, 2000, 4000, 8000, 16000]
let listening = false

function snapshot(job: SaveJob, record?: CitationShelfRecord): ShelfScopeSnapshot {
  const body = job.pending.body
  return { convId: body.convId, projectId: body.projectId,
    open: Boolean(record ? record.open : body.open), items: restoreShelfItems(record ? record.items : body.items) }
}

function sameJob(a: SaveJob | undefined, b: SaveJob): boolean {
  return !!a && a.intent?.id === b.intent?.id && JSON.stringify(a.pending.body) === JSON.stringify(b.pending.body)
}

function resume() {
  for (const [key, lane] of lanes) {
    if (!lane.retry || lane.active) continue
    if (lane.timer !== undefined) window.clearTimeout(lane.timer)
    lane.timer = undefined
    lane.failures = 0
    void run(key, lane)
  }
}

function maintainListeners() {
  if (lanes.size > 0 && !listening) {
    window.addEventListener('online', resume)
    window.addEventListener('focus', resume)
    listening = true
  } else if (!lanes.size && listening) {
    window.removeEventListener('online', resume)
    window.removeEventListener('focus', resume)
    listening = false
  }
}

async function run(key: string, lane: SaveLane): Promise<void> {
  if (lane.active) return
  const job = lane.next || lane.retry
  if (!job) return
  const retrying = !!lane.retry
  lane.next = undefined
  lane.retry = undefined
  lane.active = job
  try {
    let state = snapshot(job)
    let expectedRevision = job.pending.body.expectedRevision
    if (retrying) {
      // Refresh before replay so an outage cannot erase backend-only evidence.
      const record = await chatApi.getCitationShelf(job.pending.body)
      state = snapshot(job, record)
      expectedRevision = record.revision
    }
    state = applyShelfMutationIntent(state, job.intent)
    const record = await persistShelfWrite({ ...job.pending.body, open: state.open,
      items: shelfItemsForBackend(state.items), expectedRevision, mutationId: job.intent?.id,
      mutations: shelfOperationPayload(job.intent) }, job.intent)
    if (job.intent) {
      const stored = snapshot(job, record)
      if (record.acknowledged_mutation_id !== job.intent.id && !shelfMutationWasApplied(stored, job.intent)) {
        throw new Error('Shelf mutation was not acknowledged by storage')
      }
      acknowledgeShelfMutation(job.pending.body.projectId, job.intent.id, job.intent.operations)
    }
    job.onSaved(record)
    lane.failures = 0
  } catch {
    // Only actual retained operations merit retries, never an incidental empty cache.
    const newer = lanes.get(key)?.next
    if (job.intent || newer?.intent) {
      lane.retry = newer || job
      lane.next = undefined
      const delay = RETRY_DELAYS_MS[lane.failures++]
      if (delay !== undefined) lane.timer = window.setTimeout(() => {
        lane.timer = undefined
        void run(key, lane)
      }, delay)
    }
  } finally {
    lane.active = undefined
    if (lane.next && !lane.retry) void run(key, lane)
    else if (!lane.retry) {
      lanes.delete(key)
      maintainListeners()
    }
  }
}

// One lane per project for this document, including unmount/remount. Clearing
// uses an explicit empty PATCH in this same lane, never a racing DELETE request.
export function enqueueShelfSave(pending: ShelfPendingSave, onSaved: SaveJob['onSaved']): void {
  const key = shelfStorageKey(pending.body.projectId)
  const job = { pending, onSaved, intent: readShelfMutationIntent(pending.body.projectId) }
  const lane = lanes.get(key) || { failures: 0 }
  if (sameJob(lane.next || lane.active, job)) return
  lane.next = job
  if (lane.timer !== undefined) window.clearTimeout(lane.timer)
  lane.timer = undefined
  lanes.set(key, lane)
  maintainListeners()
  void run(key, lane)
}

export function loadShelfWithRetry(
  request: CitationShelfRequest,
  onRecord: (record: CitationShelfRecord) => void,
  onFailure: () => void,
): () => void {
  let stopped = false
  let active = false
  let failures = 0
  let timer: number | undefined
  const stop = () => {
    stopped = true
    if (timer !== undefined) window.clearTimeout(timer)
    window.removeEventListener('online', wake)
    window.removeEventListener('focus', wake)
  }
  const load = async () => {
    if (stopped || active) return
    active = true
    try {
      const record = await chatApi.getCitationShelf(request)
      if (stopped) return
      onRecord(record)
      stop()
    } catch {
      if (stopped) return
      onFailure()
      const delay = RETRY_DELAYS_MS[failures++]
      if (delay !== undefined) timer = window.setTimeout(() => { void load() }, delay)
    } finally {
      active = false
    }
  }
  function wake() {
    if (stopped || active) return
    if (timer !== undefined) window.clearTimeout(timer)
    failures = 0
    void load()
  }
  window.addEventListener('online', wake)
  window.addEventListener('focus', wake)
  timer = window.setTimeout(() => { void load() }, 0)
  return stop
}
