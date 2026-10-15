import { useEffect, useRef, useState } from 'react'
import {
  referencesApi,
  type ShelfReaderLocatorMaintenanceItem,
  type ShelfReaderLocatorMaintenanceResult,
} from '../../api/references'
import type { CiteShelfItem } from './citationState'
import {
  shelfReaderLocatorMaintenanceFingerprint,
  shelfReaderLocatorMaintenanceInput,
  shelfReaderLocatorSourceKey,
} from './shelfReaderLocatorMaintenance'


const START_DELAY_MS = 320
const SOURCE_CONCURRENCY = 2
const FAILED_REQUEST_RETRY_MS = 60_000

export interface ShelfReaderLocatorMaintenanceUpdate {
  expectedFingerprint: string
  result: ShelfReaderLocatorMaintenanceResult
}

interface UseShelfReaderLocatorMaintenanceOptions {
  items: CiteShelfItem[]
  scopeId: string
  sourceQualityRefreshToken: number
  onApply: (updates: ShelfReaderLocatorMaintenanceUpdate[]) => void
}

interface SourceGroup {
  sourceKey: string
  inputs: ShelfReaderLocatorMaintenanceItem[]
  fingerprints: Map<string, string>
}

async function mapWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  mapper: (item: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length)
  let nextIndex = 0
  const worker = async () => {
    while (nextIndex < items.length) {
      const index = nextIndex
      nextIndex += 1
      results[index] = await mapper(items[index])
    }
  }
  await Promise.all(Array.from(
    { length: Math.min(Math.max(1, concurrency), items.length) },
    () => worker(),
  ))
  return results
}

function buildSourceGroups(items: CiteShelfItem[], attempted: Set<string>, attemptPrefix: string): SourceGroup[] {
  const groups = new Map<string, SourceGroup>()
  for (const item of items) {
    const input = shelfReaderLocatorMaintenanceInput(item)
    const fingerprint = shelfReaderLocatorMaintenanceFingerprint(item)
    if (!input || !fingerprint) continue
    const attemptKey = `${attemptPrefix}|${fingerprint}`
    if (attempted.has(attemptKey)) continue
    const sourceKey = shelfReaderLocatorSourceKey(input)
    if (!sourceKey) continue
    const group = groups.get(sourceKey) || {
      sourceKey,
      inputs: [],
      fingerprints: new Map<string, string>(),
    }
    group.inputs.push(input)
    group.fingerprints.set(input.key, fingerprint)
    groups.set(sourceKey, group)
  }
  return Array.from(groups.values())
}

export function useShelfReaderLocatorMaintenance({
  items,
  scopeId,
  sourceQualityRefreshToken,
  onApply,
}: UseShelfReaderLocatorMaintenanceOptions): boolean {
  const [maintaining, setMaintaining] = useState(false)
  const [retryCycle, setRetryCycle] = useState(0)
  const attemptedRef = useRef(new Set<string>())
  const timerRef = useRef<number | null>(null)
  const retryTimerRef = useRef<number | null>(null)

  useEffect(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current)
      timerRef.current = null
    }
    const attempted = attemptedRef.current
    const attemptPrefix = `${scopeId}|${sourceQualityRefreshToken}`
    const groups = buildSourceGroups(items, attempted, attemptPrefix)
    if (groups.length <= 0) return
    const ctrl = new AbortController()
    let started = false
    let completed = false
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null
      started = true
      for (const group of groups) {
        for (const fingerprint of group.fingerprints.values()) {
          attempted.add(`${attemptPrefix}|${fingerprint}`)
        }
      }
      setMaintaining(true)
      const failedGroups: SourceGroup[] = []
      void mapWithConcurrency(groups, SOURCE_CONCURRENCY, async (group) => {
        try {
          const response = await referencesApi.maintainShelfReaderLocators(group.inputs, { signal: ctrl.signal })
          return (response.items || []).map((result) => ({
            expectedFingerprint: group.fingerprints.get(result.key) || '',
            result,
          }))
        } catch {
          if (!ctrl.signal.aborted) failedGroups.push(group)
          return []
        }
      }).then((batches) => {
        if (ctrl.signal.aborted) return
        const updates = batches.flat().filter((update) => update.expectedFingerprint)
        if (updates.length > 0) onApply(updates)
        if (failedGroups.length > 0) {
          for (const group of failedGroups) {
            for (const fingerprint of group.fingerprints.values()) {
              attempted.delete(`${attemptPrefix}|${fingerprint}`)
            }
          }
          if (retryTimerRef.current !== null) window.clearTimeout(retryTimerRef.current)
          retryTimerRef.current = window.setTimeout(() => {
            retryTimerRef.current = null
            setRetryCycle((value) => value + 1)
          }, FAILED_REQUEST_RETRY_MS)
        }
      }).finally(() => {
        completed = true
        if (!ctrl.signal.aborted) setMaintaining(false)
      })
    }, START_DELAY_MS)
    return () => {
      ctrl.abort()
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
        timerRef.current = null
      }
      if (started && !completed) {
        for (const group of groups) {
          for (const fingerprint of group.fingerprints.values()) {
            attempted.delete(`${attemptPrefix}|${fingerprint}`)
          }
        }
      }
      setMaintaining(false)
    }
  }, [items, onApply, retryCycle, scopeId, sourceQualityRefreshToken])

  useEffect(() => () => {
    if (retryTimerRef.current !== null) window.clearTimeout(retryTimerRef.current)
  }, [])

  return maintaining
}
