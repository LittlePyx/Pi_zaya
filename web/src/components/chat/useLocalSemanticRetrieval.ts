import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  retrievalApi,
  type LocalSemanticComponentResponse,
} from '../../api/retrieval'

export type LocalRetrievalMode = 'bm25' | 'hybrid_local'

const PREFERENCE_KEY = 'pi_zaya_local_retrieval_mode_v1'

function readPreference(): LocalRetrievalMode {
  try {
    return window.localStorage.getItem(PREFERENCE_KEY) === 'hybrid_local'
      ? 'hybrid_local'
      : 'bm25'
  } catch {
    return 'bm25'
  }
}

function writePreference(mode: LocalRetrievalMode) {
  try {
    window.localStorage.setItem(PREFERENCE_KEY, mode)
  } catch {
    /* Preference persistence is optional. */
  }
}

export function useLocalSemanticRetrieval() {
  const [status, setStatus] = useState<LocalSemanticComponentResponse | null>(null)
  const [preferredMode, setPreferredModeState] = useState<LocalRetrievalMode>(readPreference)
  const [busy, setBusy] = useState(false)

  const setPreferredMode = useCallback((mode: LocalRetrievalMode) => {
    setPreferredModeState(mode)
    writePreference(mode)
  }, [])

  const refresh = useCallback(async () => {
    const next = await retrievalApi.getLocalSemanticComponent()
    setStatus(next)
    return next
  }, [])

  useEffect(() => {
    let active = true
    void retrievalApi.getLocalSemanticComponent()
      .then((next) => {
        if (active) setStatus(next)
      })
      .catch(() => {
        if (active) setStatus(null)
      })
    return () => {
      active = false
    }
  }, [])

  // Treat incomplete/older API payloads as unavailable instead of taking down
  // the chat surface. BM25 remains the safe default while clients and servers
  // roll forward independently.
  const publicState = status?.hybrid_local?.state || 'unavailable'
  const polling = publicState === 'preparing' || publicState === 'removing'
  useEffect(() => {
    if (!polling) return undefined
    const timer = window.setInterval(() => {
      void refresh().catch(() => {})
    }, 700)
    return () => window.clearInterval(timer)
  }, [polling, refresh])

  const run = useCallback(async (
    operation: () => Promise<LocalSemanticComponentResponse>,
  ) => {
    setBusy(true)
    try {
      const next = await operation()
      setStatus(next)
      return next
    } finally {
      setBusy(false)
    }
  }, [])

  const enable = useCallback(async () => {
    setPreferredMode('hybrid_local')
    return run(retrievalApi.enableLocalSemanticComponent)
  }, [run, setPreferredMode])
  const resume = useCallback(async () => {
    setPreferredMode('hybrid_local')
    return run(retrievalApi.resumeLocalSemanticComponent)
  }, [run, setPreferredMode])
  const cancel = useCallback(async () => {
    const taskId = String(status?.hybrid_local.task_id || '')
    if (!taskId) return status
    setPreferredMode('bm25')
    return run(() => retrievalApi.cancelLocalSemanticComponent(taskId))
  }, [run, setPreferredMode, status])
  const remove = useCallback(async () => {
    setPreferredMode('bm25')
    return run(retrievalApi.removeLocalSemanticComponent)
  }, [run, setPreferredMode])

  return useMemo(() => ({
    status,
    preferredMode,
    busy,
    setPreferredMode,
    enable,
    resume,
    cancel,
    remove,
    refresh,
  }), [
    busy,
    cancel,
    enable,
    preferredMode,
    refresh,
    remove,
    resume,
    setPreferredMode,
    status,
  ])
}
