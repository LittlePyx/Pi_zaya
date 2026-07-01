import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { chatApi } from '../../../api/chat'
import { sameHighlightTarget } from './readerDomUtils'
import { normalizeReaderSourcePathForMatch } from './readerLocateGuard'
import { readerHighlightsSignature } from './readerSessionState'
import {
  READER_SESSION_SYNC_CHANNEL,
  type ReaderOpenPayload,
  type ReaderSessionHighlight,
} from './readerTypes'

function readerHighlightScopeKey(convId: string | null | undefined, sourcePath: string) {
  const path = normalizeReaderSourcePathForMatch(sourcePath)
  if (!path) return ''
  const conv = String(convId || '__detached__').trim().toLowerCase()
  return `${conv}::${path}`
}

function sameReaderSessionHighlight(
  left: Pick<ReaderSessionHighlight, 'text' | 'startOffset' | 'endOffset' | 'blockId' | 'anchorId' | 'occurrence' | 'readableIndex' | 'documentOccurrence' | 'startReadableIndex' | 'endReadableIndex'>,
  right: Pick<ReaderSessionHighlight, 'text' | 'startOffset' | 'endOffset' | 'blockId' | 'anchorId' | 'occurrence' | 'readableIndex' | 'documentOccurrence' | 'startReadableIndex' | 'endReadableIndex'>,
) {
  return sameHighlightTarget(left, right)
}

function normalizeReaderSessionHighlights(value: unknown): ReaderSessionHighlight[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((item): item is ReaderSessionHighlight => Boolean(item) && typeof item === 'object')
    .filter((item) => Boolean(String(item.id || '').trim() && String(item.text || '').trim()))
}

export function useReaderSessionHighlights({
  activeConversationId,
  readerPayload,
}: {
  activeConversationId?: string | null
  readerPayload: ReaderOpenPayload | null
}) {
  const [readerSessionHighlights, setReaderSessionHighlights] = useState<Record<string, ReaderSessionHighlight[]>>({})
  const activeReaderSessionHighlightsRef = useRef<ReaderSessionHighlight[]>([])
  const readerStateHydratedKeysRef = useRef<Set<string>>(new Set())
  const readerStateSaveTimersRef = useRef<Record<string, number>>({})

  const activeReaderSourcePath = useMemo(
    () => String(readerPayload?.sourcePath || '').trim(),
    [readerPayload?.sourcePath],
  )
  const activeReaderHighlightScope = useMemo(
    () => readerHighlightScopeKey(activeConversationId, activeReaderSourcePath),
    [activeConversationId, activeReaderSourcePath],
  )
  const activeReaderSessionHighlights = useMemo(
    () => (activeReaderHighlightScope ? readerSessionHighlights[activeReaderHighlightScope] || [] : []),
    [activeReaderHighlightScope, readerSessionHighlights],
  )

  useEffect(() => {
    activeReaderSessionHighlightsRef.current = activeReaderSessionHighlights
  }, [activeReaderSessionHighlights])

  const persistReaderHighlights = useCallback((convId: string, sourcePath: string, highlights: ReaderSessionHighlight[]) => {
    const cid = String(convId || '').trim()
    const src = String(sourcePath || '').trim()
    if (!cid || !src) return
    void chatApi.updateConversationReaderState(cid, src, {
      highlights,
      evidenceNotes: highlights,
      updatedAt: Date.now(),
    }).catch(() => {})
  }, [])

  useEffect(() => {
    const convId = String(activeConversationId || '').trim()
    const sourcePath = String(activeReaderSourcePath || '').trim()
    const scopeKey = activeReaderHighlightScope
    if (!convId || !sourcePath || !scopeKey) return undefined
    let cancelled = false
    readerStateHydratedKeysRef.current.delete(scopeKey)
    chatApi.getConversationReaderState(convId, sourcePath)
      .then((record) => {
        if (cancelled) return
        const highlights = normalizeReaderSessionHighlights(record.state?.highlights || record.state?.evidenceNotes)
        setReaderSessionHighlights((current) => {
          const prev = current[scopeKey] || []
          if (readerHighlightsSignature(prev) === readerHighlightsSignature(highlights)) return current
          if (highlights.length === 0 && prev.length > 0) return current
          return { ...current, [scopeKey]: highlights }
        })
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) readerStateHydratedKeysRef.current.add(scopeKey)
      })
    return () => {
      cancelled = true
    }
  }, [activeConversationId, activeReaderHighlightScope, activeReaderSourcePath])

  useEffect(() => {
    const convId = String(activeConversationId || '').trim()
    const sourcePath = String(activeReaderSourcePath || '').trim()
    const scopeKey = activeReaderHighlightScope
    if (!convId || !sourcePath || !scopeKey) return undefined
    if (!readerStateHydratedKeysRef.current.has(scopeKey)) return undefined
    const highlights = activeReaderSessionHighlights
    const previousTimer = readerStateSaveTimersRef.current[scopeKey]
    if (previousTimer) window.clearTimeout(previousTimer)
    const timer = window.setTimeout(() => {
      if (readerStateSaveTimersRef.current[scopeKey] === timer) {
        delete readerStateSaveTimersRef.current[scopeKey]
      }
      persistReaderHighlights(convId, sourcePath, highlights)
    }, 700)
    readerStateSaveTimersRef.current[scopeKey] = timer
    return undefined
  }, [
    activeConversationId,
    activeReaderHighlightScope,
    activeReaderSessionHighlights,
    activeReaderSourcePath,
    persistReaderHighlights,
  ])

  useEffect(() => {
    if (typeof BroadcastChannel === 'undefined') return undefined
    const channel = new BroadcastChannel(READER_SESSION_SYNC_CHANNEL)
    channel.onmessage = (event) => {
      const data = (event?.data && typeof event.data === 'object')
        ? event.data as Record<string, unknown>
        : {}
      if (String(data.type || '') !== 'reader-session-state') return
      const sourcePath = String(data.sourcePath || '').trim()
      if (!sourcePath) return
      const conversationId = String(data.conversationId || '').trim()
      if (conversationId && activeConversationId && conversationId !== activeConversationId) return
      const highlights = Array.isArray(data.highlights)
        ? data.highlights.filter((item): item is ReaderSessionHighlight => Boolean(item) && typeof item === 'object')
        : null
      if (!highlights) return
      const scopeKey = readerHighlightScopeKey(activeConversationId, sourcePath)
      if (!scopeKey) return
      readerStateHydratedKeysRef.current.add(scopeKey)
      setReaderSessionHighlights((current) => {
        const prev = current[scopeKey] || []
        if (readerHighlightsSignature(prev) === readerHighlightsSignature(highlights)) return current
        return { ...current, [scopeKey]: highlights }
      })
    }
    return () => {
      channel.close()
    }
  }, [activeConversationId])

  useEffect(() => () => {
    Object.values(readerStateSaveTimersRef.current).forEach((timer) => window.clearTimeout(timer))
    readerStateSaveTimersRef.current = {}
  }, [])

  const addReaderSessionHighlight = useCallback((highlight: ReaderSessionHighlight) => {
    const scopeKey = activeReaderHighlightScope
    if (!scopeKey) return
    setReaderSessionHighlights((current) => {
      const list = Array.isArray(current[scopeKey]) ? current[scopeKey] : []
      if (list.some((item) => sameReaderSessionHighlight(item, highlight))) {
        return current
      }
      return {
        ...current,
        [scopeKey]: [...list, highlight],
      }
    })
  }, [activeReaderHighlightScope])

  const removeReaderSessionHighlight = useCallback((highlightId: string) => {
    const scopeKey = activeReaderHighlightScope
    const targetId = String(highlightId || '').trim()
    if (!scopeKey || !targetId) return
    setReaderSessionHighlights((current) => {
      const list = Array.isArray(current[scopeKey]) ? current[scopeKey] : []
      const next = list.filter((item) => String(item.id || '').trim() !== targetId)
      if (next.length === list.length) return current
      return {
        ...current,
        [scopeKey]: next,
      }
    })
  }, [activeReaderHighlightScope])

  const updateReaderSessionHighlight = useCallback((highlight: ReaderSessionHighlight) => {
    const scopeKey = activeReaderHighlightScope
    const targetId = String(highlight?.id || '').trim()
    if (!scopeKey || !targetId) return
    setReaderSessionHighlights((current) => {
      const list = Array.isArray(current[scopeKey]) ? current[scopeKey] : []
      let changed = false
      const next = list.map((item) => {
        if (String(item.id || '').trim() !== targetId) return item
        changed = true
        return { ...item, ...highlight }
      })
      if (!changed) return current
      return {
        ...current,
        [scopeKey]: next,
      }
    })
  }, [activeReaderHighlightScope])

  return {
    activeReaderSessionHighlights,
    activeReaderSessionHighlightsRef,
    addReaderSessionHighlight,
    removeReaderSessionHighlight,
    updateReaderSessionHighlight,
  }
}
