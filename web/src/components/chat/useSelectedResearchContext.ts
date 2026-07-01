/* eslint-disable react-hooks/set-state-in-effect */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { chatApi } from '../../api/chat'
import {
  normalizeSelectedResearchContextPack,
  type SelectedResearchContextPack,
} from './researchContextPack'

const SELECTED_RESEARCH_CONTEXT_STORAGE_PREFIX = 'kb:chat:selected-research-context:v1'
const SELECTED_RESEARCH_CONTEXT_STATE_KEY = 'selected_research_context'
const SELECTED_RESEARCH_CONTEXT_SCOPE_STATE_KEY = 'selected_research_context_scope'
const SELECTED_RESEARCH_CONTEXT_PROJECT_STATE_KEY = 'selected_research_context_project_id'
const SELECTED_RESEARCH_CONTEXT_CLEARED_AT_STATE_KEY = 'selected_research_context_cleared_at'

function selectedResearchContextStorageKey(conversationId?: string | null, shelfScope?: string | null) {
  const conv = String(conversationId || '').trim()
  if (!conv) return ''
  const scope = String(shelfScope || '__default__').trim() || '__default__'
  return `${SELECTED_RESEARCH_CONTEXT_STORAGE_PREFIX}:${encodeURIComponent(conv)}:${encodeURIComponent(scope)}`
}

function loadStoredSelectedResearchContext(storageKey: string): SelectedResearchContextPack | null {
  if (!storageKey || typeof window === 'undefined') return null
  try {
    const raw = window.localStorage.getItem(storageKey)
    if (!raw) return null
    const pack = normalizeSelectedResearchContextPack(JSON.parse(raw))
    if (!pack) {
      window.localStorage.removeItem(storageKey)
      return null
    }
    return pack
  } catch {
    try {
      window.localStorage.removeItem(storageKey)
    } catch {
      // Best-effort cleanup only.
    }
    return null
  }
}

function saveStoredSelectedResearchContext(storageKey: string, pack: SelectedResearchContextPack | null) {
  if (!storageKey || typeof window === 'undefined') return
  try {
    if (!pack) {
      window.localStorage.removeItem(storageKey)
      return
    }
    window.localStorage.setItem(storageKey, JSON.stringify(pack))
  } catch {
    // Storage can fail in private mode or under quota pressure; the in-memory state still works.
  }
}

function selectedResearchContextFromState(state: Record<string, unknown> | undefined | null): SelectedResearchContextPack | null {
  const raw = state && typeof state === 'object' ? state[SELECTED_RESEARCH_CONTEXT_STATE_KEY] : null
  return normalizeSelectedResearchContextPack(raw)
}

function researchContextStateMatchesShelf(
  state: Record<string, unknown> | undefined | null,
  shelfScope?: string | null,
  shelfProjectId?: string | null,
) {
  if (!state || typeof state !== 'object') return true
  const storedScope = String(state[SELECTED_RESEARCH_CONTEXT_SCOPE_STATE_KEY] || '').trim()
  const currentScope = String(shelfScope || '').trim()
  if (storedScope && currentScope && storedScope !== currentScope) return false
  const storedProjectId = String(state[SELECTED_RESEARCH_CONTEXT_PROJECT_STATE_KEY] || '').trim()
  const currentProjectId = String(shelfProjectId || '').trim()
  if (storedProjectId && storedProjectId !== currentProjectId) return false
  return true
}

export function useSelectedResearchContext({
  activeConversationId,
  shelfProjectId,
  shelfScope,
  onBasketContextReady,
}: {
  activeConversationId?: string | null
  shelfProjectId?: string | null
  shelfScope?: string | null
  onBasketContextReady?: () => void
}) {
  const [selectedResearchContext, setSelectedResearchContext] = useState<SelectedResearchContextPack | null>(null)
  const [selectedResearchContextLoadedKey, setSelectedResearchContextLoadedKey] = useState('')
  const [selectedResearchContextOwnerKey, setSelectedResearchContextOwnerKey] = useState('')
  const selectedResearchContextLoadSeqRef = useRef(0)

  const selectedResearchContextDraftKey = useMemo(
    () => selectedResearchContextStorageKey(activeConversationId, shelfScope),
    [activeConversationId, shelfScope],
  )
  const currentSelectedResearchContext = selectedResearchContextOwnerKey === selectedResearchContextDraftKey
    ? selectedResearchContext
    : null
  const selectedResearchContextKeys = useMemo(() => {
    const out: Record<string, boolean> = {}
    for (const item of currentSelectedResearchContext?.items || []) {
      if (item.key) out[item.key] = true
    }
    return out
  }, [currentSelectedResearchContext])

  const handleResearchContextPackChange = useCallback((pack: SelectedResearchContextPack | null) => {
    setSelectedResearchContextOwnerKey(selectedResearchContextDraftKey)
    setSelectedResearchContext(pack)
    if (pack?.items?.length) onBasketContextReady?.()
  }, [onBasketContextReady, selectedResearchContextDraftKey])

  const clearSelectedResearchContext = useCallback(() => {
    setSelectedResearchContextOwnerKey(selectedResearchContextDraftKey)
    setSelectedResearchContext(null)
  }, [selectedResearchContextDraftKey])

  const clearSelectedResearchContextIfCurrent = useCallback((packId?: string | null) => {
    const targetId = String(packId || '').trim()
    if (!targetId) return
    setSelectedResearchContext((current) => (
      current?.id === targetId ? null : current
    ))
  }, [])

  useEffect(() => {
    const draftKey = selectedResearchContextDraftKey
    const convId = String(activeConversationId || '').trim()
    const loadSeq = selectedResearchContextLoadSeqRef.current + 1
    selectedResearchContextLoadSeqRef.current = loadSeq
    const localPack = loadStoredSelectedResearchContext(draftKey)
    setSelectedResearchContextLoadedKey('')
    setSelectedResearchContextOwnerKey(draftKey)
    setSelectedResearchContext(localPack)
    if (localPack?.items?.length) onBasketContextReady?.()
    if (!draftKey || !convId) {
      setSelectedResearchContextLoadedKey(draftKey)
      return undefined
    }
    let cancelled = false
    void chatApi.getConversationResearchState(convId).then((record) => {
      if (cancelled || selectedResearchContextLoadSeqRef.current !== loadSeq) return
      const state = record?.state && typeof record.state === 'object' ? record.state : {}
      const backendMatchesShelf = researchContextStateMatchesShelf(state, shelfScope, shelfProjectId)
      const backendPack = backendMatchesShelf ? selectedResearchContextFromState(state) : null
      const backendTouched = Boolean(
        Object.prototype.hasOwnProperty.call(state, SELECTED_RESEARCH_CONTEXT_STATE_KEY)
        || Object.prototype.hasOwnProperty.call(state, SELECTED_RESEARCH_CONTEXT_SCOPE_STATE_KEY)
        || Object.prototype.hasOwnProperty.call(state, SELECTED_RESEARCH_CONTEXT_PROJECT_STATE_KEY)
        || Object.prototype.hasOwnProperty.call(state, SELECTED_RESEARCH_CONTEXT_CLEARED_AT_STATE_KEY)
      )
      const nextPack = backendTouched ? backendPack : localPack
      setSelectedResearchContextOwnerKey(draftKey)
      setSelectedResearchContext(nextPack)
      if (nextPack?.items?.length) onBasketContextReady?.()
      if (backendTouched) {
        saveStoredSelectedResearchContext(draftKey, backendPack)
      }
      setSelectedResearchContextLoadedKey(draftKey)
    }).catch(() => {
      if (cancelled || selectedResearchContextLoadSeqRef.current !== loadSeq) return
      setSelectedResearchContextLoadedKey(draftKey)
    })
    return () => {
      cancelled = true
    }
  }, [activeConversationId, onBasketContextReady, selectedResearchContextDraftKey, shelfProjectId, shelfScope])

  useEffect(() => {
    if (selectedResearchContextLoadedKey !== selectedResearchContextDraftKey) return
    const packForCurrentScope = selectedResearchContextOwnerKey === selectedResearchContextDraftKey
      ? selectedResearchContext
      : null
    saveStoredSelectedResearchContext(selectedResearchContextDraftKey, packForCurrentScope)
    const convId = String(activeConversationId || '').trim()
    if (!convId) return undefined
    const timer = window.setTimeout(() => {
      void chatApi.patchConversationResearchState(convId, {
        [SELECTED_RESEARCH_CONTEXT_STATE_KEY]: packForCurrentScope || null,
        [SELECTED_RESEARCH_CONTEXT_SCOPE_STATE_KEY]: packForCurrentScope ? shelfScope : null,
        [SELECTED_RESEARCH_CONTEXT_PROJECT_STATE_KEY]: packForCurrentScope ? shelfProjectId : null,
        [SELECTED_RESEARCH_CONTEXT_CLEARED_AT_STATE_KEY]: packForCurrentScope ? null : Date.now(),
      }).catch(() => {
        // The local draft remains the fallback if the backend is temporarily unavailable.
      })
    }, 160)
    return () => window.clearTimeout(timer)
  }, [
    activeConversationId,
    selectedResearchContext,
    selectedResearchContextDraftKey,
    selectedResearchContextLoadedKey,
    selectedResearchContextOwnerKey,
    shelfProjectId,
    shelfScope,
  ])

  return {
    currentSelectedResearchContext,
    selectedResearchContextKeys,
    handleResearchContextPackChange,
    clearSelectedResearchContext,
    clearSelectedResearchContextIfCurrent,
  }
}
