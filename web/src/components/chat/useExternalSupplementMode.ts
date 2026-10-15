/* eslint-disable react-hooks/set-state-in-effect */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { ExternalSupplementMode } from '../../api/chat'

const STORAGE_PREFIX = 'kb:chat-external-supplement-mode:v1'

function normalizedMode(value: unknown): ExternalSupplementMode {
  const raw = String(value || '').trim().toLowerCase().replace(/-/g, '_')
  if (['model', 'llm', 'model_knowledge'].includes(raw)) return 'model'
  if (['web', 'web_search', 'search'].includes(raw)) return 'web'
  return 'disabled'
}

function readUrlOverride(): ExternalSupplementMode | null {
  if (typeof window === 'undefined') return null
  try {
    const params = new URLSearchParams(window.location.search)
    const raw = params.get('external_supplement') || params.get('supplement')
    return raw === null ? null : normalizedMode(raw)
  } catch {
    return null
  }
}

function storageKey(conversationId?: string | null) {
  const conv = String(conversationId || '').trim()
  return conv ? `${STORAGE_PREFIX}:${encodeURIComponent(conv)}` : ''
}

function loadMode(conversationId?: string | null): ExternalSupplementMode {
  const override = readUrlOverride()
  if (override !== null) return override
  if (typeof window === 'undefined') return 'disabled'
  const key = storageKey(conversationId)
  if (!key) return 'disabled'
  try {
    return normalizedMode(window.localStorage.getItem(key))
  } catch {
    return 'disabled'
  }
}

function saveMode(conversationId: string, mode: ExternalSupplementMode) {
  if (typeof window === 'undefined') return
  const key = storageKey(conversationId)
  if (!key) return
  try {
    window.localStorage.setItem(key, mode)
  } catch {
    // Private browsing may reject storage; the in-memory choice still works.
  }
}

export function useExternalSupplementMode(activeConversationId?: string | null) {
  const [mode, setMode] = useState<ExternalSupplementMode>(() => loadMode(null))
  const [ownerConversationId, setOwnerConversationId] = useState('')
  const pendingForNewConversationRef = useRef<ExternalSupplementMode | null>(null)

  useEffect(() => {
    const convId = String(activeConversationId || '').trim()
    if (!convId) {
      setOwnerConversationId('')
      return
    }
    const pending = pendingForNewConversationRef.current
    if (pending !== null) {
      pendingForNewConversationRef.current = null
      setMode(pending)
      setOwnerConversationId(convId)
      saveMode(convId, pending)
      return
    }
    setMode(loadMode(convId))
    setOwnerConversationId(convId)
  }, [activeConversationId])

  useEffect(() => {
    const convId = String(activeConversationId || '').trim()
    if (!convId || ownerConversationId !== convId) return
    saveMode(convId, mode)
  }, [activeConversationId, mode, ownerConversationId])

  const setModeForConversation = useCallback((next: ExternalSupplementMode) => {
    const normalized = normalizedMode(next)
    const convId = String(activeConversationId || '').trim()
    if (!convId) {
      pendingForNewConversationRef.current = normalized
    } else {
      setOwnerConversationId(convId)
    }
    setMode(normalized)
  }, [activeConversationId])

  return { mode, setMode: setModeForConversation }
}
