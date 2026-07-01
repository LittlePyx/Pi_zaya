/* eslint-disable react-hooks/set-state-in-effect */

import { useCallback, useEffect, useRef, useState } from 'react'

const AGENT_MODE_STORAGE_PREFIX = 'kb:chat-research-agent-mode:v1'

function readAgentModeUrlOverride(): boolean | null {
  if (typeof window === 'undefined') return null
  try {
    const params = new URLSearchParams(window.location.search)
    const queryValue = String(params.get('agent_mode') || params.get('research_agent') || '').trim().toLowerCase()
    if (['1', 'true', 'yes', 'on'].includes(queryValue)) return true
    if (['0', 'false', 'no', 'off'].includes(queryValue)) return false
  } catch {
    // URL parsing is best-effort; stored conversation state remains the fallback.
  }
  return null
}

function agentModeStorageKey(conversationId?: string | null) {
  const conv = String(conversationId || '').trim()
  return conv ? `${AGENT_MODE_STORAGE_PREFIX}:${encodeURIComponent(conv)}` : ''
}

function loadStoredAgentModeForConversation(conversationId?: string | null) {
  const urlOverride = readAgentModeUrlOverride()
  if (urlOverride !== null) return urlOverride
  if (typeof window === 'undefined') return false
  const key = agentModeStorageKey(conversationId)
  if (!key) return false
  try {
    return window.localStorage.getItem(key) === '1'
  } catch {
    return false
  }
}

function saveStoredAgentModeForConversation(conversationId: string, enabled: boolean) {
  if (typeof window === 'undefined') return
  const key = agentModeStorageKey(conversationId)
  if (!key) return
  try {
    window.localStorage.setItem(key, enabled ? '1' : '0')
  } catch {
    // Storage can fail in private mode; the in-memory toggle still works.
  }
}

export function useAgentMode(activeConversationId?: string | null) {
  const [agentMode, setAgentMode] = useState(() => loadStoredAgentModeForConversation(null))
  const [ownerConversationId, setOwnerConversationId] = useState('')
  const pendingForNewConversationRef = useRef<boolean | null>(null)

  useEffect(() => {
    const convId = String(activeConversationId || '').trim()
    if (!convId) {
      setOwnerConversationId('')
      return
    }
    const pending = pendingForNewConversationRef.current
    if (pending !== null) {
      pendingForNewConversationRef.current = null
      setAgentMode(pending)
      setOwnerConversationId(convId)
      saveStoredAgentModeForConversation(convId, pending)
      return
    }
    setAgentMode(loadStoredAgentModeForConversation(convId))
    setOwnerConversationId(convId)
  }, [activeConversationId])

  useEffect(() => {
    const convId = String(activeConversationId || '').trim()
    if (!convId || ownerConversationId !== convId) return
    saveStoredAgentModeForConversation(convId, agentMode)
  }, [activeConversationId, agentMode, ownerConversationId])

  const setAgentModeForConversation = useCallback((enabled: boolean) => {
    const convId = String(activeConversationId || '').trim()
    if (!convId) {
      pendingForNewConversationRef.current = enabled
    } else {
      setOwnerConversationId(convId)
    }
    setAgentMode(enabled)
  }, [activeConversationId])

  return {
    agentMode,
    setAgentMode: setAgentModeForConversation,
  }
}
