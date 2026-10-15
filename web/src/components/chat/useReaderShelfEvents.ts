import { useEffect, useLayoutEffect, useRef } from 'react'
import { shelfProjectScopeId } from './citationState'
import {
  normalizeReaderCitationShelfPayload,
  normalizeReaderSelectionShelfPayload,
} from './readerShelfPayload'
import {
  READER_CITATION_SHELF_CHANNEL,
  READER_CITATION_SHELF_EVENT,
  READER_SELECTION_SHELF_CHANNEL,
  READER_SELECTION_SHELF_EVENT,
  type ReaderCitationShelfPayload,
  type ReaderSelectionShelfPayload,
} from './reader/readerTypes'

interface ReaderShelfEventHandlers {
  projectId?: string | null
  // True acknowledges local acceptance, not completion of backend persistence.
  onCitation: (payload: ReaderCitationShelfPayload) => boolean
  onSelection: (payload: ReaderSelectionShelfPayload) => boolean
}

export function useReaderShelfEvents(handlers: ReaderShelfEventHandlers) {
  const handlersRef = useRef(handlers)
  useLayoutEffect(() => {
    handlersRef.current = handlers
  }, [handlers])

  useEffect(() => {
    let active = true
    const inScope = (projectId?: string) => (
      !projectId || shelfProjectScopeId(projectId) === shelfProjectScopeId(handlersRef.current.projectId)
    )
    const subscribe = (
      eventName: string,
      channelName: string,
      type: string,
      receive: (raw: unknown) => boolean,
    ) => {
      const handleWindowEvent = (event: Event) => {
        if (active) receive((event as CustomEvent<unknown>).detail)
      }
      window.addEventListener(eventName, handleWindowEvent)
      const channel = typeof BroadcastChannel !== 'undefined'
        ? new BroadcastChannel(channelName)
        : null
      if (channel) {
        channel.onmessage = (event) => {
          if (!active) return
          const data = event?.data && typeof event.data === 'object'
            ? event.data as Record<string, unknown>
            : {}
          if (String(data.type || '') !== type || !receive(data)) return
          const requestId = String(data.requestId || '').trim()
          if (requestId) channel.postMessage({ type: `${type}-ack`, requestId })
        }
      }
      return () => {
        window.removeEventListener(eventName, handleWindowEvent)
        if (channel) {
          channel.onmessage = null
          channel.close()
        }
      }
    }
    const unsubscribeCitation = subscribe(
      READER_CITATION_SHELF_EVENT, READER_CITATION_SHELF_CHANNEL, 'reader-citation-shelf',
      (raw) => {
        const payload = normalizeReaderCitationShelfPayload(raw)
        return Boolean(payload && inScope(payload.projectId) && handlersRef.current.onCitation(payload))
      },
    )
    const unsubscribeSelection = subscribe(
      READER_SELECTION_SHELF_EVENT, READER_SELECTION_SHELF_CHANNEL, 'reader-selection-shelf',
      (raw) => {
        const payload = normalizeReaderSelectionShelfPayload(raw)
        return Boolean(payload && inScope(payload.projectId) && handlersRef.current.onSelection(payload))
      },
    )
    return () => {
      active = false
      unsubscribeCitation()
      unsubscribeSelection()
    }
  }, [])
}
