import { createElement } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'
import { useReaderShelfEvents } from '../components/chat/useReaderShelfEvents'
import {
  READER_CITATION_SHELF_EVENT,
  READER_SELECTION_SHELF_EVENT,
} from '../components/chat/reader/readerTypes'

export function runReaderShelfEventsSmoke() {
  const originalChannel = window.BroadcastChannel
  const channels: TestChannel[] = []
  const calls: string[] = []
  class TestChannel {
    onmessage: ((event: MessageEvent) => void) | null = null
    sent: unknown[] = []
    closed = 0
    readonly name: string
    constructor(name: string) {
      this.name = name
      channels.push(this)
    }
    postMessage(data: unknown) { this.sent.push(data) }
    close() { this.closed += 1 }
  }
  const host = document.createElement('div')
  document.body.append(host)
  let root = createRoot(host)
  function Harness({ projectId, label }: { projectId: string; label: string }) {
    useReaderShelfEvents({
      projectId,
      onCitation: (payload) => {
        calls.push(`${label}:citation:${payload.detail.anchor}`)
        return true
      },
      onSelection: (payload) => {
        if (payload.text === 'declined') return false
        calls.push(`${label}:selection:${payload.text}`)
        return true
      },
    })
    return null
  }
  const mount = (projectId: string, label: string) => {
    flushSync(() => root.render(createElement(Harness, { projectId, label })))
  }
  const citation = { detail: { anchor: 'source-citation' }, projectId: 'a' }
  const selection = { text: 'source quote', sourcePath: '/tmp/source.md', projectId: 'a' }
  const dispatch = (name: string, detail: unknown) => {
    window.dispatchEvent(new CustomEvent(name, { detail }))
  }
  try {
    window.BroadcastChannel = TestChannel as unknown as typeof BroadcastChannel
    mount('a', 'first')
    dispatch(READER_CITATION_SHELF_EVENT, citation)
    const [citationChannel, selectionChannel] = channels
    const receive = (channel: TestChannel, data: unknown) => channel.onmessage?.(new MessageEvent('message', { data }))
    receive(selectionChannel, { type: 'reader-selection-shelf', requestId: 'accepted-a', ...selection })
    receive(selectionChannel, { type: 'reader-selection-shelf', requestId: 'malformed', text: 'missing source' })
    receive(selectionChannel, { type: 'unrelated', requestId: 'wrong-type', ...selection })
    mount('b', 'latest')
    dispatch(READER_SELECTION_SHELF_EVENT, selection)
    receive(citationChannel, { type: 'reader-citation-shelf', requestId: 'wrong-project', ...citation })
    receive(citationChannel, { type: 'reader-citation-shelf', requestId: 'accepted-b', ...citation, projectId: 'b' })
    receive(selectionChannel, { type: 'reader-selection-shelf', requestId: 'declined', ...selection, projectId: 'b', text: 'declined' })
    // A queued callback may already exist when cleanup runs.
    const queuedCitation = citationChannel.onmessage
    const queuedSelection = selectionChannel.onmessage
    root.unmount()
    dispatch(READER_CITATION_SHELF_EVENT, { ...citation, projectId: 'b' })
    dispatch(READER_SELECTION_SHELF_EVENT, { ...selection, projectId: 'b' })
    queuedCitation?.(new MessageEvent('message', { data: { type: 'reader-citation-shelf', requestId: 'late-citation', ...citation, projectId: 'b' } }))
    queuedSelection?.(new MessageEvent('message', { data: { type: 'reader-selection-shelf', requestId: 'late-selection', ...selection, projectId: 'b' } }))

    window.BroadcastChannel = undefined as unknown as typeof BroadcastChannel
    root = createRoot(host)
    mount('b', 'window-only')
    dispatch(READER_SELECTION_SHELF_EVENT, { ...selection, projectId: undefined })
    root.unmount()
    dispatch(READER_SELECTION_SHELF_EVENT, { ...selection, projectId: undefined })
    return {
      calls,
      channels: channels.map((channel) => ({
        name: channel.name, sent: channel.sent, closed: channel.closed, detached: channel.onmessage === null,
      })),
    }
  } finally {
    window.BroadcastChannel = originalChannel
    host.remove()
  }
}
