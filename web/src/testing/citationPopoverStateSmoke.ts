import { createElement, useLayoutEffect } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import {
  toShelfItem,
  type CiteDetail,
} from '../components/chat/citationState'
import {
  useCitationPopoverState,
  type CitationPopoverStateController,
} from '../components/chat/useCitationPopoverState'

export interface CitationPopoverStateSmokeSnapshot {
  doi: string
  guideLoading: boolean
  loading: boolean
  pinned: boolean
  requestKey: string
  title: string
  x: number | null
  y: number | null
}

export interface CitationPopoverStateSmokeResult {
  renderedText: string
  snapshots: CitationPopoverStateSmokeSnapshot[]
  usableMetaCount: number
}

function snapshotCitationPopoverState(
  state: CitationPopoverStateController,
): CitationPopoverStateSmokeSnapshot {
  return {
    doi: state.detail?.doi || '',
    guideLoading: state.guideLoading,
    loading: state.loading,
    pinned: state.pinned,
    requestKey: state.activeRequestKeyRef.current,
    title: state.detail?.title || '',
    x: state.position?.x ?? null,
    y: state.position?.y ?? null,
  }
}

export function runCitationPopoverStateSmoke(): CitationPopoverStateSmokeResult {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  const detail = {
    anchor: 'cite-a',
    citeFmt: '',
    doi: '',
    isInpaper: false,
    num: 4,
    raw: '',
    sourceName: 'State Paper',
    sourcePath: '/tmp/state-paper.md',
    title: 'State Paper Title',
  } as CiteDetail
  const key = toShelfItem(detail).key
  const snapshots: CitationPopoverStateSmokeSnapshot[] = []
  let controller: CitationPopoverStateController | null = null
  const readController = () => {
    if (!controller) throw new Error('citation popover state smoke did not mount')
    return controller
  }

  function Harness() {
    const state = useCitationPopoverState()
    useLayoutEffect(() => {
      controller = state
      snapshots.push(snapshotCitationPopoverState(state))
    })
    return createElement('div', { id: 'citation-popover-state-smoke' }, state.detail?.title || 'empty')
  }

  flushSync(() => {
    root.render(createElement(Harness))
  })
  flushSync(() => {
    readController().open(detail, { x: 12, y: 34 }, { loading: true, pinned: true, requestKey: key })
  })
  let usableMetaCount = 0
  flushSync(() => {
    usableMetaCount = readController().mergeDetailForKey(key, [{ doi: '10.1000/state' }, {}]).length
  })
  flushSync(() => {
    readController().setGuideLoading(true)
  })
  flushSync(() => {
    readController().close()
  })
  const renderedText = host.textContent || ''
  root.unmount()
  host.remove()

  return {
    renderedText,
    snapshots,
    usableMetaCount,
  }
}
