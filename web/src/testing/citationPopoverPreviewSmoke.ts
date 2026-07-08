import { createElement, useLayoutEffect } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'
import type { MutableRefObject } from 'react'

import type { CiteDetail } from '../components/chat/citationState'
import {
  useCitationPopoverPreview,
  type CitationPopoverPreviewController,
  type CitationPopoverPolishFetcher,
} from '../components/chat/useCitationPopoverPreview'

export interface CitationPopoverPreviewSmokeResult {
  events: string[]
  fetchCalls: number
  polishWaitSeconds: number[]
}

function waitForTimers(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

export async function runCitationPopoverPreviewSmoke(): Promise<CitationPopoverPreviewSmokeResult> {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  const events: string[] = []
  const polishWaitSeconds: number[] = []
  let fetchCalls = 0
  let controller: CitationPopoverPreviewController | null = null
  const detail = {
    anchor: 'cite-preview',
    citeFmt: '',
    doi: '',
    isInpaper: false,
    num: 7,
    raw: 'Preview raw citation.',
    sourceName: 'Preview Paper',
    sourcePath: '/tmp/preview-paper.md',
    title: 'Preview Paper Title',
  } as CiteDetail
  const activeRequestKeyRef = { current: 'preview-key' } as MutableRefObject<string>
  const fetcher: CitationPopoverPolishFetcher = async (_detail, waitSeconds) => {
    fetchCalls += 1
    polishWaitSeconds.push(waitSeconds)
    return fetchCalls === 1
      ? { citation_card_polish_status: 'pending' }
      : { doi: '10.1000/preview' }
  }
  const readController = () => {
    if (!controller) throw new Error('citation popover preview smoke did not mount')
    return controller
  }

  function Harness() {
    const preview = useCitationPopoverPreview()
    useLayoutEffect(() => {
      controller = preview
    })
    return createElement('div', { id: 'citation-popover-preview-smoke' }, 'ready')
  }

  flushSync(() => {
    root.render(createElement(Harness))
  })
  readController().schedulePreviewOpen(() => {
    events.push('open')
  }, 5)
  await waitForTimers(12)
  readController().schedulePreviewClose(() => {
    events.push('close-after-keep')
  }, 5)
  readController().keepPreviewOpen()
  await waitForTimers(12)
  readController().schedulePreviewClose(() => {
    events.push('close')
  }, 5)
  await waitForTimers(12)
  readController().requestPolish({
    activeRequestKeyRef,
    detail,
    fetcher,
    itemKey: 'preview-key',
    onMeta: (itemKey, metas) => {
      events.push(`polish:${itemKey}:${String(metas[0]?.doi || '')}`)
    },
    retryDelayMs: () => 1,
  })
  await waitForTimers(20)
  root.unmount()
  host.remove()

  return {
    events,
    fetchCalls,
    polishWaitSeconds,
  }
}
