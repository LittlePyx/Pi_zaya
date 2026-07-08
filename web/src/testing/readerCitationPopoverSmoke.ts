import { act, createElement, useLayoutEffect } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import type { CiteDetail } from '../components/chat/citationState'
import {
  useReaderCitationPopover,
  type ReaderCitationPopoverController,
  type ReaderCitationPopoverMetadataLoader,
} from '../components/chat/useReaderCitationPopover'
import type { ReaderCitationPopoverMetadataResult } from '../components/chat/readerCitationPopoverMetadata'

export interface ReaderCitationPopoverSmokeSnapshot {
  loading: boolean
  title: string
  x: number | null
  y: number | null
}

export interface ReaderCitationPopoverSmokeResult {
  calls: string[]
  renderedText: string
  snapshots: ReaderCitationPopoverSmokeSnapshot[]
}

function waitForTimers(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

function snapshotReaderCitationPopover(
  controller: ReaderCitationPopoverController,
): ReaderCitationPopoverSmokeSnapshot {
  return {
    loading: controller.loading,
    title: controller.detail?.title || '',
    x: controller.position?.x ?? null,
    y: controller.position?.y ?? null,
  }
}

export async function runReaderCitationPopoverSmoke(): Promise<ReaderCitationPopoverSmokeResult> {
  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  const calls: string[] = []
  const snapshots: ReaderCitationPopoverSmokeSnapshot[] = []
  let controller: ReaderCitationPopoverController | null = null
  const resolveSlowRef: {
    current: ((value: ReaderCitationPopoverMetadataResult) => void) | null
  } = { current: null }

  const slowDetail = {
    anchor: 'reader-slow',
    citeFmt: '',
    doi: '10.1000/slow',
    isInpaper: true,
    num: 1,
    raw: '',
    sourceName: 'Current Paper',
    sourcePath: '/tmp/current.md',
    title: 'Slow Citation',
  } as CiteDetail
  const fastDetail = {
    anchor: 'reader-fast',
    citeFmt: '',
    doi: '10.1000/fast',
    isInpaper: true,
    num: 2,
    raw: '',
    sourceName: 'Current Paper',
    sourcePath: '/tmp/current.md',
    title: 'Fast Citation',
  } as CiteDetail

  const loadMetadata: ReaderCitationPopoverMetadataLoader = async (detail, options) => {
    calls.push(`${detail.title || ''}:${options?.plan?.itemKey || ''}`)
    if (detail.title === 'Slow Citation') {
      return new Promise((resolve) => {
        resolveSlowRef.current = resolve
      })
    }
    return {
      metas: [{ citation_card_polish_status: 'ready' }],
      plan: options?.plan || {
        itemKey: '',
        missingReferenceEntry: false,
        needsSummaryBackfill: true,
        requestCount: 1,
        shouldFetchBibliometrics: true,
        shouldFetchPolish: false,
      },
    }
  }

  const readController = () => {
    if (!controller) throw new Error('reader citation popover smoke did not mount')
    return controller
  }

  function Harness() {
    const popover = useReaderCitationPopover({ loadMetadata })
    useLayoutEffect(() => {
      controller = popover
      snapshots.push(snapshotReaderCitationPopover(popover))
    })
    return createElement(
      'div',
      { id: 'reader-citation-popover-smoke' },
      popover.detail?.citationCardPolishStatus || 'empty',
    )
  }

  flushSync(() => {
    root.render(createElement(Harness))
  })
  flushSync(() => {
    readController().showCitation(slowDetail, { clientX: 12, clientY: 34 })
  })
  flushSync(() => {
    readController().showCitation(fastDetail, { clientX: 56, clientY: 78 })
  })
  await act(async () => {
    await waitForTimers(0)
  })
  await act(async () => {
    resolveSlowRef.current?.({
      metas: [{ doi: '10.1000/slow-merged' }],
      plan: {
        itemKey: '',
        missingReferenceEntry: false,
        needsSummaryBackfill: true,
        requestCount: 1,
        shouldFetchBibliometrics: true,
        shouldFetchPolish: false,
      },
    })
    await waitForTimers(0)
  })

  const renderedText = host.textContent || ''
  root.unmount()
  host.remove()

  return {
    calls,
    renderedText,
    snapshots,
  }
}
