import { createElement } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import type { ReaderLocateResult } from '../components/chat/reader/readerTypes'
import {
  buildReaderLocateFailureReport,
  buildReaderLocateSuccessReport,
  useReaderLocateResultReporting,
} from '../components/chat/useReaderLocateResultReporting'

export interface ReaderLocateResultReportingSmokeResult {
  fallbackFailure: ReaderLocateResult
  failureReport: ReaderLocateResult
  hookReports: ReaderLocateResult[]
  renderedText: string
  successReport: ReaderLocateResult
  successWithoutPayloadKey: ReaderLocateResult
}

const locatedResult = {
  activeAltIndex: 1,
  anchorId: 'anchor-a',
  anchorKind: 'paragraph',
  blockId: 'block-a',
  headingPath: 'Methods',
  hint: 'Located block-a',
  locateFeedbackKey: 'engine-key',
  locateRequestId: 12,
  ok: true,
  precision: 'block',
  reason: 'Located block-a',
  repairable: false,
  sourcePath: '/tmp/reader.md',
  status: 'block',
  strictLocate: false,
} as ReaderLocateResult

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve())
  })
}

async function settleEffects(): Promise<void> {
  await nextFrame()
  await nextFrame()
}

export async function runReaderLocateResultReportingSmoke(): Promise<ReaderLocateResultReportingSmokeResult> {
  const successReport = buildReaderLocateSuccessReport({
    locateFeedbackKey: ' payload-key ',
    locateResult: locatedResult,
    sourceName: '',
    title: 'Reader Title',
  })
  const successWithoutPayloadKey = buildReaderLocateSuccessReport({
    locateFeedbackKey: '',
    locateResult: locatedResult,
    sourceName: 'Reader Source',
    title: 'Reader Title',
  })
  const failureReport = buildReaderLocateFailureReport({
    activeAltIndex: 2,
    activeAnchorId: 'anchor-failed',
    activeAnchorKind: 'equation',
    activeBlockId: 'block-failed',
    activeHeadingPath: 'Results',
    error: ' Source file missing ',
    locateFeedbackKey: ' failed-key ',
    locateRequestId: 17,
    sourceName: '',
    sourcePath: '/tmp/missing.md',
    strictLocate: true,
    title: 'Missing Reader',
  })
  const fallbackFailure = buildReaderLocateFailureReport({
    activeAltIndex: 0,
    activeAnchorId: '',
    activeAnchorKind: '',
    activeBlockId: '',
    activeHeadingPath: '',
    error: '   ',
    locateFeedbackKey: '',
    locateRequestId: 18,
    sourceName: '',
    sourcePath: '/tmp/blank-error.md',
    strictLocate: false,
    title: '',
  })

  const host = document.createElement('div')
  document.body.append(host)
  const root = createRoot(host)
  const hookReports: ReaderLocateResult[] = []

  function Harness({
    error,
    locateResult,
    open,
  }: {
    error: string
    locateResult: ReaderLocateResult | null
    open: boolean
  }) {
    useReaderLocateResultReporting({
      activeAltIndex: 3,
      activeAnchorId: 'hook-anchor',
      activeAnchorKind: 'figure',
      activeBlockId: 'hook-block',
      activeHeadingPath: 'Hook Heading',
      error,
      locateFeedbackKey: ' hook-key ',
      locateRequestId: 21,
      locateResult,
      onLocateResult: (result) => {
        hookReports.push(result)
      },
      open,
      sourceName: '',
      sourcePath: '/tmp/hook.md',
      strictLocate: true,
      title: 'Hook Reader',
    })
    return createElement('div', { id: 'reader-locate-result-reporting-smoke' }, open ? 'open' : 'closed')
  }

  flushSync(() => {
    root.render(createElement(Harness, {
      error: '',
      locateResult: locatedResult,
      open: true,
    }))
  })
  await settleEffects()
  flushSync(() => {
    root.render(createElement(Harness, {
      error: ' Hook load failed ',
      locateResult: null,
      open: true,
    }))
  })
  await settleEffects()

  const renderedText = host.textContent || ''
  root.unmount()
  host.remove()

  return {
    fallbackFailure,
    failureReport,
    hookReports,
    renderedText,
    successReport,
    successWithoutPayloadKey,
  }
}
