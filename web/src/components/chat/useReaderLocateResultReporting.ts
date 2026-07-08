import { useEffect } from 'react'

import type { ReaderLocateResult } from './reader/readerTypes'

const READER_LOAD_FAILED_MESSAGE = 'Reader source could not be loaded.'

export interface BuildReaderLocateSuccessReportOptions {
  locateFeedbackKey?: string
  locateResult: ReaderLocateResult
  sourceName: string
  title: string
}

export interface BuildReaderLocateFailureReportOptions {
  activeAltIndex: number
  activeAnchorId: string
  activeAnchorKind: string
  activeBlockId: string
  activeHeadingPath: string
  error: string
  locateFeedbackKey?: string
  locateRequestId: number
  sourceName: string
  sourcePath: string
  strictLocate: boolean
  title: string
}

export interface UseReaderLocateResultReportingOptions extends BuildReaderLocateFailureReportOptions {
  locateResult: ReaderLocateResult | null
  onLocateResult?: (result: ReaderLocateResult) => void
  open: boolean
}

export function buildReaderLocateSuccessReport({
  locateFeedbackKey,
  locateResult,
  sourceName,
  title,
}: BuildReaderLocateSuccessReportOptions): ReaderLocateResult {
  return {
    ...locateResult,
    sourceName: sourceName || title || undefined,
    locateFeedbackKey: String(locateFeedbackKey || locateResult.locateFeedbackKey || '').trim() || undefined,
  }
}

export function buildReaderLocateFailureReport({
  activeAltIndex,
  activeAnchorId,
  activeAnchorKind,
  activeBlockId,
  activeHeadingPath,
  error,
  locateFeedbackKey,
  locateRequestId,
  sourceName,
  sourcePath,
  strictLocate,
  title,
}: BuildReaderLocateFailureReportOptions): ReaderLocateResult {
  const message = String(error || '').trim() || READER_LOAD_FAILED_MESSAGE
  return {
    locateRequestId,
    sourcePath,
    sourceName: sourceName || title || undefined,
    locateFeedbackKey: String(locateFeedbackKey || '').trim() || undefined,
    status: 'failed',
    precision: 'failed',
    ok: false,
    repairable: true,
    strictLocate,
    hint: message,
    reason: message,
    activeAltIndex,
    blockId: activeBlockId || undefined,
    anchorId: activeAnchorId || undefined,
    anchorKind: activeAnchorKind || undefined,
    headingPath: activeHeadingPath || undefined,
  }
}

export function useReaderLocateResultReporting({
  activeAltIndex,
  activeAnchorId,
  activeAnchorKind,
  activeBlockId,
  activeHeadingPath,
  error,
  locateFeedbackKey,
  locateRequestId,
  locateResult,
  onLocateResult,
  open,
  sourceName,
  sourcePath,
  strictLocate,
  title,
}: UseReaderLocateResultReportingOptions) {
  useEffect(() => {
    if (!open || !locateResult || !onLocateResult) return
    onLocateResult(buildReaderLocateSuccessReport({
      locateFeedbackKey,
      locateResult,
      sourceName,
      title,
    }))
  }, [locateFeedbackKey, locateResult, onLocateResult, open, sourceName, title])

  useEffect(() => {
    if (!open || !error || !onLocateResult || !sourcePath) return
    onLocateResult(buildReaderLocateFailureReport({
      activeAltIndex,
      activeAnchorId,
      activeAnchorKind,
      activeBlockId,
      activeHeadingPath,
      error,
      locateFeedbackKey,
      locateRequestId,
      sourceName,
      sourcePath,
      strictLocate,
      title,
    }))
  }, [
    activeAltIndex,
    activeAnchorId,
    activeAnchorKind,
    activeBlockId,
    activeHeadingPath,
    error,
    locateFeedbackKey,
    locateRequestId,
    onLocateResult,
    open,
    sourceName,
    sourcePath,
    strictLocate,
    title,
  ])
}
