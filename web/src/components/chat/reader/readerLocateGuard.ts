import type { ReaderLocateResult, ReaderOpenPayload } from './readerTypes'
import { normalizeSourcePathForMatch } from '../../../utils/sourcePath'

export interface ReaderLocateRequestGuard {
  locateRequestId: number
  sourcePath: string
  conversationId: string
}

export function normalizeReaderSourcePathForMatch(value: unknown): string {
  return normalizeSourcePathForMatch(value)
}

export function readerSourcePathsMatch(left: unknown, right: unknown): boolean {
  const leftNorm = normalizeReaderSourcePathForMatch(left)
  const rightNorm = normalizeReaderSourcePathForMatch(right)
  return Boolean(leftNorm && rightNorm && leftNorm === rightNorm)
}

export function normalizeReaderLocateRequestId(value: unknown): number {
  return Number.isFinite(Number(value || 0))
    ? Math.max(0, Math.floor(Number(value || 0)))
    : 0
}

export function readerLocateResultMatchesActiveRequest(opts: {
  result: ReaderLocateResult
  guard?: ReaderLocateRequestGuard | null
  currentPayload?: ReaderOpenPayload | null
  currentConversationId?: string | null
  readerOpen: boolean
}): boolean {
  const feedbackKey = String(opts.result.locateFeedbackKey || '').trim()
  const sourcePath = normalizeReaderSourcePathForMatch(opts.result.sourcePath)
  const locateRequestId = normalizeReaderLocateRequestId(opts.result.locateRequestId)
  const guard = opts.guard || null
  const payload = opts.currentPayload || null
  if (!opts.readerOpen || !feedbackKey || !sourcePath || locateRequestId <= 0 || !guard || !payload) return false
  if (guard.locateRequestId !== locateRequestId) return false
  if (String(guard.conversationId || '').trim() !== String(opts.currentConversationId || '').trim()) return false
  if (!readerSourcePathsMatch(guard.sourcePath, sourcePath)) return false
  if (String(payload.locateFeedbackKey || '').trim() !== feedbackKey) return false
  if (normalizeReaderLocateRequestId(payload.locateRequestId) !== locateRequestId) return false
  if (!readerSourcePathsMatch(payload.sourcePath, sourcePath)) return false
  return true
}

export function readerLocateRepairRunMatchesActiveRequest(opts: {
  expectedRunToken: number
  currentRunToken: number
  result: ReaderLocateResult
  guard?: ReaderLocateRequestGuard | null
  currentPayload?: ReaderOpenPayload | null
  currentConversationId?: string | null
  readerOpen: boolean
}): boolean {
  if (opts.currentRunToken !== opts.expectedRunToken) return false
  return readerLocateResultMatchesActiveRequest(opts)
}
