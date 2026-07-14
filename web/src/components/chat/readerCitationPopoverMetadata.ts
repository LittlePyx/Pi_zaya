import { referencesApi } from '../../api/references'
import { withBibliometricsLocale } from './bibliometricsLocale'
import { looksLowValueShelfSummary } from './citeShelfRuntime'
import { type CiteDetail } from './citationState'
import {
  isSystemBArticleSummarySource,
  isSystemBContextSummarySource,
  resolveSystemBArticleSummary,
} from './systemBArticleSummary'

export interface ReaderCitationPopoverMetadataClient {
  bibliometrics: (meta: Record<string, unknown>) => Promise<Record<string, unknown>>
  bibliometricsCached: (meta: Record<string, unknown>) => Promise<Record<string, unknown>>
  citationCardPolishCached: (
    detail: Record<string, unknown>,
    waitSeconds: number,
  ) => Promise<Record<string, unknown>>
}

export interface ReaderCitationPopoverMetadataPlan {
  itemKey: string
  missingReferenceEntry: boolean
  needsSummaryBackfill: boolean
  requestCount: number
  shouldFetchBibliometrics: boolean
  shouldFetchPolish: boolean
}

export interface LoadReaderCitationPopoverMetadataOptions {
  client?: ReaderCitationPopoverMetadataClient
  plan?: ReaderCitationPopoverMetadataPlan
}

export interface ReaderCitationPopoverMetadataResult {
  metas: Array<Record<string, unknown>>
  plan: ReaderCitationPopoverMetadataPlan
}

function emptyMetaOnFailure(request: Promise<Record<string, unknown>>): Promise<Record<string, unknown>> {
  return request.catch(() => ({}))
}

function readerCitationLooksContextOnly(detail: CiteDetail, summaryLine: string, summarySource: string): boolean {
  if (isSystemBArticleSummarySource(summarySource)) return false
  const contextSource = String(
    detail.citationContextSource
    || detail.evidenceSource
    || detail.shelfOrigin
    || '',
  ).trim().toLowerCase()
  if (isSystemBContextSummarySource(summarySource) || isSystemBContextSummarySource(contextSource)) {
    return true
  }
  if (detail.isInpaper && !summarySource) return true
  return /opened paper cites|bibliography entry is linked|current paper cites|当前论文|本文引用|上游文献|参考文献条目/i.test(summaryLine)
}

export function readerCitationHasArticleSummary(detail: CiteDetail): boolean {
  const summaryLine = String(detail.summaryLine || '').trim()
  if (!summaryLine || looksLowValueShelfSummary(summaryLine)) return false
  const systemBDecision = resolveSystemBArticleSummary(detail)
  if (systemBDecision.isSystemB) return systemBDecision.visible
  const summarySource = String(detail.summarySource || '').trim().toLowerCase()
  if (readerCitationLooksContextOnly(detail, summaryLine, summarySource)) return false
  const quality = detail.summaryQuality || {}
  const qualityOk = quality.ok === true || String(quality.status || '').trim().toLowerCase() === 'grounded'
  return Boolean(isSystemBArticleSummarySource(summarySource) || (!detail.isInpaper && qualityOk))
}

export function readerMetaHasArticleSummary(meta: Record<string, unknown>): boolean {
  const summaryLine = String(meta.summary_line || meta.summaryLine || '').trim()
  const summarySource = String(meta.summary_source || meta.summarySource || '').trim().toLowerCase()
  return Boolean(summaryLine && isSystemBArticleSummarySource(summarySource))
}

export function readerCitationHasMissingReferenceEntry(detail: CiteDetail): boolean {
  const status = String(detail.bindingStatus || '').trim().toLowerCase()
  if (status === 'missing_reference_entry') return true
  return Array.isArray(detail.cardQualityFlags) && detail.cardQualityFlags.includes('missing_reference_entry')
}

export function orderReaderCitationPopoverMetas(
  metas: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
  const usable = metas.filter((meta) => meta && Object.keys(meta).length > 0)
  return [
    ...usable.filter((meta) => !readerMetaHasArticleSummary(meta)),
    ...usable.filter(readerMetaHasArticleSummary),
  ]
}

export function buildReaderCitationPopoverMetadataPlan(
  detail: CiteDetail,
  itemKey: string,
): ReaderCitationPopoverMetadataPlan {
  const hasDoi = Boolean(String(detail.doi || detail.doiUrl || '').trim())
  const missingReferenceEntry = readerCitationHasMissingReferenceEntry(detail)
  const needsSummaryBackfill = !readerCitationHasArticleSummary(detail)
  const hasBibliometricsSeed = Boolean(hasDoi || detail.title || detail.raw || detail.citeFmt)
  const shouldFetchBibliometrics = Boolean(
    !missingReferenceEntry
    && (needsSummaryBackfill || !detail.bibliometricsChecked)
    && hasBibliometricsSeed,
  )
  const shouldFetchPolish = !missingReferenceEntry
  const requestCount = Number(shouldFetchBibliometrics) + Number(shouldFetchPolish)

  return {
    itemKey,
    missingReferenceEntry,
    needsSummaryBackfill,
    requestCount,
    shouldFetchBibliometrics,
    shouldFetchPolish,
  }
}

export async function loadReaderCitationPopoverMetadata(
  detail: CiteDetail,
  options: LoadReaderCitationPopoverMetadataOptions = {},
): Promise<ReaderCitationPopoverMetadataResult> {
  const client = options.client ?? referencesApi
  const plan = options.plan ?? buildReaderCitationPopoverMetadataPlan(detail, '')
  if (plan.requestCount <= 0) {
    return { metas: [], plan }
  }

  const reqs: Array<Promise<Record<string, unknown>>> = []
  if (plan.shouldFetchBibliometrics) {
    const payload = withBibliometricsLocale(detail as unknown as Record<string, unknown>)
    const loadBibliometrics = plan.needsSummaryBackfill
      ? client.bibliometrics
      : client.bibliometricsCached
    reqs.push(emptyMetaOnFailure(loadBibliometrics(payload)))
  }
  if (plan.shouldFetchPolish) {
    reqs.push(emptyMetaOnFailure(client.citationCardPolishCached(
      detail as unknown as Record<string, unknown>,
      1.5,
    )))
  }

  return {
    metas: orderReaderCitationPopoverMetas(await Promise.all(reqs)),
    plan,
  }
}
