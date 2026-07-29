import { referencesApi } from '../../api/references'
import { withBibliometricsLocale } from './bibliometricsLocale'
import { toShelfItem, type CiteDetail } from './citationState'
import { shelfItemNeedsSummaryBackfill } from './citeShelfRuntime'

export interface CitationPopoverMetadataClient {
  bibliometrics: (meta: Record<string, unknown>) => Promise<Record<string, unknown>>
  bibliometricsCached: (meta: Record<string, unknown>) => Promise<Record<string, unknown>>
  citationMetaCached: (sourcePath: string) => Promise<Record<string, unknown>>
}

export interface CitationPopoverMetadataPlan {
  itemKey: string
  needsSummaryBackfill: boolean
  requestCount: number
  shouldFetchBibliometrics: boolean
  shouldFetchCitationMeta: boolean
  sourcePath: string
}

export interface LoadCitationPopoverMetadataOptions {
  client?: CitationPopoverMetadataClient
  plan?: CitationPopoverMetadataPlan
}

export interface CitationPopoverMetadataResult {
  metas: Array<Record<string, unknown>>
  plan: CitationPopoverMetadataPlan
}

function emptyMetaOnFailure(request: Promise<Record<string, unknown>>): Promise<Record<string, unknown>> {
  return request.catch(() => ({}))
}

export function buildCitationPopoverMetadataPlan(
  detail: CiteDetail,
  itemKey = toShelfItem(detail).key,
): CitationPopoverMetadataPlan {
  const sourcePath = String(detail.sourcePath || '').trim()
  const isInPaperReference = Boolean(detail.isInpaper)
  const shouldFetchCitationMeta = Boolean(sourcePath) && !isInPaperReference
  const hasSourceBoundMetadata = shouldFetchCitationMeta
  const hasDoi = Boolean(String(detail.doi || '').trim())
  const shelfItem = toShelfItem(detail)
  const needsSummaryBackfill = shelfItemNeedsSummaryBackfill(shelfItem)
  const hasBibliometricsSeed = isInPaperReference
    ? hasDoi
    : Boolean(detail.doi || detail.title || detail.venue || detail.raw || detail.citeFmt)
  const shouldFetchBibliometrics = Boolean(
    !hasSourceBoundMetadata
    && (!detail.bibliometricsChecked || needsSummaryBackfill)
    && hasBibliometricsSeed,
  )
  const requestCount = Number(shouldFetchCitationMeta) + Number(shouldFetchBibliometrics)

  return {
    itemKey,
    needsSummaryBackfill,
    requestCount,
    shouldFetchBibliometrics,
    shouldFetchCitationMeta,
    sourcePath,
  }
}

export async function loadCitationPopoverMetadata(
  detail: CiteDetail,
  options: LoadCitationPopoverMetadataOptions = {},
): Promise<CitationPopoverMetadataResult> {
  const client = options.client ?? referencesApi
  const plan = options.plan ?? buildCitationPopoverMetadataPlan(detail)
  if (plan.requestCount <= 0) {
    return { metas: [], plan }
  }

  const reqs: Array<Promise<Record<string, unknown>>> = []
  if (plan.shouldFetchCitationMeta && plan.sourcePath) {
    reqs.push(emptyMetaOnFailure(client.citationMetaCached(plan.sourcePath)))
  }
  if (plan.shouldFetchBibliometrics) {
    const payload = withBibliometricsLocale(detail as unknown as Record<string, unknown>)
    const loadBibliometrics = plan.needsSummaryBackfill
      ? client.bibliometrics
      : client.bibliometricsCached
    reqs.push(emptyMetaOnFailure(loadBibliometrics(payload)))
  }

  return {
    metas: await Promise.all(reqs),
    plan,
  }
}
