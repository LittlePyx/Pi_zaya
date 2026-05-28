export interface ReaderLocateCandidate {
  headingPath?: string
  snippet?: string
  highlightSnippet?: string
  anchorId?: string
  blockId?: string
  anchorKind?: string
  anchorNumber?: number
}

export interface ReaderLocateTarget extends ReaderLocateCandidate {
  segmentId?: string
  sourceSegmentId?: string
  evidenceQuote?: string
  anchorText?: string
  hitLevel?: 'exact' | 'block' | 'heading' | 'none' | string
  claimType?: string
  locatePolicy?: string
  locateSurfacePolicy?: string
  snippetAliases?: string[]
  relatedBlockIds?: string[]
}

export interface ReaderLocateClaimGroup {
  id?: string
  kind?: string
  leadText?: string
  distance?: number
}

export interface ReaderOpenPayload {
  sourcePath: string
  sourceName?: string
  headingPath?: string
  snippet?: string
  highlightSnippet?: string
  anchorId?: string
  blockId?: string
  relatedBlockIds?: string[]
  anchorKind?: string
  anchorNumber?: number
  strictLocate?: boolean
  locateMode?: 'heuristic'
  locateTarget?: ReaderLocateTarget
  claimGroup?: ReaderLocateClaimGroup
  locateRequestId?: number
  alternatives?: ReaderLocateCandidate[]
  visibleAlternatives?: ReaderLocateCandidate[]
  evidenceAlternatives?: ReaderLocateCandidate[]
  initialAltIndex?: number
  locateFeedbackKey?: string
}

export type ReaderLocateResultStatus = 'exact' | 'block' | 'fuzzy' | 'section' | 'source_only' | 'failed'

export type ReaderLocateResultPrecision =
  | 'exact_anchor'
  | 'block'
  | 'phrase'
  | 'fuzzy'
  | 'section'
  | 'source_only'
  | 'failed'

export interface ReaderLocateResult {
  locateRequestId: number
  sourcePath: string
  sourceName?: string
  locateFeedbackKey?: string
  status: ReaderLocateResultStatus
  precision: ReaderLocateResultPrecision
  ok: boolean
  repairable: boolean
  strictLocate: boolean
  hint: string
  reason: string
  activeAltIndex?: number
  blockId?: string
  anchorId?: string
  anchorKind?: string
  headingPath?: string
}

export interface ReaderSessionHighlight {
  id: string
  text: string
  startOffset?: number
  endOffset?: number
  // Legacy compatibility fields for pre-range highlights. Prefer start/end offsets.
  blockId?: string
  anchorId?: string
  occurrence?: number
  readableIndex?: number
  documentOccurrence?: number
  startReadableIndex?: number
  endReadableIndex?: number
}
