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
  noteKind?: 'highlight' | 'quote' | string
  sourcePath?: string
  sourceName?: string
  conversationId?: string
  messageId?: number
  locateRequestId?: number
  locateFeedbackKey?: string
  createdAt?: number
  updatedAt?: number
  feedback?: 'useful' | 'needs_check' | 'wrong' | string
  feedbackAt?: number
  headingPath?: string
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

export const READER_SELECTION_SHELF_EVENT = 'kb:reader-selection-shelf'
export const READER_SELECTION_SHELF_CHANNEL = 'kb:reader-selection-shelf'
export const READER_CITATION_SHELF_EVENT = 'kb:reader-citation-shelf'
export const READER_CITATION_SHELF_CHANNEL = 'kb:reader-citation-shelf'
export const READER_SESSION_SYNC_CHANNEL = 'kb:reader-session-sync'
export const READER_SESSION_NAV_CHANNEL = 'kb:reader-session-nav'
export const CHAT_MAIN_WINDOW_NAME = 'kb-chat-main'
export const READER_STANDALONE_WINDOW_NAME = 'kb-reader-standalone'

export interface ReaderSelectionShelfPayload {
  text: string
  sourcePath: string
  sourceName?: string
  headingPath?: string
  blockId?: string
  anchorId?: string
  anchorKind?: string
  startOffset?: number
  endOffset?: number
  occurrence?: number
  readableIndex?: number
  documentOccurrence?: number
  startReadableIndex?: number
  endReadableIndex?: number
  conversationId?: string
  projectId?: string
  createdAt?: number
}

export interface ReaderCitationShelfPayload {
  detail: Record<string, unknown>
  conversationId?: string
  projectId?: string
  createdAt?: number
}
