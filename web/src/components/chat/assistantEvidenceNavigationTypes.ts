import type { CiteDetail } from './citationState'
import type { ReaderOpenPayload } from './reader/readerTypes'
import type { ProvenanceLocateEntry } from './reader/messageStructuredProvenance'
import type { SelectedResearchContextItem, SelectedResearchContextPack } from './researchContextPack'

export interface AssistantEvidenceNavigationInteractions {
  onAddReferenceToShelf: (detail: CiteDetail) => void
  onFollowUp?: (pack: SelectedResearchContextPack, item: SelectedResearchContextItem) => void
  onOpenProvenanceEntry: (entry: ProvenanceLocateEntry, snippet: string) => void
  onOpenReader?: (payload: ReaderOpenPayload) => void
  onOpenReference: (detail: CiteDetail) => void
}
