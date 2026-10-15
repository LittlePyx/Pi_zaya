import type { UnlinkedReferenceView } from './messageCitationViews'
import { MessageProvenanceChips } from './MessageProvenanceChips'
import { MessageReferenceCandidates } from './MessageReferenceCandidates'
import { ResearchContextReceipt } from './ResearchContextReceipt'
import type { ProvenanceLocateEntry } from './reader/messageStructuredProvenance'
import type { SelectedResearchContextPack } from './researchContextPack'
import type { AssistantEvidenceNavigationInteractions } from './assistantEvidenceNavigationTypes'

interface AssistantEvidenceNavigationPanelsProps {
  contextPack?: SelectedResearchContextPack | null
  interactions: AssistantEvidenceNavigationInteractions
  messageId: number
  provenanceEntries: ProvenanceLocateEntry[]
  referenceViews: UnlinkedReferenceView[]
  S: Record<string, string>
}

export function AssistantEvidenceNavigationPanels({
  contextPack,
  interactions,
  messageId,
  provenanceEntries,
  referenceViews,
  S,
}: AssistantEvidenceNavigationPanelsProps) {
  return (
    <>
      <ResearchContextReceipt
        pack={contextPack}
        onOpenReader={interactions.onOpenReader}
        onFollowUp={interactions.onFollowUp}
        S={S}
      />
      <MessageReferenceCandidates
        views={referenceViews}
        messageId={messageId}
        canOpenReader={Boolean(interactions.onOpenReader)}
        onOpenReader={interactions.onOpenReference}
        onAddToShelf={interactions.onAddReferenceToShelf}
        S={S}
      />
      <MessageProvenanceChips
        entries={provenanceEntries}
        messageId={messageId}
        onOpenEntry={interactions.onOpenProvenanceEntry}
      />
    </>
  )
}
