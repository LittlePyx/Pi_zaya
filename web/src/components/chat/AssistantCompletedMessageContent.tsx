import type { MouseEvent } from 'react'
import type { Message } from '../../api/chat'
import type { AnswerSourceNoticeViewModel } from './answerSourceNoticeViewModel'
import { AssistantMessageNotices } from './AssistantMessageNotices'
import type { CiteDetail } from './citationState'
import { MarkdownRenderer } from './MarkdownRenderer'
import type { MessageMarkdownLocateProps } from './messageMarkdownLocateProps'
import type { LowConfidenceMetaLite } from './messageLowConfidence'

interface AssistantCompletedMessageContentProps {
  bodyContent: string
  citeDetails: CiteDetail[]
  linkifyPlainCitations: boolean
  lowConfidenceMeta: LowConfidenceMetaLite | null
  markdownLocateProps: MessageMarkdownLocateProps
  message: Message
  onCitationClick: (detail: CiteDetail, event: MouseEvent<HTMLElement>) => void
  onCitationHover: (detail: CiteDetail, event: MouseEvent<HTMLElement>) => void
  onCitationLeave: () => void
  onOpenEvidence: (sourceNotice: AnswerSourceNoticeViewModel) => void
  provenanceModeLabel: string
  S: Record<string, string>
}

export function AssistantCompletedMessageContent({
  bodyContent,
  citeDetails,
  linkifyPlainCitations,
  lowConfidenceMeta,
  markdownLocateProps,
  message,
  onCitationClick,
  onCitationHover,
  onCitationLeave,
  onOpenEvidence,
  provenanceModeLabel,
  S,
}: AssistantCompletedMessageContentProps) {
  return (
    <>
      <AssistantMessageNotices
        message={message}
        lowConfidenceMeta={lowConfidenceMeta}
        provenanceModeLabel={provenanceModeLabel}
        onOpenEvidence={onOpenEvidence}
        S={S}
      />
      <MarkdownRenderer
        content={bodyContent}
        citeDetails={citeDetails}
        linkifyPlainCitations={linkifyPlainCitations}
        onCitationClick={onCitationClick}
        onCitationHover={onCitationHover}
        onCitationLeave={onCitationLeave}
        {...markdownLocateProps}
      />
    </>
  )
}
