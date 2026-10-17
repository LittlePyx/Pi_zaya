import type { Message } from '../../api/chat'
import type { AnswerSourceNoticeViewModel } from './answerSourceNoticeViewModel'
import { buildAnswerSourceNoticeViewModel } from './answerSourceNoticeViewModel'
import type { LowConfidenceMetaLite } from './messageLowConfidence'
import { getMessageNoticeValue, isAssistantSourceNoticeText } from './messageRenderPacket'
import { getMessageAgentSourceSummary, getMessageAnswerContract } from './messageTraceUtils'

export interface AssistantMessageNoticeViewModel {
  sourceNoticeViewModel: AnswerSourceNoticeViewModel | null
  legacySourceNoticeText: string
  plainNoticeText: string
  lowConfidenceMeta: LowConfidenceMetaLite | null
  showLowConfidence: boolean
  provenanceModeLabel: string
  showProvenanceLabel: boolean
  hasVisibleNotice: boolean
}

export function buildAssistantMessageNoticeViewModel({
  message,
  lowConfidenceMeta,
  provenanceModeLabel,
  showProvenanceModeLabel,
  S,
  localEvidenceCount = 0,
}: {
  message: Message
  lowConfidenceMeta: LowConfidenceMetaLite | null
  provenanceModeLabel: string
  showProvenanceModeLabel: boolean
  S: Record<string, string>
  localEvidenceCount?: number
}): AssistantMessageNoticeViewModel {
  const noticeText = getMessageNoticeValue(message) || ''
  const isSourceNotice = Boolean(noticeText && isAssistantSourceNoticeText(noticeText))
  const sourceNoticeViewModel = buildAnswerSourceNoticeViewModel({
    answerContract: getMessageAnswerContract(message),
    legacySourceSummary: getMessageAgentSourceSummary(message),
    fallbackNoticeText: noticeText,
    allowFallbackNotice: isSourceNotice,
    S,
  }) || (localEvidenceCount > 0 ? {
    label: `${S.msg_evidence_label || 'Evidence'} · ${localEvidenceCount}`,
    title: S.evidence_local_sources_heading || 'Local evidence sources',
    kind: 'local_evidence_cards',
    usesLocalKnowledgeBase: true,
    usesExternalModel: false,
    requiresUserNotice: false,
  } : null)
  const legacySourceNoticeText = !sourceNoticeViewModel && isSourceNotice ? noticeText : ''
  const plainNoticeText = noticeText && !isSourceNotice ? noticeText : ''
  const showLowConfidence = Boolean(lowConfidenceMeta)
  const showProvenanceLabel = Boolean(showProvenanceModeLabel && provenanceModeLabel)

  return {
    sourceNoticeViewModel,
    legacySourceNoticeText,
    plainNoticeText,
    lowConfidenceMeta,
    showLowConfidence,
    provenanceModeLabel,
    showProvenanceLabel,
    hasVisibleNotice: Boolean(
      sourceNoticeViewModel
        || legacySourceNoticeText
        || plainNoticeText
        || showLowConfidence
        || showProvenanceLabel,
    ),
  }
}
