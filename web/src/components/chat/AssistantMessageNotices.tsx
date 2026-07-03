import { Typography } from 'antd'
import type { Message } from '../../api/chat'
import { getMessageNoticeValue, isAssistantSourceNoticeText } from './messageRenderPacket'
import { getMessageAgentSourceSummary, getMessageAnswerContract } from './messageTraceUtils'
import type { LowConfidenceMetaLite } from './messageLowConfidence'
import {
  type AnswerSourceNoticeViewModel,
  buildAnswerSourceNoticeViewModel,
  labelForSourceNoticeText,
} from './answerContractViewModel'

const { Text } = Typography

interface AssistantMessageNoticesProps {
  message: Message
  lowConfidenceMeta: LowConfidenceMetaLite | null
  provenanceModeLabel: string
  S: Record<string, string>
  onOpenEvidence?: (sourceNotice: AnswerSourceNoticeViewModel) => void
}

interface ProvenanceModeDebugWindow {
  __KB_SHOW_PROVENANCE_MODE_LABEL__?: boolean
}

function shouldShowProvenanceModeLabel(): boolean {
  return Boolean((globalThis as ProvenanceModeDebugWindow).__KB_SHOW_PROVENANCE_MODE_LABEL__)
}

export function AssistantSourceNotice({
  noticeText,
  S,
  labelText,
  titleText,
  onClick,
}: {
  noticeText: string
  S: Record<string, string>
  labelText?: string
  titleText?: string
  onClick?: () => void
}) {
  if (!noticeText) return null
  const className = `kb-assistant-source-notice${onClick ? ' is-clickable' : ''}`
  const content = (
    <>
      <span className="kb-assistant-source-dot" />
      <span>{labelText || labelForSourceNoticeText(noticeText, S)}</span>
    </>
  )
  if (onClick) {
    return (
      <button
        type="button"
        className={className}
        title={titleText || noticeText}
        onClick={onClick}
        data-testid="assistant-source-notice"
      >
        {content}
      </button>
    )
  }
  return (
    <div className={className} title={titleText || noticeText} data-testid="assistant-source-notice">
      {content}
    </div>
  )
}

export function AssistantSourceSummaryNotice({
  answerContract,
  sourceSummary,
  fallbackNoticeText = '',
  S,
  onOpenEvidence,
}: {
  answerContract?: Record<string, unknown> | null | undefined
  sourceSummary: Record<string, unknown> | null | undefined
  fallbackNoticeText?: string
  S: Record<string, string>
  onOpenEvidence?: (sourceNotice: AnswerSourceNoticeViewModel) => void
}) {
  const viewModel = buildAnswerSourceNoticeViewModel({
    answerContract,
    legacySourceSummary: sourceSummary,
    fallbackNoticeText,
    S,
  })
  if (!viewModel) return null
  return (
    <AssistantSourceNotice
      noticeText={viewModel.title}
      titleText={viewModel.title}
      labelText={viewModel.label}
      onClick={onOpenEvidence ? () => onOpenEvidence(viewModel) : undefined}
      S={S}
    />
  )
}

export function AssistantMessageNotices({
  message,
  lowConfidenceMeta,
  provenanceModeLabel,
  S,
  onOpenEvidence,
}: AssistantMessageNoticesProps) {
  const noticeText = getMessageNoticeValue(message)
  const sourceNotice = Boolean(noticeText && isAssistantSourceNoticeText(noticeText))
  const answerContract = getMessageAnswerContract(message)
  const sourceSummary = getMessageAgentSourceSummary(message)
  const sourceNoticeViewModel = buildAnswerSourceNoticeViewModel({
    answerContract,
    legacySourceSummary: sourceSummary,
    fallbackNoticeText: noticeText || '',
    allowFallbackNotice: sourceNotice,
    S,
  })
  const showProvenanceLabel = shouldShowProvenanceModeLabel() && Boolean(provenanceModeLabel)

  if (!noticeText && !sourceNoticeViewModel && !lowConfidenceMeta && !showProvenanceLabel) return null

  return (
    <>
      {sourceNoticeViewModel ? (
        <AssistantSourceNotice
          noticeText={sourceNoticeViewModel.title}
          titleText={sourceNoticeViewModel.title}
          labelText={sourceNoticeViewModel.label}
          onClick={onOpenEvidence ? () => onOpenEvidence(sourceNoticeViewModel) : undefined}
          S={S}
        />
      ) : noticeText && sourceNotice ? (
        <AssistantSourceNotice noticeText={noticeText} S={S} />
      ) : null}
      {noticeText && !sourceNotice ? (
        <div className="mb-4 rounded-2xl border border-[var(--border)] bg-black/[0.03] px-4 py-3 text-sm text-black/70 dark:bg-white/[0.04] dark:text-white/70">
          {noticeText}
        </div>
      ) : null}
      {lowConfidenceMeta ? (
        <div className="mb-4 rounded-2xl border border-amber-300/70 bg-amber-50/80 px-4 py-3 text-sm text-amber-900 dark:border-amber-300/50 dark:bg-amber-300/10 dark:text-amber-100">
          <div className="font-medium">
            {lowConfidenceMeta.isZh ? S.msg_retrieval_low_confidence : 'Lower retrieval confidence'}
          </div>
          <div className="mt-1">
            {lowConfidenceMeta.isZh
              ? S.msg_retrieval_low_reason.replace('{text}', lowConfidenceMeta.reasonText)
              : `Reason: ${lowConfidenceMeta.reasonText}.`}
          </div>
          {lowConfidenceMeta.candidateRefs.length > 0 ? (
            <div className="mt-1">
              {lowConfidenceMeta.isZh
                ? S.msg_retrieval_candidate_refs.replace('{refs}', lowConfidenceMeta.candidateRefs.map((num) => `[${num}]`).join(', '))
                : `Candidate refs for cross-check: ${lowConfidenceMeta.candidateRefs.map((num) => `[${num}]`).join(', ')}.`}
            </div>
          ) : null}
        </div>
      ) : null}
      {showProvenanceLabel ? (
        <div className="mb-2">
          <Text type="secondary" className="text-xs">{provenanceModeLabel}</Text>
        </div>
      ) : null}
    </>
  )
}
