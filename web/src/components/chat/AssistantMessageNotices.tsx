import { Typography } from 'antd'
import type { Message } from '../../api/chat'
import type { LowConfidenceMetaLite } from './messageLowConfidence'
import {
  type AnswerSourceNoticeViewModel,
  buildAnswerSourceNoticeViewModel,
  labelForSourceNoticeText,
} from './answerSourceNoticeViewModel'
import { buildAssistantMessageNoticeViewModel } from './assistantMessageNoticeViewModel'
import {
  hasAutomaticLocalSemanticFallback,
  messageHasAutomaticLocalSemanticFallback,
} from './localRetrievalOutcome'

const { Text } = Typography

interface AssistantMessageNoticesProps {
  message: Message
  lowConfidenceMeta: LowConfidenceMetaLite | null
  provenanceModeLabel: string
  S: Record<string, string>
  onOpenEvidence?: (sourceNotice: AnswerSourceNoticeViewModel) => void
  localEvidenceCount?: number
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
  testId = 'assistant-source-notice',
}: {
  noticeText: string
  S: Record<string, string>
  labelText?: string
  titleText?: string
  onClick?: () => void
  testId?: string
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
        data-testid={testId}
      >
        {content}
      </button>
    )
  }
  return (
    <div className={className} title={titleText || noticeText} data-testid={testId}>
      {content}
    </div>
  )
}

export function AssistantRetrievalFallbackNotice({
  trace,
  S,
}: {
  trace: unknown
  S: Record<string, string>
}) {
  if (!hasAutomaticLocalSemanticFallback(trace)) return null
  return (
    <AssistantSourceNotice
      noticeText={S.chat_retrieval_auto_fallback_tip}
      titleText={S.chat_retrieval_auto_fallback_tip}
      labelText={S.chat_retrieval_auto_fallback}
      testId="assistant-retrieval-fallback"
      S={S}
    />
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
  localEvidenceCount = 0,
}: AssistantMessageNoticesProps) {
  const viewModel = buildAssistantMessageNoticeViewModel({
    message,
    lowConfidenceMeta,
    provenanceModeLabel,
    showProvenanceModeLabel: shouldShowProvenanceModeLabel(),
    localEvidenceCount,
    S,
  })
  const sourceNoticeViewModel = viewModel.sourceNoticeViewModel
  const showRetrievalFallback = messageHasAutomaticLocalSemanticFallback(message)

  if (!viewModel.hasVisibleNotice && !showRetrievalFallback) return null

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
      ) : viewModel.legacySourceNoticeText ? (
        <AssistantSourceNotice noticeText={viewModel.legacySourceNoticeText} S={S} />
      ) : null}
      {showRetrievalFallback ? (
        <AssistantRetrievalFallbackNotice trace={message.meta?.research_trace} S={S} />
      ) : null}
      {viewModel.plainNoticeText ? (
        <div className="mb-4 rounded-2xl border border-[var(--border)] bg-black/[0.03] px-4 py-3 text-sm text-black/70 dark:bg-white/[0.04] dark:text-white/70">
          {viewModel.plainNoticeText}
        </div>
      ) : null}
      {viewModel.showLowConfidence && viewModel.lowConfidenceMeta ? (
        <div className="mb-4 rounded-2xl border border-amber-300/70 bg-amber-50/80 px-4 py-3 text-sm text-amber-900 dark:border-amber-300/50 dark:bg-amber-300/10 dark:text-amber-100">
          <div className="font-medium">
            {viewModel.lowConfidenceMeta.isZh ? S.msg_retrieval_low_confidence : 'Lower retrieval confidence'}
          </div>
          <div className="mt-1">
            {viewModel.lowConfidenceMeta.isZh
              ? S.msg_retrieval_low_reason.replace('{text}', viewModel.lowConfidenceMeta.reasonText)
              : `Reason: ${viewModel.lowConfidenceMeta.reasonText}.`}
          </div>
          {viewModel.lowConfidenceMeta.candidateRefs.length > 0 ? (
            <div className="mt-1">
              {viewModel.lowConfidenceMeta.isZh
                ? S.msg_retrieval_candidate_refs.replace('{refs}', viewModel.lowConfidenceMeta.candidateRefs.map((num) => `[${num}]`).join(', '))
                : `Candidate refs for cross-check: ${viewModel.lowConfidenceMeta.candidateRefs.map((num) => `[${num}]`).join(', ')}.`}
            </div>
          ) : null}
        </div>
      ) : null}
      {viewModel.showProvenanceLabel ? (
        <div className="mb-2">
          <Text type="secondary" className="text-xs">{viewModel.provenanceModeLabel}</Text>
        </div>
      ) : null}
    </>
  )
}
