import { Typography } from 'antd'
import type { Message } from '../../api/chat'
import { getMessageNoticeValue, isAssistantSourceNoticeText } from './messageRenderPacket'
import type { LowConfidenceMetaLite } from './messageLowConfidence'

const { Text } = Typography

interface AssistantMessageNoticesProps {
  message: Message
  lowConfidenceMeta: LowConfidenceMetaLite | null
  provenanceModeLabel: string
  S: Record<string, string>
}

interface ProvenanceModeDebugWindow {
  __KB_SHOW_PROVENANCE_MODE_LABEL__?: boolean
}

function shouldShowProvenanceModeLabel(): boolean {
  return Boolean((globalThis as ProvenanceModeDebugWindow).__KB_SHOW_PROVENANCE_MODE_LABEL__)
}

function sourceNoticeLabel(noticeText: string, S: Record<string, string>): string {
  if (/local citations\s*\[n\]\s*come from the knowledge base/i.test(noticeText) || /带\s*\[n\]\s*的内容来自本地知识库/.test(noticeText)) {
    return S.agent_trace_source_local_external || 'Local + external'
  }
  return S.agent_trace_evidence_not_from_kb || 'Not from KB'
}

export function AssistantSourceNotice({
  noticeText,
  S,
}: {
  noticeText: string
  S: Record<string, string>
}) {
  if (!noticeText) return null
  return (
    <div className="kb-assistant-source-notice" title={noticeText} data-testid="assistant-source-notice">
      <span className="kb-assistant-source-dot" />
      <span>{sourceNoticeLabel(noticeText, S)}</span>
    </div>
  )
}

export function AssistantMessageNotices({
  message,
  lowConfidenceMeta,
  provenanceModeLabel,
  S,
}: AssistantMessageNoticesProps) {
  const noticeText = getMessageNoticeValue(message)
  const sourceNotice = noticeText && isAssistantSourceNoticeText(noticeText)
  const showProvenanceLabel = shouldShowProvenanceModeLabel() && Boolean(provenanceModeLabel)

  if (!noticeText && !lowConfidenceMeta && !showProvenanceLabel) return null

  return (
    <>
      {noticeText && sourceNotice ? (
        <AssistantSourceNotice noticeText={noticeText} S={S} />
      ) : noticeText ? (
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
