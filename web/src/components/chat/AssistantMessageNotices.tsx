import { Typography } from 'antd'
import type { Message } from '../../api/chat'
import { getMessageNoticeValue, isAssistantSourceNoticeText } from './messageRenderPacket'
import { getMessageAgentSourceSummary } from './messageTraceUtils'
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
  const mentionsLocalKb = /local citations\s*\[n\]\s*come from the knowledge base/i.test(noticeText)
    || (/[\u672c\u5730\u77e5\u8bc6\u6587\u732e\u5e93]/.test(noticeText) && /\[n\]/.test(noticeText))
  if (mentionsLocalKb) {
    return S.agent_trace_source_local_external || 'Local + external'
  }
  return S.agent_trace_evidence_not_from_kb || 'Not from KB'
}

function sourceSummaryLabel(summary: Record<string, unknown>, S: Record<string, string>): string {
  const labelKey = String(summary.label_key || summary.labelKey || '').trim()
  if (labelKey && S[labelKey]) return S[labelKey]
  const kind = String(summary.kind || '').trim()
  if (kind === 'local_kb') return S.agent_trace_source_local_only || 'Local KB'
  if (kind === 'local_plus_external') return S.agent_trace_source_local_external || 'Local + external'
  if (kind === 'external_not_kb' || kind === 'general_api') return S.agent_trace_evidence_not_from_kb || 'Not from KB'
  return String(summary.label || '').trim() || S.agent_trace_source_fallback || 'Source'
}

function shouldShowSourceSummary(summary: Record<string, unknown> | null): summary is Record<string, unknown> {
  if (!summary) return false
  if (summary.should_show === false || summary.shouldShow === false) return false
  const kind = String(summary.kind || '').trim()
  if (kind && kind !== 'unknown') return true
  return Boolean(String(summary.label || summary.label_key || summary.labelKey || '').trim())
}

function sourceSummaryTitle(summary: Record<string, unknown>, fallbackNotice: string): string {
  const detail = String(summary.detail || '').trim()
  if (detail) return detail
  return fallbackNotice || String(summary.label || '').trim()
}

export function AssistantSourceNotice({
  noticeText,
  S,
  labelText,
  titleText,
}: {
  noticeText: string
  S: Record<string, string>
  labelText?: string
  titleText?: string
}) {
  if (!noticeText) return null
  return (
    <div className="kb-assistant-source-notice" title={titleText || noticeText} data-testid="assistant-source-notice">
      <span className="kb-assistant-source-dot" />
      <span>{labelText || sourceNoticeLabel(noticeText, S)}</span>
    </div>
  )
}

export function AssistantSourceSummaryNotice({
  sourceSummary,
  fallbackNoticeText = '',
  S,
}: {
  sourceSummary: Record<string, unknown> | null | undefined
  fallbackNoticeText?: string
  S: Record<string, string>
}) {
  const summary = sourceSummary || null
  if (!shouldShowSourceSummary(summary)) return null
  const label = sourceSummaryLabel(summary, S)
  const title = sourceSummaryTitle(summary, fallbackNoticeText) || label
  return (
    <AssistantSourceNotice
      noticeText={title}
      titleText={title}
      labelText={label}
      S={S}
    />
  )
}

export function AssistantMessageNotices({
  message,
  lowConfidenceMeta,
  provenanceModeLabel,
  S,
}: AssistantMessageNoticesProps) {
  const noticeText = getMessageNoticeValue(message)
  const sourceNotice = Boolean(noticeText && isAssistantSourceNoticeText(noticeText))
  const sourceSummary = getMessageAgentSourceSummary(message)
  const showSourceSummary = shouldShowSourceSummary(sourceSummary)
  const showProvenanceLabel = shouldShowProvenanceModeLabel() && Boolean(provenanceModeLabel)

  if (!noticeText && !showSourceSummary && !lowConfidenceMeta && !showProvenanceLabel) return null

  return (
    <>
      {showSourceSummary ? (
        <AssistantSourceSummaryNotice sourceSummary={sourceSummary} fallbackNoticeText={noticeText || ''} S={S} />
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
