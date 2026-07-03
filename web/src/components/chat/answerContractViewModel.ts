export interface AnswerSourceNoticeViewModel {
  label: string
  title: string
  kind: string
  usesLocalKnowledgeBase: boolean
  usesExternalModel: boolean
  requiresUserNotice: boolean
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function firstRecord(...values: unknown[]): Record<string, unknown> {
  for (const value of values) {
    const record = asRecord(value)
    if (Object.keys(record).length > 0) return record
  }
  return {}
}

function textValue(...values: unknown[]): string {
  for (const value of values) {
    const text = String(value || '').trim()
    if (text) return text
  }
  return ''
}

function boolValue(value: unknown): boolean | null {
  if (typeof value === 'boolean') return value
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase()
    if (['1', 'true', 'yes', 'on'].includes(normalized)) return true
    if (['0', 'false', 'no', 'off'].includes(normalized)) return false
  }
  return null
}

function sourcePolicyPayload(summary: Record<string, unknown>): Record<string, unknown> {
  return asRecord(summary.source_policy_payload || summary.sourcePolicyPayload)
}

export function sourceSummaryFromAnswerContract(answerContract: unknown): Record<string, unknown> | null {
  const contract = asRecord(answerContract)
  if (Object.keys(contract).length <= 0) return null
  const summary = asRecord(contract.source_summary || contract.sourceSummary)
  return Object.keys(summary).length > 0 ? summary : null
}

export function labelForSourceNoticeText(noticeText: string, S: Record<string, string>): string {
  const mentionsLocalKb = /local citations\s*\[n\]\s*come from the knowledge base/i.test(noticeText)
    || (/[\u672c\u5730\u77e5\u8bc6\u6587\u732e\u5e93]/.test(noticeText) && /\[n\]/.test(noticeText))
  if (mentionsLocalKb) {
    return S.agent_trace_source_local_external || 'Local + external'
  }
  return S.agent_trace_evidence_not_from_kb || 'Not from KB'
}

function sourceKindLabel(kind: string, S: Record<string, string>): string {
  if (kind === 'local_kb') return S.agent_trace_source_local_only || 'Local KB'
  if (kind === 'local_plus_external') return S.agent_trace_source_local_external || 'Local + external'
  if (kind === 'external_not_kb' || kind === 'general_api') return S.agent_trace_evidence_not_from_kb || 'Not from KB'
  return ''
}

export function buildAnswerSourceNoticeViewModel({
  answerContract,
  legacySourceSummary,
  fallbackNoticeText = '',
  allowFallbackNotice = false,
  S,
}: {
  answerContract?: unknown
  legacySourceSummary?: unknown
  fallbackNoticeText?: string
  allowFallbackNotice?: boolean
  S: Record<string, string>
}): AnswerSourceNoticeViewModel | null {
  const contract = asRecord(answerContract)
  const contractSummary = sourceSummaryFromAnswerContract(contract)
  const summary = contractSummary || asRecord(legacySourceSummary)
  const contractUi = asRecord(contract.ui)
  const uiBadge = asRecord(contractUi.source_badge || contractUi.sourceBadge)
  const policy = firstRecord(
    contract.source_policy_payload || contract.sourcePolicyPayload,
    sourcePolicyPayload(summary),
  )
  const policyBadge = asRecord(policy.badge)
  const badge = firstRecord(policyBadge, uiBadge)
  const shouldShow = boolValue(
    badge.should_show ?? badge.shouldShow
      ?? uiBadge.should_show ?? uiBadge.shouldShow
      ?? summary.should_show ?? summary.shouldShow,
  )
  if (shouldShow === false) return null

  const kind = textValue(policy.kind, summary.kind)
  const labelKey = textValue(badge.label_key, badge.labelKey, uiBadge.label_key, uiBadge.labelKey, summary.label_key, summary.labelKey)
  const fallbackLabel = textValue(badge.label, uiBadge.label, summary.label)
  const detail = textValue(badge.detail, uiBadge.detail, summary.detail)
  const hasSourceSignal = Boolean(
    (kind && kind !== 'unknown')
      || labelKey
      || fallbackLabel
      || detail
      || (allowFallbackNotice && fallbackNoticeText),
  )
  if (!hasSourceSignal) return null

  const label = (labelKey && S[labelKey])
    || sourceKindLabel(kind, S)
    || fallbackLabel
    || (allowFallbackNotice ? labelForSourceNoticeText(fallbackNoticeText, S) : '')
    || S.agent_trace_source_fallback
    || 'Source'
  const title = detail || fallbackNoticeText || fallbackLabel || label
  const policyUsesLocal = boolValue(policy.uses_local_knowledge_base ?? policy.usesLocalKnowledgeBase)
  const policyUsesExternal = boolValue(policy.uses_external_model ?? policy.usesExternalModel)
  const policyRequiresNotice = boolValue(policy.requires_user_notice ?? policy.requiresUserNotice)
  return {
    label,
    title,
    kind,
    usesLocalKnowledgeBase: policyUsesLocal ?? (kind === 'local_kb' || kind === 'local_plus_external'),
    usesExternalModel: policyUsesExternal ?? (kind === 'local_plus_external' || kind === 'external_not_kb' || kind === 'general_api'),
    requiresUserNotice: policyRequiresNotice ?? (kind === 'local_plus_external' || kind === 'external_not_kb'),
  }
}
