import type { CiteDetail } from './citationState'
import type { ExternalWebSource } from '../../api/chat'

export interface AnswerSourceNoticeViewModel {
  label: string
  title: string
  kind: string
  usesLocalKnowledgeBase: boolean
  usesExternalModel: boolean
  usesExternalWeb?: boolean
  requiresUserNotice: boolean
  externalSourceKind?: string
  externalSupplementMode?: string
  webSources?: ExternalWebSource[]
  externalSupplementUnavailable?: string
}

export interface EvidenceDrawerViewModel {
  title: string
  subtitle: string
  sourceLabel: string
  sourceDetail: string
  visibleDetails: CiteDetail[]
  webSources: ExternalWebSource[]
  externalSourceKind: string
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

function safeWebSources(value: unknown): ExternalWebSource[] {
  if (!Array.isArray(value)) return []
  const seen = new Set<string>()
  const out: ExternalWebSource[] = []
  for (const raw of value) {
    const item = asRecord(raw)
    const url = textValue(item.url)
    try {
      const parsed = new URL(url)
      if (parsed.protocol !== 'https:' || parsed.username || parsed.password) continue
      if (textValue(item.verification_status, item.verificationStatus) !== 'verified') continue
      parsed.hash = ''
      const cleanUrl = parsed.toString()
      if (seen.has(cleanUrl)) continue
      seen.add(cleanUrl)
      let pdfUrl = ''
      try {
        const parsedPdf = new URL(textValue(item.pdf_url, item.pdfUrl))
        if (parsedPdf.protocol === 'https:' && !parsedPdf.username && !parsedPdf.password) {
          parsedPdf.hash = ''
          pdfUrl = parsedPdf.toString()
        }
      } catch {
        pdfUrl = ''
      }
      out.push({
        title: textValue(item.title).slice(0, 500),
        url: cleanUrl,
        domain: textValue(item.domain) || parsed.hostname,
        verification_status: 'verified',
        verification_level: textValue(item.verification_level, item.verificationLevel).slice(0, 40),
        claim_text: textValue(item.claim_text, item.claimText).slice(0, 320),
        scholarly_kind: textValue(item.scholarly_kind, item.scholarlyKind).slice(0, 40),
        doi: textValue(item.doi).slice(0, 300),
        arxiv_id: textValue(item.arxiv_id, item.arxivId).slice(0, 80),
        pdf_url: pdfUrl || undefined,
        import_eligible: Boolean(pdfUrl && boolValue(item.import_eligible ?? item.importEligible)),
      })
      if (out.length >= 8) break
    } catch {
      // Ignore malformed or non-http(s) URLs at the presentation boundary.
    }
  }
  return out
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
  const rawDetail = textValue(badge.detail, uiBadge.detail, summary.detail)
  const hasSourceSignal = Boolean(
    (kind && kind !== 'unknown')
      || labelKey
      || fallbackLabel
      || rawDetail
      || (allowFallbackNotice && fallbackNoticeText),
  )
  if (!hasSourceSignal) return null

  const label = (labelKey && S[labelKey])
    || sourceKindLabel(kind, S)
    || fallbackLabel
    || (allowFallbackNotice ? labelForSourceNoticeText(fallbackNoticeText, S) : '')
    || S.agent_trace_source_fallback
    || 'Source'
  const detail = (labelKey && S[`${labelKey}_detail`]) || rawDetail
  const title = detail || fallbackNoticeText || fallbackLabel || label
  const policyUsesLocal = boolValue(policy.uses_local_knowledge_base ?? policy.usesLocalKnowledgeBase)
  const policyUsesExternal = boolValue(policy.uses_external_model ?? policy.usesExternalModel)
  const policyUsesWeb = boolValue(policy.uses_external_web ?? policy.usesExternalWeb)
  const policyRequiresNotice = boolValue(policy.requires_user_notice ?? policy.requiresUserNotice)
  const externalSourceKind = textValue(
    policy.external_source_kind,
    policy.externalSourceKind,
    summary.external_source_kind,
    summary.externalSourceKind,
  )
  const webSources = safeWebSources(
    policy.web_sources || policy.webSources || summary.web_sources || summary.webSources,
  )
  return {
    label,
    title,
    kind,
    usesLocalKnowledgeBase: policyUsesLocal ?? (kind === 'local_kb' || kind === 'local_plus_external'),
    usesExternalModel: policyUsesExternal ?? (
      externalSourceKind !== 'web'
      && (kind === 'local_plus_external' || kind === 'external_not_kb' || kind === 'general_api')
    ),
    usesExternalWeb: policyUsesWeb ?? (externalSourceKind === 'web'),
    requiresUserNotice: policyRequiresNotice ?? (kind === 'local_plus_external' || kind === 'external_not_kb'),
    externalSourceKind,
    externalSupplementMode: textValue(
      policy.external_supplement_mode,
      policy.externalSupplementMode,
      summary.external_supplement_mode,
      summary.externalSupplementMode,
    ),
    webSources,
    externalSupplementUnavailable: textValue(
      policy.external_supplement_unavailable,
      policy.externalSupplementUnavailable,
      summary.external_supplement_unavailable,
      summary.externalSupplementUnavailable,
    ),
  }
}

export function dedupeEvidenceDrawerDetails(details: CiteDetail[]): CiteDetail[] {
  const seen = new Set<string>()
  const out: CiteDetail[] = []
  for (const detail of details) {
    const key = [
      Number(detail.displayNum || detail.num || 0),
      detail.sourcePath,
      detail.sourceName,
      detail.cardEvidence || detail.evidenceQuote || detail.summaryLine,
    ].join('|')
    if (seen.has(key)) continue
    seen.add(key)
    out.push(detail)
    if (out.length >= 8) break
  }
  return out
}

export function sourcePolicyLine(sourceNotice: AnswerSourceNoticeViewModel | null, S: Record<string, string> = {}): string {
  if (!sourceNotice) return ''
  if (sourceNotice.externalSourceKind === 'web') {
    return sourceNotice.usesLocalKnowledgeBase
      ? S.evidence_source_policy_local_web || 'Library citations are local evidence; linked web sources are separate external background.'
      : S.evidence_source_policy_web_only || 'This answer uses linked web sources and is not grounded in the local library.'
  }
  if (sourceNotice.externalSourceKind === 'model') {
    return sourceNotice.usesLocalKnowledgeBase
      ? S.evidence_source_policy_local_model || 'Library citations are local evidence; uncited model background is not verifiable evidence.'
      : S.evidence_source_policy_model_only || 'This is model background, not verifiable library evidence.'
  }
  if (sourceNotice.usesLocalKnowledgeBase && sourceNotice.usesExternalModel) {
    return 'Local citations are grounded in the knowledge base; external context may supplement uncited background.'
  }
  if (sourceNotice.usesLocalKnowledgeBase) {
    return 'This answer is grounded in local knowledge-base evidence.'
  }
  if (sourceNotice.usesExternalModel) {
    return 'This answer is not grounded in local knowledge-base evidence.'
  }
  return sourceNotice.title
}

export function buildEvidenceDrawerViewModel({
  sourceNotice,
  citeDetails,
  S,
}: {
  sourceNotice: AnswerSourceNoticeViewModel | null
  citeDetails: CiteDetail[]
  S: Record<string, string>
}): EvidenceDrawerViewModel {
  return {
    title: S.msg_evidence_label || S.agent_trace_label_evidence || 'Evidence',
    subtitle: sourceNotice?.label || S.agent_trace_source_fallback || 'Source',
    sourceLabel: sourceNotice?.label || '',
    sourceDetail: sourcePolicyLine(sourceNotice, S),
    visibleDetails: dedupeEvidenceDrawerDetails(citeDetails),
    webSources: sourceNotice?.webSources || [],
    externalSourceKind: sourceNotice?.externalSourceKind || '',
  }
}
