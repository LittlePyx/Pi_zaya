import { useSettingsStore } from '../../stores/settingsStore'

interface LowConfidenceMetaLite {
  isZh: boolean
  reasonCode: string
  reasonText: string
  candidateRefs: number[]
}

const LOW_CONF_REASON_MAP_EN: Record<string, string> = {
  empty_hits: 'no scoped evidence was retrieved',
  target_miss: 'the requested target section was not matched directly',
  reference_only_hits: 'retrieval mostly returned reference-like snippets',
  weak_signal: 'retrieval signal is weak for the requested claim',
  strict_family_without_targeted_support: 'strict question type lacks targeted support',
  strict_family_weak_overlap: 'strict question type has weak lexical overlap',
  strict_family_sparse_hits: 'strict question type has sparse evidence hits',
  broad_family_weak_overlap: 'broad summary question has weak evidence overlap',
}

function toPositiveInt(input: unknown): number {
  const n = Number(input)
  if (!Number.isFinite(n)) return 0
  const out = Math.floor(n)
  return out > 0 ? out : 0
}

function hasCjkText(input: string): boolean {
  return /[\u4e00-\u9fff]/.test(String(input || ''))
}

export function resolveLowConfidenceMeta(
  metaRaw: Record<string, unknown> | null | undefined,
  localeHintText: string,
  S: Record<string, string>,
): LowConfidenceMetaLite | null {
  const meta = metaRaw && typeof metaRaw === 'object' ? metaRaw : null
  if (!meta) return null
  const answerQuality = meta.answer_quality
  if (!answerQuality || typeof answerQuality !== 'object') return null
  const retrieval = (answerQuality as Record<string, unknown>).retrieval_confidence
  if (!retrieval || typeof retrieval !== 'object') return null
  const retrievalRecord = retrieval as Record<string, unknown>
  const lowRaw = retrievalRecord.low_confidence
  const lowConfidence = lowRaw === true || lowRaw === 1 || String(lowRaw || '').trim().toLowerCase() === 'true'
  if (!lowConfidence) return null
  const reasonCode = String(
    retrievalRecord.low_confidence_reason
    || retrievalRecord.force_rescue_reason
    || '',
  ).trim()
  const reasonNorm = reasonCode.toLowerCase()
  const uiLocale = useSettingsStore.getState().uiLocale
  const isZh = uiLocale === 'zh' ? true : (uiLocale === 'en' ? false : hasCjkText(localeHintText))
  const zhMap: Record<string, string> = {
    empty_hits: S.msg_empty_hits,
    target_miss: S.msg_target_miss,
    reference_only_hits: S.msg_reference_only,
    weak_signal: S.msg_weak_signal,
    strict_family_without_targeted_support: S.msg_strict_no_support,
    strict_family_weak_overlap: S.msg_strict_weak_overlap,
    strict_family_sparse_hits: S.msg_strict_sparse,
    broad_family_weak_overlap: S.msg_broad_weak_overlap,
  }
  const reasonText = isZh
    ? (zhMap[reasonNorm] || reasonNorm || S.msg_low_confidence)
    : (LOW_CONF_REASON_MAP_EN[reasonNorm] || (reasonNorm ? reasonNorm.replace(/_/g, ' ') : 'evidence matching is lower confidence'))
  const refsRaw = Array.isArray(retrievalRecord.candidate_refs_for_notice)
    ? retrievalRecord.candidate_refs_for_notice
    : (Array.isArray(retrievalRecord.candidate_refs) ? retrievalRecord.candidate_refs : [])
  const candidateRefs: number[] = []
  const seen = new Set<number>()
  for (const item of refsRaw) {
    const num = toPositiveInt(item)
    if (num <= 0 || seen.has(num)) continue
    seen.add(num)
    candidateRefs.push(num)
    if (candidateRefs.length >= 8) break
  }
  return {
    isZh,
    reasonCode: reasonNorm,
    reasonText,
    candidateRefs,
  }
}

export function stripLeadingLowConfidenceNotice(body: string): string {
  const text = String(body || '')
  if (!text.trim()) return text
  const normalized = text.trimStart()
  const split = normalized.split(/\n\s*\n/, 2)
  if (split.length < 2) return text
  const lead = String(split[0] || '').trim()
  const leadLower = lead.toLowerCase()
  const looksLowConfidenceNotice = (
    leadLower.startsWith('note: this answer is based on lower-confidence evidence matching')
    || (lead.includes('低置信') && lead.includes('证据'))
  )
  if (!looksLowConfidenceNotice) return text
  return String(split[1] || '').trimStart()
}
