import type { Message } from '../../api/chat'
import { getMessageAgentSourceSummary, getMessageAnswerContract } from './messageTraceUtils'

export interface MessageEvidenceActionPolicy {
  allowLocalCitations: boolean
  allowStructuredLocalLocate: boolean
  allowLooseLocalLocate: boolean
  allowResearchNote: boolean
  reason: 'local' | 'mixed' | 'external' | 'quality_failed' | 'insufficient_evidence' | 'legacy'
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim().toLowerCase() : ''
}

function explicitBool(value: unknown): boolean | null {
  if (value === true || value === 1) return true
  if (value === false || value === 0) return false
  const normalized = text(value)
  if (['true', 'yes', 'on', '1'].includes(normalized)) return true
  if (['false', 'no', 'off', '0'].includes(normalized)) return false
  return null
}

function failedQualityPolicy(message: Message): MessageEvidenceActionPolicy | null {
  const meta = record(message.meta)
  const contract = record(getMessageAnswerContract(message))
  const contractQuality = record(contract.quality)
  const answerQuality = record(meta.answer_quality ?? meta.answerQuality)
  const minimumOk = explicitBool(answerQuality.minimum_ok ?? answerQuality.minimumOk
    ?? contractQuality.minimum_ok ?? contractQuality.minimumOk)
  if (minimumOk === false) {
    return {
      allowLocalCitations: false,
      allowStructuredLocalLocate: false,
      allowLooseLocalLocate: false,
      allowResearchNote: false,
      reason: 'quality_failed',
    }
  }
  const resolution = text(
    answerQuality.resolution_status ?? answerQuality.resolutionStatus
      ?? contractQuality.resolution_status ?? contractQuality.resolutionStatus,
  )
  if (['insufficient_evidence', 'not_found', 'scoped_not_found'].includes(resolution)) {
    return {
      allowLocalCitations: false,
      allowStructuredLocalLocate: false,
      allowLooseLocalLocate: false,
      allowResearchNote: false,
      reason: 'insufficient_evidence',
    }
  }
  return null
}

export function messageEvidenceActionPolicy(message: Message): MessageEvidenceActionPolicy {
  const failed = failedQualityPolicy(message)
  if (failed) return failed

  const contract = record(getMessageAnswerContract(message))
  const contractSummary = record(contract.source_summary ?? contract.sourceSummary)
  const summary = Object.keys(contractSummary).length > 0
    ? contractSummary
    : record(getMessageAgentSourceSummary(message))
  const contractPolicy = record(contract.source_policy_payload ?? contract.sourcePolicyPayload)
  const summaryPolicy = record(summary.source_policy_payload ?? summary.sourcePolicyPayload)
  const policy = Object.keys(contractPolicy).length > 0 ? contractPolicy : summaryPolicy
  const kind = text(policy.kind ?? summary.kind)
  const sourceBlend = text(
    policy.source_blend ?? policy.sourceBlend
      ?? summary.source_blend ?? summary.sourceBlend
      ?? contract.source_blend ?? contract.sourceBlend,
  )
  const externalKind = text(
    policy.external_source_kind ?? policy.externalSourceKind
      ?? summary.external_source_kind ?? summary.externalSourceKind,
  )
  const usesLocal = explicitBool(policy.uses_local_knowledge_base ?? policy.usesLocalKnowledgeBase)

  if (
    usesLocal === false
    || ['external_not_kb', 'general_api'].includes(kind)
    || ['external_academic', 'general_llm'].includes(sourceBlend)
  ) {
    return {
      allowLocalCitations: false,
      allowStructuredLocalLocate: false,
      allowLooseLocalLocate: false,
      allowResearchNote: false,
      reason: 'external',
    }
  }

  if (
    kind === 'local_plus_external'
    || sourceBlend === 'hybrid_local_external'
    || ['web', 'model'].includes(externalKind)
  ) {
    return {
      allowLocalCitations: true,
      allowStructuredLocalLocate: true,
      allowLooseLocalLocate: false,
      allowResearchNote: false,
      reason: 'mixed',
    }
  }

  if (usesLocal === true || kind === 'local_kb' || sourceBlend === 'local_grounded') {
    return {
      allowLocalCitations: true,
      allowStructuredLocalLocate: true,
      // Contract-era answers must use the same structured binding as their
      // citation cards. The open paper is never a claim-level fallback.
      allowLooseLocalLocate: false,
      allowResearchNote: true,
      reason: 'local',
    }
  }

  // Keep pre-contract local answers working. New answers should always carry
  // an explicit source policy, but historical messages predate that contract.
  return {
    allowLocalCitations: true,
    allowStructuredLocalLocate: true,
    allowLooseLocalLocate: true,
    allowResearchNote: true,
    reason: 'legacy',
  }
}
