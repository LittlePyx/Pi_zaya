import { compact } from './citationPopoverUtils'

interface CitationPopoverLocalizationStrings extends Record<string, string> {
  cite_answer_point: string
  cite_anchor_label: string
  cite_binding_candidate: string
  cite_binding_mismatch: string
  cite_candidate_support_default: string
  cite_context: string
  cite_context_summary: string
  cite_evidence_chain: string
  cite_evidence_focus: string
  cite_frontend_candidate_reason: string
  cite_kind_evidence: string
  cite_kind_upstream: string
  cite_location_current: string
  cite_location_paper: string
  cite_meta_author: string
  cite_meta_published: string
  cite_meta_source: string
  cite_missing_reference_entry: string
  cite_missing_reference_entry_body: string
  cite_note: string
  cite_original_evidence: string
  cite_position: string
  cite_reference_entry: string
  cite_reliability: string
  cite_trace_complete: string
  cite_trace_review: string
  cite_upstream_reference: string
  cite_upstream_role: string
}

export interface CitationPopoverLocalizers {
  localizeKnownBody: (value: string) => string
  localizeKnownLabel: (value: string) => string
}

export function buildCitationPopoverLocalizers(S: CitationPopoverLocalizationStrings): CitationPopoverLocalizers {
  const localizeKnownLabel = (value: string): string => {
    const text = compact(value)
    if (!text) return ''
    const labels: Record<string, string> = {
      上游引用: S.cite_kind_upstream,
      答案依据: S.cite_kind_evidence,
      答案中的话: S.cite_answer_point,
      对应回答: S.cite_answer_point,
      答案要点: S.cite_answer_point,
      引用语境: S.cite_context,
      语境摘要: S.cite_context_summary,
      链路已闭合: S.cite_trace_complete,
      链路需核对: S.cite_trace_review,
      疑似错配: S.cite_binding_mismatch,
      候选依据: S.cite_binding_candidate,
      上游参考文献: S.cite_upstream_reference,
      引用所在论文: S.cite_location_paper,
      当前论文引用处: S.cite_location_current,
      来源: S.cite_meta_source,
      发表: S.cite_meta_published,
      作者: S.cite_meta_author,
      位置: S.cite_position,
      锚点: S.cite_anchor_label,
      证据重点: S.cite_evidence_focus,
      原文证据: S.cite_original_evidence,
      可靠度: S.cite_reliability,
      证据链: S.cite_evidence_chain,
      上游作用: S.cite_upstream_role,
      上游文献条目: S.cite_reference_entry,
      说明: S.cite_note,
      'Missing reference entry': S.cite_missing_reference_entry,
    }
    return labels[text] || text
  }

  const localizeKnownBody = (value: string): string => {
    const text = compact(value)
    if (!text) return ''
    const missingReferenceMatch = text.match(/^Reference \[(\d{1,4})]\s+is cited in the opened Reader document, but the converted References section does not contain a matching bibliography entry\.?$/i)
    if (missingReferenceMatch) {
      return S.cite_missing_reference_entry_body.replace('{n}', missingReferenceMatch[1])
    }
    if (/前端缺少后端 cite_details/.test(text)) return S.cite_frontend_candidate_reason
    if (/前端根据本轮 References 临时补齐/.test(text)) return S.cite_candidate_support_default
    if (/这条引用只能作为候选依据/.test(text)) return S.cite_candidate_support_default
    return text
  }

  return {
    localizeKnownBody,
    localizeKnownLabel,
  }
}
