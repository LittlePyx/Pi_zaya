import { Button, Drawer, Empty, Typography } from 'antd'
import { BookOutlined, PlusOutlined } from '@ant-design/icons'
import type { AnswerSourceNoticeViewModel } from './answerContractViewModel'
import {
  citationInlineLabel,
  cleanCitationDisplayText,
  type CiteDetail,
} from './citationState'

const { Text } = Typography

interface EvidenceDrawerProps {
  open: boolean
  sourceNotice: AnswerSourceNoticeViewModel | null
  citeDetails: CiteDetail[]
  onClose: () => void
  onOpenReader?: (detail: CiteDetail) => void
  onAddToShelf?: (detail: CiteDetail) => void
  S: Record<string, string>
}

function text(...values: unknown[]): string {
  for (const value of values) {
    const cleaned = cleanCitationDisplayText(String(value || ''))
    if (cleaned) return cleaned
  }
  return ''
}

function shortText(value: string, limit: number): string {
  const cleaned = text(value)
  if (!cleaned || cleaned.length <= limit) return cleaned
  return `${cleaned.slice(0, limit - 1).trim()}...`
}

function dedupeDetails(details: CiteDetail[]): CiteDetail[] {
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

function sourcePolicyLine(sourceNotice: AnswerSourceNoticeViewModel | null): string {
  if (!sourceNotice) return ''
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

export function EvidenceDrawer({
  open,
  sourceNotice,
  citeDetails,
  onClose,
  onOpenReader,
  onAddToShelf,
  S,
}: EvidenceDrawerProps) {
  const visibleDetails = dedupeDetails(citeDetails)
  const title = S.msg_evidence_label || S.agent_trace_label_evidence || 'Evidence'
  const subtitle = sourceNotice?.label || S.agent_trace_source_fallback || 'Source'
  return (
    <Drawer
      title={(
        <div className="kb-evidence-drawer-title">
          <span>{title}</span>
          <Text className="kb-evidence-drawer-subtitle">{subtitle}</Text>
        </div>
      )}
      open={open}
      onClose={onClose}
      width={420}
      mask={false}
      rootClassName="kb-evidence-drawer-root"
      className="kb-evidence-drawer"
    >
      <div className="kb-evidence-drawer-shell" data-testid="evidence-drawer">
        {sourceNotice ? (
          <section className="kb-evidence-source-summary" data-testid="evidence-source-summary">
            <div className="kb-evidence-source-label">{sourceNotice.label}</div>
            <div className="kb-evidence-source-detail">{sourcePolicyLine(sourceNotice)}</div>
          </section>
        ) : null}

        {visibleDetails.length > 0 ? (
          <div className="kb-evidence-list">
            {visibleDetails.map((detail, index) => {
              const label = citationInlineLabel(detail) || `[${index + 1}]`
              const source = text(detail.cardTitle, detail.title, detail.sourceName, detail.sourcePath)
              const claim = text(detail.cardClaim, detail.answerClaim, detail.cardTakeaway)
              const evidence = text(detail.cardEvidence, detail.evidenceQuote, detail.citationContext, detail.summaryLine, detail.raw)
              const location = text(detail.headingPath, detail.cardLocator, detail.locationLabel)
              const support = text(detail.cardSupportExplanation, detail.supportRelation, detail.whyLine, detail.bindingReason)
              return (
                <article className="kb-evidence-item" key={`${detail.anchor || detail.num || index}-${index}`} data-testid="evidence-drawer-item">
                  <div className="kb-evidence-item-head">
                    <span className="kb-evidence-cite-label">{label}</span>
                    <span className="kb-evidence-source-name">{source || S.cite_meta_source || 'Source'}</span>
                  </div>
                  {claim ? (
                    <div className="kb-evidence-block">
                      <div className="kb-evidence-block-label">{S.cite_answer_point || 'Answer claim'}</div>
                      <div className="kb-evidence-block-text">{shortText(claim, 220)}</div>
                    </div>
                  ) : null}
                  {evidence ? (
                    <div className="kb-evidence-block">
                      <div className="kb-evidence-block-label">{S.cite_original_evidence || 'Evidence'}</div>
                      <blockquote>{shortText(evidence, 320)}</blockquote>
                    </div>
                  ) : null}
                  {location || support ? (
                    <div className="kb-evidence-meta">
                      {location ? <span>{location}</span> : null}
                      {support ? <span>{shortText(support, 160)}</span> : null}
                    </div>
                  ) : null}
                  <div className="kb-evidence-actions">
                    {onOpenReader ? (
                      <Button
                        size="small"
                        icon={<BookOutlined />}
                        onClick={() => onOpenReader(detail)}
                        data-testid="evidence-open-source"
                      >
                        {S.cite_open_reader || 'Open source'}
                      </Button>
                    ) : null}
                    {onAddToShelf ? (
                      <Button
                        size="small"
                        icon={<PlusOutlined />}
                        onClick={() => onAddToShelf(detail)}
                        data-testid="evidence-add-shelf"
                      >
                        {S.cite_add_to_shelf || 'Add'}
                      </Button>
                    ) : null}
                  </div>
                </article>
              )
            })}
          </div>
        ) : (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={S.agent_trace_evidence_not_from_kb || 'No local evidence cards for this answer.'}
          />
        )}
      </div>
    </Drawer>
  )
}
