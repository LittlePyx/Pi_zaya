import { BookOutlined, PlusOutlined } from '@ant-design/icons'
import { Button } from 'antd'
import type { CiteDetail } from './citationState'
import { buildEvidenceCardViewModel } from './evidenceCardViewModel'

export function EvidenceDrawerItem({
  detail,
  index,
  onOpenReader,
  onAddToShelf,
  S,
}: {
  detail: CiteDetail
  index: number
  onOpenReader?: (detail: CiteDetail) => void
  onAddToShelf?: (detail: CiteDetail) => void
  S: Record<string, string>
}) {
  const card = buildEvidenceCardViewModel(detail, {
    S,
    fallbackLabel: `[${index + 1}]`,
    evidenceLimit: 320,
    claimLimit: 220,
    supportLimit: 160,
    includeRawFallback: true,
  })

  return (
    <article className="kb-evidence-item" data-testid="evidence-drawer-item">
      <div className="kb-evidence-item-head">
        <span className="kb-evidence-cite-label">{card.label}</span>
        <span className="kb-evidence-source-name">{card.source || S.cite_meta_source || 'Source'}</span>
      </div>
      {card.claim ? (
        <div className="kb-evidence-block">
          <div className="kb-evidence-block-label">{card.claimLabel}</div>
          <div className="kb-evidence-block-text">{card.claimPreview}</div>
        </div>
      ) : null}
      {card.evidence ? (
        <div className="kb-evidence-block">
          <div className="kb-evidence-block-label">{card.evidenceLabel}</div>
          <blockquote>{card.evidencePreview}</blockquote>
        </div>
      ) : null}
      {card.location || card.support ? (
        <div className="kb-evidence-meta">
          {card.location ? <span>{card.location}</span> : null}
          {card.support ? <span>{card.supportPreview}</span> : null}
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
}
