import { Button, Drawer, Empty, Typography } from 'antd'
import { BookOutlined, PlusOutlined } from '@ant-design/icons'
import {
  buildEvidenceDrawerViewModel,
  type AnswerSourceNoticeViewModel,
} from './answerSourceNoticeViewModel'
import { buildEvidenceCardViewModel } from './evidenceCardViewModel'
import type { CiteDetail } from './citationState'

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

export function EvidenceDrawer({
  open,
  sourceNotice,
  citeDetails,
  onClose,
  onOpenReader,
  onAddToShelf,
  S,
}: EvidenceDrawerProps) {
  const drawer = buildEvidenceDrawerViewModel({
    sourceNotice,
    citeDetails,
    S,
  })
  return (
    <Drawer
      title={(
        <div className="kb-evidence-drawer-title">
          <span>{drawer.title}</span>
          <Text className="kb-evidence-drawer-subtitle">{drawer.subtitle}</Text>
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
            <div className="kb-evidence-source-label">{drawer.sourceLabel}</div>
            <div className="kb-evidence-source-detail">{drawer.sourceDetail}</div>
          </section>
        ) : null}

        {drawer.visibleDetails.length > 0 ? (
          <div className="kb-evidence-list">
            {drawer.visibleDetails.map((detail, index) => {
              const card = buildEvidenceCardViewModel(detail, {
                S,
                fallbackLabel: `[${index + 1}]`,
                evidenceLimit: 320,
                claimLimit: 220,
                supportLimit: 160,
                includeRawFallback: true,
              })
              return (
                <article className="kb-evidence-item" key={`${detail.anchor || detail.num || index}-${index}`} data-testid="evidence-drawer-item">
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
