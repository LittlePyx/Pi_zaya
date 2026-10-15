import { Drawer, Empty, Typography } from 'antd'
import {
  buildEvidenceDrawerViewModel,
  type AnswerSourceNoticeViewModel,
} from './answerSourceNoticeViewModel'
import type { CiteDetail } from './citationState'
import { EvidenceDrawerItem } from './EvidenceDrawerItem'
import { EvidenceDrawerSourceSummary } from './EvidenceDrawerSourceSummary'
import { EvidenceDrawerWebSources } from './EvidenceDrawerWebSources'

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
      size={420}
      mask={false}
      rootClassName="kb-evidence-drawer-root"
      className="kb-evidence-drawer"
    >
      <div className="kb-evidence-drawer-shell" data-testid="evidence-drawer">
        {sourceNotice ? (
          <EvidenceDrawerSourceSummary
            label={drawer.sourceLabel}
            detail={drawer.sourceDetail}
          />
        ) : null}

        {drawer.visibleDetails.length > 0 ? (
          <div className="kb-evidence-list">
            <div className="kb-evidence-section-heading">{S.evidence_local_sources_heading}</div>
            {drawer.visibleDetails.map((detail, index) => (
              <EvidenceDrawerItem
                key={`${detail.anchor || detail.num || index}-${index}`}
                detail={detail}
                index={index}
                onOpenReader={onOpenReader}
                onAddToShelf={onAddToShelf}
                S={S}
              />
            ))}
          </div>
        ) : drawer.webSources.length <= 0 && drawer.externalSourceKind !== 'model' ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={S.evidence_no_local_cards || 'No local evidence cards for this answer.'}
          />
        ) : null}

        <EvidenceDrawerWebSources sources={drawer.webSources} S={S} />

        {drawer.externalSourceKind === 'model' ? (
          <div className="kb-evidence-model-notice" data-testid="evidence-model-background-notice">
            {S.evidence_model_background_notice}
          </div>
        ) : null}
      </div>
    </Drawer>
  )
}
