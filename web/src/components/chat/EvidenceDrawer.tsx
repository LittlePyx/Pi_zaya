import { Drawer, Empty, Typography } from 'antd'
import {
  buildEvidenceDrawerViewModel,
  type AnswerSourceNoticeViewModel,
} from './answerSourceNoticeViewModel'
import type { CiteDetail } from './citationState'
import { EvidenceDrawerItem } from './EvidenceDrawerItem'
import { EvidenceDrawerSourceSummary } from './EvidenceDrawerSourceSummary'

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
          <EvidenceDrawerSourceSummary
            label={drawer.sourceLabel}
            detail={drawer.sourceDetail}
          />
        ) : null}

        {drawer.visibleDetails.length > 0 ? (
          <div className="kb-evidence-list">
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
