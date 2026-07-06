import { Button, Card, Typography } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import './LibraryBatchSelectionBar.css'

const { Text } = Typography

type LibraryBatchSelectionBarProps = {
  visible: boolean
  S: Record<string, string>
  selectedCount: number
  currentCount: number
  qualityDiagnosticsVisible: boolean
  repairableQualityCount: number
  repairingSelectedQuality: boolean
  onSelectCurrentList: () => void
  onClearSelection: () => void
  onRepairSelectedQuality: () => void
  onOpenBatchEditor: () => void
}

export function LibraryBatchSelectionBar({
  visible,
  S,
  selectedCount,
  currentCount,
  qualityDiagnosticsVisible,
  repairableQualityCount,
  repairingSelectedQuality,
  onSelectCurrentList,
  onClearSelection,
  onRepairSelectedQuality,
  onOpenBatchEditor,
}: LibraryBatchSelectionBarProps) {
  if (!visible || selectedCount <= 0) return null

  return (
    <Card size="small" className="kb-lib-card kb-lib-batch-card">
      <div className="kb-lib-batch-bar">
        <div className="kb-lib-batch-summary">
          <div className="kb-lib-batch-badges">
            <span className="kb-lib-batch-badge is-strong">{S.lib_batch_selected_count.replace('{n}', String(selectedCount))}</span>
            <span className="kb-lib-batch-badge">{S.lib_batch_current_count.replace('{n}', String(currentCount))}</span>
          </div>
          <Text className="kb-lib-batch-count">{S.lib_batch_title_selected}</Text>
          <Text type="secondary" className="kb-lib-batch-hint">{S.lib_batch_hint_scope}</Text>
        </div>
        <div className="kb-lib-batch-actions">
          <Button onClick={onSelectCurrentList}>{S.lib_btn_select_current_list}</Button>
          <Button onClick={onClearSelection} disabled={selectedCount <= 0}>{S.lib_btn_clear_selection}</Button>
          {qualityDiagnosticsVisible && repairableQualityCount > 0 ? (
            <Button
              icon={<ReloadOutlined />}
              loading={repairingSelectedQuality}
              onClick={onRepairSelectedQuality}
              data-testid="library-quality-repair-selected"
            >
              {S.lib_btn_repair_quality_selected}
            </Button>
          ) : null}
          <Button type="primary" onClick={onOpenBatchEditor} disabled={selectedCount <= 0}>{S.lib_batch_title}</Button>
        </div>
      </div>
    </Card>
  )
}
