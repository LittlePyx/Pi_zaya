import { Button, Dropdown } from 'antd'
import { DeleteOutlined, MoreOutlined, ReloadOutlined, StopOutlined } from '@ant-design/icons'
import type { LibraryFileItem } from '../../api/library'
import './LibraryFileActions.css'

type LibraryFileActionsProps = {
  S: Record<string, string>
  item: LibraryFileItem
  onOpenMeta: () => void
  onStartPaperGuide: () => void
  onConvert: () => void
  onCancel: () => void
  onResume: () => void
  onRetry: () => void
  retrying: boolean
  onOpenPdf: () => void
  onOpenMarkdown: () => void
  onDelete: () => void
}

export function LibraryFileActions({
  S,
  item,
  onOpenMeta,
  onStartPaperGuide,
  onConvert,
  onCancel,
  onResume,
  onRetry,
  retrying,
  onOpenPdf,
  onOpenMarkdown,
  onDelete,
}: LibraryFileActionsProps) {
  const showPrimaryConvertAction = !item.md_exists
  const conversionBusy = item.task_state === 'queued' || item.task_state === 'running'
  const recoveryAvailable = item.task_state === 'interrupted'
  const conversionCancelling = item.conversion_stage === 'cancelling'
  const retryAction = conversionBusy ? '' : String(item.last_conversion?.retry_action || '')

  return (
    <div className={`kb-lib-file-actions${showPrimaryConvertAction ? ' has-convert' : ' is-compact'}`}>
      <Button className="kb-lib-file-action-main" size="small" onClick={onOpenMeta}>
        {S.lib_btn_categorize}
      </Button>
      {item.md_exists ? (
        <Button
          className="kb-lib-file-action-link"
          type="text"
          size="small"
          disabled={!item.md_path}
          onClick={onStartPaperGuide}
        >
          {S.lib_btn_read}
        </Button>
      ) : null}
      {recoveryAvailable ? (
        <Button
          className="kb-lib-file-action-link is-accent"
          type="text"
          size="small"
          icon={<ReloadOutlined />}
          loading={retrying}
          disabled={retrying || !item.task_id}
          onClick={onResume}
          data-testid="library-resume-conversion"
        >
          {S.lib_btn_resume_conversion}
        </Button>
      ) : conversionBusy ? (
        <Button
          className="kb-lib-file-action-link"
          type="text"
          size="small"
          danger
          icon={<StopOutlined />}
          disabled={!item.task_id || conversionCancelling}
          onClick={onCancel}
          data-testid="library-cancel-conversion"
        >
          {conversionCancelling ? S.lib_convert_cancelling : S.lib_btn_cancel_conversion}
        </Button>
      ) : retryAction ? (
        <Button
          className="kb-lib-file-action-link is-accent"
          type="text"
          size="small"
          icon={<ReloadOutlined />}
          loading={retrying}
          disabled={retrying}
          onClick={onRetry}
          data-testid="library-retry-conversion"
        >
          {retryAction === 'reindex' ? S.lib_btn_retry_index : S.lib_btn_retry_conversion}
        </Button>
      ) : showPrimaryConvertAction ? (
        <Button
          className="kb-lib-file-action-link is-accent"
          type="text"
          size="small"
          onClick={onConvert}
        >
          {S.lib_btn_convert}
        </Button>
      ) : null}
      <Button className="kb-lib-file-action-link" type="text" size="small" onClick={onOpenPdf}>
        PDF
      </Button>
      <div className="kb-lib-file-more">
        <Dropdown
          trigger={['click']}
          menu={{
            items: [
              ...(item.md_exists
                ? [{ key: 'reconvert', label: S.lib_btn_reconvert, disabled: item.task_state !== 'idle', icon: <ReloadOutlined /> }]
                : []),
              ...(recoveryAvailable
                ? [{ key: 'dismiss-recovery', label: S.lib_btn_dismiss_conversion_recovery, danger: true, icon: <StopOutlined /> }]
                : []),
              { key: 'open-md', label: S.lib_btn_open_md, disabled: !item.md_exists },
              { type: 'divider' as const },
              { key: 'delete', label: S.lib_btn_delete, danger: true, disabled: item.task_state !== 'idle', icon: <DeleteOutlined /> },
            ],
            onClick: ({ key }) => {
              if (key === 'reconvert') {
                onConvert()
                return
              }
              if (key === 'open-md') {
                onOpenMarkdown()
                return
              }
              if (key === 'dismiss-recovery') {
                onCancel()
                return
              }
              if (key === 'delete') {
                onDelete()
              }
            },
          }}
        >
          <Button size="small" className="kb-lib-file-more-btn" icon={<MoreOutlined />} />
        </Dropdown>
      </div>
    </div>
  )
}
