import { Button, Dropdown } from 'antd'
import { DeleteOutlined, MoreOutlined, ReloadOutlined } from '@ant-design/icons'
import type { LibraryFileItem } from '../../api/library'
import './LibraryFileActions.css'

type LibraryFileActionsProps = {
  S: Record<string, string>
  item: LibraryFileItem
  onOpenMeta: () => void
  onStartPaperGuide: () => void
  onConvert: () => void
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
  onOpenPdf,
  onOpenMarkdown,
  onDelete,
}: LibraryFileActionsProps) {
  const showPrimaryConvertAction = !item.md_exists

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
      {showPrimaryConvertAction ? (
        <Button
          className="kb-lib-file-action-link is-accent"
          type="text"
          size="small"
          disabled={item.task_state !== 'idle'}
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
