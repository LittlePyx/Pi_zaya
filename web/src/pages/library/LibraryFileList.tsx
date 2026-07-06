import type { ReactNode } from 'react'
import { Empty, Typography } from 'antd'
import VirtualList from 'rc-virtual-list'
import type { LibraryFileItem } from '../../api/library'
import './LibraryFileList.css'

const { Text } = Typography
const FILE_VIRTUAL_THRESHOLD = 60
const FILE_VIRTUAL_HEIGHT = 620
const FILE_VIRTUAL_ROW_HEIGHT = 88

type LibraryFileListProps = {
  items: LibraryFileItem[]
  emptyText: string
  virtualScrollHint: string
  renderRow: (item: LibraryFileItem) => ReactNode
}

export function LibraryFileList({
  items,
  emptyText,
  virtualScrollHint,
  renderRow,
}: LibraryFileListProps) {
  if (!items.length) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyText} />
  }

  if (items.length < FILE_VIRTUAL_THRESHOLD) {
    return (
      <div className="kb-lib-file-list" role="list">
        {items.map((item) => (
          <div key={item.name} className="kb-lib-file-item" role="listitem">
            {renderRow(item)}
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="kb-lib-file-virtual-shell">
      <div className="kb-lib-file-virtual-tip">
        <Text type="secondary" className="text-xs">{virtualScrollHint.replace('{n}', String(items.length))}</Text>
      </div>
      <VirtualList
        data={items}
        itemKey="name"
        height={FILE_VIRTUAL_HEIGHT}
        itemHeight={FILE_VIRTUAL_ROW_HEIGHT}
      >
        {(item: LibraryFileItem) => (
          <div className="ant-list-item kb-lib-file-item kb-lib-file-virtual-item">
            {renderRow(item)}
          </div>
        )}
      </VirtualList>
    </div>
  )
}
