import { Typography } from 'antd'
import type { LibraryFileItem } from '../../api/library'
import { fileTag } from './libraryPageUtils'
import { LibraryFileQualityChips } from './LibraryFileQualityLine'
import './LibraryFileHeader.css'

const { Text } = Typography

type LibraryFileHeaderProps = {
  S: Record<string, string>
  item: LibraryFileItem
  suggestionCount: number
  qualityStatusVisible: boolean
  qualityDiagnosticsVisible: boolean
}

export function LibraryFileHeader({
  S,
  item,
  suggestionCount,
  qualityStatusVisible,
  qualityDiagnosticsVisible,
}: LibraryFileHeaderProps) {
  const tag = fileTag(item, S)
  const statusTone =
    tag.color === 'success'
      ? 'is-success'
      : tag.color === 'processing'
        ? 'is-processing'
        : tag.color === 'warning'
          ? 'is-warning'
          : 'is-default'

  return (
    <div className="kb-lib-file-head">
      <div className="kb-lib-file-title-wrap">
        <Text className="kb-lib-file-title" title={item.name}>{item.name}</Text>
      </div>
      <div className="kb-lib-file-submeta">
        <span className={`kb-lib-file-status-chip ${statusTone}`}>{tag.text}</span>
        <LibraryFileQualityChips
          S={S}
          item={item}
          qualityStatusVisible={qualityStatusVisible}
          qualityDiagnosticsVisible={qualityDiagnosticsVisible}
        />
        {!item.md_exists ? <span className="kb-lib-file-meta-muted">{S.lib_file_no_md}</span> : null}
        {suggestionCount > 0 ? (
          <span className="kb-lib-file-submeta-chip is-suggestion">
            {S.lib_file_suggestions.replace('{n}', String(suggestionCount))}
          </span>
        ) : null}
      </div>
    </div>
  )
}
