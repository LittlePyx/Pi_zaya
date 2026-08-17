import { Checkbox } from 'antd'
import type { LibraryFileItem } from '../../api/library'
import type { QualityRepairHistoryRecord } from './libraryPageUtils'
import { LibraryFileActions } from './LibraryFileActions'
import { LibraryFileHeader } from './LibraryFileHeader'
import { LibraryFileProgressNote } from './LibraryFileProgressNote'
import { LibraryFileQualityLine } from './LibraryFileQualityLine'
import { LibraryFileTaxonomy } from './LibraryFileTaxonomy'
import './LibraryFileRow.css'

type LibraryFileRowProps = {
  S: Record<string, string>
  item: LibraryFileItem
  selected: boolean
  readingLabel: string
  onlyUnclassified: boolean
  paperCategoryFilter: string
  readingStatusFilter: string
  paperTagFilter: string
  qualityStatusVisible: boolean
  qualityDiagnosticsVisible: boolean
  qualityRepairing: boolean
  qualityRepairResult?: string
  qualityRepairRecord?: QualityRepairHistoryRecord
  onSelectionChange: (name: string, checked: boolean) => void
  onApplyPaperCategoryFilter: (category: string) => void
  onSetReadingStatusFilter: (status: LibraryFileItem['reading_status']) => void
  onApplyPaperTagFilter: (tag: string) => void
  onRepairQuality: (item: LibraryFileItem) => void
  onReindex: () => void
  onOpenMeta: (item: LibraryFileItem) => void
  onStartPaperGuide: (item: LibraryFileItem) => void
  onConvert: (item: LibraryFileItem) => void
  onCancel: (item: LibraryFileItem) => void
  onRetry: (item: LibraryFileItem) => void
  retrying: boolean
  onOpenPdf: (name: string) => void
  onOpenMarkdown: (name: string) => void
  onDelete: (item: LibraryFileItem) => void
}

export function LibraryFileRow({
  S,
  item,
  selected,
  readingLabel,
  onlyUnclassified,
  paperCategoryFilter,
  readingStatusFilter,
  paperTagFilter,
  qualityStatusVisible,
  qualityDiagnosticsVisible,
  qualityRepairing,
  qualityRepairResult,
  qualityRepairRecord,
  onSelectionChange,
  onApplyPaperCategoryFilter,
  onSetReadingStatusFilter,
  onApplyPaperTagFilter,
  onRepairQuality,
  onReindex,
  onOpenMeta,
  onStartPaperGuide,
  onConvert,
  onCancel,
  onRetry,
  retrying,
  onOpenPdf,
  onOpenMarkdown,
  onDelete,
}: LibraryFileRowProps) {
  const suggestionCount = (item.suggested_category ? 1 : 0) + (item.suggested_tags || []).length

  return (
    <div
      className={`kb-lib-file-row${selected ? ' is-selected' : ''}${suggestionCount > 0 ? ' has-suggestions' : ''}`}
      data-testid="library-file-row"
      data-library-file-name={item.name}
    >
      <div className="kb-lib-file-select">
        <Checkbox
          checked={selected}
          onChange={(event) => onSelectionChange(item.name, event.target.checked)}
        />
      </div>

      <div className="kb-lib-file-main">
        <LibraryFileHeader
          S={S}
          item={item}
          suggestionCount={suggestionCount}
          qualityStatusVisible={qualityStatusVisible}
          qualityDiagnosticsVisible={qualityDiagnosticsVisible}
        />

        <LibraryFileTaxonomy
          item={item}
          readingLabel={readingLabel}
          onlyUnclassified={onlyUnclassified}
          paperCategoryFilter={paperCategoryFilter}
          readingStatusFilter={readingStatusFilter}
          paperTagFilter={paperTagFilter}
          onApplyPaperCategoryFilter={onApplyPaperCategoryFilter}
          onSetReadingStatusFilter={onSetReadingStatusFilter}
          onApplyPaperTagFilter={onApplyPaperTagFilter}
        />

        <LibraryFileQualityLine
          S={S}
          item={item}
          diagnosticsVisible={qualityDiagnosticsVisible}
          repairing={qualityRepairing}
          repairResult={qualityRepairResult}
          repairRecord={qualityRepairRecord}
          onRepairQuality={() => onRepairQuality(item)}
          onReindex={onReindex}
        />

        <LibraryFileProgressNote item={item} />
      </div>

      <LibraryFileActions
        S={S}
        item={item}
        onOpenMeta={() => onOpenMeta(item)}
        onStartPaperGuide={() => onStartPaperGuide(item)}
        onConvert={() => onConvert(item)}
        onCancel={() => onCancel(item)}
        onRetry={() => onRetry(item)}
        retrying={retrying}
        onOpenPdf={() => onOpenPdf(item.name)}
        onOpenMarkdown={() => onOpenMarkdown(item.name)}
        onDelete={() => onDelete(item)}
      />
    </div>
  )
}
