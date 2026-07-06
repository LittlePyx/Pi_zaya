import type { LibraryFileItem } from '../../api/library'
import './LibraryFileTaxonomy.css'

type LibraryFileTaxonomyProps = {
  item: LibraryFileItem
  readingLabel: string
  onlyUnclassified: boolean
  paperCategoryFilter: string
  readingStatusFilter: string
  paperTagFilter: string
  onApplyPaperCategoryFilter: (category: string) => void
  onSetReadingStatusFilter: (status: LibraryFileItem['reading_status']) => void
  onApplyPaperTagFilter: (tag: string) => void
}

export function LibraryFileTaxonomy({
  item,
  readingLabel,
  onlyUnclassified,
  paperCategoryFilter,
  readingStatusFilter,
  paperTagFilter,
  onApplyPaperCategoryFilter,
  onSetReadingStatusFilter,
  onApplyPaperTagFilter,
}: LibraryFileTaxonomyProps) {
  const metaTags = item.user_tags || []
  const categoryActive = Boolean(!onlyUnclassified && paperCategoryFilter && String(item.paper_category || '') === paperCategoryFilter)
  const statusActive = Boolean(readingStatusFilter && item.reading_status === readingStatusFilter)

  if (!item.paper_category && !readingLabel && metaTags.length <= 0) return null

  return (
    <div className="kb-lib-file-taxonomy">
      {item.paper_category ? (
        <button
          type="button"
          className={`kb-lib-taxonomy-pill is-category${categoryActive ? ' is-active' : ''}`}
          onClick={() => onApplyPaperCategoryFilter(String(item.paper_category || ''))}
        >
          {item.paper_category}
        </button>
      ) : null}
      {readingLabel ? (
        <button
          type="button"
          className={`kb-lib-taxonomy-pill is-status${statusActive ? ' is-active' : ''}`}
          onClick={() => onSetReadingStatusFilter(item.reading_status)}
        >
          {readingLabel}
        </button>
      ) : null}
      {metaTags.map((tagValue) => (
        <button
          key={`${item.name}-tag-${tagValue}`}
          type="button"
          className={`kb-lib-taxonomy-pill is-tag${paperTagFilter && tagValue.toLowerCase() === paperTagFilter.toLowerCase() ? ' is-active' : ''}`}
          onClick={() => onApplyPaperTagFilter(tagValue)}
        >
          #{tagValue}
        </button>
      ))}
    </div>
  )
}
