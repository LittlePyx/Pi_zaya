import { Empty } from 'antd'
import './LibraryTaxonomyViews.css'

export type CategoryCardItem = {
  key: string
  label: string
  count: number
  unreadCount: number
  convertedCount: number
  pendingCount: number
  commonTags: string[]
  recentPapers: string[]
}

export type TagCardItem = {
  key: string
  label: string
  count: number
  unreadCount: number
  categories: string[]
  recentPapers: string[]
}

type LibraryCategoryCardsProps = {
  S: Record<string, string>
  cards: CategoryCardItem[]
  onlyUnclassified: boolean
  paperCategoryFilter: string
  onSelectCategory: (card: CategoryCardItem) => void
}

type LibraryTagCardsProps = {
  S: Record<string, string>
  cards: TagCardItem[]
  paperTagFilter: string
  onSelectTag: (card: TagCardItem) => void
}

export function LibraryCategoryCards({
  S,
  cards,
  onlyUnclassified,
  paperCategoryFilter,
  onSelectCategory,
}: LibraryCategoryCardsProps) {
  if (!cards.length) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_empty_category} />
  }

  return (
    <div className="kb-lib-category-grid">
      {cards.map((card) => {
        const isUnclassified = card.key === 'category:__unclassified__'
        const active = isUnclassified ? onlyUnclassified : (!onlyUnclassified && paperCategoryFilter === card.label)
        return (
          <button
            key={card.key}
            type="button"
            className={`kb-lib-category-card${active ? ' is-active' : ''}`}
            onClick={() => onSelectCategory(card)}
          >
            <div className="kb-lib-category-card-head">
              <div className="kb-lib-category-card-title">
                <span>{card.label}</span>
                <strong>{card.count}</strong>
              </div>
              <div className="kb-lib-category-card-meta">
                <span>{card.unreadCount} unread</span>
                <span>{card.convertedCount} converted</span>
                {card.pendingCount > 0 ? <span>{card.pendingCount} pending</span> : null}
              </div>
            </div>

            {card.commonTags.length > 0 ? (
              <div className="kb-lib-category-card-tags">
                {card.commonTags.map((tagValue) => (
                  <span key={`${card.key}-${tagValue}`} className="kb-lib-category-tag">
                    #{tagValue}
                  </span>
                ))}
              </div>
            ) : (
              <div className="kb-lib-category-card-empty">{S.lib_tag_empty_common}</div>
            )}

            <div className="kb-lib-category-card-recent">
              {card.recentPapers.map((paper) => (
                <span key={`${card.key}-${paper}`} className="kb-lib-category-paper">
                  {paper}
                </span>
              ))}
            </div>
          </button>
        )
      })}
    </div>
  )
}

export function LibraryTagCards({
  S,
  cards,
  paperTagFilter,
  onSelectTag,
}: LibraryTagCardsProps) {
  if (!cards.length) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_empty_tag} />
  }

  return (
    <div className="kb-lib-tag-grid">
      {cards.map((card) => {
        const active = paperTagFilter && card.label.toLowerCase() === paperTagFilter.toLowerCase()
        return (
          <button
            key={card.key}
            type="button"
            className={`kb-lib-tag-card${active ? ' is-active' : ''}`}
            onClick={() => onSelectTag(card)}
          >
            <div className="kb-lib-tag-card-head">
              <div className="kb-lib-tag-card-title">
                <span>#{card.label}</span>
                <strong>{card.count}</strong>
              </div>
              <div className="kb-lib-tag-card-meta">
                <span>{S.lib_tag_unread_count.replace('{n}', String(card.unreadCount))}</span>
              </div>
            </div>

            {card.categories.length > 0 ? (
              <div className="kb-lib-tag-card-cats">
                {card.categories.map((category) => (
                  <span key={`${card.key}-${category}`} className="kb-lib-tag-category">
                    {category}
                  </span>
                ))}
              </div>
            ) : null}

            <div className="kb-lib-tag-card-recent">
              {card.recentPapers.map((paper) => (
                <span key={`${card.key}-${paper}`} className="kb-lib-tag-paper">
                  {paper}
                </span>
              ))}
            </div>
          </button>
        )
      })}
    </div>
  )
}
