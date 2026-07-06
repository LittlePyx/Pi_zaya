import { Button, Card, Input, Segmented, Select, Typography } from 'antd'
import { SearchOutlined } from '@ant-design/icons'
import './LibraryTaxonomyToolbar.css'

const { Text } = Typography

type LibraryBrowseMode = 'list' | 'categories' | 'tags'

type TextOption = {
  value: string
  label: string
}

type LibraryTaxonomyToolbarProps = {
  S: Record<string, string>
  browseMode: LibraryBrowseMode
  visibleCount: number
  totalCount: number
  hasActiveFilters: boolean
  activeFilterCount: number
  canSelectCurrent: boolean
  canRefreshSuggestions: boolean
  canClearFilters: boolean
  suggestionsRefreshing: boolean
  fileKeyword: string
  paperCategoryFilter: string
  paperCategoryOptions: TextOption[]
  paperTagFilter: string
  paperTagOptions: TextOption[]
  readingStatusFilter: string
  readingStatusOptions: TextOption[]
  onlyUnread: boolean
  onlyUnclassified: boolean
  onlySuggested: boolean
  diagnosticsVisible: boolean
  onlyQualityIssues: boolean
  qualityReviewCount: number
  qualityHistoryFocusCount: number
  onBrowseModeChange: (value: LibraryBrowseMode) => void
  onSelectCurrentList: () => void
  onRefreshSuggestions: () => void
  onClearFilters: () => void
  onFileKeywordChange: (value: string) => void
  onPaperCategoryFilterChange: (value: string) => void
  onPaperTagFilterChange: (value: string) => void
  onReadingStatusFilterChange: (value: string) => void
  onToggleOnlyUnread: () => void
  onToggleOnlyUnclassified: () => void
  onToggleOnlySuggested: () => void
  onToggleOnlyQualityIssues: () => void
  onClearQualityHistoryFocus: () => void
}

export function LibraryTaxonomyToolbar({
  S,
  browseMode,
  visibleCount,
  totalCount,
  hasActiveFilters,
  activeFilterCount,
  canSelectCurrent,
  canRefreshSuggestions,
  canClearFilters,
  suggestionsRefreshing,
  fileKeyword,
  paperCategoryFilter,
  paperCategoryOptions,
  paperTagFilter,
  paperTagOptions,
  readingStatusFilter,
  readingStatusOptions,
  onlyUnread,
  onlyUnclassified,
  onlySuggested,
  diagnosticsVisible,
  onlyQualityIssues,
  qualityReviewCount,
  qualityHistoryFocusCount,
  onBrowseModeChange,
  onSelectCurrentList,
  onRefreshSuggestions,
  onClearFilters,
  onFileKeywordChange,
  onPaperCategoryFilterChange,
  onPaperTagFilterChange,
  onReadingStatusFilterChange,
  onToggleOnlyUnread,
  onToggleOnlyUnclassified,
  onToggleOnlySuggested,
  onToggleOnlyQualityIssues,
  onClearQualityHistoryFocus,
}: LibraryTaxonomyToolbarProps) {
  const showTopActions = canSelectCurrent || canRefreshSuggestions || canClearFilters

  return (
    <Card size="small" className="kb-lib-card kb-lib-taxonomy-bar" title={S.lib_taxonomy_title}>
      <div className="kb-lib-taxonomy-shell">
        <div className="kb-lib-taxonomy-top">
          <div className="kb-lib-taxonomy-view">
            <Segmented
              className="kb-lib-browse-switch"
              value={browseMode}
              onChange={(value) => onBrowseModeChange(value as LibraryBrowseMode)}
              options={[
                { label: S.lib_browse_list, value: 'list' },
                { label: S.lib_browse_categories, value: 'categories' },
                { label: S.lib_browse_tags, value: 'tags' },
              ]}
            />
          </div>
          <div className="kb-lib-taxonomy-meta">
            <div className="kb-lib-taxonomy-summary">
              <Text type="secondary" className="kb-lib-taxonomy-result">
                {S.lib_taxonomy_result.replace('{n}', String(visibleCount)).replace('{total}', String(totalCount))}
              </Text>
              {hasActiveFilters ? (
                <span className="kb-lib-taxonomy-status-pill">
                  {S.lib_taxonomy_filtering.replace('{n}', String(activeFilterCount))}
                </span>
              ) : null}
            </div>
            {showTopActions ? (
              <div className="kb-lib-taxonomy-top-actions">
                {canSelectCurrent ? (
                  <Button className="kb-lib-action-quiet" onClick={onSelectCurrentList}>
                    {S.lib_btn_select_current_list}
                  </Button>
                ) : null}
                {canRefreshSuggestions ? (
                  <Button
                    className="kb-lib-action-tonal"
                    loading={suggestionsRefreshing}
                    onClick={onRefreshSuggestions}
                  >
                    {S.lib_btn_auto_organize}
                  </Button>
                ) : null}
                {canClearFilters ? (
                  <Button className="kb-lib-action-quiet" onClick={onClearFilters}>
                    {S.lib_btn_clear_filters}
                  </Button>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>

        <div className="kb-lib-taxonomy-controls">
          <div className="kb-lib-taxonomy-filters">
            <Input
              value={fileKeyword}
              onChange={(event) => onFileKeywordChange(event.target.value)}
              allowClear
              prefix={<SearchOutlined className="opacity-50" />}
              placeholder={S.lib_search_placeholder}
              className="kb-lib-taxonomy-search"
            />
            <Select
              value={paperCategoryFilter || undefined}
              allowClear
              placeholder={S.lib_search_category}
              className="kb-lib-taxonomy-select"
              options={paperCategoryOptions}
              onChange={(value) => onPaperCategoryFilterChange(String(value || ''))}
            />
            <Select
              value={paperTagFilter || undefined}
              allowClear
              showSearch
              placeholder={S.lib_search_tag}
              className="kb-lib-taxonomy-select"
              options={paperTagOptions}
              optionFilterProp="label"
              onChange={(value) => onPaperTagFilterChange(String(value || ''))}
            />
            <Select
              value={readingStatusFilter || undefined}
              allowClear
              placeholder={S.lib_search_reading}
              className="kb-lib-taxonomy-select"
              options={readingStatusOptions}
              onChange={(value) => onReadingStatusFilterChange(String(value || ''))}
            />
          </div>

          <div className="kb-lib-taxonomy-quick">
            <div className="kb-lib-taxonomy-toggle-row">
              <button
                type="button"
                className={`kb-lib-taxonomy-pill is-status${onlyUnread ? ' is-active' : ''}`}
                onClick={onToggleOnlyUnread}
              >
                {S.lib_taxonomy_unread}
              </button>
              <button
                type="button"
                className={`kb-lib-taxonomy-pill is-category${onlyUnclassified ? ' is-active' : ''}`}
                onClick={onToggleOnlyUnclassified}
              >
                {S.lib_category_unclassified}
              </button>
              <button
                type="button"
                className={`kb-lib-taxonomy-pill is-suggestion${onlySuggested ? ' is-active' : ''}`}
                onClick={onToggleOnlySuggested}
              >
                {S.lib_taxonomy_has_suggestions}
              </button>
              {diagnosticsVisible ? (
                <button
                  type="button"
                  className={`kb-lib-taxonomy-pill is-quality${onlyQualityIssues ? ' is-active' : ''}`}
                  data-testid="library-quality-issues-filter"
                  onClick={onToggleOnlyQualityIssues}
                >
                  {S.lib_quality_quick_filter_review.replace('{n}', String(qualityReviewCount))}
                </button>
              ) : null}
              {qualityHistoryFocusCount > 0 ? (
                <button
                  type="button"
                  className="kb-lib-taxonomy-pill is-quality is-active"
                  data-testid="library-quality-history-active-filter"
                  onClick={onClearQualityHistoryFocus}
                >
                  {S.lib_quality_history_focus_badge.replace('{n}', String(qualityHistoryFocusCount))}
                </button>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </Card>
  )
}
