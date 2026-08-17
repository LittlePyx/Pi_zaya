import { Button, Card, Input, Select } from 'antd'
import { ReloadOutlined, SearchOutlined, StopOutlined } from '@ant-design/icons'
import { SCOPE_OPTIONS } from './libraryPageUtils'
import './LibraryLegacyConvertCard.css'

type TextOption = {
  value: string
  label: string
}

type LibraryLegacyConvertCardProps = {
  S: Record<string, string>
  scope: string
  fileKeyword: string
  paperCategoryFilter: string
  paperCategoryFilterOptions: ReadonlyArray<TextOption>
  paperTagFilter: string
  paperTagFilterOptions: ReadonlyArray<TextOption>
  readingStatusFilter: string
  readingStatusOptions: ReadonlyArray<TextOption>
  converting: boolean
  onScopeChange: (value: string) => void | Promise<unknown>
  onFileKeywordChange: (value: string) => void
  onPaperCategoryFilterChange: (value: string) => void
  onPaperTagFilterChange: (value: string) => void
  onReadingStatusFilterChange: (value: string) => void
  onClearMetadataFilters: () => void
  onRefresh: () => void | Promise<unknown>
  onConvertPending: () => void | Promise<unknown>
  onStopConvert: () => void | Promise<unknown>
}

export function LibraryLegacyConvertCard({
  S,
  scope,
  fileKeyword,
  paperCategoryFilter,
  paperCategoryFilterOptions,
  paperTagFilter,
  paperTagFilterOptions,
  readingStatusFilter,
  readingStatusOptions,
  converting,
  onScopeChange,
  onFileKeywordChange,
  onPaperCategoryFilterChange,
  onPaperTagFilterChange,
  onReadingStatusFilterChange,
  onClearMetadataFilters,
  onRefresh,
  onConvertPending,
  onStopConvert,
}: LibraryLegacyConvertCardProps) {
  return (
    <Card size="small" className="kb-lib-card kb-lib-legacy-convert-card" title={S.lib_convert_scope}>
      <div className="kb-lib-convert-shell">
        <div className="kb-lib-convert-row kb-lib-convert-row-top">
          <Select
            value={scope}
            onChange={(value) => { void onScopeChange(String(value)) }}
            data-testid="library-convert-scope"
            className="kb-lib-convert-scope"
            options={SCOPE_OPTIONS(S)}
          />
          <Input
            value={fileKeyword}
            onChange={(event) => onFileKeywordChange(event.target.value)}
            allowClear
            prefix={<SearchOutlined className="opacity-50" />}
            placeholder={S.lib_filter_filename}
            className="kb-lib-convert-search"
          />
          <Button className="kb-lib-convert-refresh" icon={<ReloadOutlined />} onClick={() => { void onRefresh() }}>
            {S.lib_btn_refresh}
          </Button>
        </div>

        <div className="kb-lib-convert-row kb-lib-convert-row-filters">
          <Select
            value={paperCategoryFilter || undefined}
            allowClear
            placeholder={S.lib_filter_category}
            className="kb-lib-convert-filter"
            options={Array.from(paperCategoryFilterOptions)}
            onChange={(value) => onPaperCategoryFilterChange(String(value || ''))}
          />
          <Select
            value={paperTagFilter || undefined}
            allowClear
            showSearch
            placeholder={S.lib_filter_tag}
            className="kb-lib-convert-filter"
            options={Array.from(paperTagFilterOptions)}
            optionFilterProp="label"
            onChange={(value) => onPaperTagFilterChange(String(value || ''))}
          />
          <Select
            value={readingStatusFilter || undefined}
            allowClear
            placeholder={S.lib_filter_reading}
            className="kb-lib-convert-filter"
            options={Array.from(readingStatusOptions)}
            onChange={(value) => onReadingStatusFilterChange(String(value || ''))}
          />
          <Button className="kb-lib-convert-refresh" onClick={onClearMetadataFilters}>
            {S.lib_btn_clear_metadata_filter}
          </Button>
        </div>

        <div className="kb-lib-convert-row kb-lib-convert-row-actions">
          <Button type="primary" onClick={() => { void onConvertPending() }}>
            {S.lib_btn_convert_pending}
          </Button>
          {converting ? (
            <Button icon={<StopOutlined />} danger onClick={() => { void onStopConvert() }}>
              {S.lib_btn_stop_all}
            </Button>
          ) : null}
        </div>
      </div>
    </Card>
  )
}
