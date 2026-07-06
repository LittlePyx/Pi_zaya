import { Button, Checkbox, Input, Pagination, Select, Tag, Typography } from 'antd'
import type { RenameSuggestionItem } from '../../api/library'
import { RENAME_SCOPE_OPTIONS, suggestionBasisTagColor } from './libraryPageUtils'
import './LibraryRenameWorkbench.css'

const { Text } = Typography

type LibraryRenameWorkbenchProps = {
  S: Record<string, string>
  renameScope: string
  renameItems: RenameSuggestionItem[]
  renameVisible: RenameSuggestionItem[]
  pagedRenameVisible: RenameSuggestionItem[]
  renameSelected: Record<string, boolean>
  renameOverrides: Record<string, string>
  renameResultsOpen: boolean
  renameLoading: boolean
  renameApplying: boolean
  renamePage: number
  renamePageSize: number
  selectedRenameCount: number
  onRenameScopeChange: (value: string) => void
  onScanRenameSuggestions: () => void | Promise<unknown>
  onToggleResultsOpen: () => void
  onSelectDiffItems: () => void
  onClearSelection: () => void
  onApplyRenameSuggestions: () => void | Promise<unknown>
  onSelectedChange: (name: string, selected: boolean) => void
  onOverrideChange: (name: string, value: string) => void
  onPageChange: (page: number) => void
}

export function LibraryRenameWorkbench({
  S,
  renameScope,
  renameItems,
  renameVisible,
  pagedRenameVisible,
  renameSelected,
  renameOverrides,
  renameResultsOpen,
  renameLoading,
  renameApplying,
  renamePage,
  renamePageSize,
  selectedRenameCount,
  onRenameScopeChange,
  onScanRenameSuggestions,
  onToggleResultsOpen,
  onSelectDiffItems,
  onClearSelection,
  onApplyRenameSuggestions,
  onSelectedChange,
  onOverrideChange,
  onPageChange,
}: LibraryRenameWorkbenchProps) {
  const renameHasResults = renameItems.length > 0
  const renameHasVisibleItems = renameVisible.length > 0
  const hasRenameSelection = selectedRenameCount > 0

  return (
    <section className="kb-lib-workbench-section kb-lib-workbench-section-rename">
      <div className="kb-lib-section-head">
        <div className="kb-lib-section-copy">
          <Text className="kb-lib-section-title">{S.lib_section_rename}</Text>
        </div>
      </div>

      <div className="kb-lib-rename-summary">
        <div className="kb-lib-rename-summary-main">
          <Select
            value={renameScope}
            onChange={onRenameScopeChange}
            className="kb-lib-rename-scope"
            options={RENAME_SCOPE_OPTIONS(S)}
          />
          <Button
            size="small"
            className="kb-lib-action-tonal"
            loading={renameLoading}
            onClick={() => { void onScanRenameSuggestions() }}
          >
            {renameHasResults ? S.lib_rename_recheck : S.lib_btn_rename_check}
          </Button>
          {renameHasVisibleItems ? (
            <Button className="kb-lib-action-quiet" size="small" onClick={onToggleResultsOpen}>
              {renameResultsOpen ? S.lib_rename_collapse : S.lib_rename_expand}
            </Button>
          ) : null}
          {renameHasVisibleItems ? (
            <Button className="kb-lib-action-quiet" size="small" onClick={onSelectDiffItems}>
              {S.lib_btn_select_all}
            </Button>
          ) : null}
          {hasRenameSelection ? (
            <Button className="kb-lib-action-quiet" size="small" onClick={onClearSelection}>
              {S.lib_btn_clear}
            </Button>
          ) : null}
          {hasRenameSelection ? (
            <Button
              className="kb-lib-action-tonal"
              size="small"
              type="primary"
              loading={renameApplying}
              onClick={() => { void onApplyRenameSuggestions() }}
            >
              {S.lib_btn_apply_rename}
            </Button>
          ) : null}
        </div>
        {renameHasResults ? (
          <div className="kb-lib-rename-summary-side">
            <div className="kb-lib-rename-badges">
              <span className="kb-lib-rename-meta">
                {S.lib_rename_meta_format
                  .replace('{sel}', String(selectedRenameCount))
                  .replace('{vis}', String(renameVisible.length))
                  .replace('{total}', String(renameItems.length))}
              </span>
            </div>
          </div>
        ) : null}
      </div>

      {renameHasResults && renameHasVisibleItems && renameResultsOpen ? (
        <div className="kb-lib-rename-list">
          <div className="kb-lib-rename-list-body" role="list">
            {pagedRenameVisible.map((item) => (
              <div key={item.name} className="kb-lib-rename-list-item" role="listitem">
                <div className="kb-lib-rename-item">
                  <div className="kb-lib-rename-item-head">
                    <Checkbox
                      checked={Boolean(renameSelected[item.name])}
                      onChange={(event) => onSelectedChange(item.name, event.target.checked)}
                    />
                    <Text className="kb-lib-rename-item-name">{item.name}</Text>
                    <Tag color={item.diff ? 'warning' : 'default'}>
                      {item.diff ? S.lib_rename_suggest_rename : S.lib_rename_no_rename}
                    </Tag>
                  </div>
                  <Input
                    value={renameOverrides[item.name] || ''}
                    onChange={(event) => onOverrideChange(item.name, event.target.value)}
                    className="kb-lib-rename-item-input"
                  />
                  <div className="flex flex-wrap items-center gap-2">
                    <Text type="secondary" className="kb-lib-rename-item-source">
                      {item.display_full_name}
                    </Text>
                    {item.meta?.basis_label ? (
                      <Tag color={suggestionBasisTagColor(item.meta)}>
                        {item.meta.basis_label}
                      </Tag>
                    ) : null}
                  </div>
                  {item.meta?.basis_detail ? (
                    <Text type="secondary" className="kb-lib-rename-item-source">
                      {item.meta.basis_detail}
                    </Text>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
          {renameVisible.length > renamePageSize ? (
            <Pagination
              className="kb-lib-list-pagination"
              size="small"
              current={renamePage}
              pageSize={renamePageSize}
              total={renameVisible.length}
              showSizeChanger={false}
              onChange={onPageChange}
            />
          ) : null}
        </div>
      ) : null}
      {renameHasResults && !renameHasVisibleItems ? (
        <Text type="secondary" className="kb-lib-section-note">
          {S.lib_rename_no_files}
        </Text>
      ) : null}
    </section>
  )
}
