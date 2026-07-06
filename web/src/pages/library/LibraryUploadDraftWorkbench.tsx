import type { ReactNode } from 'react'
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Empty,
  Input,
  Pagination,
  Select,
  Space,
  Switch,
  Tag,
  Tooltip,
  Typography,
} from 'antd'
import {
  ApiOutlined,
  CheckOutlined,
  ClearOutlined,
  CopyOutlined,
  ExclamationCircleOutlined,
  FolderOpenOutlined,
  LockOutlined,
} from '@ant-design/icons'
import {
  classifyFailedReason,
  suggestionBasisTagColor,
  type UploadDraft,
  type UploadDraftFilter,
  type UploadErrorReason,
} from './libraryPageUtils'
import './LibraryUploadDraftWorkbench.css'

const { Text } = Typography

type UploadFailureReason = Exclude<UploadErrorReason, 'all'>

type UploadFilterOption = {
  value: string
  label: string
}

type FailedReasonBucket = {
  key: UploadFailureReason
  count: number
}

type ReasonMeta = {
  label: string
  icon: ReactNode
}

type LibraryUploadDraftWorkbenchProps = {
  S: Record<string, string>
  uploadDrafts: UploadDraft[]
  filteredUploadDrafts: UploadDraft[]
  pagedUploadDrafts: UploadDraft[]
  selectedUploadCount: number
  uploadDraftFilter: UploadDraftFilter
  uploadErrorReason: UploadErrorReason
  uploadDraftFilterOptions: UploadFilterOption[]
  uploadUseLlm: boolean
  uploadInspecting: boolean
  uploadSaving: boolean
  uploadLocked: boolean
  failedUploadDraftCount: number
  failedUploadNotes: string[]
  failedReasonBuckets: FailedReasonBucket[]
  duplicateFailedDraftCount: number
  retryableFailedUploadDraftCount: number
  uploadDraftPage: number
  uploadDraftPageSize: number
  onCollapse: () => void
  onUploadUseLlmChange: (value: boolean) => void
  onFilterChange: (value: UploadDraftFilter) => void
  onClearErrorReason: () => void
  onSelectAllDrafts: () => void
  onInvertDraftSelection: () => void
  onInspectSelectedDrafts: () => void | Promise<unknown>
  onSaveSelectedDrafts: (convertNow: boolean) => void | Promise<unknown>
  onSelectFailedDrafts: () => void
  onShowDuplicateFailedDrafts: () => void
  onRetryFailedDrafts: (convertNow: boolean) => void | Promise<unknown>
  onClearSavedDrafts: () => void
  onSelectFailedReason: (reason: UploadFailureReason) => void
  onDraftSelectedChange: (key: string, selected: boolean) => void
  onDraftStemChange: (key: string, stem: string) => void
  onInspectDraft: (key: string) => void
  onSaveDraft: (key: string, convertNow: boolean) => void | Promise<unknown>
  onPageChange: (page: number) => void
}

function draftStatusText(S: Record<string, string>): Record<UploadDraft['status'], string> {
  return {
    queued: S.lib_draft_queued,
    inspecting: S.lib_draft_inspecting,
    ready: S.lib_draft_ready,
    saving: S.lib_draft_saving,
    saved: S.lib_draft_saved,
    error: S.lib_draft_error,
  }
}

function failedReasonMeta(S: Record<string, string>): Record<UploadFailureReason, ReasonMeta> {
  return {
    duplicate: { label: S.lib_fail_duplicate, icon: <CopyOutlined /> },
    path: { label: S.lib_fail_path, icon: <FolderOpenOutlined /> },
    permission: { label: S.lib_fail_permission, icon: <LockOutlined /> },
    network: { label: S.lib_fail_network, icon: <ApiOutlined /> },
    other: { label: S.lib_fail_other, icon: <ExclamationCircleOutlined /> },
  }
}

function failedReasonLabel(S: Record<string, string>, reason: UploadErrorReason) {
  if (reason === 'all') return S.lib_error_filter_all
  return failedReasonMeta(S)[reason].label
}

export function LibraryUploadDraftWorkbench({
  S,
  uploadDrafts,
  filteredUploadDrafts,
  pagedUploadDrafts,
  selectedUploadCount,
  uploadDraftFilter,
  uploadErrorReason,
  uploadDraftFilterOptions,
  uploadUseLlm,
  uploadInspecting,
  uploadSaving,
  uploadLocked,
  failedUploadDraftCount,
  failedUploadNotes,
  failedReasonBuckets,
  duplicateFailedDraftCount,
  retryableFailedUploadDraftCount,
  uploadDraftPage,
  uploadDraftPageSize,
  onCollapse,
  onUploadUseLlmChange,
  onFilterChange,
  onClearErrorReason,
  onSelectAllDrafts,
  onInvertDraftSelection,
  onInspectSelectedDrafts,
  onSaveSelectedDrafts,
  onSelectFailedDrafts,
  onShowDuplicateFailedDrafts,
  onRetryFailedDrafts,
  onClearSavedDrafts,
  onSelectFailedReason,
  onDraftSelectedChange,
  onDraftStemChange,
  onInspectDraft,
  onSaveDraft,
  onPageChange,
}: LibraryUploadDraftWorkbenchProps) {
  const statusText = draftStatusText(S)
  const reasonMeta = failedReasonMeta(S)
  const activeErrorReasonText = failedReasonLabel(S, uploadErrorReason)
  const showReasonFilter = (uploadDraftFilter === 'error' || uploadDraftFilter === 'dup_error') && uploadErrorReason !== 'all'

  return (
    <Card
      size="small"
      className="kb-lib-card kb-lib-upload-workbench-card"
      title={S.lib_section_upload_workbench}
      extra={(
        <Space size={8}>
          <Text type="secondary" className="text-xs">{S.lib_upload_selected_count.replace('{n}', String(selectedUploadCount))}</Text>
          <Text type="secondary" className="text-xs">{S.lib_upload_show_count.replace('{n}', String(filteredUploadDrafts.length)).replace('{total}', String(uploadDrafts.length))}</Text>
          <Button size="small" onClick={onCollapse}>{S.lib_btn_collapse}</Button>
        </Space>
      )}
    >
      <div className="space-y-3">
        <div className="kb-lib-upload-toolbar flex flex-wrap items-center gap-2">
          <Switch checked={uploadUseLlm} onChange={onUploadUseLlmChange} />
          <Text className="text-sm text-[var(--muted)]">{S.lib_upload_use_llm}</Text>
          <Select
            value={uploadDraftFilter}
            onChange={(value) => onFilterChange(value as UploadDraftFilter)}
            options={uploadDraftFilterOptions}
            className="kb-lib-upload-filter"
          />
          <Tooltip title={S.lib_btn_select_all}>
            <Button icon={<CheckOutlined />} onClick={onSelectAllDrafts}>{S.lib_btn_select_all}</Button>
          </Tooltip>
          <Tooltip title={S.lib_btn_invert_select}>
            <Button icon={<ClearOutlined />} onClick={onInvertDraftSelection}>{S.lib_btn_invert_select}</Button>
          </Tooltip>
          <Button loading={uploadInspecting} disabled={uploadLocked} onClick={() => { void onInspectSelectedDrafts() }}>{S.lib_btn_scan_selected}</Button>
          <Button loading={uploadSaving} disabled={uploadLocked} onClick={() => { void onSaveSelectedDrafts(false) }}>{S.lib_btn_save_selected}</Button>
          <Button type="primary" loading={uploadSaving} disabled={uploadLocked} onClick={() => { void onSaveSelectedDrafts(true) }}>{S.lib_btn_save_and_convert}</Button>
          <Button disabled={uploadLocked} onClick={onSelectFailedDrafts}>{S.lib_btn_select_failed}</Button>
          <Button disabled={uploadLocked || duplicateFailedDraftCount === 0} onClick={onShowDuplicateFailedDrafts}>{S.lib_btn_view_dup_failed}</Button>
          <Button loading={uploadSaving || uploadInspecting} disabled={uploadLocked || retryableFailedUploadDraftCount === 0} onClick={() => { void onRetryFailedDrafts(false) }}>{S.lib_btn_retry_failed}</Button>
          <Button type="primary" loading={uploadSaving || uploadInspecting} disabled={uploadLocked || retryableFailedUploadDraftCount === 0} onClick={() => { void onRetryFailedDrafts(true) }}>{S.lib_btn_retry_and_convert}</Button>
          <Button disabled={uploadLocked} onClick={onClearSavedDrafts}>{S.lib_btn_clear_saved}</Button>
        </div>

        {showReasonFilter ? (
          <div className="kb-lib-upload-meta flex flex-wrap items-center gap-3">
            <Button size="small" onClick={onClearErrorReason}>
              {S.lib_upload_filter_reason.replace('{reason}', activeErrorReasonText)}
            </Button>
          </div>
        ) : null}

        {failedUploadDraftCount > 0 ? (
          <Alert
            type="warning"
            showIcon
            message={S.lib_upload_failed_drafts.replace('{n}', String(failedUploadDraftCount))}
            description={(
              <div className="kb-lib-failed-summary">
                <div className="kb-lib-failed-reasons">
                  {failedReasonBuckets.map((bucket) => (
                    <Button
                      key={bucket.key}
                      size="small"
                      icon={reasonMeta[bucket.key].icon}
                      className={`kb-lib-failed-reason-btn kb-lib-reason-tone is-${bucket.key}${uploadErrorReason === bucket.key ? ' is-active' : ''}`}
                      onClick={() => onSelectFailedReason(bucket.key)}
                    >
                      {reasonMeta[bucket.key].label} ({bucket.count})
                    </Button>
                  ))}
                </div>
                <Text type="secondary" className="text-xs">
                  {failedUploadNotes.length > 0 ? failedUploadNotes.join(' | ') : S.lib_upload_error_hint}
                </Text>
              </div>
            )}
          />
        ) : null}

        {filteredUploadDrafts.length > 0 ? (
          <div className="kb-lib-upload-draft-list">
            <div className="kb-lib-upload-draft-list-body" role="list">
              {pagedUploadDrafts.map((draft) => {
                const reasonKey = draft.status === 'error' ? classifyFailedReason(draft.note) : null
                return (
                  <div key={draft.key} className="kb-lib-upload-draft-item" role="listitem">
                    <div className="w-full space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <Checkbox checked={draft.selected} onChange={(event) => onDraftSelectedChange(draft.key, event.target.checked)} />
                        <Text className="min-w-0 flex-1 truncate text-sm">{draft.name}</Text>
                        <Tag color={draft.status === 'saved' ? 'success' : draft.status === 'error' ? 'error' : (draft.status === 'saving' || draft.status === 'inspecting') ? 'processing' : 'default'}>
                          {statusText[draft.status]}
                        </Tag>
                        {reasonKey ? (
                          <span className={`kb-lib-inline-reason-chip kb-lib-reason-tone is-${reasonKey}`}>
                            {reasonMeta[reasonKey].icon}
                            <span>{reasonMeta[reasonKey].label}</span>
                          </span>
                        ) : null}
                      </div>
                      <div className="flex flex-wrap items-center gap-2 pl-6">
                        <Text type="secondary" className="text-xs">{S.lib_upload_suggest_name}</Text>
                        <Input value={draft.stem} onChange={(event) => onDraftStemChange(draft.key, event.target.value)} className="w-[24rem] max-w-full" />
                        <Button size="small" disabled={uploadLocked || draft.status === 'saving' || draft.status === 'inspecting'} onClick={() => onInspectDraft(draft.key)}>{S.lib_btn_scan}</Button>
                        <Button size="small" disabled={uploadLocked || draft.status === 'saving' || draft.status === 'saved' || draft.status === 'inspecting'} onClick={() => { void onSaveDraft(draft.key, false) }}>{S.lib_btn_save}</Button>
                        <Button size="small" type="primary" disabled={uploadLocked || draft.status === 'saving' || draft.status === 'saved' || draft.status === 'inspecting'} onClick={() => { void onSaveDraft(draft.key, true) }}>{S.lib_btn_save_and_convert}</Button>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 pl-6">
                        <Text type="secondary" className="text-xs">{draft.displayName}</Text>
                        {draft.suggestionBasisLabel ? (
                          <Tag color={suggestionBasisTagColor({ match_method: draft.suggestionMatchMethod, year_source: draft.suggestionYearSource })}>
                            {draft.suggestionBasisLabel}
                          </Tag>
                        ) : null}
                      </div>
                      {draft.suggestionBasisDetail ? (
                        <Text type="secondary" className="block pl-6 text-xs">{draft.suggestionBasisDetail}</Text>
                      ) : null}
                      {draft.note ? (
                        <Text type="secondary" className={`block pl-6 text-xs${reasonKey ? ' kb-lib-fail-note' : ''}`}>
                          {draft.note}
                        </Text>
                      ) : null}
                    </div>
                  </div>
                )
              })}
            </div>
            {filteredUploadDrafts.length > uploadDraftPageSize ? (
              <Pagination
                className="kb-lib-list-pagination"
                size="small"
                current={uploadDraftPage}
                pageSize={uploadDraftPageSize}
                total={filteredUploadDrafts.length}
                showSizeChanger={false}
                onChange={onPageChange}
              />
            ) : null}
          </div>
        ) : (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={S.lib_upload_empty} />
        )}
      </div>
    </Card>
  )
}
