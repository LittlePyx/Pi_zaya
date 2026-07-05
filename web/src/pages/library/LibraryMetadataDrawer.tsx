import { type Dispatch, type SetStateAction } from 'react'
import { Alert, AutoComplete, Button, Drawer, Input, Select, Space, Tag, Typography } from 'antd'
import type { LibraryFileItem } from '../../api/library'
import {
  normalizeTextList,
  normalizeTextValue,
  optionMatchesInput,
  stripKnownSourceExt,
} from './libraryPageUtils'
import './LibraryMetadataDrawer.css'

const { Text } = Typography

type ReadingStatusValue = '' | 'unread' | 'reading' | 'done' | 'revisit'

type LibraryMetaDraft = {
  paper_category: string
  reading_status: ReadingStatusValue
  note: string
  user_tags: string[]
}

type TextOption = {
  value: string
  label: string
}

type MetadataSuggestionAction = {
  category_action?: '' | 'accept' | 'dismiss'
  accept_tags?: string[]
  dismiss_tags?: string[]
  accept_all_tags?: boolean
  dismiss_all_tags?: boolean
}

type LibraryMetadataDrawerProps = {
  open: boolean
  item: LibraryFileItem | null
  draft: LibraryMetaDraft
  draftCategory: string
  draftTags: string[]
  suggestionCount: number
  saving: boolean
  suggestionSaving: boolean
  S: Record<string, string>
  paperCategoryOptions: TextOption[]
  paperTagOptions: TextOption[]
  readingStatusOptions: TextOption[]
  tagInputSeparators: string[]
  onClose: () => void
  onDraftChange: Dispatch<SetStateAction<LibraryMetaDraft>>
  onSave: () => void
  onRegenerateSuggestions: () => void
  onApplySuggestionAction: (body: MetadataSuggestionAction) => void
  readingStatusLabel: (value: string, S: Record<string, string>) => string
}

export function LibraryMetadataDrawer({
  open,
  item,
  draft,
  draftCategory,
  draftTags,
  suggestionCount,
  saving,
  suggestionSaving,
  S,
  paperCategoryOptions,
  paperTagOptions,
  readingStatusOptions,
  tagInputSeparators,
  onClose,
  onDraftChange,
  onSave,
  onRegenerateSuggestions,
  onApplySuggestionAction,
  readingStatusLabel,
}: LibraryMetadataDrawerProps) {
  return (
    <Drawer
      title={item ? S.lib_meta_title.replace('{name}', item.name) : S.lib_meta_title_fallback}
      open={open}
      size={420}
      onClose={onClose}
      destroyOnClose={false}
    >
      <div className="kb-lib-meta-drawer">
        {item ? (
          <div className="kb-lib-meta-hero">
            <div className="kb-lib-meta-hero-copy">
              <Text className="kb-lib-meta-hero-title">{stripKnownSourceExt(item.name) || item.name}</Text>
              <Text type="secondary" className="kb-lib-meta-hero-note">
                {S.lib_meta_hero_hint}
              </Text>
            </div>
            <Space wrap size={[6, 6]} className="kb-lib-meta-chip-row">
              <Tag color={draftCategory ? 'blue' : 'default'}>{draftCategory || S.lib_category_unclassified}</Tag>
              {draft.reading_status ? (
                <Tag color="gold">{readingStatusLabel(draft.reading_status, S)}</Tag>
              ) : (
                <Tag>{S.lib_meta_status_not_set}</Tag>
              )}
              <Tag color={suggestionCount ? 'processing' : 'default'}>
                {suggestionCount ? S.lib_meta_suggestions.replace('{n}', String(suggestionCount)) : S.lib_meta_no_suggestions}
              </Tag>
            </Space>
            {draftTags.length ? (
              <div className="kb-lib-meta-chip-row">
                {draftTags.slice(0, 8).map((tagValue) => (
                  <Tag key={`meta-current-${tagValue}`}>{tagValue}</Tag>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        <section className="kb-lib-meta-section">
          <div className="kb-lib-meta-section-head">
            <div className="kb-lib-meta-section-copy">
              <Text className="kb-lib-meta-section-title">{S.lib_meta_section_my_org}</Text>
              <Text type="secondary" className="kb-lib-meta-section-note">
                {S.lib_meta_org_hint}
              </Text>
            </div>
          </div>

          <div className="kb-lib-meta-field">
            <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_category}</Text>
            <AutoComplete
              value={draft.paper_category}
              allowClear
              options={paperCategoryOptions}
              placeholder={S.lib_meta_category_placeholder}
              filterOption={optionMatchesInput}
              onChange={(value) => onDraftChange((cur) => ({ ...cur, paper_category: String(value || '') }))}
              onBlur={() => onDraftChange((cur) => ({ ...cur, paper_category: normalizeTextValue(cur.paper_category) }))}
            />
            <Text type="secondary" className="kb-lib-meta-help">
              {S.lib_meta_category_hint}
            </Text>
          </div>

          <div className="kb-lib-meta-field">
            <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_status}</Text>
            <Select
              value={draft.reading_status || undefined}
              allowClear
              placeholder={S.lib_meta_reading_placeholder}
              options={readingStatusOptions}
              onChange={(value) => onDraftChange((cur) => ({ ...cur, reading_status: String(value || '') as ReadingStatusValue }))}
            />
          </div>

          <div className="kb-lib-meta-field">
            <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_tags}</Text>
            <Select
              mode="tags"
              value={draft.user_tags}
              showSearch
              maxTagCount="responsive"
              tokenSeparators={tagInputSeparators}
              placeholder={S.lib_meta_tag_placeholder}
              options={paperTagOptions}
              optionFilterProp="label"
              onChange={(value) => onDraftChange((cur) => ({ ...cur, user_tags: normalizeTextList(value as unknown[]) }))}
            />
            <Text type="secondary" className="kb-lib-meta-help">
              {S.lib_meta_tags_hint}
            </Text>
          </div>

          <div className="kb-lib-meta-field">
            <Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_note}</Text>
            <Input.TextArea
              autoSize={{ minRows: 5, maxRows: 9 }}
              value={draft.note}
              placeholder={S.lib_meta_note_placeholder}
              onChange={(event) => onDraftChange((cur) => ({ ...cur, note: event.target.value }))}
            />
          </div>
        </section>

        <section className="kb-lib-meta-section kb-lib-meta-section-suggest">
          <div className="kb-lib-suggest-head">
            <div className="kb-lib-meta-section-copy">
              <Text className="kb-lib-meta-section-title">{S.lib_meta_section_system}</Text>
              <Text type="secondary" className="kb-lib-meta-section-note">
                {S.lib_meta_system_hint}
              </Text>
            </div>
            <Space size={8} wrap>
              <Button size="small" loading={suggestionSaving} onClick={() => { onRegenerateSuggestions() }}>
                {S.lib_btn_refresh_suggestions}
              </Button>
              {item?.has_suggestions ? (
                <>
                  <Button
                    size="small"
                    type="primary"
                    ghost
                    loading={suggestionSaving}
                    onClick={() => {
                      onApplySuggestionAction({
                        category_action: item?.suggested_category ? 'accept' : '',
                        accept_all_tags: true,
                      })
                    }}
                  >
                    {S.lib_btn_accept_all}
                  </Button>
                  <Button
                    size="small"
                    loading={suggestionSaving}
                    onClick={() => {
                      onApplySuggestionAction({
                        category_action: item?.suggested_category ? 'dismiss' : '',
                        dismiss_all_tags: true,
                      })
                    }}
                  >
                    {S.lib_btn_dismiss_all}
                  </Button>
                </>
              ) : null}
            </Space>
          </div>

          {item?.has_suggestions ? (
            <div className="kb-lib-suggest-list">
              {item.suggested_category ? (
                <div className="kb-lib-suggest-item">
                  <div className="kb-lib-suggest-copy">
                    <Text className="kb-lib-suggest-title">{S.lib_meta_suggest_category}</Text>
                    <div className="kb-lib-meta-chip-row">
                      <Tag color="blue">{item.suggested_category}</Tag>
                    </div>
                  </div>
                  <Space size={8}>
                    <Button
                      size="small"
                      type="primary"
                      ghost
                      loading={suggestionSaving}
                      onClick={() => { onApplySuggestionAction({ category_action: 'accept' }) }}
                    >
                      {S.lib_btn_accept}
                    </Button>
                    <Button
                      size="small"
                      loading={suggestionSaving}
                      onClick={() => { onApplySuggestionAction({ category_action: 'dismiss' }) }}
                    >
                      {S.lib_btn_dismiss}
                    </Button>
                  </Space>
                </div>
              ) : null}

              {(item?.suggested_tags || []).map((tagValue) => (
                <div key={`meta-suggest-${tagValue}`} className="kb-lib-suggest-item">
                  <div className="kb-lib-suggest-copy">
                    <Text className="kb-lib-suggest-title">{S.lib_meta_suggest_tags}</Text>
                    <div className="kb-lib-meta-chip-row">
                      <Tag>{tagValue}</Tag>
                    </div>
                  </div>
                  <Space size={8}>
                    <Button
                      size="small"
                      type="primary"
                      ghost
                      loading={suggestionSaving}
                      onClick={() => { onApplySuggestionAction({ accept_tags: [tagValue] }) }}
                    >
                      {S.lib_btn_accept}
                    </Button>
                    <Button
                      size="small"
                      loading={suggestionSaving}
                      onClick={() => { onApplySuggestionAction({ dismiss_tags: [tagValue] }) }}
                    >
                      {S.lib_btn_dismiss}
                    </Button>
                  </Space>
                </div>
              ))}
            </div>
          ) : (
            <Alert
              type="info"
              showIcon
              className="kb-lib-suggest-empty"
              message={S.lib_meta_no_suggestions_msg}
              description={S.lib_batch_hint}
            />
          )}
        </section>

        <div className="kb-lib-meta-actions">
          <Button onClick={onClose}>
            {S.lib_btn_cancel}
          </Button>
          <Button type="primary" loading={saving} onClick={() => { onSave() }}>
            {S.lib_btn_save}
          </Button>
        </div>
      </div>
    </Drawer>
  )
}
