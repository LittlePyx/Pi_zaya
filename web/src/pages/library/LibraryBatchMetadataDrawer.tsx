import { type Dispatch, type SetStateAction } from 'react'
import { AutoComplete, Button, Checkbox, Drawer, Select, Space, Tag, Typography } from 'antd'
import {
  normalizeTextList,
  normalizeTextValue,
  optionMatchesInput,
} from './libraryPageUtils'
import './LibraryMetadataDrawer.css'

const { Text } = Typography

type ReadingStatusValue = '' | 'unread' | 'reading' | 'done' | 'revisit'

export type LibraryBatchMetaDraft = {
  apply_paper_category: boolean
  paper_category: string
  apply_reading_status: boolean
  reading_status: ReadingStatusValue
  add_tags: string[]
  remove_tags: string[]
}

type TextOption = {
  value: string
  label: string
}

type LibraryBatchMetadataDrawerProps = {
  open: boolean
  selectedCount: number
  draft: LibraryBatchMetaDraft
  saving: boolean
  S: Record<string, string>
  paperCategoryOptions: ReadonlyArray<TextOption>
  paperTagOptions: ReadonlyArray<TextOption>
  paperTagFilterOptions: ReadonlyArray<TextOption>
  readingStatusOptions: ReadonlyArray<TextOption>
  tagInputSeparators: string[]
  onClose: () => void
  onDraftChange: Dispatch<SetStateAction<LibraryBatchMetaDraft>>
  onSave: () => void
  readingStatusLabel: (value: string, S: Record<string, string>) => string
}

export function LibraryBatchMetadataDrawer({
  open,
  selectedCount,
  draft,
  saving,
  S,
  paperCategoryOptions,
  paperTagOptions,
  paperTagFilterOptions,
  readingStatusOptions,
  tagInputSeparators,
  onClose,
  onDraftChange,
  onSave,
  readingStatusLabel,
}: LibraryBatchMetadataDrawerProps) {
  const draftCategory = normalizeTextValue(draft.paper_category)
  const draftAddTags = normalizeTextList(draft.add_tags)
  const draftRemoveTags = normalizeTextList(draft.remove_tags)
  const draftWillClearCategory = draft.apply_paper_category && !draftCategory
  const draftWillClearStatus = draft.apply_reading_status && !draft.reading_status
  const draftReadingLabel = draft.apply_reading_status
    ? readingStatusLabel(draft.reading_status, S)
    : ''

  return (
    <Drawer
      title={S.lib_batch_edit_count_format.replace('{n}', String(selectedCount))}
      open={open}
      size={420}
      onClose={onClose}
      destroyOnClose={false}
    >
      <div className="kb-lib-meta-drawer">
        <div className="kb-lib-meta-hero kb-lib-meta-hero-batch">
          <div className="kb-lib-meta-hero-copy">
            <Text className="kb-lib-meta-hero-title">{S.lib_batch_edit_hero.replace('{n}', String(selectedCount))}</Text>
            <Text type="secondary" className="kb-lib-meta-hero-note">
              {S.lib_batch_notice}
            </Text>
          </div>
          <Space wrap size={[6, 6]} className="kb-lib-meta-chip-row">
            <Tag color={selectedCount ? 'blue' : 'default'}>{S.lib_batch_selected_tag.replace('{n}', String(selectedCount))}</Tag>
            {draft.apply_paper_category ? (
              draftWillClearCategory ? (
                <Tag color="warning">{S.lib_batch_clear_category_label}</Tag>
              ) : (
                <Tag color="processing">{S.lib_batch_set_category_label.replace('{category}', draftCategory)}</Tag>
              )
            ) : null}
            {draft.apply_reading_status ? (
              draftWillClearStatus ? (
                <Tag color="warning">{S.lib_batch_clear_status_label}</Tag>
              ) : (
                <Tag color="gold">{S.lib_batch_set_status_label.replace('{status}', draftReadingLabel)}</Tag>
              )
            ) : null}
            {draftAddTags.length ? (
              <Tag color="green">{S.lib_batch_add_tag_count.replace('{n}', String(draftAddTags.length))}</Tag>
            ) : null}
            {draftRemoveTags.length ? (
              <Tag color="red">{S.lib_batch_remove_tag_count.replace('{n}', String(draftRemoveTags.length))}</Tag>
            ) : null}
          </Space>
        </div>

        <section className="kb-lib-meta-section">
          <div className="kb-lib-meta-section-head">
            <div className="kb-lib-meta-section-copy">
              <Text className="kb-lib-meta-section-title">{S.lib_batch_section_setting}</Text>
              <Text type="secondary" className="kb-lib-meta-section-note">
                {S.lib_batch_setting_hint}
              </Text>
            </div>
          </div>

          <div className={`kb-lib-meta-field ${draft.apply_paper_category ? '' : 'is-muted'}`}>
            <Checkbox
              checked={draft.apply_paper_category}
              onChange={(event) => onDraftChange((cur) => ({ ...cur, apply_paper_category: event.target.checked }))}
            >
              {S.lib_batch_set_category_cb}
            </Checkbox>
            <AutoComplete
              value={draft.paper_category}
              allowClear
              disabled={!draft.apply_paper_category}
              options={Array.from(paperCategoryOptions)}
              placeholder={S.lib_meta_category_placeholder}
              filterOption={optionMatchesInput}
              onChange={(value) => onDraftChange((cur) => ({ ...cur, paper_category: String(value || '') }))}
              onBlur={() => onDraftChange((cur) => ({ ...cur, paper_category: normalizeTextValue(cur.paper_category) }))}
            />
            <Text type="secondary" className="kb-lib-meta-help">
              {S.lib_batch_category_hint}
            </Text>
          </div>

          <div className={`kb-lib-meta-field ${draft.apply_reading_status ? '' : 'is-muted'}`}>
            <Checkbox
              checked={draft.apply_reading_status}
              onChange={(event) => onDraftChange((cur) => ({ ...cur, apply_reading_status: event.target.checked }))}
            >
              {S.lib_batch_set_status_cb}
            </Checkbox>
            <Select
              value={draft.reading_status || undefined}
              allowClear
              disabled={!draft.apply_reading_status}
              placeholder={S.lib_meta_reading_placeholder}
              options={Array.from(readingStatusOptions)}
              onChange={(value) => onDraftChange((cur) => ({ ...cur, reading_status: String(value || '') as ReadingStatusValue }))}
            />
          </div>
        </section>

        <section className="kb-lib-meta-section">
          <div className="kb-lib-meta-section-head">
            <div className="kb-lib-meta-section-copy">
              <Text className="kb-lib-meta-section-title">{S.lib_batch_section_tags}</Text>
              <Text type="secondary" className="kb-lib-meta-section-note">
                {S.lib_batch_tags_hint}
              </Text>
            </div>
          </div>

          <div className="kb-lib-meta-field">
            <Text type="secondary" className="kb-lib-meta-label">{S.lib_batch_label_add_tags}</Text>
            <Select
              mode="tags"
              value={draft.add_tags}
              showSearch
              maxTagCount="responsive"
              tokenSeparators={tagInputSeparators}
              placeholder={S.lib_batch_add_tag_placeholder}
              options={Array.from(paperTagOptions)}
              optionFilterProp="label"
              onChange={(value) => onDraftChange((cur) => ({ ...cur, add_tags: normalizeTextList(value as unknown[]) }))}
            />
          </div>

          <div className="kb-lib-meta-field">
            <Text type="secondary" className="kb-lib-meta-label">{S.lib_batch_label_remove_tags}</Text>
            <Select
              mode="multiple"
              value={draft.remove_tags}
              maxTagCount="responsive"
              placeholder={S.lib_batch_remove_tag_placeholder}
              options={Array.from(paperTagFilterOptions)}
              optionFilterProp="label"
              onChange={(value) => onDraftChange((cur) => ({ ...cur, remove_tags: normalizeTextList(value as unknown[]) }))}
            />
          </div>
        </section>

        <div className="kb-lib-meta-actions">
          <Button onClick={onClose}>
            {S.lib_btn_cancel}
          </Button>
          <Button type="primary" loading={saving} onClick={() => { onSave() }}>
            {S.lib_btn_apply_to_selected}
          </Button>
        </div>
      </div>
    </Drawer>
  )
}
