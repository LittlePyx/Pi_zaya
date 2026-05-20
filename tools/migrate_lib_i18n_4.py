"""Final comprehensive migration — handle ALL remaining Chinese strings in LibraryPage.tsx."""
FP = r'f:\research-papers\2026\Jan\else\kb_chat\web\src\pages\LibraryPage.tsx'
with open(FP, 'r', encoding='utf-8') as f:
    content = f.read()

repl = [
    # --- Refresh button in convert card ---
    ('<Button className="kb-lib-convert-refresh" icon={<ReloadOutlined />} onClick={() => { void store.loadFiles(scope) }}>\n              刷新\n            </Button>',
     '<Button className="kb-lib-convert-refresh" icon={<ReloadOutlined />} onClick={() => { void store.loadFiles(scope) }}>\n              {S.lib_btn_refresh}\n            </Button>'),

    # --- Clear metadata filter button ---
    ('<Button\n              className="kb-lib-convert-refresh"\n              onClick={() => {\n                setPaperCategoryFilter(\'\')\n                setPaperTagFilter(\'\')\n                setReadingStatusFilter(\'\')\n              }}\n            >\n              清空元数据筛选\n            </Button>',
     '<Button\n              className="kb-lib-convert-refresh"\n              onClick={() => {\n                setPaperCategoryFilter(\'\')\n                setPaperTagFilter(\'\')\n                setReadingStatusFilter(\'\')\n              }}\n            >\n              {S.lib_btn_clear_metadata_filter}\n            </Button>'),

    # --- Taxonomy card title ---
    ('<Card size="small" className="kb-lib-card kb-lib-taxonomy-bar" title="文献分类与标签">',
     '<Card size="small" className="kb-lib-card kb-lib-taxonomy-bar" title={S.lib_taxonomy_title}>'),

    # --- Taxonomy result count ---
    ('<Text type="secondary" className="kb-lib-taxonomy-result">\n                  已显示 {visibleAll.length}/{store.files.length} 篇文献\n                </Text>',
     '<Text type="secondary" className="kb-lib-taxonomy-result">\n                  {S.lib_taxonomy_result.replace(\'{n}\', String(visibleAll.length)).replace(\'{total}\', String(store.files.length))}\n                </Text>'),

    # --- Taxonomy filtering pill ---
    ('<span className="kb-lib-taxonomy-status-pill">\n                    筛选中 {activeTaxonomyFilterCount}\n                  </span>',
     '<span className="kb-lib-taxonomy-status-pill">\n                    {S.lib_taxonomy_filtering.replace(\'{n}\', String(activeTaxonomyFilterCount))}\n                  </span>'),

    # --- Select current list ---
    ('<Button className="kb-lib-action-quiet" onClick={selectCurrentListItems}>\n                      选中当前列表\n                    </Button>',
     '<Button className="kb-lib-action-quiet" onClick={selectCurrentListItems}>\n                      {S.lib_btn_select_current_list}\n                    </Button>'),

    # --- Refresh suggestions ---
    ('<Button\n                      className="kb-lib-action-tonal"\n                      loading={suggestionsRefreshing}\n                      onClick={() => { void regenerateSuggestionsForVisible() }}\n                    >\n                      刷新建议\n                    </Button>',
     '<Button\n                      className="kb-lib-action-tonal"\n                      loading={suggestionsRefreshing}\n                      onClick={() => { void regenerateSuggestionsForVisible() }}\n                    >\n                      {S.lib_btn_refresh_suggestions}\n                    </Button>'),

    # --- Clear filters ---
    ('<Button className="kb-lib-action-quiet" onClick={clearTaxonomyFilters}>\n                      清空筛选\n                    </Button>',
     '<Button className="kb-lib-action-quiet" onClick={clearTaxonomyFilters}>\n                      {S.lib_btn_clear_filters}\n                    </Button>'),

    # --- Quick filter: 未读 ---
    ('<button\n                  type="button"\n                  className={`kb-lib-taxonomy-pill is-status${onlyUnread ? \' is-active\' : \'\'}`}\n                  onClick={() => setOnlyUnread((value) => !value)}\n                >\n                  未读\n                </button>',
     '<button\n                  type="button"\n                  className={`kb-lib-taxonomy-pill is-status${onlyUnread ? \' is-active\' : \'\'}`}\n                  onClick={() => setOnlyUnread((value) => !value)}\n                >\n                  {S.lib_taxonomy_unread}\n                </button>'),

    # --- Quick filter: 未分类 ---
    ('<button\n                  type="button"\n                  className={`kb-lib-taxonomy-pill is-category${onlyUnclassified ? \' is-active\' : \'\'}`}\n                  onClick={() => {\n                    const next = !onlyUnclassified\n                    setOnlyUnclassified(next)\n                    if (next) setPaperCategoryFilter(\'\')\n                  }}\n                >\n                  未分类\n                </button>',
     '<button\n                  type="button"\n                  className={`kb-lib-taxonomy-pill is-category${onlyUnclassified ? \' is-active\' : \'\'}`}\n                  onClick={() => {\n                    const next = !onlyUnclassified\n                    setOnlyUnclassified(next)\n                    if (next) setPaperCategoryFilter(\'\')\n                  }}\n                >\n                  {S.lib_category_unclassified}\n                </button>'),

    # --- Quick filter: 有建议 ---
    ('<button\n                  type="button"\n                  className={`kb-lib-taxonomy-pill is-suggestion${onlySuggested ? \' is-active\' : \'\'}`}\n                  onClick={() => setOnlySuggested((value) => !value)}\n                >\n                  有建议\n                </button>',
     '<button\n                  type="button"\n                  className={`kb-lib-taxonomy-pill is-suggestion${onlySuggested ? \' is-active\' : \'\'}`}\n                  onClick={() => setOnlySuggested((value) => !value)}\n                >\n                  {S.lib_taxonomy_has_suggestions}\n                </button>'),

    # --- Batch bar: selected count ---
    ('<span className="kb-lib-batch-badge is-strong">已选 {selectedLibraryCount} 篇</span>',
     '<span className="kb-lib-batch-badge is-strong">{S.lib_batch_selected_count.replace(\'{n}\', String(selectedLibraryCount))}</span>'),

    # --- Batch bar: current list count ---
    ('<span className="kb-lib-batch-badge">{currentListItems.length} 篇在当前列表</span>',
     '<span className="kb-lib-batch-badge">{S.lib_batch_current_count.replace(\'{n}\', String(currentListItems.length))}</span>'),

    # --- Batch bar: title ---
    ('<Text className="kb-lib-batch-count">批量整理当前选择</Text>',
     '<Text className="kb-lib-batch-count">{S.lib_batch_title_selected}</Text>'),

    # --- Batch bar: hint ---
    ('<Text type="secondary" className="kb-lib-batch-hint">批量编辑只会作用于已选文献，适合先批量设分类，再统一加减标签。</Text>',
     '<Text type="secondary" className="kb-lib-batch-hint">{S.lib_batch_hint_scope}</Text>'),

    # --- Batch bar: select current list btn ---
    ('<Button onClick={selectCurrentListItems}>选中当前列表</Button>',
     '<Button onClick={selectCurrentListItems}>{S.lib_btn_select_current_list}</Button>'),

    # --- Batch bar: clear selection ---
    ('<Button onClick={clearLibrarySelection} disabled={!selectedLibraryCount}>清空选中</Button>',
     '<Button onClick={clearLibrarySelection} disabled={!selectedLibraryCount}>{S.lib_btn_clear_selection}</Button>'),

    # --- Batch bar: batch edit ---
    ('<Button type="primary" onClick={openBatchEditor} disabled={!selectedLibraryCount}>批量编辑</Button>',
     '<Button type="primary" onClick={openBatchEditor} disabled={!selectedLibraryCount}>{S.lib_batch_title}</Button>'),

    # --- Meta hero note ---
    ('<Text type="secondary" className="kb-lib-meta-hero-note">\n                  分类和标签完全由你掌控。可以沿用已有词汇，也可以直接录入你自己的整理方式。\n                </Text>',
     '<Text type="secondary" className="kb-lib-meta-hero-note">\n                  {S.lib_meta_hero_hint}\n                </Text>'),

    # --- Meta: 阅读状态未设置 ---
    ('<Tag>阅读状态未设置</Tag>',
     '<Tag>{S.lib_meta_status_not_set}</Tag>'),

    # --- Meta section: 我的整理 ---
    ('<Text className="kb-lib-meta-section-title">我的整理</Text>',
     '<Text className="kb-lib-meta-section-title">{S.lib_meta_section_my_org}</Text>'),

    ('<Text type="secondary" className="kb-lib-meta-section-note">\n                  主分类放稳定归属，标签放可复用的检索切面。\n                </Text>',
     '<Text type="secondary" className="kb-lib-meta-section-note">\n                  {S.lib_meta_org_hint}\n                </Text>'),

    # --- Meta label: 主分类 ---
    ('<Text type="secondary" className="kb-lib-meta-label">主分类</Text>',
     '<Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_category}</Text>'),

    ('<Text type="secondary" className="kb-lib-meta-help">\n                可直接新建分类。建议保持短、稳定、能跨多篇论文复用。\n              </Text>',
     '<Text type="secondary" className="kb-lib-meta-help">\n                {S.lib_meta_category_hint}\n              </Text>'),

    # --- Meta label: 阅读状态 ---
    ('<Text type="secondary" className="kb-lib-meta-label">阅读状态</Text>',
     '<Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_status}</Text>'),

    # --- Meta label: 标签 ---
    ('<Text type="secondary" className="kb-lib-meta-label">标签</Text>',
     '<Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_tags}</Text>'),

    ('<Text type="secondary" className="kb-lib-meta-help">\n                标签更适合放 modality、task、constraint、method property 这类可复用 facet。\n              </Text>',
     '<Text type="secondary" className="kb-lib-meta-help">\n                {S.lib_meta_tags_hint}\n              </Text>'),

    # --- Meta label: 备注 ---
    ('<Text type="secondary" className="kb-lib-meta-label">备注</Text>',
     '<Text type="secondary" className="kb-lib-meta-label">{S.lib_meta_label_note}</Text>'),

    # --- Meta section: 系统建议 ---
    ('<Text className="kb-lib-meta-section-title">系统建议</Text>',
     '<Text className="kb-lib-meta-section-title">{S.lib_meta_section_system}</Text>'),

    ('<Text type="secondary" className="kb-lib-meta-section-note">\n                  系统只建议，不会自动覆盖你已经确认的分类和标签。\n                </Text>',
     '<Text type="secondary" className="kb-lib-meta-section-note">\n                  {S.lib_meta_system_hint}\n                </Text>'),

    # --- Meta: refresh suggestions ---
    ('<Button size="small" loading={metaSuggestionSaving} onClick={() => { void regenerateMetaSuggestions() }}>\n                  刷新建议\n                </Button>',
     '<Button size="small" loading={metaSuggestionSaving} onClick={() => { void regenerateMetaSuggestions() }}>\n                  {S.lib_btn_refresh_suggestions}\n                </Button>'),

    # --- Meta: accept all ---
    ('<Button\n                      size="small"\n                      type="primary"\n                      ghost\n                      loading={metaSuggestionSaving}\n                      onClick={() => {\n                        void applyMetaSuggestionAction({\n                          category_action: metaItem?.suggested_category ? \'accept\' : \'\',\n                          accept_all_tags: true,\n                        })\n                      }}\n                    >\n                      接受全部\n                    </Button>',
     '<Button\n                      size="small"\n                      type="primary"\n                      ghost\n                      loading={metaSuggestionSaving}\n                      onClick={() => {\n                        void applyMetaSuggestionAction({\n                          category_action: metaItem?.suggested_category ? \'accept\' : \'\',\n                          accept_all_tags: true,\n                        })\n                      }}\n                    >\n                      {S.lib_btn_accept_all}\n                    </Button>'),

    # --- Meta: dismiss all ---
    ('<Button\n                      size="small"\n                      loading={metaSuggestionSaving}\n                      onClick={() => {\n                        void applyMetaSuggestionAction({\n                          category_action: metaItem?.suggested_category ? \'dismiss\' : \'\',\n                          dismiss_all_tags: true,\n                        })\n                      }}\n                    >\n                      忽略全部\n                    </Button>',
     '<Button\n                      size="small"\n                      loading={metaSuggestionSaving}\n                      onClick={() => {\n                        void applyMetaSuggestionAction({\n                          category_action: metaItem?.suggested_category ? \'dismiss\' : \'\',\n                          dismiss_all_tags: true,\n                        })\n                      }}\n                    >\n                      {S.lib_btn_dismiss_all}\n                    </Button>'),

    # --- Meta: suggest category title ---
    ('<Text className="kb-lib-suggest-title">建议分类</Text>',
     '<Text className="kb-lib-suggest-title">{S.lib_meta_suggest_category}</Text>'),

    # --- Meta: suggest tags title ---
    ('<Text className="kb-lib-suggest-title">建议标签</Text>',
     '<Text className="kb-lib-suggest-title">{S.lib_meta_suggest_tags}</Text>'),

    # --- Meta: accept btn ---
    ('<Button\n                        size="small"\n                        type="primary"\n                        ghost\n                        loading={metaSuggestionSaving}\n                        onClick={() => { void applyMetaSuggestionAction({ category_action: \'accept\' }) }}\n                      >\n                        接受\n                      </Button>',
     '<Button\n                        size="small"\n                        type="primary"\n                        ghost\n                        loading={metaSuggestionSaving}\n                        onClick={() => { void applyMetaSuggestionAction({ category_action: \'accept\' }) }}\n                      >\n                        {S.lib_btn_accept}\n                      </Button>'),

    # --- Meta: dismiss btn ---
    ('<Button\n                        size="small"\n                        loading={metaSuggestionSaving}\n                        onClick={() => { void applyMetaSuggestionAction({ category_action: \'dismiss\' }) }}\n                      >\n                        忽略\n                      </Button>',
     '<Button\n                        size="small"\n                        loading={metaSuggestionSaving}\n                        onClick={() => { void applyMetaSuggestionAction({ category_action: \'dismiss\' }) }}\n                      >\n                        {S.lib_btn_dismiss}\n                      </Button>'),

    # --- Meta: accept btn for tags ---
    ('<Button\n                        size="small"\n                        type="primary"\n                        ghost\n                        loading={metaSuggestionSaving}\n                        onClick={() => { void applyMetaSuggestionAction({ accept_tags: [tagValue] }) }}\n                      >\n                        接受\n                      </Button>',
     '<Button\n                        size="small"\n                        type="primary"\n                        ghost\n                        loading={metaSuggestionSaving}\n                        onClick={() => { void applyMetaSuggestionAction({ accept_tags: [tagValue] }) }}\n                      >\n                        {S.lib_btn_accept}\n                      </Button>'),

    # --- Meta: dismiss btn for tags ---
    ('<Button\n                        size="small"\n                        loading={metaSuggestionSaving}\n                        onClick={() => { void applyMetaSuggestionAction({ dismiss_tags: [tagValue] }) }}\n                      >\n                        忽略\n                      </Button>',
     '<Button\n                        size="small"\n                        loading={metaSuggestionSaving}\n                        onClick={() => { void applyMetaSuggestionAction({ dismiss_tags: [tagValue] }) }}\n                      >\n                        {S.lib_btn_dismiss}\n                      </Button>'),

    # --- Meta: no suggestions alert ---
    ('message="当前还没有分类建议"',
     'message={S.lib_meta_no_suggestions_msg}'),

    # --- Fix broken description: "{S.lib_batch_hint}" (quoted string, should be expression) ---
    ('description="{S.lib_batch_hint}"',
     'description={S.lib_batch_hint}'),

    # --- Meta drawer cancel btn ---
    ('<Button onClick={() => setMetaDrawerOpen(false)}>\n              取消\n            </Button>',
     '<Button onClick={() => setMetaDrawerOpen(false)}>\n              {S.lib_btn_cancel}\n            </Button>'),

    # --- Meta drawer save btn ---
    ('<Button type="primary" loading={metaSaving} onClick={() => { void saveMetaEditor() }}>\n              保存\n            </Button>',
     '<Button type="primary" loading={metaSaving} onClick={() => { void saveMetaEditor() }}>\n              {S.lib_btn_save}\n            </Button>'),

    # --- Batch drawer title ---
    ("title={`批量编辑 · ${selectedLibraryCount} 篇文献`}",
     "title={S.lib_batch_edit_count_format.replace('{n}', String(selectedLibraryCount))}"),

    # --- Batch hero title ---
    ('<Text className="kb-lib-meta-hero-title">批量编辑 {selectedLibraryCount} 篇文献</Text>',
     '<Text className="kb-lib-meta-hero-title">{S.lib_batch_edit_hero.replace(\'{n}\', String(selectedLibraryCount))}</Text>'),

    # --- Batch hero note ---
    ('<Text type="secondary" className="kb-lib-meta-hero-note">\n                适合先统一主分类和阅读状态，再一次性补充或移除标签。\n              </Text>',
     '<Text type="secondary" className="kb-lib-meta-hero-note">\n                {S.lib_batch_notice}\n              </Text>'),

    # --- Batch selected tag ---
    ('<Tag color={selectedLibraryCount ? \'blue\' : \'default\'}>{selectedLibraryCount} 篇已选</Tag>',
     '<Tag color={selectedLibraryCount ? \'blue\' : \'default\'}>{S.lib_batch_selected_tag.replace(\'{n}\', String(selectedLibraryCount))}</Tag>'),

    # --- Batch set category label ---
    ('<Tag color="processing">将设置分类: {normalizeTextValue(batchDraft.paper_category)}</Tag>',
     '<Tag color="processing">{S.lib_batch_set_category_label.replace(\'{category}\', normalizeTextValue(batchDraft.paper_category))}</Tag>'),

    # --- Batch add tag count ---
    ('<Tag color="green">新增 {normalizeTextList(batchDraft.add_tags).length} 个标签</Tag>',
     '<Tag color="green">{S.lib_batch_add_tag_count.replace(\'{n}\', String(normalizeTextList(batchDraft.add_tags).length))}</Tag>'),

    # --- Batch section: 批量设置 ---
    ('<Text className="kb-lib-meta-section-title">批量设置</Text>',
     '<Text className="kb-lib-meta-section-title">{S.lib_batch_section_setting}</Text>'),

    ('<Text type="secondary" className="kb-lib-meta-section-note">\n                  只会影响当前选中的文献，不会改到未选中的内容。\n                </Text>',
     '<Text type="secondary" className="kb-lib-meta-section-note">\n                  {S.lib_batch_setting_hint}\n                </Text>'),

    # --- Batch set category checkbox ---
    ('<Checkbox\n                checked={batchDraft.apply_paper_category}\n                onChange={(event) => setBatchDraft((cur) => ({ ...cur, apply_paper_category: event.target.checked }))}\n              >\n                批量设置主分类\n              </Checkbox>',
     '<Checkbox\n                checked={batchDraft.apply_paper_category}\n                onChange={(event) => setBatchDraft((cur) => ({ ...cur, apply_paper_category: event.target.checked }))}\n              >\n                {S.lib_batch_set_category_cb}\n              </Checkbox>'),

    # --- Batch set reading status checkbox ---
    ('<Checkbox\n                checked={batchDraft.apply_reading_status}\n                onChange={(event) => setBatchDraft((cur) => ({ ...cur, apply_reading_status: event.target.checked }))}\n              >\n                批量设置阅读状态\n              </Checkbox>',
     '<Checkbox\n                checked={batchDraft.apply_reading_status}\n                onChange={(event) => setBatchDraft((cur) => ({ ...cur, apply_reading_status: event.target.checked }))}\n              >\n                {S.lib_batch_set_status_cb}\n              </Checkbox>'),

    # --- Batch category hint ---
    ('<Text type="secondary" className="kb-lib-meta-help">\n                这里也支持手动录入新分类，会写入到所有已选文献。\n              </Text>',
     '<Text type="secondary" className="kb-lib-meta-help">\n                {S.lib_batch_category_hint}\n              </Text>'),

    # --- Batch section: 标签批处理 ---
    ('<Text className="kb-lib-meta-section-title">标签批处理</Text>',
     '<Text className="kb-lib-meta-section-title">{S.lib_batch_section_tags}</Text>'),

    ('<Text type="secondary" className="kb-lib-meta-section-note">\n                  新增标签支持自由输入；移除标签只从已存在标签里选，避免误删。\n                </Text>',
     '<Text type="secondary" className="kb-lib-meta-section-note">\n                  {S.lib_batch_tags_hint}\n                </Text>'),

    # --- Batch label: add tags ---
    ('<Text type="secondary" className="kb-lib-meta-label">批量新增标签</Text>',
     '<Text type="secondary" className="kb-lib-meta-label">{S.lib_batch_label_add_tags}</Text>'),

    # --- Batch label: remove tags ---
    ('<Text type="secondary" className="kb-lib-meta-label">批量移除标签</Text>',
     '<Text type="secondary" className="kb-lib-meta-label">{S.lib_batch_label_remove_tags}</Text>'),

    # --- Batch drawer cancel ---
    ('<Button onClick={() => setBatchDrawerOpen(false)}>\n              取消\n            </Button>',
     '<Button onClick={() => setBatchDrawerOpen(false)}>\n              {S.lib_btn_cancel}\n            </Button>'),

    # --- Batch drawer apply ---
    ('<Button type="primary" loading={batchSaving} onClick={() => { void saveBatchEditor() }}>\n              应用到已选文献\n            </Button>',
     '<Button type="primary" loading={batchSaving} onClick={() => { void saveBatchEditor() }}>\n              {S.lib_btn_apply_to_selected}\n            </Button>'),

    # --- Module-level functions with S_ fallback (accept these as intentional, but convert the refsync catch) ---
]

count = 0
for old, new in repl:
    if old in content:
        content = content.replace(old, new)
        count += 1
        print(f'  OK ({count})')
    else:
        print(f'  MISS: {old[:60]}...')

with open(FP, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'\nReplaced {count}/{len(repl)} strings')
