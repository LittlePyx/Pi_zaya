"""Migrate ALL remaining Chinese strings in LibraryPage.tsx — pass 3 comprehensive."""
import re

FP = r'f:\research-papers\2026\Jan\else\kb_chat\web\src\pages\LibraryPage.tsx'
with open(FP, 'r', encoding='utf-8') as f:
    content = f.read()

repl = [
    # --- Page head ---
    ('<Text className="text-2xl font-semibold">文献管理</Text>',  # 文献管理
     '<Text className="text-2xl font-semibold">{S.page_library}</Text>'),

    ('<Text type="secondary" className="text-sm">先配置目录，再批量处理，最后在列表逐条检查。</Text>',  # 先配置目录...
     '<Text type="secondary" className="text-sm">{S.lib_page_subtitle}</Text>'),

    # --- Head buttons ---
    ('>更新知识库<',  # 更新知识库
     '>{S.reindex_now}<'),
    ('>同步引用信息<',  # 同步引用信息
     '>{S.lib_btn_sync_refs}<'),

    # --- Stats grid ---
    ('<Text type="secondary">当前视图</Text>',  # 当前视图
     '<Text type="secondary">{S.lib_stats_view}</Text>'),
    ('<Text type="secondary">待转换</Text>',  # 待转换
     '<Text type="secondary">{S.lib_stats_pending}</Text>'),
    ('<Text type="secondary">已转换</Text>',  # 已转换
     '<Text type="secondary">{S.lib_stats_converted}</Text>'),
    ('<Text type="secondary">排队中</Text>',  # 排队中
     '<Text type="secondary">{S.lib_stats_queued}</Text>'),
    ('<Text type="secondary">运行中</Text>',  # 运行中
     '<Text type="secondary">{S.lib_stats_running}</Text>'),

    # --- Sticky convert title ---
    ('<Text className="kb-lib-sticky-title">转换中 {store.progress.completed}/{store.progress.total}</Text>',  # 转换中
     '<Text className="kb-lib-sticky-title">{S.lib_convert_progress.replace(\'{done}\', String(store.progress.completed)).replace(\'{total}\', String(store.progress.total))}</Text>'),

    # --- Sticky stop button ---
    ('<Button size="small" danger icon={<StopOutlined />} onClick={() => { void store.cancelConvert() }}>\n                  停止\n                </Button>',  # 停止
     '<Button size="small" danger icon={<StopOutlined />} onClick={() => { void store.cancelConvert() }}>\n                  {S.lib_btn_stop}\n                </Button>'),

    # --- File row: no md ---
    ('<span className="kb-lib-file-meta-muted">待生成 MD</span>',  # 待生成 MD
     '<span className="kb-lib-file-meta-muted">{S.lib_file_no_md}</span>'),

    # --- File row: suggestion count ---
    ('<span className="kb-lib-file-submeta-chip is-suggestion">\n                  {suggestionCount} 建议\n                </span>',  # {suggestionCount} 建议
     '<span className="kb-lib-file-submeta-chip is-suggestion">\n                  {S.lib_file_suggestions.replace(\'{n}\', String(suggestionCount))}\n                </span>'),

    # --- File row: 分类 button ---
    ('<Button className="kb-lib-file-action-main" size="small" onClick={() => openMetaEditor(item)}>\n            分类\n          </Button>',  # 分类
     '<Button className="kb-lib-file-action-main" size="small" onClick={() => openMetaEditor(item)}>\n            {S.lib_btn_categorize}\n          </Button>'),

    # --- File row: 阅读 button ---
    ('<Button\n              className="kb-lib-file-action-link"\n              type="text"\n              size="small"\n              disabled={!item.md_path}\n              onClick={() => { void handleStartPaperGuide(item) }}\n            >\n              阅读\n            </Button>',  # 阅读
     '<Button\n              className="kb-lib-file-action-link"\n              type="text"\n              size="small"\n              disabled={!item.md_path}\n              onClick={() => { void handleStartPaperGuide(item) }}\n            >\n              {S.lib_btn_read}\n            </Button>'),

    # --- File row: 转换 button ---
    ('<Button\n              className="kb-lib-file-action-link is-accent"\n              type="text"\n              size="small"\n              disabled={item.task_state !== \'idle\'}\n              onClick={() => { void handleConvertOne(item) }}\n            >\n              转换\n            </Button>',  # 转换
     '<Button\n              className="kb-lib-file-action-link is-accent"\n              type="text"\n              size="small"\n              disabled={item.task_state !== \'idle\'}\n              onClick={() => { void handleConvertOne(item) }}\n            >\n              {S.lib_btn_convert}\n            </Button>'),

    # --- Category card empty ---
    ('<div className="kb-lib-category-card-empty">暂时还没有明显的常用标签</div>',  # 暂时还没有明显的常用标签
     '<div className="kb-lib-category-card-empty">{S.lib_tag_empty_common}</div>'),

    # --- Tag card unread ---
    ('<span>{card.unreadCount} 未读</span>',  # {n} 未读
     '<span>{S.lib_tag_unread_count.replace(\'{n}\', String(card.unreadCount))}</span>'),

    # --- Virtual scroll hint ---
    ('<Text type="secondary" className="text-xs">已启用虚拟滚动（{items.length} 条）</Text>',  # 已启用虚拟滚动
     '<Text type="secondary" className="text-xs">{S.lib_virtual_scroll_hint.replace(\'{n}\', String(items.length))}</Text>'),

    # --- Section title: 文件名管理 ---
    ('<Text className="kb-lib-section-title">文件名管理</Text>',  # 文件名管理
     '<Text className="kb-lib-section-title">{S.lib_section_rename}</Text>'),

    # --- Rename toolbar buttons ---
    ('<Button className="kb-lib-action-quiet" size="small" onClick={selectRenameDiffItems}>全选</Button>',  # 全选
     '<Button className="kb-lib-action-quiet" size="small" onClick={selectRenameDiffItems}>{S.lib_btn_select_all}</Button>'),
    ('<Button className="kb-lib-action-quiet" size="small" onClick={clearRenameSelection}>清空</Button>',  # 清空
     '<Button className="kb-lib-action-quiet" size="small" onClick={clearRenameSelection}>{S.lib_btn_clear}</Button>'),

    # --- Rename: 应用改名 ---
    ('<Button type="primary" size="small" disabled={!hasRenameSelection} onClick={() => { void applyRenameSuggestions() }}>应用改名</Button>',  # 应用改名
     '<Button type="primary" size="small" disabled={!hasRenameSelection} onClick={() => { void applyRenameSuggestions() }}>{S.lib_btn_apply_rename}</Button>'),

    # --- Rename meta format ---
    ('<span className="kb-lib-rename-meta">{selectedRenameCount} 已选 · {renameVisible.length}/{renameItems.length} 显示</span>',  # 已选 · 显示
     '<span className="kb-lib-rename-meta">{S.lib_rename_meta_format.replace(\'{sel}\', String(selectedRenameCount)).replace(\'{vis}\', String(renameVisible.length)).replace(\'{total}\', String(renameItems.length))}</span>'),

    # --- Rename no files ---
    ('<Text type="secondary" className="kb-lib-section-note">\n          当前范围内没有需要改名的文件。\n        </Text>',  # 当前范围内...
     '<Text type="secondary" className="kb-lib-section-note">\n          {S.lib_rename_no_files}\n        </Text>'),

    # --- Prep workbench card title ---
    ('<Card size="small" className="kb-lib-card kb-lib-workbench-card" title="准备工作台">',  # 准备工作台
     '<Card size="small" className="kb-lib-card kb-lib-workbench-card" title={S.lib_prep_workbench}>'),

    # --- Section: 目录设置 ---
    ('<Text className="kb-lib-section-title">目录设置</Text>',  # 目录设置
     '<Text className="kb-lib-section-title">{S.lib_section_dir}</Text>'),

    # --- Draft count --- Note: S.lib_workbench_draft_count already exists
    ('<span className="kb-lib-rename-meta">草稿 {uploadDrafts.length}</span>',  # 草稿
     '<span className="kb-lib-rename-meta">{S.lib_workbench_draft_count.replace(\'{n}\', String(uploadDrafts.length))}</span>'),

    # --- Upload locked hint - these look like they're in template literal quotes incorrectly ---
    # Actually looking at line 2023:
    ("{store.converting ? '{S.lib_upload_locked_converting}' : '{S.lib_upload_locked_refsync}'}",
     "{store.converting ? S.lib_upload_locked_converting : S.lib_upload_locked_refsync}"),

    # --- Section: 批量处理 ---
    ('<Text className="kb-lib-section-title">批量处理</Text>',  # 批量处理
     '<Text className="kb-lib-section-title">{S.lib_section_batch}</Text>'),

    # --- Convert pending button (short) ---
    ('<Button className="kb-lib-action-tonal" type="primary" onClick={() => { void handleConvertPending() }}>转换待处理</Button>',  # 转换待处理
     '<Button className="kb-lib-action-tonal" type="primary" onClick={() => { void handleConvertPending() }}>{S.lib_btn_convert_pending_short}</Button>'),

    # --- Upload workbench card title ---
    ('<Card\n      size="small"\n      className="kb-lib-card kb-lib-upload-workbench-card"\n      title="上传工作台"',  # 上传工作台
     '<Card\n      size="small"\n      className="kb-lib-card kb-lib-upload-workbench-card"\n      title={S.lib_section_upload_workbench}'),

    # --- Upload card extra: selected count ---
    ('<Text type="secondary" className="text-xs">已选 {selectedUploadCount} 项</Text>',  # 已选 {n} 项
     '<Text type="secondary" className="text-xs">{S.lib_upload_selected_count.replace(\'{n}\', String(selectedUploadCount))}</Text>'),
    ('<Text type="secondary" className="text-xs">显示 {filteredUploadDrafts.length}/{uploadDrafts.length} 项</Text>',  # 显示 {n}/{total} 项
     '<Text type="secondary" className="text-xs">{S.lib_upload_show_count.replace(\'{n}\', String(filteredUploadDrafts.length)).replace(\'{total}\', String(uploadDrafts.length))}</Text>'),
    ('<Button size="small" onClick={() => setUploadWorkbenchOpen(false)}>收起</Button>',  # 收起
     '<Button size="small" onClick={() => setUploadWorkbenchOpen(false)}>{S.lib_btn_collapse}</Button>'),

    # --- Upload toolbar: LLM hint ---
    ('<Text className="text-sm text-[var(--muted)]">使用 LLM 补全信息</Text>',  # 使用 LLM 补全信息
     '<Text className="text-sm text-[var(--muted)]">{S.lib_upload_use_llm}</Text>'),

    # --- Upload toolbar buttons ---
    ('<Tooltip title="全选草稿"><Button icon={<CheckOutlined />} onClick={selectAllUploadDrafts}>全选</Button></Tooltip>',  # 全选草稿 / 全选
     '<Tooltip title={S.lib_btn_select_all}><Button icon={<CheckOutlined />} onClick={selectAllUploadDrafts}>{S.lib_btn_select_all}</Button></Tooltip>'),
    ('<Tooltip title="反选草稿"><Button icon={<ClearOutlined />} onClick={invertUploadDraftSelection}>反选</Button></Tooltip>',  # 反选草稿 / 反选
     '<Tooltip title={S.lib_btn_invert_select}><Button icon={<ClearOutlined />} onClick={invertUploadDraftSelection}>{S.lib_btn_invert_select}</Button></Tooltip>'),
    ('<Button loading={uploadInspecting} disabled={uploadLocked} onClick={() => { void inspectSelectedDrafts() }}>扫描已选</Button>',  # 扫描已选
     '<Button loading={uploadInspecting} disabled={uploadLocked} onClick={() => { void inspectSelectedDrafts() }}>{S.lib_btn_scan_selected}</Button>'),
    ('<Button loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(false) }}>保存已选</Button>',  # 保存已选
     '<Button loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(false) }}>{S.lib_btn_save_selected}</Button>'),
    ('<Button type="primary" loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(true) }}>保存并转换</Button>',  # 保存并转换
     '<Button type="primary" loading={uploadSaving} disabled={uploadLocked} onClick={() => { void saveSelectedDrafts(true) }}>{S.lib_btn_save_and_convert}</Button>'),
    ('<Button disabled={uploadLocked} onClick={selectFailedDrafts}>选择失败项</Button>',  # 选择失败项
     '<Button disabled={uploadLocked} onClick={selectFailedDrafts}>{S.lib_btn_select_failed}</Button>'),
    ('<Button disabled={uploadLocked || duplicateFailedDrafts.length === 0} onClick={showDuplicateFailedDrafts}>仅看重复失败</Button>',  # 仅看重复失败
     '<Button disabled={uploadLocked || duplicateFailedDrafts.length === 0} onClick={showDuplicateFailedDrafts}>{S.lib_btn_view_dup_failed}</Button>'),
    ('<Button loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(false) }}>重试失败项</Button>',  # 重试失败项
     '<Button loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(false) }}>{S.lib_btn_retry_failed}</Button>'),
    ('<Button type="primary" loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(true) }}>重试并转换</Button>',  # 重试并转换
     '<Button type="primary" loading={uploadSaving} disabled={uploadLocked || failedUploadDrafts.length === 0} onClick={() => { void retryFailedDrafts(true) }}>{S.lib_btn_retry_and_convert}</Button>'),
    ('<Button disabled={uploadLocked} onClick={() => setUploadDrafts((cur) => cur.filter((x) => x.status !== \'saved\'))}>清理已保存</Button>',  # 清理已保存
     '<Button disabled={uploadLocked} onClick={() => setUploadDrafts((cur) => cur.filter((x) => x.status !== \'saved\'))}>{S.lib_btn_clear_saved}</Button>'),

    # --- Upload filter reason ---
    ('<Button size="small" onClick={() => setUploadErrorReason(\'all\')}>\n              原因筛选：{activeErrorReasonText}（清除）\n            </Button>',  # 原因筛选：{reason}（清除）
     '<Button size="small" onClick={() => setUploadErrorReason(\'all\')}>\n              {S.lib_upload_filter_reason.replace(\'{reason}\', activeErrorReasonText)}\n            </Button>'),

    # --- Failed drafts alert ---
    ("message={`失败草稿：${failedUploadDrafts.length}`}",  # 失败草稿：{n}
     "message={S.lib_upload_failed_drafts.replace('{n}', String(failedUploadDrafts.length))}"),

    # --- Draft row: 建议存储名 ---
    ('<Text type="secondary" className="text-xs">建议存储名</Text>',  # 建议存储名
     '<Text type="secondary" className="text-xs">{S.lib_upload_suggest_name}</Text>'),

    # --- Draft row: 扫描 button ---
    ('<Button size="small" disabled={uploadLocked || x.status === \'saving\' || x.status === \'inspecting\'} onClick={() => { void inspectDraft(x.key) }}>扫描</Button>',  # 扫描
     '<Button size="small" disabled={uploadLocked || x.status === \'saving\' || x.status === \'inspecting\'} onClick={() => { void inspectDraft(x.key) }}>{S.lib_btn_scan}</Button>'),

    # --- Draft row: 保存 button ---
    ('<Button size="small" disabled={uploadLocked || x.status === \'saving\' || x.status === \'saved\' || x.status === \'inspecting\'} onClick={() => { void saveDraft(x.key, false) }}>保存</Button>',  # 保存
     '<Button size="small" disabled={uploadLocked || x.status === \'saving\' || x.status === \'saved\' || x.status === \'inspecting\'} onClick={() => { void saveDraft(x.key, false) }}>{S.lib_btn_save}</Button>'),

    # --- Draft row: 保存并转换 button ---
    ('<Button size="small" type="primary" disabled={uploadLocked || x.status === \'saving\' || x.status === \'saved\' || x.status === \'inspecting\'} onClick={() => { void saveDraft(x.key, true) }}>保存并转换</Button>',  # 保存并转换
     '<Button size="small" type="primary" disabled={uploadLocked || x.status === \'saving\' || x.status === \'saved\' || x.status === \'inspecting\'} onClick={() => { void saveDraft(x.key, true) }}>{S.lib_btn_save_and_convert}</Button>'),

    # --- Message toast: 已刷新 {updated} 篇文献的分类建议 ---
    ("message.success(`已刷新 ${updated} 篇文献的分类建议`)",
     "message.success(S.lib_msg_suggestions_refreshed_count.replace('{n}', String(updated)))"),

    # --- Message toast: 已批量更新 {updated} 篇文献 ---
    ("message.success(`已批量更新 ${updated} 篇文献`)",
     "message.success(S.lib_msg_batch_updated_count.replace('{n}', String(updated)))"),
]

# First pass: exact replacements
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
