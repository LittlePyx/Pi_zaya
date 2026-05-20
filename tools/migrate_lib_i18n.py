"""Migrate remaining Chinese strings in LibraryPage.tsx to use i18n S keys."""
import re

FP = r'f:\research-papers\2026\Jan\else\kb_chat\web\src\pages\LibraryPage.tsx'
with open(FP, 'r', encoding='utf-8') as f:
    content = f.read()

repl = [
    # --- Upload workbench ---
    ("showUploadWorkbench ? '收起队列' : '查看队列'",
     "showUploadWorkbench ? S.lib_workbench_hide_queue : S.lib_workbench_upload_queue"),

    # --- File action buttons ---
    ('>分类<', '>{S.lib_btn_categorize}<'),
    ('>阅读<', '>{S.lib_btn_read}<'),
    ('>转换<', '>{S.lib_btn_convert}<'),
    ('>重新转换<', '>{S.lib_btn_reconvert}<'),
    ('>打开 MD<', '>{S.lib_btn_open_md}<'),
    ('>删除文献<', '>{S.lib_btn_delete}<'),

    # --- Dropdown menu items ---
    ("label: '重新转换'", "label: S.lib_btn_reconvert"),
    ("label: '打开 MD'", "label: S.lib_btn_open_md"),
    ("label: '删除文献'", "label: S.lib_btn_delete"),

    # --- Empty state descriptions ---
    ('description="当前筛选下暂无分类结果"', 'description={S.lib_empty_category}'),
    ('description="当前筛选下暂无标签结果"', 'description={S.lib_empty_tag}'),
    ('description="当前筛选下暂无改名项"', 'description={S.lib_empty_rename}'),
    ('description="暂无上传草稿"', 'description={S.lib_upload_empty}'),

    # --- Rename section ---
    ('{renameHasResults ? \'重新检查\' : \'检查文件名\'}',
     '{renameHasResults ? S.lib_rename_recheck : S.lib_btn_rename_check}'),
    ('{renameResultsOpen ? \'收起结果\' : \'展开结果\'}',
     '{renameResultsOpen ? S.lib_rename_collapse : S.lib_rename_expand}'),

    # --- Tag color ---
    ("item.diff ? '建议改名' : '无需改名'",
     "item.diff ? S.lib_rename_suggest_rename : S.lib_rename_no_rename"),

    # --- Directory editor ---
    ('{showDirEditor ? \'收起编辑\' : \'编辑目录\'}',
     '{showDirEditor ? S.lib_dir_collapse : S.lib_dir_edit}'),
    ('placeholder="选择 PDF 文献目录"', 'placeholder={S.lib_dir_select_pdf}'),
    ('placeholder="选择 Markdown 输出目录"', 'placeholder={S.lib_dir_select_md}'),
    ("ellipsis={{ tooltip: pdfDirDraft || '未设置 PDF 目录' }}",
     "ellipsis={{ tooltip: pdfDirDraft || S.lib_dir_no_pdf }}"),
    ("{pdfDirDraft || '未设置 PDF 目录'}",
     "{pdfDirDraft || S.lib_dir_no_pdf}"),
    ("ellipsis={{ tooltip: mdDirDraft || '未设置 Markdown 目录' }}",
     "ellipsis={{ tooltip: mdDirDraft || S.lib_dir_no_md }}"),
    ("{mdDirDraft || '未设置 Markdown 目录'}",
     "{mdDirDraft || S.lib_dir_no_md}"),
    ('>选择目录<', '>{S.lib_dir_pick}<'),
    ('>打开目录<', '>{S.lib_dir_open}<'),
    ('>保存目录设置<', '>{S.lib_dir_save}<'),

    # --- Upload section title ---
    ('<Text className="kb-lib-section-title">上传 PDF</Text>',
     '<Text className="kb-lib-section-title">{S.lib_upload_title}</Text>'),
    ('>拖拽 PDF 到这里<', '>{S.lib_upload_drop_hint}<'),
    ('>或点击选择文件<', '>{S.lib_upload_click_hint}<'),

    # --- Upload locked hint ---
    ("转换进行中，上传暂时锁定。",
     "{S.lib_upload_locked_converting}"),
    ("引用同步进行中，上传暂时锁定。",
     "{S.lib_upload_locked_refsync}"),

    # --- Sticky status ---
    ('>转换中 <', '>{S.lib_convert_title} <'),
    ('>引用同步中<', '>{S.lib_refsync_title}<'),
    ("{store.refSync.stage || '运行中'}", "{store.refSync.stage || S.lib_refsync_running}"),
    ("{store.refSync.message || '等待同步任务'}", "{store.refSync.message || S.lib_refsync_waiting}"),
    ("{store.refSync.stage || '运行中'}", "{store.refSync.stage || S.lib_refsync_running}"),

    # --- Convert card title ---
    ('title="转换与列表筛选"', 'title={S.lib_convert_scope}'),

    # --- Filter placeholders ---
    ('placeholder="筛选文件名"', 'placeholder={S.lib_filter_filename}'),
    ('placeholder="按分类筛选"', 'placeholder={S.lib_filter_category}'),
    ('placeholder="按标签筛选"', 'placeholder={S.lib_filter_tag}'),
    ('placeholder="按阅读状态筛选"', 'placeholder={S.lib_filter_reading}'),

    # --- Refresh / Stop buttons ---
    ('>刷新<', '>{S.lib_btn_refresh}<'),
    ('>停止<', '>{S.lib_btn_stop}<'),

    # --- Browse mode tabs ---
    ("{ label: '列表', value: 'list' }", "{ label: S.lib_browse_list, value: 'list' }"),
    ("{ label: '分类', value: 'categories' }", "{ label: S.lib_browse_categories, value: 'categories' }"),
    ("{ label: '标签', value: 'tags' }", "{ label: S.lib_browse_tags, value: 'tags' }"),

    # --- Search/filter placeholders ---
    ('placeholder="搜索标题、分类、标签或备注"', 'placeholder={S.lib_search_placeholder}'),
    ('placeholder="分类"', 'placeholder={S.lib_search_category}'),
    ('placeholder="标签"', 'placeholder={S.lib_search_tag}'),
    ('placeholder="阅读状态"', 'placeholder={S.lib_search_reading}'),

    # --- Stats labels ---
    ("label: '当前视图'", "label: S.lib_stats_view"),
    ("label: '待转换'", "label: S.lib_stats_pending"),
    ("label: '已转换'", "label: S.lib_stats_converted"),
    ("label: '排队中'", "label: S.lib_stats_queued"),
    ("label: '运行中'", "label: S.lib_stats_running"),

    # --- Page progress text ---
    ('篇内进度', '{S.lib_convert_page_progress}'),

    # --- Upload draft filter options (template literals) ---
    # These use template literals with `${}` so they need careful handling
    # "全部 (${uploadDrafts...}" -> S.lib_upload_filter_all pattern
    # We'll handle these via the template string

    # --- Running/idle status ---
    ("store.refSync.running ? '运行中' : (store.refSync.status === 'idle' ? '空闲' : store.refSync.status)",
     "store.refSync.running ? S.lib_refsync_running : (store.refSync.status === 'idle' ? S.lib_refsync_idle : store.refSync.status)"),

    # --- Meta drawer placeholders ---
    ('placeholder="选择已有分类，或直接输入自己的分类"', 'placeholder={S.lib_meta_category_placeholder}'),
    ('placeholder="选择阅读状态"', 'placeholder={S.lib_meta_reading_placeholder}'),
    ('placeholder="输入标签后回车，也支持逗号 / 分号分隔"', 'placeholder={S.lib_meta_tag_placeholder}'),
    ('placeholder="记录这篇文献的用途、结论或后续阅读计划"', 'placeholder={S.lib_meta_note_placeholder}'),

    # --- Batch meta placeholders ---
    ('placeholder="选择已有分类，或直接输入自己的分类"', 'placeholder={S.lib_batch_category_placeholder}'),
    ('placeholder="选择阅读状态"', 'placeholder={S.lib_batch_reading_placeholder}'),
    ('placeholder="输入新增标签后回车，也支持逗号 / 分号分隔"', 'placeholder={S.lib_batch_add_tag_placeholder}'),
    ('placeholder="选择要移除的标签"', 'placeholder={S.lib_batch_remove_tag_placeholder}'),

    # --- Meta suggestions count ---
    ("{metaSuggestionCount ? `${metaSuggestionCount} 条系统建议` : '暂无系统建议'}",
     "{metaSuggestionCount ? S.lib_meta_suggestions.replace('{n}', String(metaSuggestionCount)) : S.lib_meta_no_suggestions}"),

    # --- Unclassified tag ---
    ("{metaDraftCategory || '未分类'}", "{metaDraftCategory || S.lib_category_unclassified}"),

    # --- Batch hint ---
    ('建议会结合你确认过的分类、标签和论文信号生成；你始终可以直接手动录入自己的分类与标签。',
     '{S.lib_batch_hint}'),

    # --- Draft status texts ---
    ("status === 'error' ? '失败' : `" , "status === 'error' ? S.lib_draft_error : `"),

    # --- Note: 未分类 in module-level ---
    ("? '未分类'", "? S.lib_category_unclassified"),
    ("label: label || '未分类'", "label: label || S.lib_category_unclassified"),
    ("const category = String(item.paper_category || '').trim() || '未分类'",
     "const category = String(item.paper_category || '').trim() || S.lib_category_unclassified"),

    # --- Delete confirmation ---
    ("title: '确认删除这个文献吗？'", "title: S.lib_menu_delete_confirm_title"),
    ("okText: '删除'", "okText: S.lib_menu_delete_ok"),
    ("cancelText: '取消'", "cancelText: S.lib_menu_delete_cancel"),

    # --- Scan/save note prefixes ---
    ("note: res.duplicate ? `重复：${String(res.existing || '')}` : '扫描完成'",
     "note: res.duplicate ? `${S.lib_upload_dup_prefix}${String(res.existing || '')}` : S.lib_upload_scan_done"),
    ("? { ...x, status: 'error', note: err instanceof Error ? err.message : '扫描失败' }",
     "? { ...x, status: 'error', note: err instanceof Error ? err.message : S.lib_upload_scan_fail }"),
    ("? { ...x, status: 'error', note: err instanceof Error ? err.message : '保存失败' }",
     "? { ...x, status: 'error', note: err instanceof Error ? err.message : S.lib_upload_save_fail }"),

    # --- Error hint ---
    ("failedUploadNotes.length > 0 ? failedUploadNotes.join(' | ') : '请查看行内错误信息后重试。'",
     "failedUploadNotes.length > 0 ? failedUploadNotes.join(' | ') : S.lib_upload_error_hint"),
]

count = 0
for old, new in repl:
    if old in content:
        content = content.replace(old, new)
        count += 1
        print(f'  OK {old[:60]}')
    else:
        print(f'  MISS: {old[:60]}')

with open(FP, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'\nReplaced {count}/{len(repl)} strings')
