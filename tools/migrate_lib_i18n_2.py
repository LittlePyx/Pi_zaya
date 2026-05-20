"""Migrate remaining Chinese strings in LibraryPage.tsx — pass 2."""
import re

FP = r'f:\research-papers\2026\Jan\else\kb_chat\web\src\pages\LibraryPage.tsx'
with open(FP, 'r', encoding='utf-8') as f:
    content = f.read()

repl = [
    # --- Upload draft filter options (template literals) ---
    ("{ value: 'all', label: `全部 (${uploadDrafts.length})` }",
     "{ value: 'all', label: S.lib_upload_filter_all.replace('{n}', String(uploadDrafts.length)) }"),
    ("{ value: 'todo', label: `待处理 (${uploadDrafts.filter((x) => ['queued', 'inspecting', 'ready', 'saving'].includes(x.status)).length})` }",
     "{ value: 'todo', label: S.lib_upload_filter_todo.replace('{n}', String(uploadDrafts.filter((x) => ['queued', 'inspecting', 'ready', 'saving'].includes(x.status)).length)) }"),
    ("{ value: 'error', label: `失败 (${uploadDrafts.filter((x) => x.status === 'error').length})` }",
     "{ value: 'error', label: S.lib_upload_filter_error.replace('{n}', String(uploadDrafts.filter((x) => x.status === 'error').length)) }"),
    ("{ value: 'dup_error', label: `重复失败 (${uploadDrafts.filter((x) => x.status === 'error' && isDuplicateFailure(x.note)).length})` }",
     "{ value: 'dup_error', label: S.lib_upload_filter_dup.replace('{n}', String(uploadDrafts.filter((x) => x.status === 'error' && isDuplicateFailure(x.note)).length)) }"),
    ("{ value: 'saved', label: `已保存 (${uploadDrafts.filter((x) => x.status === 'saved').length})` }",
     "{ value: 'saved', label: S.lib_upload_filter_saved.replace('{n}', String(uploadDrafts.filter((x) => x.status === 'saved').length)) }"),

    # --- Save dir: PDF/MD cannot be empty ---
    ("message.warning('PDF/MD 目录不能为空')",
     "message.warning(S.lib_msg_dir_empty)"),

    # --- Save dir success ---
    ("message.success('目录已保存')",
     "message.success(S.lib_msg_save_dir_success)"),

    # --- Save dir fail ---
    ("message.error(err instanceof Error ? err.message : '目录保存失败')",
     "message.error(err instanceof Error ? err.message : S.lib_msg_save_dir_fail)"),

    # --- No dir selected ---
    ("message.info('未选择目录')",
     "message.info(S.lib_msg_no_dir_selected)"),

    # --- Pick dir fail ---
    ("message.error(err instanceof Error ? err.message : '打开目录选择器失败')",
     "message.error(err instanceof Error ? err.message : S.lib_msg_pick_dir_fail)"),

    # --- Select files to scan ---
    ("message.info('请先选择要扫描的文件')",
     "message.info(S.lib_msg_select_scan)"),

    # --- Scanned count ---
    ("message.success(`已扫描 ${selected.length} 个文件`)",
     "message.success(S.lib_msg_scanned_count.replace('{n}', String(selected.length)))"),

    # --- Select files to save ---
    ("message.info('请先选择要保存的文件')",
     "message.info(S.lib_msg_select_save)"),

    # --- Processed count ---
    ("message.success(`已处理 ${selected.length} 个文件`)",
     "message.success(S.lib_msg_processed_count.replace('{n}', String(selected.length)))"),

    # --- Scan rename result ---
    ("message.success(`扫描完成：${res.changed}/${res.total_scanned} 需要改名`)",
     "message.success(S.lib_msg_scan_result.replace('{changed}', String(res.changed)).replace('{total}', String(res.total_scanned)))"),

    # --- Scan rename fail ---
    ("message.error(err instanceof Error ? err.message : '扫描改名建议失败')",
     "message.error(err instanceof Error ? err.message : S.lib_msg_scan_rename_fail)"),

    # --- No failed items ---
    ("message.info('暂无失败项')",
     "message.info(S.lib_msg_no_failed_items)"),

    # --- Selected failed count ---
    ("message.info(`已选择 ${failedUploadDrafts.length} 个失败项`)",
     "message.info(S.lib_msg_selected_failed.replace('{n}', String(failedUploadDrafts.length)))"),

    # --- No dup failures ---
    ("message.info('当前没有重复文件失败项')",
     "message.info(S.lib_msg_no_dup_failures)"),

    # --- Switched to dup ---
    ("message.info(`已切换到重复失败项（${duplicateFailedDrafts.length}）`)",
     "message.info(S.lib_msg_switched_dup.replace('{n}', String(duplicateFailedDrafts.length)))"),

    # --- No retryable ---
    ("message.info('没有可重试的失败项')",
     "message.info(S.lib_msg_no_retryable)"),

    # --- Retried count ---
    ("message.success(`已重试 ${failed.length} 个失败项`)",
     "message.success(S.lib_msg_retried_count.replace('{n}', String(failed.length)))"),

    # --- Select rename items ---
    ("message.info('请先选择要改名的条目')",
     "message.info(S.lib_msg_select_rename)"),

    # --- Rename result template ---
    ("message[res.failed > 0 ? 'warning' : 'success'](`改名完成：成功 ${res.renamed}，跳过 ${res.skipped}，失败 ${res.failed}`)",
     "message[res.failed > 0 ? 'warning' : 'success'](S.lib_msg_rename_result.replace('{ok}', String(res.renamed)).replace('{skip}', String(res.skipped)).replace('{fail}', String(res.failed)))"),

    # --- Rename suggest reindex ---
    ("if (res.needs_reindex) message.info('改名后建议更新知识库')",
     "if (res.needs_reindex) message.info(S.lib_msg_rename_suggest_reindex)"),

    # --- Apply rename fail ---
    ("message.error(err instanceof Error ? err.message : '应用改名失败')",
     "message.error(err instanceof Error ? err.message : S.lib_msg_apply_rename_fail)"),

    # --- No convertible ---
    ("message[res.enqueued > 0 ? 'success' : 'info'](\n        res.enqueued > 0\n          ? `已加入队列 ${res.enqueued} 个待转换文件`\n          : '没有可入队的待转换文件',",
     "message[res.enqueued > 0 ? 'success' : 'info'](\n        res.enqueued > 0\n          ? S.lib_msg_enqueued_count.replace('{n}', String(res.enqueued))\n          : S.lib_msg_no_convertible,"),

    # --- Deleted name ---
    ("message.success(`已删除 ${item.name}`)",
     "message.success(S.lib_msg_deleted_name.replace('{name}', item.name))"),

    # --- Delete suggest reindex ---
    ("if (res.needs_reindex) {\n        message.info('删除/改名后建议更新知识库')",
     "if (res.needs_reindex) {\n        message.info(S.lib_msg_delete_suggest_reindex)"),

    # --- Delete warning ---
    ("message.warning(`删除未完全成功${warning}`)",
     "message.warning(S.lib_msg_delete_not_complete.replace('{warning}', warning))"),

    # --- Reindex loading ---
    ("const hide = message.loading('正在更新知识库...', 0)",
     "const hide = message.loading(S.lib_msg_updating_kb, 0)"),

    # --- Exec fail ---
    ("message.error('执行失败')",
     "message.error(S.lib_msg_exec_fail)"),

    # --- Exec done ---
    ("message.success('执行完成')",
     "message.success(S.lib_msg_exec_done)"),

    # --- Refsync started bg ---
    ("message.info('已在后台启动引用同步')",
     "message.info(S.lib_msg_refsync_started_bg)"),

    # --- Refsync fail detail ---
    ("message.warning(`引用同步启动失败：${res.refsync_error}`)",
     "message.warning(S.lib_msg_refsync_fail_detail.replace('{error}', String(res.refsync_error)))"),

    # --- Starting refsync ---
    ("const hide = message.loading('正在启动引用同步...', 0)",
     "const hide = message.loading(S.lib_msg_starting_refsync, 0)"),

    # --- Refsync started ---
    ("message.success('引用同步已启动')",
     "message.success(S.lib_msg_refsync_started)"),

    # --- Refsync already running ---
    ("message.info('引用同步已在运行')",
     "message.info(S.lib_msg_refsync_already_running)"),

    # --- Refsync not started ---
    ("message.warning('引用同步未启动')",
     "message.warning(S.lib_msg_refsync_not_started)"),

    # --- Start refsync fail ---
    ("message.error(err instanceof Error ? err.message : '启动引用同步失败')",
     "message.error(err instanceof Error ? err.message : S.lib_msg_start_refsync_fail)"),

    # --- Guide not converted ---
    ("message.info('该文献尚未完成入库转换，请先转换后再进入阅读指导。')",
     "message.info(S.lib_msg_guide_not_converted)"),

    # --- Creating guide ---
    ("const hide = message.loading('正在创建阅读指导会话...', 0)",
     "const hide = message.loading(S.lib_msg_creating_guide, 0)"),

    # --- Guide source fallback ---
    ("message.warning('阅读指导源解析失败，已回退到当前文献源。建议重启后端后再试。')",
     "message.warning(S.lib_msg_guide_source_fallback)"),

    # --- Guide entered ---
    ("message.success('已进入阅读指导会话')",
     "message.success(S.lib_msg_guide_entered)"),

    # --- Guide create fail ---
    ("message.error(err instanceof Error ? err.message : '创建阅读指导会话失败')",
     "message.error(err instanceof Error ? err.message : S.lib_msg_guide_create_fail)"),

    # --- Meta saved ---
    ("message.success('文献元数据已保存')",
     "message.success(S.lib_msg_meta_saved)"),

    # --- Meta save fail ---
    ("message.error(err instanceof Error ? err.message : '保存文献元数据失败')",
     "message.error(err instanceof Error ? err.message : S.lib_msg_meta_save_fail)"),

    # --- No suggestion candidates ---
    ("message.info('当前筛选结果里没有可生成建议的文献')",
     "message.info(S.lib_msg_no_suggestion_candidates)"),

    # --- Refresh suggestion fail ---
    ("message.error(err instanceof Error ? err.message : '刷新建议失败')",
     "message.error(err instanceof Error ? err.message : S.lib_msg_refresh_suggestion_fail)"),

    # --- Update suggestion fail ---
    ("message.error(err instanceof Error ? err.message : '更新建议失败')",
     "message.error(err instanceof Error ? err.message : S.lib_msg_update_suggestion_fail)"),

    # --- Suggestion refreshed ---
    ("message.success('文献建议已刷新')",
     "message.success(S.lib_msg_suggestion_refreshed)"),

    # --- No selectable ---
    ("message.info('当前列表没有可选文献')",
     "message.info(S.lib_msg_no_selectable)"),

    # --- Select batch edit ---
    ("message.info('请先选择要批量编辑的文献')",
     "message.info(S.lib_msg_select_batch_edit)"),

    # --- Set batch content ---
    ("message.info('请先设置至少一项批量修改内容')",
     "message.info(S.lib_msg_set_batch_content)"),

    # --- Batch edit fail ---
    ("message.error(err instanceof Error ? err.message : '批量编辑失败')",
     "message.error(err instanceof Error ? err.message : S.lib_msg_batch_edit_fail)"),

    # --- Save draft duplicate note ---
    ("if (res.duplicate) return { ...x, status: 'error', note: `重复：${String(res.existing || '')}` }",
     "if (res.duplicate) return { ...x, status: 'error', note: `${S.lib_upload_dup_prefix}${String(res.existing || '')}` }"),

    # --- Save draft saved note ---
    ("note: enqueued ? `已保存并加入转换队列：${savedName}` : `已保存：${savedName}`",
     "note: enqueued ? S.lib_msg_saved_enqueued.replace('{name}', savedName) : S.lib_msg_saved_only.replace('{name}', savedName)"),

    # --- RefSync sticky: waiting message ---
    ("(store.refSync.message || '等待同步任务')",
     "(store.refSync.message || S.lib_refsync_waiting)"),

    # --- RefSync card title ---
    ('<Card size="small" className="kb-lib-card" title="引用同步">',
     '<Card size="small" className="kb-lib-card" title={S.lib_card_refsync}>'),

    # --- Convert pending button ---
    ('<Button type="primary" onClick={() => { void handleConvertPending() }}>立即转换待处理</Button>',
     '<Button type="primary" onClick={() => { void handleConvertPending() }}>{S.lib_btn_convert_pending}</Button>'),

    # --- Sticky Tag running ---
    ('<Tag color="processing">运行中</Tag>',
     '<Tag color="processing">{S.lib_refsync_running}</Tag>'),

    # --- Tabs pending ---
    ("{ key: 'pending', label: `待转换 (${visiblePending.length})`, children: renderFiles(visiblePending, '暂无待转换文件') }",
     "{ key: 'pending', label: S.lib_tab_pending.replace('{n}', String(visiblePending.length)), children: renderFiles(visiblePending, S.lib_empty_pending) }"),

    # --- Tabs converted ---
    ("{ key: 'converted', label: `已转换 (${visibleConverted.length})`, children: renderFiles(visibleConverted, '暂无已转换文件') }",
     "{ key: 'converted', label: S.lib_tab_converted.replace('{n}', String(visibleConverted.length)), children: renderFiles(visibleConverted, S.lib_empty_converted) }"),

    # --- Tabs all ---
    ("{ key: 'all', label: `全部 (${visibleAll.length})`, children: renderFiles(visibleAll, '暂无文件') }",
     "{ key: 'all', label: S.lib_tab_all.replace('{n}', String(visibleAll.length)), children: renderFiles(visibleAll, S.lib_empty_all) }"),

    # --- Meta drawer title ---
    ("title={metaItem ? `文献元数据 · ${metaItem.name}` : '文献元数据'}",
     "title={metaItem ? S.lib_meta_title.replace('{name}', metaItem.name) : S.lib_meta_title_fallback}"),

]

count = 0
for old, new in repl:
    if old in content:
        content = content.replace(old, new)
        count += 1
        print(f'  OK ({count})')
    else:
        print(f'  MISS: {old[:80]}...')

with open(FP, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'\nReplaced {count}/{len(repl)} strings')
