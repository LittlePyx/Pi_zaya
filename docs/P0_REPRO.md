# P0 Repro Manual

更新时间：2026-03-06

## 1. 目的

统一复现文献篮丢失、状态降级、页面切换卡顿、引用错配等问题，保证每次修复都能回归。

## 2. 前置准备

1. 执行环境体检：`.\tools\stability\doctor.ps1 -Strict`
2. 建议清理运行态噪音：`.\tools\stability\reset_state.ps1`
3. 启动开发环境：`.\run_new.ps1 -StopExisting`

## 3. 复现场景

### Case-01：切会话后文献篮丢失

步骤：

1. 在会话 A 中添加 2 条文献到文献篮。
2. 切换到会话 B，再切回会话 A。
3. 打开文献篮观察条目。

期望：

1. 会话 A 的文献篮条目完整保留。
2. 不出现条目数量变化或内容置空。

---

### Case-02：切 Chat/Library 往返后文献篮丢失

步骤：

1. 在会话 A 文献篮中保留 2 条条目。
2. 切到 Library 页面，再回 Chat。
3. 重复往返 5 次。

期望：

1. 文献篮条目不丢失。
2. 不出现“先正常后清空”的现象。

---

### Case-03：刷新后文献篮“信息降级”

步骤：

1. 在文献篮中加入带结构化字段的条目（作者、venue、year、IF 等）。
2. 刷新页面。
3. 再次打开文献篮并查看同条目字段。

期望：

1. 结构化字段保留，不退化成纯题录文本。
2. 标题/DOI 显示格式一致。

---

### Case-04：连续切换导致卡顿/假死

步骤：

1. 在两个会话间快速切换 30-50 次。
2. 期间穿插切到 Library 再回 Chat。
3. 观察界面响应与点击可用性。

期望：

1. 无明显卡死、无长时间白屏/无响应。
2. 会话内容加载正确，不串会话。

---

### Case-05：引用定位图/公式错配

步骤：

1. 提问触发目标论文的图/公式定位（含已知样本 NatPhoton Fig.3）。
2. 对照文献内容核验返回的资源与编号是否一致。
3. 如需定位 remap 决策细节，可开启：

```powershell
$env:KB_PDF_DEBUG_FIG_REMAP="1"
```

期望：

1. 命中编号与资源一致。
2. 不出现“文字解释正确但图像资源错误”。

---

### Case-06：跨平台路径场景（Windows 样式 source_path）

步骤：

1. 构造/使用 `db\xxx\xxx.en.md` 风格 `source_path`。
2. 在 Linux CI 或本地测试中执行引用元数据与展示逻辑。

期望：

1. venue/year/title 解析正确。
2. 不出现 `db\ICIP` 这类错误截断。

---

### Case-07：文献篮存储 revision 单调递增（防并发覆盖）

步骤：

1. 在会话 A 中添加/删除文献篮条目各 1 次。
2. 打开浏览器控制台，执行：

```javascript
Object.entries(localStorage)
  .filter(([k]) => k.startsWith('kb_cite_shelf:'))
  .map(([k, v]) => {
    try {
      const p = JSON.parse(v)
      return { key: k, revision: Number(p?.revision || 0), updatedAt: Number(p?.updatedAt || 0), count: Array.isArray(p?.items) ? p.items.length : 0 }
    } catch {
      return { key: k, revision: -1, updatedAt: 0, count: -1 }
    }
  })
```

3. 快速切换会话 20 次，并在 Chat/Library 间往返 10 次后重复执行上面脚本。

期望：

1. 当前会话对应的 `kb_cite_shelf:*` 记录 `revision` 只增不减。
2. `count` 与界面文献篮条目数一致。
3. 不出现解析失败（`revision=-1`）。

---

### Case-08：会话/页面连续切换压测（前端 debug 入口）

步骤：

1. 进入 Chat 页面，确保至少有 2 个会话。
2. 在浏览器控制台执行：

```javascript
await window.__kbDebug.runSwitchStress({ rounds: 60, delayMs: 25, includeLibrary: true, awaitSelect: true })
```

3. 查看切换性能日志：

```javascript
window.__kbDebug.getSwitchPerf()
window.__kbSwitchPerf?.summary()
```

期望：

1. 压测过程中界面无明显卡死、无长时间无响应。
2. `summary.success > 0` 且 `summary.error === 0`。
3. `summary.stale` 可存在（代表并发切换被正确丢弃），但不应导致会话串写或文献篮清空。

## 4. 结果记录模板

每个 case 记录：

1. 执行日期
2. 分支/commit
3. 是否通过（Pass/Fail）
4. 失败截图/日志
5. 备注（复现概率、触发条件）
