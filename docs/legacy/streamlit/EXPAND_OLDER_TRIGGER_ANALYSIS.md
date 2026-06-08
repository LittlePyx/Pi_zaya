# 点击「展开更早会话」后触发的完整链路分析

## 一、用户操作 → Python 执行

1. **用户点击**「展开更早会话」按钮
2. **Streamlit** 调用 `on_click=_toggle_history_older_click`
3. **回调执行**：`st.session_state["history_show_older_convs"] = True`
4. **Streamlit 触发 rerun**：整个 `app.py` 重新执行
5. **条件分支**：`if show_older:` 为 True，渲染 `older_section_slot`、`older_scroll_slot`、更早会话行

## 二、Streamlit 前端 Delta 更新

1. **服务端** 返回新的 widget tree（含新增的 older_section）
2. **前端** 收到 delta，与当前 DOM 做 diff
3. **DOM 补丁**：可能的行为包括：
   - **A**：只往 `list_slot` 下追加新块（older_section）
   - **B**：替换整个 `list_slot`（含子块）
   - **C**：替换整个 `history_slot`（含 actions、list、CSS）
   - **D**：替换整个侧边栏

## 三、关键发现

### 1. CSS 注入位置

- **theme_history_overrides**：通过 `history_slot.markdown(_history_sidebar_compact_css())` 注入
- 该 markdown 是 **history_slot 的子块**
- 若 Streamlit 采用 **C 或 D**，整个 history 区域被替换，**旧 CSS 会随旧 DOM 一起被移除**
- 新 DOM 会重新渲染该 markdown，理论上会带上新的 `<style>` 标签

### 2. components.html 与脚本执行时机

- `_inject_runtime_ui_fixes` 使用 `components.html()` 注入脚本
- 脚本在 **iframe 加载时** 执行
- 脚本通过 `host.document` 操作 **父文档**（Streamlit 主文档）
- **MutationObserver** 监听 `doc.body`，DOM 变化时调用 `schedule()` → `applyNow()` → `decorateConversationHistoryButtons()`

### 3. 可能的竞态

| 场景 | 描述 |
|------|------|
| **Delta 应用顺序** | components.html 的 delta 若在 sidebar delta **之后** 应用，脚本会在新 DOM 上执行；若在 **之前**，脚本可能仍作用在旧 DOM |
| **MutationObserver 时机** | 在 DOM 变更的同一帧或下一帧触发，此时布局可能尚未完成 |
| **injectActionButtonsCSS 的 early return** | `if (targetDoc.getElementById(id)) return` 导致只注入一次；若主文档被整体替换，旧 style 消失，会再次注入；若只是局部 patch，旧 style 仍在，不会更新 |

### 4. 新建会话按钮样式失效的可能原因

1. **DOM 结构变化**：展开后 Streamlit 可能改变块结构，导致 `div:has(.kb-history-actions) + div` 等选择器不再匹配
2. **CSS 被替换**：若 history_slot 被整体替换，旧 `<style>` 被移除，新 DOM 中的 `<style>` 可能尚未插入或尚未生效
3. **JS 执行时机**：`decorateConversationHistoryButtons` 在 DOM 未完全稳定时执行，`findNextHorizontalBlock` 可能匹配到错误块
4. **样式被覆盖**：theme_legacy 的通用按钮样式可能在后续加载，覆盖我们的扁平样式

## 四、建议的修复方向

1. **强制 re-inject**：去掉 `injectActionButtonsCSS` 的 early return，每次执行时先移除旧 style 再注入新的，保证样式始终存在
2. **延长重试**：在 80ms、250ms 之外，增加 500ms 的 `schedule()` 调用，应对较晚完成的 DOM 更新
3. **直接内联样式**：在 `bindMarkerToBlock` 中，对新建会话按钮的父块直接设置 `style` 属性，减少对 CSS 选择器的依赖
4. **用 data 属性定位**：在 app.py 中给新建会话按钮的容器加 `data-kb-new-chat`，用该属性做选择器，提高稳定性
