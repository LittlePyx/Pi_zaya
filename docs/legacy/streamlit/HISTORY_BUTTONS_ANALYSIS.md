# 点击「展开更早会话」后两个顶部按钮排布异常 — 分析

## 一、点击后发生的结构性变化

### 1. 渲染内容变化（app.py）

| 状态 | `show_older` | 新增渲染内容 |
|------|--------------|--------------|
| 收起 | `False` | 无 |
| 展开 | `True` | `older_section_slot` → `older_scroll_slot`(height=280) → 更早会话行 |

展开后新增：
- `older_section_slot`（container）
- `older_scroll_slot`（`container(height=280)`，带滚动）
- `.kb-history-older-list` 标记
- 若干 `.kb-history-row` 行（含 dummy 行）

### 2. DOM 结构变化

**收起时**（简化）：
```
history_slot
├── markdown (hr)
├── subheader (对话记录)
├── markdown (CSS)
├── markdown (.kb-history-root)
├── actions_slot
│   ├── markdown (.kb-history-actions)
│   └── columns [新建会话]
└── list_slot
    ├── markdown (.kb-history-list)
    ├── 七天内行...
    ├── markdown (.kb-history-toggle-marker)
    └── button (展开更早会话)
```

**展开时**：
```
history_slot
├── ... (同上)
└── list_slot
    ├── ...
    ├── button (收起更早会话)
    └── older_section_slot  ← 新增
        ├── markdown (.kb-history-older)
        └── older_scroll_slot (height=280, 可滚动)
            ├── markdown (.kb-history-older-list)
            ├── dummy row
            └── 更早会话行...
```

### 3. 侧边栏布局变化

- **滚动条**：`older_scroll_slot` 使用 `container(height=280)`，内部出现垂直滚动条。
- **整体高度**：展开后内容变高，侧边栏若 `overflow-y: auto`，会在右侧出现主滚动条。
- **主滚动条**：主滚动条会占用约 8–17px 宽度，导致侧边栏内容区变窄。

## 二、可能导致按钮排布异常的原因

### 原因 1：侧边栏主滚动条导致宽度变化（最可能）

- 收起：内容少，侧边栏可能无主滚动条。
- 展开：内容多，侧边栏出现主滚动条。
- 结果：内容区宽度减小，`columns([1, 1])` 的 flex 子项宽度随之变化。
- 表现：两个按钮变窄、换行或布局错乱。

### 原因 2：Streamlit 块结构变化

- 展开后 `list_slot` 多了 `older_section_slot`，块树变化。
- Streamlit 可能重新编号或调整块结构。
- 若 `actions_slot` 与 `list_slot` 的兄弟关系或嵌套发生变化，CSS 选择器可能匹配到错误元素。

### 原因 3：JS `bindMarkerToBlock` 匹配错误

- `findNextHorizontalBlock(container)` 通过 `nextElementSibling` 找下一个 `stHorizontalBlock`。
- 若 DOM 顺序或结构在展开后改变，可能匹配到「展开/收起」按钮所在块或第一行块，而不是两个顶部按钮的 columns 块。
- 结果：`kb-history-actions-block` 加错位置，样式应用到错误元素。

### 原因 4：CSS 选择器误匹配

- 使用 `div:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"]` 依赖相邻兄弟关系。
- 若 Streamlit 在展开时插入额外 wrapper 或改变兄弟关系，选择器可能失效或匹配到其他块。

### 原因 5：`all: unset` 与 flex 布局冲突

- 按钮使用 `all: unset`，会清除继承和默认样式。
- 父级 `columns([1, 1])` 为 flex 布局，子项依赖 flex 属性。
- 在宽度变化时，`unset` 后的 flex 行为可能与预期不符，导致排布异常。

## 三、建议的验证步骤

1. **检查主滚动条**：展开后看侧边栏右侧是否出现主滚动条；若有，可尝试 `scrollbar-gutter: stable` 固定预留空间。
2. **检查 JS 绑定**：在控制台执行 `document.querySelectorAll('.kb-history-actions-block')`，确认是否只有一个元素，且为两个按钮的父块。
3. **检查 CSS 匹配**：在 DevTools 中选中两个按钮的父块，查看哪些规则生效，是否有意外覆盖。
4. **对比 DOM**：分别保存收起/展开时的侧边栏 HTML，对比 `actions_slot` 及其兄弟结构是否一致。

## 四、建议的修复方向

1. **固定内容区宽度**：对侧边栏主容器使用 `scrollbar-gutter: stable`，避免主滚动条出现时宽度突变。
2. **强化 actions 选择器**：用更精确的选择器（如 `.kb-history-actions-block`）或给 actions 容器加唯一 class，减少对 DOM 结构的依赖。
3. **限制 `all: unset` 影响**：只重置必要属性，保留 flex 相关属性，或为按钮容器单独设置 flex 布局。
4. **隔离 actions 布局**：为 actions 容器设置固定或最小宽度，避免受下方内容高度和滚动条影响。
