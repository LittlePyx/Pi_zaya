# Project 功能实现指南

类似 ChatGPT 的 Project：用户可新建多个项目，每个项目有独立会话列表；同时支持「全局会话」（不属于任何项目）。细节与交互尽量仿照 ChatGPT 的 Project 体验。

---

## 〇、ChatGPT Project 行为参考（对齐目标）

| 行为 | ChatGPT 做法 | 本实现 |
|------|--------------|--------|
| 侧边栏入口 | 顶部有「New project」，项目列表在下方，点击项目切换 | 项目选择区在「新建会话」下方，可选「全局会话」或某项目；支持「新建项目」 |
| 新建项目 | 点 New project → 输入名称、选图标/颜色 → Create | 名称必填；图标/颜色可 Phase 2 再做 |
| 当前上下文 | 选中某项目后，下方为该项目的会话列表 | 同：`project_id` 决定列表数据源 |
| 新建会话 | 在项目内点「New chat in this project」→ 会话归属当前项目 | 同：新建会话时 `project_id = 当前选中项目`（全局则 NULL） |
| 会话移动 | 会话菜单「Add to project」/「Move to project」选目标项目；「Remove」移出项目 | 会话行菜单：「移动到项目」选目标；「移出项目」→ 变为全局会话 |
| 项目菜单 | 项目名旁三点：重命名、设置、删除 | 项目行右侧菜单：重命名、删除（设置/说明可占位） |
| 删除项目 | Delete Project → 移除项目及其中会话（不可恢复） | 软删除：项目删除后，其下会话变为全局会话（可恢复数据） |

---

## 一、功能目标

| 能力 | 说明 |
|------|------|
| 新建项目 | 用户可创建多个项目，每个项目有名称（后续可加图标/颜色） |
| 会话归属 | 会话可属于某项目，或为「全局会话」（`project_id = NULL`） |
| 侧边栏展示 | 先选「全局会话」或某项目，再展示该上下文下的会话列表 |
| 新建会话 | 新建会话归属当前选中的项目（全局则 NULL） |
| 切换项目 | 切换后会话列表只显示该项目下的会话；当前会话若不属于新项目，可保留选中但列表会切走，或自动切到该项目的第一个会话（见下） |
| 会话移动 | 会话可「移动到项目」或「移出项目」（变为全局） |
| 项目操作 | 项目重命名、删除（删除后其下会话变为全局） |

---

## 二、当前架构简要

- **数据**：`kb/chat_store.py`，SQLite，表 `conversations`、`messages`、`message_refs`
- **会话列表**：`app.py` 约 2674–2875 行，侧边栏 `history_slot` 渲染
- **状态**：`st.session_state["conv_id"]` 当前会话；`chat_store.list_conversations(limit=200)` 拉取列表

---

## 三、实现步骤

### 3.1 数据库层

**新增 `projects` 表：**

```sql
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
```

（可选，后续仿 ChatGPT 图标/颜色：增加 `icon TEXT`、`color TEXT`，默认 NULL。）

**修改 `conversations` 表：**

- 新增列 `project_id TEXT`（可为 NULL，表示全局会话）
- 外键：`FOREIGN KEY(project_id) REFERENCES projects(id)`（SQLite 不强制，可省略，保留语义即可）
- 索引：`CREATE INDEX IF NOT EXISTS idx_conversations_project_id ON conversations(project_id);`

**迁移策略（兼容已有 DB）：**

- 若表 `conversations` 已存在且无 `project_id` 列：  
  `ALTER TABLE conversations ADD COLUMN project_id TEXT;`  
  已有行自动为 NULL，即全部视为全局会话。
- 若表 `projects` 不存在：在 `_init_db()` 里执行上述 `CREATE TABLE IF NOT EXISTS projects`。

**ChatStore 扩展接口：**

| 方法 | 说明 |
|------|------|
| `create_project(name: str) -> str` | 返回新项目 id |
| `list_projects() -> list[dict]` | 按 `updated_at` 或 `created_at` 排序，返回 `id, name, created_at, updated_at` |
| `get_project(project_id: str) -> dict \| None` | 单条查询，重命名/删除前校验用 |
| `rename_project(project_id: str, name: str) -> bool` | 更新名称与 `updated_at` |
| `delete_project(project_id: str) -> None` | 软删除：将该项目下会话的 `project_id` 置为 NULL，再删除项目行 |
| `create_conversation(title: str = "新对话", project_id: str \| None = None) -> str` | 与现有一致，仅新增参数 |
| `list_conversations(project_id: str \| None = None, limit: int = 50) -> list[dict]` | `project_id=None` 查全局；否则查该项目下 |
| `set_conversation_project(conv_id: str, project_id: str \| None) -> bool` | 移动会话到项目或移出为全局 |

---

### 3.2 会话状态扩展

在 `st.session_state` 中增加：

- `project_id`：当前选中的项目 ID，`None` 表示「全局会话」
- 新建会话时：根据当前 `project_id` 决定写入哪个项目

---

### 3.3 侧边栏 UI 改造（仿 ChatGPT）

**目标布局（从上到下）：**

```
[新建会话]           ← 不变，仍为「新对话」主按钮
────────────────────
[项目选择区]
  • 当前：全局会话 ▼  或  当前：<项目名> ▼   （下拉或可点击展开）
  或改为：单选列表
    ○ 全局会话
    ○ 项目A         ⋮  ← 项目右侧菜单（重命名 / 删除）
    ○ 项目B         ⋮
    [+ 新建项目]
────────────────────
[会话列表]           ← 仅当前 project_id 下的会话
  03-01 12:00 | 标题1   🗑
  02-28 09:00 | 标题2   🗑
  [展开更早会话] / [收起更早会话]
```

**项目选择器实现建议：**

- **方案 A（简单）**：`st.selectbox("当前项目", options=["全局会话"] + [p["name"] for p in projects], key="project_selector")`  
  - 用 `index` 或选项值反查：选「全局会话」→ `project_id = None`；选项目名 → 从 `list_projects()` 里取对应 `id` 写入 `st.session_state["project_id"]`。
- **方案 B（更像 ChatGPT）**：项目以列表形式展示，每行「项目名 + 右侧 ⋮ 菜单」；顶部单独一行「+ 新建项目」按钮。当前选中项高亮（如现有会话行的 `kb-history-row-current`）。  
  - 点击项目行 → `st.session_state["project_id"] = pid`；点击「全局会话」行 → `project_id = None`。  
  - 新建项目：`st.text_input` 在 expander 或 `st.dialog` 中，确认后 `create_project(name)` 并设为当前项目。

**切换项目时的当前会话处理：**

- 若当前 `conv_id` 属于新选中的项目（或全局）→ 保持不变。
- 若当前会话不属于新选中的项目（例如从项目 A 切到项目 B）→ 两种策略二选一：  
  - **策略 1**：保持 `conv_id` 不变，主区仍显示该会话，但侧边列表变为项目 B 的会话（当前会话可能不在列表中，可接受）。  
  - **策略 2**：自动切到新项目下的「最近一条会话」或空状态并新建一条。  
- 建议先采用策略 1，实现简单且不丢上下文。

**3.3.1 侧边栏项目列表 UX 细节（文件夹图标、间距、选中样式）**

| 要求 | 实现方式 |
|------|----------|
| 未选中时样式一致 | 仅带 `kb-history-project-row-selected` 的行有高亮（背景 + 左侧条）；其余项目行无区分，按钮背景透明或统一浅色。 |
| 选中 = 展开 | 选中某项目后，下方展示「项目内会话」列表（已有逻辑：`_pid` 时渲染 `_proj_conv_slot`）。 |
| 未展开时行距更紧 | 在 `theme_history_overrides.py` 中为项目行及其后的列块减小 `margin-top` / `margin-bottom`。 |
| 文件夹图标 | 项目名左侧：未选中用闭合文件夹 📁 (U+1F4C1)，选中用打开文件夹 📂 (U+1F4C2)。在 `app.py` 渲染项目按钮时用 `f"{'📂' if _is_selected else '📁'} {_proj_name}"`。 |

**新建会话逻辑：**

- 与现在一致，仅改调用：`chat_store.create_conversation(project_id=st.session_state.get("project_id"))`（全局时传 `None`）。

**会话列表逻辑：**

- `convs = chat_store.list_conversations(project_id=st.session_state.get("project_id"), limit=200)`。
- 7 天内 / 更早 分组、cap 30 条等逻辑保持不变，仅数据源按 `project_id` 过滤。

---

### 3.4 会话移动（仿 ChatGPT「Add to project」/「Remove」）

- **入口**：会话行右侧除删除按钮外，增加「⋮」或「移动到」菜单（可用 `st.popover` 或第二行小按钮）。
- **菜单项**：  
  - 「移动到项目」→ 子列表为 `list_projects()`，选一个则 `set_conversation_project(conv_id, project_id)`。  
  - 「移出项目」→ `set_conversation_project(conv_id, None)`，会话变为全局会话。
- **Store 接口**：`set_conversation_project(conv_id: str, project_id: str | None) -> bool`（更新 `conversations.project_id`，并可选更新 `updated_at`）。

---

### 3.5 项目删除策略

两种方案：

1. **软删除**：删除项目时，将其下会话的 `project_id` 置为 NULL，变为全局会话
2. **级联删除**：删除项目时一并删除其下所有会话（需谨慎，可加二次确认）

推荐方案 1，避免误删会话。

---

## 四、涉及文件清单

| 文件 | 修改内容 |
|------|----------|
| `kb/chat_store.py` | 新增 `projects` 表、`conversations.project_id` 迁移、上述 CRUD 与 `set_conversation_project` |
| `app.py` | 初始化 `project_id` 状态；项目选择区（含新建项目）；会话列表按 `project_id` 过滤；新建会话传 `project_id`；切换项目时 `conv_id` 策略；可选会话菜单「移动到项目/移出项目」、项目菜单「重命名/删除」 |
| `ui/strings.py` | 新增 `S["project_*"]`（见下表） |
| `ui/theme_history_overrides.py` | 若项目列表/当前项需要样式，可加 `.kb-project-*` 等 class |
| `docs/PROJECT_FEATURE_GUIDE.md` | 本文档 |

**建议文案键（`ui/strings.py`）：**

| 键 | 建议中文 |
|----|----------|
| `project_current` | 当前项目 |
| `project_global` | 全局会话 |
| `project_new` | 新建项目 |
| `project_rename` | 重命名项目 |
| `project_delete` | 删除项目 |
| `project_move_here` | 移动到项目 |
| `project_remove` | 移出项目 |
| `project_delete_confirm` | 确定删除该项目？其下会话将变为「全局会话」。 |

---

## 五、实现顺序建议（分步可测）

1. **Phase 1：数据库层**  
   - 在 `_init_db()` 中：创建 `projects` 表；若 `conversations` 无 `project_id` 则 `ALTER TABLE` 增加列；创建索引。  
   - 实现：`create_project`、`list_projects`、`get_project`、`rename_project`、`delete_project`（软删除）；`create_conversation(..., project_id=None)`；`list_conversations(project_id=None, limit=50)`；`set_conversation_project`。  
   - 本地用已有 DB 跑一次，确认旧会话均为 `project_id IS NULL`。

2. **Phase 2：最小 UI（仅列表 + 切换）**  
   - 在 `app.py` 侧边栏「新建会话」下方增加项目选择（如 `st.selectbox`：选项「全局会话」+ `[p["name"] for p in list_projects()]`），用 `st.session_state["project_id"]` 存当前选中（全局为 `None`）。  
   - 新建会话：`create_conversation(project_id=st.session_state.get("project_id"))`。  
   - 会话列表：`list_conversations(project_id=st.session_state.get("project_id"), limit=200)`，其余 7 天内/更早、cap 逻辑不变。  
   - 切换项目时采用「策略 1」：不自动改 `conv_id`，仅列表切换。

3. **Phase 3：新建项目、删除项目**  
   - 「新建项目」：输入名称（`st.text_input` 或 dialog），确认后 `create_project(name)`，并将 `st.session_state["project_id"]` 设为新 id。  
   - 项目菜单（每项目右侧 ⋮）：重命名（弹输入框）、删除（确认后调 `delete_project`）。  
   - 删除后若当前选中被删项目，将 `project_id` 置为 `None`（切回全局）。

4. **Phase 4：会话移动**  
   - 会话行增加「移动到项目」/「移出项目」菜单，调用 `set_conversation_project`；若当前列表因移动导致该会话不在当前项目，可保持主区显示不变或刷新列表。

---

## 六、注意事项

- **兼容性**：已有 DB 需做 `ALTER TABLE` 或提供迁移脚本，`project_id` 默认 NULL
- **会话上限**：当前「30 条硬 cap」逻辑需按 project 分别处理，或全局统一（建议按 project 分别 cap）
- **知识库**：`KB_DB_DIR`、`KB_PDF_DIR` 等为全局配置，项目与知识库无绑定；若未来需要「项目级知识库」，需额外设计

---

## 七、参考：ChatStore 当前与扩展接口

```python
# 现有
create_conversation(title="新对话") -> str
list_conversations(limit=50) -> list[dict]  # 返回 id, title, created_at, updated_at
delete_conversation(conv_id)

# 扩展后
create_project(name: str) -> str
list_projects() -> list[dict]
get_project(project_id: str) -> dict | None
rename_project(project_id: str, name: str) -> bool
delete_project(project_id: str) -> None   # 软删除：会话 project_id 置 NULL
create_conversation(title="新对话", project_id: str | None = None) -> str
list_conversations(project_id: str | None = None, limit: int = 50) -> list[dict]
set_conversation_project(conv_id: str, project_id: str | None) -> bool
```

---

## 八、代码片段参考（便于直接落地）

**1. 迁移：在 `_init_db()` 末尾为已有 DB 增加 `project_id`**

```python
# 在 _init_db() 中，创建 projects 表之后：
conn.execute("CREATE TABLE IF NOT EXISTS projects (...);")  # 见上文

# 为 conversations 增加 project_id（若已有表且无该列）
try:
    conn.execute("ALTER TABLE conversations ADD COLUMN project_id TEXT;")
except sqlite3.OperationalError:
    pass  # 列已存在
conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_project_id ON conversations(project_id);")
```

**2. `list_conversations(project_id=..., limit=...)` 查询**

```python
def list_conversations(self, project_id: str | None = None, limit: int = 50) -> list[dict]:
    with self._connect() as conn:
        if project_id is None:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE project_id IS NULL ORDER BY updated_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE project_id = ? ORDER BY updated_at DESC LIMIT ?",
                (project_id, int(limit)),
            ).fetchall()
    return [dict(r) for r in rows]
```

**3. `create_conversation` 增加 `project_id` 写入**

```python
def create_conversation(self, title: str = "新对话", project_id: str | None = None) -> str:
    conv_id = uuid.uuid4().hex
    now = time.time()
    with self._connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, project_id) VALUES (?, ?, ?, ?, ?)",
            (conv_id, title.strip() or "新对话", now, now, project_id),
        )
    return conv_id
```

**4. `app.py` 中初始化与项目选择区占位**

```python
# 与 conv_id 一起初始化
if "project_id" not in st.session_state:
    st.session_state["project_id"] = None  # 默认全局

# 项目选择（Phase 2 最小版）：选项值为 project_id（None 表示全局）
projects = chat_store.list_projects()
option_ids: list[str | None] = [None] + [p["id"] for p in projects]
option_labels = [S["project_global"]] + [p["name"] for p in projects]
cur_pid = st.session_state.get("project_id")
default_idx = option_ids.index(cur_pid) if cur_pid in option_ids else 0
selected_id = st.selectbox(
    S["project_current"],
    option_ids,
    index=default_idx,
    format_func=lambda x: S["project_global"] if x is None else next((p["name"] for p in projects if p["id"] == x), str(x)),
    key="project_selector",
)
st.session_state["project_id"] = selected_id
```
