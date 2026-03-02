# Project 功能实现指南

类似 GPT 的 Project：用户可新建多个项目，每个项目有独立会话列表；同时支持「全局会话」（不属于任何项目），可直接对话。

---

## 一、功能目标

| 能力 | 说明 |
|------|------|
| 新建项目 | 用户可创建多个项目，每个项目有名称 |
| 会话归属 | 会话可属于某项目，或为「全局会话」（`project_id = NULL`） |
| 侧边栏展示 | 按项目分组显示会话；全局会话单独入口 |
| 新建会话 | 新建时可选：放入当前项目 / 作为全局会话 |
| 切换项目 | 切换当前项目后，会话列表只显示该项目下的会话 |

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

**修改 `conversations` 表：**

- 新增列 `project_id TEXT`（可为 NULL，表示全局会话）
- 外键：`FOREIGN KEY(project_id) REFERENCES projects(id)`
- 索引：`CREATE INDEX idx_conversations_project_id ON conversations(project_id);`

**迁移策略：**

- 已有会话：`project_id = NULL`，视为全局会话
- 可选：新建默认项目「默认项目」，把旧会话迁入（按需）

**ChatStore 扩展：**

- `create_project(name: str) -> str`
- `list_projects() -> list[dict]`
- `rename_project(project_id: str, name: str) -> bool`
- `delete_project(project_id: str) -> None`（会话的 `project_id` 置为 NULL 或一并删除，需约定）
- `create_conversation(title, project_id=None) -> str`
- `list_conversations(project_id=None, limit=50) -> list[dict]`  
  - `project_id=None`：返回全局会话  
  - `project_id="xxx"`：返回该项目下会话  
  - 可选：`list_all_conversations(limit=200)` 用于「展开更早」等场景

---

### 3.2 会话状态扩展

在 `st.session_state` 中增加：

- `project_id`：当前选中的项目 ID，`None` 表示「全局会话」
- 新建会话时：根据当前 `project_id` 决定写入哪个项目

---

### 3.3 侧边栏 UI 改造

**布局建议：**

```
[新建会话]
---
[项目选择器]  ← 下拉/单选：全局 | 项目A | 项目B | + 新建项目
---
[会话列表]    ← 仅显示当前 project_id 下的会话
  - 7天内
  - 展开更早
```

**项目选择器实现：**

- `st.selectbox` 或 `st.radio`：选项为 `["全局会话", "项目A", "项目B", ...]`
- 选中「全局会话」时 `project_id = None`
- 选中某项目时 `project_id = "xxx"`
- 增加「新建项目」入口：弹窗或 `st.text_input` + 确认按钮

**新建会话逻辑：**

- 新建时：`chat_store.create_conversation(project_id=st.session_state.get("project_id"))`
- 当前在「全局会话」则 `project_id=None`；在项目内则 `project_id=当前项目ID`

**会话列表逻辑：**

- 用 `chat_store.list_conversations(project_id=st.session_state.get("project_id"), limit=200)` 拉取
- 7 天内 / 更早 的分组逻辑保持不变，只改数据源

---

### 3.4 会话移动（可选）

- 在会话行增加「移动到…」菜单
- 目标：当前项目 / 其他项目 / 全局会话
- 实现：`update_conversation_project(conv_id, project_id)`，`project_id=None` 表示全局

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
| `kb/chat_store.py` | 新增 `projects` 表、`project_id` 列、迁移逻辑；扩展 CRUD |
| `app.py` | 项目选择器、会话列表按 project 过滤、新建会话传 project_id |
| `ui/strings.py` | 新增 `S["project_*"]` 等文案 |
| `ui/theme_history_overrides.py` | 若项目选择器需要新 class，可补充样式 |
| `docs/PROJECT_FEATURE_GUIDE.md` | 本文档 |

---

## 五、实现顺序建议

1. **Phase 1**：数据库层  
   - 在 `chat_store.py` 中加 `projects` 表、`conversations.project_id`  
   - 写迁移脚本（兼容已有 DB）  
   - 实现 `create_project`、`list_projects`、`create_conversation(project_id=...)`、`list_conversations(project_id=...)`

2. **Phase 2**：最小 UI  
   - 在侧边栏加项目选择器（先只显示「全局会话」+ 一个默认项目）  
   - 新建会话时传入 `project_id`  
   - 会话列表按 `project_id` 过滤

3. **Phase 3**：新建项目、删除项目  
   - 新建项目 UI  
   - 删除项目（软删除：会话置为全局）

4. **Phase 4**：可选增强  
   - 会话移动到其他项目  
   - 项目重命名  
   - 项目排序（`order` 列或 `updated_at`）

---

## 六、注意事项

- **兼容性**：已有 DB 需做 `ALTER TABLE` 或提供迁移脚本，`project_id` 默认 NULL
- **会话上限**：当前「30 条硬 cap」逻辑需按 project 分别处理，或全局统一（建议按 project 分别 cap）
- **知识库**：`KB_DB_DIR`、`KB_PDF_DIR` 等为全局配置，项目与知识库无绑定；若未来需要「项目级知识库」，需额外设计

---

## 七、参考：ChatStore 当前接口

```python
# 现有
create_conversation(title="新对话") -> str
list_conversations(limit=50) -> list[dict]  # 返回 id, title, created_at, updated_at
delete_conversation(conv_id)

# 扩展后
create_project(name: str) -> str
list_projects() -> list[dict]
create_conversation(title="新对话", project_id: str | None = None) -> str
list_conversations(project_id: str | None = None, limit: int = 50) -> list[dict]
# project_id=None 时：返回 project_id IS NULL 的会话
```
