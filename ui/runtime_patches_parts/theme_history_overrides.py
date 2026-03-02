from __future__ import annotations

# Selector prefixes for sidebar (Streamlit uses data-testid and/or .stSidebar).
_SIDEBAR = ('section[data-testid="stSidebar"]', "section.stSidebar")


def _sidebar_sel(suffix: str) -> tuple[str, ...]:
    """Return (sidebar1 + suffix, sidebar2 + suffix) for use in generated rules."""
    return tuple(s + suffix for s in _SIDEBAR)


def _css_rule(selectors: tuple[str, ...], declarations: str) -> str:
    """Build one CSS rule: selectors { declarations }."""
    return ",\n".join(selectors) + " {\n" + declarations + "\n}"


def _history_sidebar_compact_css() -> str:
    return (
        """
<style>
/* Final compact override for conversation history.
   This block is injected in the sidebar so it applies to the same document.
   Fallback: section.stSidebar (Streamlit class) if data-testid differs. */

/* 侧边栏主滚动区：预留滚动条、适度左侧留白 */
section[data-testid="stSidebar"] > div,
section.stSidebar > div{
  scrollbar-gutter: stable !important;
  padding-left: 0.5rem !important;
  margin-left: 0 !important;
}
section[data-testid="stSidebar"] > div > div,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
section.stSidebar > div > div,
section.stSidebar [data-testid="stSidebarUserContent"]{
  padding-left: 0.5rem !important;
  margin-left: 0 !important;
}

/* 当前项目区：列表化、紧凑，无突兀外框 */
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap),
section.stSidebar div:has(.kb-history-project-wrap){
  margin-top: 0.02rem !important;
  margin-bottom: 0.05rem !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
/* 当前项目 / Your chats 等区块标题：加透明度，与正文区分（用 data-testid 覆盖 theme_legacy） */
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) [data-testid="stCaptionContainer"] *,
section.stSidebar div:has(.kb-history-project-wrap) [data-testid="stCaptionContainer"],
section.stSidebar div:has(.kb-history-project-wrap) [data-testid="stCaptionContainer"] *{
  font-size: 0.7rem !important;
  color: var(--sidebar-soft-text, #888) !important;
  opacity: 0.58 !important;
  margin-top: 0 !important;
  margin-bottom: 0.08rem !important;
  padding: 0 !important;
}
/* 项目区内部：取消块间 gap，压紧 project 之间间距 */
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) div[data-testid="stVerticalBlock"],
section.stSidebar div:has(.kb-history-project-wrap) div[data-testid="stVerticalBlock"]{
  gap: 0 !important;
  row-gap: 0 !important;
}
/* 项目行容器：无边框、无多余留白 */
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) div[data-testid="element-container"],
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) div[data-testid="stHorizontalBlock"],
section.stSidebar div:has(.kb-history-project-wrap) div[data-testid="element-container"],
section.stSidebar div:has(.kb-history-project-wrap) div[data-testid="stHorizontalBlock"]{
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  margin-top: 0 !important;
  margin-bottom: 0 !important;
}
/* 项目行（含行标记 + 列块）：上下边距压到最小 */
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row),
section.stSidebar div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row){
  margin-top: 0 !important;
  margin-bottom: 0 !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row) + div[data-testid="element-container"],
section.stSidebar div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row) + div[data-testid="element-container"]{
  margin-top: 0 !important;
  margin-bottom: 0 !important;
}
/* 项目行：与历史会话行同布局；所有项目行统一渐变+圆角（对行外层容器和内部块都生效） */
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row) + div[data-testid="element-container"],
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row) + div[data-testid="stElementContainer"],
section.stSidebar div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row) + div[data-testid="element-container"],
section.stSidebar div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row) + div[data-testid="stElementContainer"]{
  border-radius: 6px !important;
  padding: 0.06rem 0 !important;
  background: linear-gradient(
    to bottom,
    color-mix(in srgb, var(--sidebar-strong-text, #eaeaea) 10%, transparent),
    color-mix(in srgb, var(--sidebar-strong-text, #eaeaea) 4%, transparent)
  ) !important;
  border: none !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"],
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"],
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"]{
  align-items: center !important;
  border-radius: 6px !important;
  padding: 0.06rem 0 !important;
  background: linear-gradient(
    to bottom,
    color-mix(in srgb, var(--sidebar-strong-text, #eaeaea) 10%, transparent),
    color-mix(in srgb, var(--sidebar-strong-text, #eaeaea) 4%, transparent)
  ) !important;
  border: none !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div,
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div,
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div{
  display: flex !important;
  align-items: center !important;
  justify-content: flex-start !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div:nth-child(2),
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div:nth-child(3),
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div:nth-child(2),
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div:nth-child(3),
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div:nth-child(2),
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div:nth-child(3),
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div:nth-child(2),
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div:nth-child(3){
  flex: 0 0 1.5rem !important;
  flex-shrink: 0 !important;
  min-width: 1.5rem !important;
  max-width: 1.5rem !important;
  justify-content: center !important;
  padding: 0 !important;
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}
/* 选中项目行：渐变蓝底 + 左侧细条（行外层容器与内部块一致） */
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row-selected) + div[data-testid="element-container"],
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row-selected) + div[data-testid="stElementContainer"],
section.stSidebar div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row-selected) + div[data-testid="element-container"],
section.stSidebar div:has(.kb-history-project-wrap) div[data-testid="element-container"]:has(.kb-history-project-row-selected) + div[data-testid="stElementContainer"]{
  background: linear-gradient(
    to bottom,
    color-mix(in srgb, var(--blue-line, #4a9eff) 22%, transparent),
    color-mix(in srgb, var(--blue-line, #4a9eff) 12%, transparent)
  ) !important;
  box-shadow: inset 3px 0 0 var(--blue-line, #4a9eff) !important;
  border-radius: 6px !important;
  padding: 0.08rem 0 !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-row-selected) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div:has(.kb-history-project-row-selected) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"],
section.stSidebar div:has(.kb-history-project-row-selected) + div[data-testid="stHorizontalBlock"],
section.stSidebar div:has(.kb-history-project-row-selected) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"]{
  background: linear-gradient(
    to bottom,
    color-mix(in srgb, var(--blue-line, #4a9eff) 22%, transparent),
    color-mix(in srgb, var(--blue-line, #4a9eff) 12%, transparent)
  ) !important;
  box-shadow: inset 3px 0 0 var(--blue-line, #4a9eff) !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 0.08rem 0 !important;
}
/* 项目行项目名按钮：未选中时透明，样式一致；选中行由整行背景强调 */
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) div[data-testid="stVerticalBlock"] > div button,
section.stSidebar div:has(.kb-history-project-wrap) div[data-testid="stVerticalBlock"] > div button{
  min-height: 28px !important;
  height: auto !important;
  padding: 0.25rem 0.5rem !important;
  border-radius: 6px !important;
  font-size: 0.75rem !important;
  text-align: left !important;
  justify-content: flex-start !important;
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--sidebar-strong-text, #eaeaea) !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-wrap) div[data-testid="stVerticalBlock"] > div button:hover,
section.stSidebar div:has(.kb-history-project-wrap) div[data-testid="stVerticalBlock"] > div button:hover{
  background: color-mix(in srgb, var(--sidebar-strong-text, #eaeaea) 14%, transparent) !important;
}
/* 当前项目区：三点 ⋮ expander 无边框、无阴影（多种选择器兜底：marker 相邻 + 项目行第 2 列） */
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div details[data-testid="stExpander"],
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div [data-testid="stExpander"],
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"],
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"],
section.stSidebar div:has(.kb-history-project-expander) + div details[data-testid="stExpander"],
section.stSidebar div:has(.kb-history-project-expander) + div [data-testid="stExpander"],
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"],
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"]{
  border: none !important;
  border-width: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
  background-color: transparent !important;
  position: relative !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div [data-testid="stExpander"] *,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] *,
section.stSidebar div:has(.kb-history-project-expander) + div [data-testid="stExpander"] *,
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] *{
  border: none !important;
  border-width: 0 !important;
  box-shadow: none !important;
  outline: none !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] summary,
section.stSidebar div:has(.kb-history-project-expander) + div [data-testid="stExpander"] summary,
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] summary{
  min-width: 1.2rem !important;
  width: 1.2rem !important;
  min-height: 1.2rem !important;
  height: 1.2rem !important;
  padding: 0 !important;
  margin: 0 !important;
  border-radius: 0 !important;
  font-size: 0.7rem !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 50%, var(--sidebar-soft-text)) !important;
  opacity: 0.65 !important;
  flex-shrink: 0 !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div [data-testid="stExpander"] summary::-webkit-details-marker,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] summary::-webkit-details-marker,
section.stSidebar div:has(.kb-history-project-expander) + div [data-testid="stExpander"] summary::-webkit-details-marker,
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] summary::-webkit-details-marker{
  display: none !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div [data-testid="stExpander"] summary::before,
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div [data-testid="stExpander"] summary::after,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] summary::before,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] summary::after,
section.stSidebar div:has(.kb-history-project-expander) + div [data-testid="stExpander"] summary::before,
section.stSidebar div:has(.kb-history-project-expander) + div [data-testid="stExpander"] summary::after,
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] summary::before,
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] summary::after{
  display: none !important;
  content: none !important;
}
/* 项目行垃圾桶：与 ⋮ 对齐，微调上移、无边框 */
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div:nth-child(3),
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div:nth-child(3),
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div:nth-child(3),
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div:nth-child(3){
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div:nth-child(3) button,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div:nth-child(3) button,
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div:nth-child(3) button,
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div:nth-child(3) button{
  min-width: 1.2rem !important;
  width: 1.2rem !important;
  min-height: 1.2rem !important;
  height: 1.2rem !important;
  padding: 0 !important;
  margin: 0 !important;
  margin-top: -3px !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 50%, var(--sidebar-soft-text)) !important;
  font-size: 0.7rem !important;
  opacity: 0.65 !important;
  line-height: 1 !important;
  vertical-align: middle !important;
}
/* 项目行第 3 列 stButton 容器：与 ⋮ 同高、内容居中，避免垃圾桶被撑偏 */
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stButton"],
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stButton"],
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stButton"],
section.stSidebar div:has(.kb-history-project-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stButton"]{
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  min-height: 0 !important;
  height: auto !important;
}

/* 项目内会话：紧挨在选中项目行下方、紧凑；仅用于 CSS 定位的 marker 整行隐藏 */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-project-convs),
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-project-convs),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-project-convs),
section.stSidebar div[data-testid="stElementContainer"]:has(.kb-history-project-convs){
  display: none !important;
}
section[data-testid="stSidebar"] .kb-history-project-convs,
section.stSidebar .kb-history-project-convs{
  display: none !important;
}
/* 项目内会话区（含 caption/会话行）紧挨上方项目行，无多余留白 */
section[data-testid="stSidebar"] div:has(.kb-history-project-convs),
section.stSidebar div:has(.kb-history-project-convs){
  margin-top: 0.04rem !important;
  margin-bottom: 0.04rem !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-convs) .stCaptionContainer,
section.stSidebar div:has(.kb-history-project-convs) .stCaptionContainer{
  font-size: 0.68rem !important;
  color: var(--sidebar-soft-text, #888) !important;
  margin-top: 0 !important;
  margin-bottom: 0.12rem !important;
  padding: 0 !important;
  line-height: 1.2 !important;
}

/* 零散会话区块（Your chats）：在下方，无多余横线 */
section[data-testid="stSidebar"] div:has(.kb-history-scattered-section),
section.stSidebar div:has(.kb-history-scattered-section){
  margin-top: 0.35rem !important;
  padding-top: 0 !important;
  border: none !important;
  border-top: none !important;
  border-bottom: none !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-scattered-section) [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] div:has(.kb-history-scattered-section) [data-testid="stCaptionContainer"] *,
section.stSidebar div:has(.kb-history-scattered-section) [data-testid="stCaptionContainer"],
section.stSidebar div:has(.kb-history-scattered-section) [data-testid="stCaptionContainer"] *{
  opacity: 0.58 !important;
  color: var(--sidebar-soft-text, #888) !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-scattered-section) > div,
section.stSidebar div:has(.kb-history-scattered-section) > div{
  border: none !important;
  border-top: none !important;
  border-bottom: none !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-scattered-section) + div,
section[data-testid="stSidebar"] div:has(.kb-history-scattered-section) + div .stCaptionContainer,
section[data-testid="stSidebar"] div:has(.kb-history-scattered-section) ~ div .stCaptionContainer:first-of-type,
section.stSidebar div:has(.kb-history-scattered-section) + div,
section.stSidebar div:has(.kb-history-scattered-section) + div .stCaptionContainer,
section.stSidebar div:has(.kb-history-scattered-section) ~ div .stCaptionContainer:first-of-type{
  border: none !important;
  border-top: none !important;
  border-bottom: none !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-scattered-section) + div [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] div:has(.kb-history-scattered-section) + div [data-testid="stCaptionContainer"] *,
section[data-testid="stSidebar"] div:has(.kb-history-scattered-section) ~ div [data-testid="stCaptionContainer"]:first-of-type,
section[data-testid="stSidebar"] div:has(.kb-history-scattered-section) ~ div [data-testid="stCaptionContainer"]:first-of-type *,
section.stSidebar div:has(.kb-history-scattered-section) + div [data-testid="stCaptionContainer"],
section.stSidebar div:has(.kb-history-scattered-section) + div [data-testid="stCaptionContainer"] *,
section.stSidebar div:has(.kb-history-scattered-section) ~ div [data-testid="stCaptionContainer"]:first-of-type,
section.stSidebar div:has(.kb-history-scattered-section) ~ div [data-testid="stCaptionContainer"]:first-of-type *{
  font-size: 0.7rem !important;
  color: var(--sidebar-soft-text, #888) !important;
  opacity: 0.58 !important;
  margin-bottom: 0.25rem !important;
}

/* All sidebar buttons: no line break when narrow, shrink text to fit (ellipsis + responsive font where applied). */
"""
        + _css_rule(
            _sidebar_sel(" button"),
            """  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  min-width: 0 !important;""",
        )
        + "\n"
        + _css_rule(
            _sidebar_sel(" button p") + _sidebar_sel(" button span"),
            """  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;""",
        )
        + """

@keyframes kb-btn-shimmer {{
  0% {{ transform: translateX(-100%); }}
  100% {{ transform: translateX(200%); }}
}}

/* Trash button: JS adds .kb-trash-wrap and .kb-trash-cell - target these for guaranteed no-border. */
section[data-testid="stSidebar"] .kb-trash-cell,
section.stSidebar .kb-trash-cell,
section[data-testid="stSidebar"] .kb-trash-cell *,
section.stSidebar .kb-trash-cell *,
section[data-testid="stSidebar"] .kb-trash-wrap,
section.stSidebar .kb-trash-wrap,
section[data-testid="stSidebar"] .kb-trash-wrap *,
section.stSidebar .kb-trash-wrap *,
section[data-testid="stSidebar"] button.kb-history-trash-btn,
section.stSidebar button.kb-history-trash-btn{
  border: none !important;
  border-width: 0 !important;
  border-style: none !important;
  box-shadow: none !important;
  outline: none !important;
  outline-width: 0 !important;
  background: transparent !important;
  background-color: transparent !important;
}

section[data-testid="stSidebar"] .kb-history-root{
  display: block !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}
section[data-testid="stSidebar"] .kb-history-row-btn-slot,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-btn-slot){
  display: block !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  line-height: 0 !important;
}

/* Collapse marker divs between rows to minimize line spacing. */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row),
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row){
  display: block !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  line-height: 0 !important;
  background: transparent !important;
  background-color: transparent !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) > div,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) > div{
  margin: 0 !important;
  padding: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
  overflow: hidden !important;
  background: transparent !important;
  background-color: transparent !important;
}

/* Tight row spacing: minimal margin between conversation rows. */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"]{
  margin-top: 0 !important;
  margin-bottom: 0 !important;
}

/* 新建会话所在块：占满侧边栏宽度（align-self: stretch），展开/收起更早会话时不收缩 */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(> div[data-testid="element-container"]:has(.kb-history-actions)),
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"]:has(.kb-history-actions)),
section.stSidebar div[data-testid="stVerticalBlock"]:has(> div[data-testid="element-container"]:has(.kb-history-actions)),
section.stSidebar div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"]:has(.kb-history-actions)){
  flex-shrink: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
  align-self: stretch !important;
  box-sizing: border-box !important;
}
/* 新建会话所在块：不参与 flex 收缩，展开/收起更早会话时布局不变 */
section[data-testid="stSidebar"] div.kb-history-actions-block,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"],
section.stSidebar div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"]{
  gap: 0.2rem !important;
  margin: 0 0 0.2rem 0 !important;
  align-items: stretch !important;
  flex-shrink: 0 !important;
  min-height: 0 !important;
  min-width: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}
/* marker 的紧邻兄弟（无 columns 时即按钮所在块）也占满宽 */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-actions) + div,
section.stSidebar div[data-testid="stElementContainer"]:has(.kb-history-actions) + div{
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}
/* 新建会话行内列：占满整行，保证按钮能 100% 宽度 */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section[data-testid="stSidebar"] div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section.stSidebar div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section.stSidebar div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]{
  flex: 1 1 0% !important;
  min-width: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}
/* 新建会话按钮容器：始终占满侧边栏宽度，展开更早会话后也不变窄、文字保持居中 */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"],
section[data-testid="stSidebar"] div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"],
section.stSidebar div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]{
  min-width: 5rem !important;
  width: 100% !important;
  max-width: 100% !important;
  flex: 1 1 auto !important;
  box-sizing: border-box !important;
}
/* 新建会话：仿参考定位里的 open/cite 按钮 — 圆角、渐变、玻璃感 */
section[data-testid="stSidebar"] button.kb-history-action-btn-inline,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button,
section.stSidebar div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button{
  all: unset !important;
  box-sizing: border-box !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  position: relative !important;
  isolation: isolate !important;
  width: 100% !important;
  min-width: 4.6rem !important;
  min-height: 2.02rem !important;
  height: 2.02rem !important;
  padding: 0 0.86rem !important;
  border: none !important;
  border-radius: 6px !important;
  outline: none !important;
  overflow: hidden !important;
  white-space: nowrap !important;
  text-overflow: clip !important;
  color: var(--sidebar-strong-text, var(--text-main, #e9eff8)) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text, var(--text-main, #e9eff8)) !important;
  font-size: clamp(0.5rem, 1.4vw, 0.875rem) !important;
  font-weight: 665 !important;
  letter-spacing: 0.008em !important;
  cursor: pointer !important;
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--sidebar-background-color, var(--sidebar-bg, #181818)) 95%, var(--blue-line) 5%),
      color-mix(in srgb, var(--sidebar-background-color, var(--sidebar-bg, #181818)) 92%, var(--blue-line) 8%)
    ) !important;
  backdrop-filter: saturate(120%) blur(8px) !important;
  -webkit-backdrop-filter: saturate(120%) blur(8px) !important;
  box-shadow: none !important;
  transition: background 150ms ease, color 150ms ease !important;
}
section[data-testid="stSidebar"] button.kb-history-action-btn-inline::before,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button::before,
section.stSidebar div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button::before{
  content: "" !important;
  position: absolute !important;
  inset: 0 !important;
  pointer-events: none !important;
  z-index: 0 !important;
  background:
    linear-gradient(
      180deg,
      rgba(255,255,255,0.24) 0%,
      rgba(255,255,255,0.10) 44%,
      rgba(255,255,255,0.00) 100%
    ) !important;
  opacity: 0.26 !important;
  border-radius: 6px !important;
}
section[data-testid="stSidebar"] button.kb-history-action-btn-inline > *,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button > *,
section.stSidebar div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button > *{
  position: relative !important;
  z-index: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button span,
section.stSidebar div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p,
section.stSidebar div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button span{
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: clip !important;
  font-size: inherit !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-actions) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover,
section.stSidebar div:has(.kb-history-actions) + div div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover{
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--sidebar-background-color, var(--sidebar-bg, #181818)) 92%, var(--blue-line) 8%),
      color-mix(in srgb, var(--sidebar-background-color, var(--sidebar-bg, #181818)) 88%, var(--blue-line) 12%)
    ) !important;
  border: none !important;
  color: var(--blue-line) !important;
  -webkit-text-fill-color: var(--blue-line) !important;
}

/* Conversation rows: no borders, compact gap. 用 .kb-history-list / .kb-history-older-list 统一最近与更早，flex-wrap: nowrap 保证 ⋮/垃圾桶 与标题同一行。 */
section[data-testid="stSidebar"] div.kb-history-row-block,
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"],
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]{
  display: flex !important;
  width: 100% !important;
  gap: 0.15rem !important;
  margin: 0 !important;
  align-items: center !important;
  flex-wrap: nowrap !important;
  overflow: visible !important;
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}
/* 会话行三列布局：第一列弹性，第二三列固定宽度并右对齐 ⋮ 和 🗑；行用 flex-end 使图标列底边与标题文字底边对齐 */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child{
  flex: 1 1 0% !important;
  min-width: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3),
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3),
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3),
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3),
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3),
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3),
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3),
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3),
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3){
  flex: 0 0 1.5rem !important;
  flex-shrink: 0 !important;
  min-width: 1.5rem !important;
  max-width: 1.5rem !important;
  width: 1.5rem !important;
  min-height: 0 !important;
  height: auto !important;
  display: flex !important;
  justify-content: flex-end !important;
  align-items: center !important;
  padding: 0 !important;
  padding-right: 0 !important;
  margin-left: 0.12rem !important;
}

/* 会话行第一列（标题）：min-width:0 允许收缩，避免把垃圾桶挤到下一行 */
section[data-testid="stSidebar"] div.kb-history-row-block > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div.kb-history-row-block > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child{
  min-width: 0 !important;
}
section[data-testid="stSidebar"] div.kb-history-row-block > div[data-testid="column"],
section[data-testid="stSidebar"] div.kb-history-row-block > div[data-testid="stColumn"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]{
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
  display: flex !important;
  align-items: center !important;
}
/* 会话行/当前行 marker 容器：无边框、描边、阴影 */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current),
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row),
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current),
section.stSidebar div[data-testid="stElementContainer"]:has(.kb-history-row),
section.stSidebar div[data-testid="stElementContainer"]:has(.kb-history-row-current){
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}
/* 当前行：去掉外层强调，仅内层窄条有背景；其直接父级也强制无背景 */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current),
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current),
section.stSidebar div[data-testid="stElementContainer"]:has(.kb-history-row-current),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div,
section.stSidebar div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div{
  background: transparent !important;
  background-color: transparent !important;
}
/* List row buttons: flat, left-aligned, no card/glass — must not look like 新建会话.
   Override theme_legacy's generic sidebar button style (min-height 42px, border-radius 14px, gradient).
   .kb-history-row-btn-slot: hidden marker in app before each row's pick button for reliable targeting. */
section[data-testid="stSidebar"] button.kb-history-row-btn,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button{
  all: unset !important;
  box-sizing: border-box !important;
  display: flex !important;
  align-items: flex-start !important;
  width: 100% !important;
  min-height: 2.5rem !important;
  max-height: none !important;
  padding: 0.25rem 0.1rem 0.2rem !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  transform: none !important;
  overflow: visible !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 88%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 88%, var(--sidebar-soft-text)) !important;
  font-size: 0.74rem !important;
  font-weight: 510 !important;
  line-height: 1.05 !important;
  letter-spacing: 0 !important;
  text-align: left !important;
  cursor: pointer !important;
}
/* 会话行第一列：min-width:0 让列宽由比例决定，文字在「当前宽度」下换行再由 line-clamp 压成两行 */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:first-child{
  min-width: 0 !important;
}
/* 会话行按钮内文本块：占满按钮宽度并收缩，用 line-clamp 把内容压成两行+省略号，不靠按钮高度裁切 */
section[data-testid="stSidebar"] button.kb-history-row-btn > *,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button > *{
  margin: 0 !important;
  flex: 1 1 0 !important;
  min-width: 0 !important;
  max-width: 100% !important;
  display: -webkit-box !important;
  -webkit-box-orient: vertical !important;
  -webkit-line-clamp: 2 !important;
  font-size: inherit !important;
  line-height: 1.05 !important;
  white-space: normal !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  text-align: left !important;
}
section[data-testid="stSidebar"] button.kb-history-row-btn p,
section[data-testid="stSidebar"] button.kb-history-row-btn span,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button span{
  margin: 0 !important;
  display: -webkit-box !important;
  -webkit-box-orient: vertical !important;
  -webkit-line-clamp: 2 !important;
  min-width: 0 !important;
  width: 100% !important;
  font-size: 0.74rem !important;
  line-height: 1.05 !important;
  white-space: normal !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  text-align: left !important;
}

/* 会话行：按钮容器和第一列不裁切，让两行完整显示 */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:first-child{
  overflow: visible !important;
  min-height: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) div[data-testid="stButton"]{
  overflow: visible !important;
  min-height: 0 !important;
}

/* Current row: 只留内层窄条 = 整行一块深色底 + 左侧强调条；无外层强调 */
section[data-testid="stSidebar"] div.kb-history-row-block.kb-history-row-block-current,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div div[data-testid="stHorizontalBlock"]{
  background: color-mix(in srgb, var(--blue-line, #4a9eff) 10%, transparent) !important;
  background-color: color-mix(in srgb, var(--blue-line, #4a9eff) 10%, transparent) !important;
  box-shadow: inset 4px 0 0 var(--blue-line, #4a9eff) !important;
  border: none !important;
  outline: none !important;
  padding-top: 0.05rem !important;
  padding-bottom: 0.05rem !important;
  min-height: 0 !important;
  border-radius: 4px !important;
}
/* 当前会话三列均无单独背景，只沿用整行窄条 */
section[data-testid="stSidebar"] div.kb-history-row-block.kb-history-row-block-current > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div.kb-history-row-block.kb-history-row-block-current > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child{
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  border-radius: 0 !important;
}
/* 当前会话的第二、三列（三点、垃圾桶）：强制无背景，与其它行一致 */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3),
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3),
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child{
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"],
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"],
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"],
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"]{
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  border: none !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] > button,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button{
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  border: none !important;
}
section[data-testid="stSidebar"] div.kb-history-row-block.kb-history-row-block-current button.kb-history-row-btn,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button{
  background: transparent !important;
  box-shadow: none !important;
  color: var(--blue-line) !important;
  -webkit-text-fill-color: var(--blue-line) !important;
  font-weight: 600 !important;
  min-height: 12px !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
section[data-testid="stSidebar"] div.kb-history-row-block.kb-history-row-block-current button.kb-history-row-btn p,
section[data-testid="stSidebar"] div.kb-history-row-block.kb-history-row-block-current button.kb-history-row-btn span,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button span{
  font-weight: 600 !important;
  color: var(--blue-line) !important;
  -webkit-text-fill-color: var(--blue-line) !important;
}

/* Trash column (last-child): 与 nth-child(2)(3) 规则一致，统一 1.5rem 固定宽、右对齐 */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:last-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child{
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
  flex: 0 0 1.5rem !important;
  flex-shrink: 0 !important;
  min-width: 1.5rem !important;
  max-width: 1.5rem !important;
  width: 1.5rem !important;
  justify-content: flex-end !important;
  align-items: center !important;
  display: flex !important;
  padding: 0 !important;
  padding-right: 0 !important;
}
/* 更早会话滚动区预留滚动条宽度，与七天内列表视觉宽度一致，垃圾桶列才能对齐 */
section[data-testid="stSidebar"] div:has(.kb-history-older-list),
section.stSidebar div:has(.kb-history-older-list){
  scrollbar-gutter: stable !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton,
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton{
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  background-color: transparent !important;
  flex-shrink: 0 !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::after,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::after,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button::after,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::after,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::after,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::after,
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::before,
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::after,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::before,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::after{
  display: none !important;
  content: none !important;
}
section[data-testid="stSidebar"] button.kb-history-trash-btn,
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:last-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton > button,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button,
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button{
  all: unset !important;
  box-sizing: border-box !important;
  width: 18px !important;
  min-width: 18px !important;
  max-width: 18px !important;
  height: 18px !important;
  min-height: 18px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  margin: 0 !important;
  border: none !important;
  border-width: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  outline: none !important;
  outline-width: 0 !important;
  color: color-mix(in srgb, var(--sidebar-soft-text) 68%, transparent) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-soft-text) 68%, transparent) !important;
  fill: currentColor !important;
  font-size: 0.72rem !important;
  line-height: 1 !important;
  opacity: 0.5 !important;
  cursor: pointer !important;
  transition: color 0.2s ease, -webkit-text-fill-color 0.2s ease, fill 0.2s ease, opacity 0.2s ease, filter 0.2s ease !important;
  -webkit-appearance: none !important;
  appearance: none !important;
  filter: none !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton > button:hover,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton > button:hover,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:hover,
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:hover,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:hover{
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  color: #6baf6b !important;
  -webkit-text-fill-color: #6baf6b !important;
  fill: #6baf6b !important;
  opacity: 0.9 !important;
  filter: brightness(0) saturate(100%) invert(58%) sepia(35%) saturate(1200%) hue-rotate(95deg) !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus-visible,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus-visible,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus-visible,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus-visible,
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus,
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus-visible,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus-visible{
  outline: none !important;
  box-shadow: none !important;
  border: none !important;
}

/* Toggle block + stButton: prevent full-width stretch. */
section[data-testid="stSidebar"] div.kb-history-toggle-block,
section.stSidebar div.kb-history-toggle-block,
section[data-testid="stSidebar"] div.kb-history-toggle-block div[data-testid="stButton"],
section.stSidebar div.kb-history-toggle-block div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-toggle-marker) + div,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-toggle-marker) + div div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-toggle-marker) + div[data-testid="stButton"]{
  width: auto !important;
  max-width: max-content !important;
  flex: none !important;
}

/* Toggle (收起/展开更早会话): same compact link-like style for both states. */
section[data-testid="stSidebar"] button.kb-history-toggle-btn,
section.stSidebar button.kb-history-toggle-btn,
section[data-testid="stSidebar"] div.kb-history-toggle-block div[data-testid="stButton"] > button,
section.stSidebar div.kb-history-toggle-block div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-toggle-marker) + div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-toggle-marker) + div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-toggle-marker) + div[data-testid="element-container"] div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-toggle-marker) + div[data-testid="stElementContainer"] div[data-testid="stButton"] > button{
  all: unset !important;
  box-sizing: border-box !important;
  display: inline-flex !important;
  align-items: center !important;
  width: auto !important;
  min-height: 18px !important;
  padding: 0.04rem 0.16rem !important;
  border: none !important;
  border-radius: 4px !important;
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 70%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 70%, var(--sidebar-soft-text)) !important;
  font-size: 0.58rem !important;
  font-weight: 500 !important;
  line-height: 1.2 !important;
  letter-spacing: 0.02em !important;
  cursor: pointer !important;
  transition: color 0.2s ease, opacity 0.2s ease !important;
}
section[data-testid="stSidebar"] button.kb-history-toggle-btn p,
section[data-testid="stSidebar"] button.kb-history-toggle-btn span,
section.stSidebar button.kb-history-toggle-btn p,
section.stSidebar button.kb-history-toggle-btn span,
section[data-testid="stSidebar"] div.kb-history-toggle-block div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div.kb-history-toggle-block div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-toggle-marker) + div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-toggle-marker) + div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-toggle-marker) + div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-toggle-marker) + div[data-testid="stButton"] > button span{
  font-size: 0.58rem !important;
}
section[data-testid="stSidebar"] button.kb-history-toggle-btn:hover,
section.stSidebar button.kb-history-toggle-btn:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-toggle-marker) + div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-toggle-marker) + div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-toggle-marker) + div[data-testid="element-container"] div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-toggle-marker) + div[data-testid="stElementContainer"] div[data-testid="stButton"] > button:hover{
  background: transparent !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 88%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 88%, var(--sidebar-soft-text)) !important;
}

/* Dummy row in older section: hide so first real row has same DOM position as others (no first-child layout). */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-dummy-row),
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-dummy-row),
section[data-testid="stSidebar"] div:has(> .kb-history-dummy-row){
  display: none !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-dummy-row) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-dummy-row) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div:has(> .kb-history-dummy-row) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div:has(> .kb-history-dummy-row) + div + div[data-testid="stHorizontalBlock"]{
  display: none !important;
}

/* Remove older block framed card and all borders in 更早会话 area. */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.kb-history-older-list),
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.kb-history-older-list) div[data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.kb-history-older-list) div,
section[data-testid="stSidebar"] div:has(.kb-history-older-list),
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div{
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  outline: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.kb-history-older-list) div[data-testid="stVerticalBlockBorderWrapper"]{
  background: transparent !important;
  padding: 0 !important;
}
/* 展开更早会话区：仅更早列表内部不额外左内边距；不修改侧栏主容器，避免整块左侧被裁切 */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.kb-history-older-list) .kb-history-older-list,
section.stSidebar div[data-testid="stVerticalBlock"]:has(.kb-history-older-list) .kb-history-older-list{
  padding-left: 0 !important;
  margin-left: 0 !important;
}

/* 历史列表区块：紧凑间距、无边框、contain 隔离（展开/收起不波及新建会话等） */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.kb-history-list),
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.kb-history-older-list),
section.stSidebar div[data-testid="stVerticalBlock"]:has(.kb-history-list),
section.stSidebar div[data-testid="stVerticalBlock"]:has(.kb-history-older-list){
  gap: 0 !important;
  border: none !important;
  box-shadow: none !important;
  contain: layout style !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="element-container"],
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stElementContainer"]{
  border: none !important;
  box-shadow: none !important;
}

/* 会话行 ⋮/垃圾桶 统一：用 .kb-history-list 与 .kb-history-older-list 同时覆盖「最近」与「更早」，与展开更早会话后样式一致，无按钮背景 */
/* ⋮ 列与垃圾桶列：stButton 无背景无轮廓 */
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"],
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) div[data-testid="stButton"],
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"],
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) div[data-testid="stButton"],
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"],
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) div[data-testid="stButton"],
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"],
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"],
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"]{
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
  background-color: transparent !important;
}
/* ⋮ 列：去掉 button 伪元素（避免主题画出的外轮廓） */
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button::after,
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] > button::after,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button::after,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) div[data-testid="stButton"] > button::after,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button::before,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button::after,
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button::before,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] > button::after{
  display: none !important;
  content: none !important;
}
/* 会话行右侧：⋮ 与删除按钮统一样式（与展开更早会话后一致），无边框无背景、同色系，右对齐 */
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button,
section[data-testid="stSidebar"] div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) button,
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button,
section.stSidebar div:has(.kb-history-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) button,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) button,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) button,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) button{
  min-width: 1.2rem !important;
  width: 1.2rem !important;
  height: 1.2rem !important;
  min-height: 1.2rem !important;
  padding: 0 !important;
  margin: 0 !important;
  border: none !important;
  border-radius: 0 !important;
  outline: none !important;
  outline-width: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 50%, var(--sidebar-soft-text)) !important;
  font-size: 0.7rem !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  opacity: 0.65 !important;
  transition: opacity 0.15s ease !important;
}
/* 悬停整行时 ⋮ 和删除更明显 */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"]:hover > div[data-testid="column"]:nth-child(2) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"]:hover > div[data-testid="column"]:nth-child(3) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"]:hover > div[data-testid="stColumn"]:nth-child(2) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"]:hover > div[data-testid="stColumn"]:nth-child(3) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"]:hover > div[data-testid="column"]:nth-child(2) button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"]:hover > div[data-testid="column"]:nth-child(3) button,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]:hover > div[data-testid="column"]:nth-child(2) button,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]:hover > div[data-testid="stColumn"]:nth-child(2) button,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]:hover > div[data-testid="column"]:nth-child(3) button,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]:hover > div[data-testid="stColumn"]:nth-child(3) button,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"]:hover > div[data-testid="column"]:nth-child(2) button,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"]:hover > div[data-testid="column"]:nth-child(3) button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]:hover > div[data-testid="column"]:nth-child(2) button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]:hover > div[data-testid="stColumn"]:nth-child(2) button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]:hover > div[data-testid="column"]:nth-child(3) button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]:hover > div[data-testid="stColumn"]:nth-child(3) button{
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) button:hover,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button:hover,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button:hover,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) button:hover,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) button:hover,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button:hover,
section.stSidebar div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button:hover,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button:hover,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) button:hover,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) button:hover,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) button:hover{
  color: var(--sidebar-strong-text) !important;
  background: color-mix(in srgb, var(--sidebar-strong-text) 12%, transparent) !important;
}

/* 侧边栏内紧凑型 expander：Streamlit 渲染为 details[data-testid="stExpander"]，去掉边框与下拉框感 */
section[data-testid="stSidebar"] details[data-testid="stExpander"],
section[data-testid="stSidebar"] div[data-testid="stExpander"],
section.stSidebar details[data-testid="stExpander"],
section.stSidebar div[data-testid="stExpander"]{
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
}
section[data-testid="stSidebar"] details[data-testid="stExpander"] > div,
section[data-testid="stSidebar"] details[data-testid="stExpander"] summary,
section[data-testid="stSidebar"] div[data-testid="stExpander"] > div,
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
section.stSidebar details[data-testid="stExpander"] > div,
section.stSidebar details[data-testid="stExpander"] summary,
section.stSidebar div[data-testid="stExpander"] > div,
section.stSidebar div[data-testid="stExpander"] summary{
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
  min-height: 0 !important;
  padding: 0.15rem 0.2rem !important;
}
section[data-testid="stSidebar"] details[data-testid="stExpander"] summary::-webkit-details-marker,
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary::-webkit-details-marker,
section.stSidebar details[data-testid="stExpander"] summary::-webkit-details-marker,
section.stSidebar div[data-testid="stExpander"] summary::-webkit-details-marker{
  display: none !important;
}

/* 展开/收起更早会话：弱化为次要操作，不抢视线 */
section[data-testid="stSidebar"] div:has(.kb-history-toggle-marker) + div button,
section[data-testid="stSidebar"] div:has(.kb-history-toggle-marker) ~ div button,
section.stSidebar div:has(.kb-history-toggle-marker) + div button,
section.stSidebar div:has(.kb-history-toggle-marker) ~ div button{
  font-size: 0.7rem !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 65%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 65%, var(--sidebar-soft-text)) !important;
  padding: 0.2rem 0.35rem !important;
  min-height: 0 !important;
  background: transparent !important;
  border: none !important;
}

/* Older list: same styles for all rows (dummy row ensures first visible row has same DOM as others). */
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]{
  display: flex !important;
  width: 100% !important;
  gap: 0 !important;
  margin: 0 !important;
  align-items: flex-end !important;
  flex-wrap: nowrap !important;
  background: transparent !important;
  background-color: transparent !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]{
  display: flex !important;
  align-items: center !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child{
  overflow: visible !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stButton"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stButton"]{
  overflow: visible !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button{
  all: unset !important;
  box-sizing: border-box !important;
  display: flex !important;
  align-items: flex-start !important;
  width: 100% !important;
  min-height: 2.5rem !important;
  max-height: none !important;
  padding: 0.25rem 0.1rem 0.2rem !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  overflow: visible !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 88%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 88%, var(--sidebar-soft-text)) !important;
  font-size: 0.74rem !important;
  font-weight: 510 !important;
  line-height: 1.05 !important;
  letter-spacing: 0 !important;
  text-align: left !important;
  cursor: pointer !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child{
  min-width: 0 !important;
}
/* 展开更早会话：按钮内文本块同逻辑，当前宽度下 line-clamp 压成两行+省略号 */
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button > *,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button > *,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button > *,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button > *{
  margin: 0 !important;
  flex: 1 1 0 !important;
  min-width: 0 !important;
  max-width: 100% !important;
  display: -webkit-box !important;
  -webkit-box-orient: vertical !important;
  -webkit-line-clamp: 2 !important;
  font-size: inherit !important;
  line-height: 1.05 !important;
  white-space: normal !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  text-align: left !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button span,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button p,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button span{
  margin: 0 !important;
  display: -webkit-box !important;
  -webkit-box-orient: vertical !important;
  -webkit-line-clamp: 2 !important;
  min-width: 0 !important;
  width: 100% !important;
  font-size: 0.74rem !important;
  line-height: 1.05 !important;
  white-space: normal !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  text-align: left !important;
}

/* 隐藏侧边栏底部「测试模型连接」按钮，避免显得突兀 */
section[data-testid="stSidebar"] div.kb-sidebar-test-llm-marker + div,
section.stSidebar div.kb-sidebar-test-llm-marker + div{
  display: none !important;
}
</style>
"""
    )


__all__ = ["_history_sidebar_compact_css"]

