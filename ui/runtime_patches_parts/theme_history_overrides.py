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

/* 侧边栏主滚动区预留滚动条，展开更早会话后宽度不变，新建会话按钮不会变拥挤 */
section[data-testid="stSidebar"] > div,
section.stSidebar > div{
  scrollbar-gutter: stable !important;
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

/* Conversation rows: no borders, compact gap. flex-wrap: nowrap 保证垃圾桶始终与标题同一行。 */
section[data-testid="stSidebar"] div.kb-history-row-block,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]{
  gap: 0 !important;
  margin: 0 !important;
  align-items: center !important;
  flex-wrap: nowrap !important;
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
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
  min-height: 20px !important;
  max-height: calc(3 * 1.05 * 0.74rem) !important;
  padding: 0.02rem 0.1rem !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  transform: none !important;
  overflow: hidden !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 88%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 88%, var(--sidebar-soft-text)) !important;
  font-size: 0.74rem !important;
  font-weight: 510 !important;
  line-height: 1.05 !important;
  letter-spacing: 0 !important;
  text-align: left !important;
  cursor: pointer !important;
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
  -webkit-line-clamp: 3 !important;
  width: 100% !important;
  font-size: 0.74rem !important;
  line-height: 1.05 !important;
  white-space: normal !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  text-align: left !important;
}

/* Current row: 仅保留左侧蓝条 + 字重，去掉蓝色背景块 */
section[data-testid="stSidebar"] div.kb-history-row-block.kb-history-row-block-current,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row-current) + div div[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row-current) + div div[data-testid="stHorizontalBlock"]{
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: inset 3px 0 0 var(--blue-line) !important;
  border: none !important;
  outline: none !important;
  padding-top: 0.05rem !important;
  padding-bottom: 0.05rem !important;
  min-height: 0 !important;
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

/* Trash column + button: 七天内与更早会话完全统一，避免两套样式导致对不齐。 */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:last-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="element-container"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stElementContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child{
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
  justify-content: flex-end !important;
  align-items: center !important;
  display: flex !important;
  padding-right: 0 !important;
  width: 26px !important;
  max-width: 26px !important;
  min-width: 26px !important;
  flex: 0 0 26px !important;
}
/* 更早会话滚动区预留滚动条宽度，与七天内列表视觉宽度一致，垃圾桶列才能对齐 */
section[data-testid="stSidebar"] div:has(.kb-history-older-list),
section.stSidebar div:has(.kb-history-older-list){
  scrollbar-gutter: stable !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"],
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton{
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  margin-left: auto !important;
  flex-shrink: 0 !important;
}
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
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::before,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button::after{
  display: none !important;
  content: none !important;
}
section[data-testid="stSidebar"] button.kb-history-trash-btn,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="column"]:last-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) > div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton > button,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button,
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
  margin-top: 9px !important;
  border: none !important;
  border-width: 0 !important;
  border-radius: 4px !important;
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
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton > button:hover,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div.stButton > button:hover,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:hover,
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
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus-visible,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.kb-history-row) + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus-visible,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stButton"] > button:focus-visible,
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

/* Older list: same styles for all rows (dummy row ensures first visible row has same DOM as others). */
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"]{
  gap: 0 !important;
  margin: 0 !important;
  align-items: center !important;
  background: transparent !important;
  background-color: transparent !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]{
  display: flex !important;
  align-items: center !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button{
  all: unset !important;
  box-sizing: border-box !important;
  display: flex !important;
  align-items: flex-start !important;
  width: 100% !important;
  min-height: 20px !important;
  max-height: calc(3 * 1.05 * 0.74rem) !important;
  padding: 0.02rem 0.1rem !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  overflow: hidden !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 88%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 88%, var(--sidebar-soft-text)) !important;
  font-size: 0.74rem !important;
  font-weight: 510 !important;
  line-height: 1.05 !important;
  letter-spacing: 0 !important;
  text-align: left !important;
  cursor: pointer !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section.stSidebar div:has(.kb-history-older-list) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span{
  margin: 0 !important;
  display: -webkit-box !important;
  -webkit-box-orient: vertical !important;
  -webkit-line-clamp: 3 !important;
  width: 100% !important;
  font-size: 0.74rem !important;
  line-height: 1.05 !important;
  white-space: normal !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  text-align: left !important;
}
</style>
"""
    )


__all__ = ["_history_sidebar_compact_css"]

