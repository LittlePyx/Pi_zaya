from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

# Single source of truth for dark theme CSS variables (used by _init_theme_css and _sync_theme_with_browser_preference).
_DARK_THEME_TOKENS = """
  --bg: #1f1f1f;
  --panel: #252526;
  --sidebar-bg: #181818;
  --line: rgba(168, 176, 189, 0.42);
  --muted: #d2d9e4;
  --text-main: #e7eaef;
  --text-soft: #e0e7f0;
  --sidebar-strong-text: #e9eff8;
  --sidebar-soft-text: #d5deea;
  --slider-tick-text: #d9e1ec;
  --accent: #4daafc;
  --blue-weak: rgba(77, 170, 252, 0.18);
  --blue-line: rgba(77, 170, 252, 0.58);
  --font-display: "Inter", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
  --font-body: "Inter", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
  --btn-bg: #2d2d30;
  --btn-border: #45494f;
  --btn-text: #e7eaef;
  --btn-hover: #37373d;
  --btn-active: #3f444c;
  --btn-shadow: 0 1px 0 rgba(0, 0, 0, 0.32), 0 12px 30px rgba(0, 0, 0, 0.42);
  --input-bg: #1f2632;
  --input-border: #505a6d;
  --msg-user-bg: rgba(77, 170, 252, 0.14);
  --msg-user-border: rgba(126, 179, 228, 0.40);
  --msg-user-text: #eaf3ff;
  --msg-ai-bg: #222934;
  --msg-ai-border: #3f4b5f;
  --snip-bg: rgba(148, 163, 184, 0.14);
  --snip-border: rgba(148, 163, 184, 0.34);
  --snip-text: #d7deea;
  --snip-quote-bg: rgba(77, 170, 252, 0.16);
  --snip-quote-border: rgba(77, 170, 252, 0.50);
  --snip-mark-bg: rgba(250, 204, 21, 0.28);
  --snip-mark-text: #f8fafc;
  --notice-text: #fde68a;
  --notice-bg: rgba(245, 158, 11, 0.20);
  --notice-border: rgba(245, 158, 11, 0.38);
  --ref-accent: rgba(77, 170, 252, 0.52);
  --dock-bg: linear-gradient(180deg, rgba(31, 31, 31, 0.72) 0%, rgba(31, 31, 31, 0.94) 20%, rgba(31, 31, 31, 0.98) 100%);
  --dock-border: rgba(148, 163, 184, 0.30);
  --dock-shadow: 0 -10px 28px rgba(0, 0, 0, 0.45);
  --copy-btn-bg: rgba(45, 45, 48, 0.94);
  --copy-btn-border: rgba(148, 163, 184, 0.34);
  --copy-btn-text: #dbe4f0;
  --toast-bg: rgba(36, 39, 45, 0.96);
  --toast-border: rgba(148, 163, 184, 0.30);
  --toast-text: #ebf1f8;
  --hint-text: #d2d9e4;
  --refs-title-text: #e7eaef;
  --refs-body-text: #dbe4f0;
  --code-bg: #171d28;
  --code-border: #3d4658;
  --code-text: #e6edf3;
  --code-inline-bg: rgba(77, 170, 252, 0.14);
  --code-syn-keyword: #c678dd;
  --code-syn-string: #98c379;
  --code-syn-comment: #7f848e;
  --code-syn-number: #d19a66;
  --code-syn-func: #61afef;
  --code-syn-type: #e5c07b;
  --code-syn-literal: #56b6c2;
  --code-syn-operator: #abb2bf;
"""

# Selector groups for generated CSS (align with theme_history_overrides).
_SIDEBAR = ('section[data-testid="stSidebar"]', "section.stSidebar")


def _sidebar_sel(suffix: str) -> tuple[str, ...]:
    """Return (sidebar1 + suffix, sidebar2 + suffix) for generated rules."""
    return tuple(s + suffix for s in _SIDEBAR)


def _css_rule(selectors: tuple[str, ...], declarations: str) -> str:
    """Build one CSS rule: selectors { declarations }."""
    return ",\n".join(selectors) + " {\n" + declarations + "\n}"


def _build_sidebar_core_css() -> str:
    """Generate core sidebar layout/close-button rules using selector helpers."""
    return (
        _css_rule(_sidebar_sel(" > div:first-child"), """  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--line) !important;""")
        + "\n"
        + _css_rule(
            _sidebar_sel(""),
            """  background: var(--sidebar-bg) !important;
  --text-color: var(--sidebar-strong-text) !important;
  --secondary-text-color: var(--sidebar-soft-text) !important;
  --body-text-color: var(--sidebar-strong-text) !important;
  container-type: inline-size;
  container-name: kb-sidebar;""",
        )
        + "\n"
        + _css_rule(
            _sidebar_sel(" > div") + _sidebar_sel(" > div > div"),
            "  background: var(--sidebar-bg) !important;",
        )
        + "\n"
        + _css_rule(
            _sidebar_sel(' [data-testid="stSidebarCollapseButton"] button')
            + _sidebar_sel(' [data-testid="stSidebarNav"] button[aria-label*="Close"]')
            + _sidebar_sel(' [data-testid="stSidebarNav"] button[aria-label*="关闭"]'),
            """  width: 30px !important;
  min-width: 30px !important;
  height: 30px !important;
  min-height: 30px !important;
  padding: 0 !important;
  border-radius: 9px !important;
  border: 1px solid var(--btn-border) !important;
  background: color-mix(in srgb, var(--sidebar-bg) 76%, var(--panel)) !important;
  box-shadow: none !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  position: relative !important;
  font-size: 0 !important;
  line-height: 0 !important;""",
        )
        + "\n"
        + _css_rule(
            _sidebar_sel(' [data-testid="stSidebarCollapseButton"] button:hover')
            + _sidebar_sel(' [data-testid="stSidebarNav"] button[aria-label*="Close"]:hover')
            + _sidebar_sel(' [data-testid="stSidebarNav"] button[aria-label*="关闭"]:hover'),
            """  background: var(--btn-hover) !important;
  border-color: var(--blue-line) !important;
  transform: none !important;""",
        )
        + "\n"
        + _css_rule(
            _sidebar_sel(' [data-testid="stSidebarCollapseButton"] button:active')
            + _sidebar_sel(' [data-testid="stSidebarNav"] button[aria-label*="Close"]:active')
            + _sidebar_sel(' [data-testid="stSidebarNav"] button[aria-label*="关闭"]:active'),
            """  background: var(--btn-active) !important;
  border-color: var(--blue-line) !important;
  transform: none !important;""",
        )
        + "\n"
        + _css_rule(
            _sidebar_sel(' [data-testid="stSidebarCollapseButton"] button svg')
            + _sidebar_sel(' [data-testid="stSidebarNav"] button[aria-label*="Close"] svg')
            + _sidebar_sel(' [data-testid="stSidebarNav"] button[aria-label*="关闭"] svg')
            + _sidebar_sel(' [data-testid="stSidebarCollapseButton"] button [data-testid="stIcon"]')
            + _sidebar_sel(' [data-testid="stSidebarNav"] button[aria-label*="Close"] [data-testid="stIcon"]')
            + _sidebar_sel(' [data-testid="stSidebarNav"] button[aria-label*="关闭"] [data-testid="stIcon"]'),
            "  display: none !important;",
        )
    )


def _init_theme_css(theme_mode: str = "dark") -> None:
    mode = "dark" if str(theme_mode or "").lower() == "dark" else "light"
    color_scheme = "dark" if mode == "dark" else "light"

    if mode == "dark":
        tokens = _DARK_THEME_TOKENS
    else:
        tokens = """
  --bg: #fcfcfd;
  --panel: #ffffff;
  --sidebar-bg: #f7f8fa;
  --line: rgba(90, 98, 112, 0.24);
  --muted: rgba(55, 65, 81, 0.76);
  --text-main: #1f2329;
  --text-soft: #4b5563;
  --sidebar-strong-text: #1f2329;
  --sidebar-soft-text: #5a6472;
  --slider-tick-text: #5a6472;
  --accent: #0f6cbd;
  --blue-weak: rgba(15, 108, 189, 0.10);
  --blue-line: rgba(15, 108, 189, 0.40);
  --font-display: "Inter", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
  --font-body: "Inter", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
  --btn-bg: #ffffff;
  --btn-border: rgba(31, 35, 41, 0.16);
  --btn-text: #1f2329;
  --btn-hover: rgba(15, 108, 189, 0.08);
  --btn-active: rgba(15, 108, 189, 0.14);
  --btn-shadow: 0 1px 0 rgba(16, 24, 40, 0.04), 0 10px 24px rgba(16, 24, 40, 0.06);
  --input-bg: #ffffff;
  --input-border: rgba(31, 35, 41, 0.18);
  --msg-user-bg: #eef3f9;
  --msg-user-border: rgba(108, 134, 170, 0.34);
  --msg-user-text: #1f2a37;
  --msg-ai-bg: #ffffff;
  --msg-ai-border: rgba(49, 51, 63, 0.12);
  --snip-bg: rgba(49, 51, 63, 0.04);
  --snip-border: rgba(49, 51, 63, 0.12);
  --snip-text: rgba(15, 23, 42, 0.90);
  --snip-quote-bg: rgba(15, 108, 189, 0.08);
  --snip-quote-border: rgba(15, 108, 189, 0.30);
  --snip-mark-bg: rgba(251, 191, 36, 0.36);
  --snip-mark-text: #0f172a;
  --notice-text: rgba(120, 53, 15, 0.95);
  --notice-bg: rgba(245, 158, 11, 0.10);
  --notice-border: rgba(245, 158, 11, 0.20);
  --ref-accent: rgba(15, 108, 189, 0.24);
  --dock-bg: linear-gradient(180deg, #f7f8fa 0%, #f4f6f8 100%);
  --dock-border: rgba(49, 51, 63, 0.12);
  --dock-shadow: 0 -8px 26px rgba(16, 24, 40, 0.08);
  --copy-btn-bg: rgba(255, 255, 255, 0.88);
  --copy-btn-border: rgba(49, 51, 63, 0.16);
  --copy-btn-text: rgba(31, 42, 55, 0.90);
  --toast-bg: rgba(255, 255, 255, 0.95);
  --toast-border: rgba(49, 51, 63, 0.16);
  --toast-text: rgba(31, 42, 55, 0.88);
  --hint-text: rgba(75, 85, 99, 0.62);
  --refs-title-text: #1f2329;
  --refs-body-text: #4b5563;
  --code-bg: #f5f7fb;
  --code-border: rgba(31, 35, 41, 0.16);
  --code-text: #1f2329;
  --code-inline-bg: rgba(15, 108, 189, 0.10);
  --code-syn-keyword: #a626a4;
  --code-syn-string: #50a14f;
  --code-syn-comment: #a0a1a7;
  --code-syn-number: #986801;
  --code-syn-func: #4078f2;
  --code-syn-type: #c18401;
  --code-syn-literal: #0184bc;
  --code-syn-operator: #383a42;
"""

    css = """
<style>
:root{
__TOKENS__
  --text-color: var(--text-main);
  --secondary-text-color: var(--text-soft);
  --body-text-color: var(--text-main);
  --content-max: 1220px;
}
html, body{
  background: var(--bg) !important;
  color: var(--text-main) !important;
  color-scheme: __SCHEME__;
  font-family: var(--font-body);
}
/* Streamlit marks previous DOM as "stale" during reruns.
   For background conversion auto-refresh, keep old nodes from cluttering the UI. */
.stale-element,
.stale-element *,
[data-stale="true"],
[data-stale="true"] *{
  transition: none !important;
  animation: none !important;
}
/* Chat streaming on Streamlit 1.12 relies on full-script reruns.
   Keep stale nodes visually stable (instead of fading/hiding) to reduce flashing. */
body.kb-live-streaming .stale-element,
body.kb-live-streaming [data-stale="true"]{
  opacity: 1 !important;
  filter: none !important;
  visibility: visible !important;
  pointer-events: none !important;
}
/* Keep stale chat messages visible to reduce flicker, but never keep a stale prompt composer visible:
   it can overlap the fresh composer after send and produce duplicated send/stop/upload controls. */
body.kb-live-streaming .stale-element[data-kb-prompt-dock-root="1"],
body.kb-live-streaming [data-stale="true"][data-kb-prompt-dock-root="1"],
body.kb-live-streaming .stale-element [data-kb-prompt-dock-root="1"],
body.kb-live-streaming [data-stale="true"] [data-kb-prompt-dock-root="1"]{
  opacity: 0 !important;
  filter: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
}
/* Extra safety: hide stale form copies in the main chat area during streaming.
   Streamlit fragment reruns can leave an old prompt form visible for a moment, which
   makes the composer degrade into a duplicated native layout (textarea/uploader/send+stop). */
body.kb-live-streaming section.main .stale-element[data-testid="stForm"],
body.kb-live-streaming section.main [data-stale="true"][data-testid="stForm"],
body.kb-live-streaming section.main .stale-element div[data-testid="stForm"],
body.kb-live-streaming section.main [data-stale="true"] div[data-testid="stForm"],
body.kb-live-streaming section.main .stale-element form,
body.kb-live-streaming section.main [data-stale="true"] form{
  opacity: 0 !important;
  filter: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
}
/* Fragment local reruns can temporarily mark non-fragment regions (including sidebar) stale.
   Never disable/hide sidebar controls because it makes radio/slider/buttons feel broken. */
body.kb-live-streaming section[data-testid="stSidebar"] .stale-element,
body.kb-live-streaming section[data-testid="stSidebar"] [data-stale="true"]{
  opacity: 1 !important;
  filter: none !important;
  visibility: visible !important;
  pointer-events: auto !important;
}
/* Only while background library auto-refresh is active: hide stale copies so controls/messages
   do not appear duplicated. Avoid this in chat streaming because it causes visible flicker. */
body.kb-hide-stale-rerun .stale-element,
body.kb-hide-stale-rerun [data-stale="true"]{
  opacity: 0 !important;
  filter: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
}
/* Keep sidebar interactive even while library background fragments are updating. */
body.kb-hide-stale-rerun section[data-testid="stSidebar"] .stale-element,
body.kb-hide-stale-rerun section[data-testid="stSidebar"] [data-stale="true"]{
  opacity: 1 !important;
  filter: none !important;
  visibility: visible !important;
  pointer-events: auto !important;
}
[data-testid="staleElementOverlay"],
[data-testid="stale-overlay"],
.stale-element-overlay{
  opacity: 0 !important;
  background: transparent !important;
  display: none !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"]{
  background: var(--bg) !important;
}
header[data-testid="stHeader"]{
  background: var(--bg) !important;
  border-bottom: 1px solid var(--line) !important;
}
div[data-testid="stToolbar"],
div[data-testid="stStatusWidget"]{
  background: transparent !important;
}
div[data-testid="stToolbar"] button,
div[data-testid="stStatusWidget"] *{
  color: var(--text-soft) !important;
}
body.kb-live-streaming [data-testid="stAppViewContainer"],
body.kb-live-streaming [data-testid="stAppViewContainer"]{
  opacity: 1 !important;
  filter: none !important;
}
body.kb-resizing [data-testid="stAppViewContainer"],
body.kb-resizing [data-testid="stAppViewContainer"]{
  opacity: 1 !important;
  filter: none !important;
}
body.kb-resizing [data-testid="stAppViewContainer"] *{
  filter: none !important;
}
body.kb-resizing section[data-testid="stSidebar"]{
  background: var(--sidebar-bg) !important;
}
body.kb-resizing section[data-testid="stSidebar"] > div,
body.kb-resizing section[data-testid="stSidebar"] > div > div{
  background: var(--sidebar-bg) !important;
}
body.kb-resizing section[data-testid="stSidebar"] div[style*="z-index"]{
  background: transparent !important;
  opacity: 0 !important;
  pointer-events: none !important;
}
.block-container{
  width: 100%;
  max-width: var(--content-max);
  margin-left: auto !important;
  margin-right: auto !important;
  padding-top: 1.6rem;
  padding-bottom: 12.2rem;
}
__SIDEBAR_CORE__
section[data-testid="stSidebar"] .kb-close-glyph,
.kb-close-glyph{
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  font-size: 22px !important;
  line-height: 1 !important;
  font-weight: 600 !important;
  color: var(--text-main) !important;
  transform: translate(-1px, -1px);
  pointer-events: none !important;
}
section[data-testid="stSidebar"] button.kb-sidebar-close-btn,
button.kb-sidebar-close-btn{
  width: 30px !important;
  min-width: 30px !important;
  height: 30px !important;
  min-height: 30px !important;
  padding: 0 !important;
  border-radius: 9px !important;
  border: 1px solid var(--btn-border) !important;
  background: color-mix(in srgb, var(--sidebar-bg) 76%, var(--panel)) !important;
  box-shadow: none !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  position: relative !important;
  font-size: 0 !important;
  line-height: 0 !important;
  color: transparent !important;
  text-shadow: none !important;
}
section[data-testid="stSidebar"] button.kb-sidebar-close-btn:hover,
button.kb-sidebar-close-btn:hover{
  background: var(--btn-hover) !important;
  border-color: var(--blue-line) !important;
}
section[data-testid="stSidebar"] button.kb-sidebar-close-btn:active,
button.kb-sidebar-close-btn:active{
  background: var(--btn-active) !important;
  border-color: var(--blue-line) !important;
}
section[data-testid="stSidebar"] button.kb-sidebar-close-btn::before,
button.kb-sidebar-close-btn::before{
  content: "\2039" !important;
  position: absolute !important;
  inset: 0 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  font-size: 24px !important;
  line-height: 1 !important;
  font-weight: 600 !important;
  color: var(--text-main) !important;
  transform: translate(-1px, -1px);
  pointer-events: none !important;
}
section[data-testid="stSidebar"] button.kb-sidebar-close-btn svg,
section[data-testid="stSidebar"] button.kb-sidebar-close-btn [data-testid="stIcon"],
button.kb-sidebar-close-btn svg,
button.kb-sidebar-close-btn [data-testid="stIcon"]{
  display: none !important;
}
section[data-testid="stSidebar"] button.kb-sidebar-close-btn *,
button.kb-sidebar-close-btn *{
  color: transparent !important;
  -webkit-text-fill-color: transparent !important;
  opacity: 0 !important;
}
/* Streamlit >=1.5x: keep sidebar collapse button native to avoid breaking click behavior. */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button{
  font-size: inherit !important;
  line-height: normal !important;
  color: inherit !important;
  -webkit-text-fill-color: inherit !important;
  visibility: visible !important;
  pointer-events: auto !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] *{
  visibility: visible !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button svg,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button [data-testid="stIcon"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"]{
  display: inline-flex !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button *{
  opacity: 1 !important;
  color: inherit !important;
  -webkit-text-fill-color: inherit !important;
}
section[data-testid="stSidebar"] button.kb-sidebar-close-btn::before,
button.kb-sidebar-close-btn::before{
  content: none !important;
}
section[data-testid="stSidebar"] button.kb-sidebar-close-btn *,
button.kb-sidebar-close-btn *{
  opacity: 1 !important;
  color: inherit !important;
  -webkit-text-fill-color: inherit !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] *{
  color: var(--sidebar-strong-text) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] *{
  color: var(--sidebar-soft-text) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label *,
section[data-testid="stSidebar"] div[data-testid="stRadio"] p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] span,
section[data-testid="stSidebar"] [role="radiogroup"] label,
section[data-testid="stSidebar"] [role="radiogroup"] label *,
section[data-testid="stSidebar"] [role="radiogroup"] p,
section[data-testid="stSidebar"] [role="radiogroup"] span{
  color: var(--text-main) !important;
  fill: var(--text-main) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label *,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] p,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] span{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stTickBarMax"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stThumbValue"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderValue"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] *,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="tick"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [class*="stSlider"] *,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [style*="color"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] div[style*="color"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] small,
section[data-testid="stSidebar"] div[data-testid="stSlider"] p,
section[data-testid="stSidebar"] div[data-testid="stSlider"] span{
  color: var(--sidebar-soft-text) !important;
  -webkit-text-fill-color: var(--sidebar-soft-text) !important;
  fill: var(--sidebar-soft-text) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] > div,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"]{
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"]::before,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"]::after{
  color: var(--sidebar-soft-text) !important;
  -webkit-text-fill-color: var(--sidebar-soft-text) !important;
  opacity: 1 !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"],
div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
div[data-testid="stSlider"] [data-testid*="TickBarMin"],
div[data-testid="stSlider"] [data-testid*="TickBarMax"],
div[data-testid="stSlider"] [class*="TickBarMin"],
div[data-testid="stSlider"] [class*="TickBarMax"],
div[data-testid="stSlider"] [class*="tickBarMin"],
div[data-testid="stSlider"] [class*="tickBarMax"],
div[data-testid="stSlider"] .stSliderTickBar,
div[data-testid="stSlider"] .stSliderTickBar *,
div[data-testid="stSlider"] [data-testid*="TickBar"]{
  color: var(--slider-tick-text) !important;
  -webkit-text-fill-color: var(--slider-tick-text) !important;
  fill: var(--slider-tick-text) !important;
  stroke: var(--slider-tick-text) !important;
  opacity: 1 !important;
  filter: brightness(1.12) contrast(1.08) !important;
}
div[data-testid="stSlider"] [data-testid*="ThumbValue"],
div[data-testid="stSlider"] [data-testid="stSliderValue"],
div[data-testid="stSlider"] [class*="ThumbValue"]{
  color: var(--accent) !important;
  -webkit-text-fill-color: var(--accent) !important;
  fill: var(--accent) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stThumbValue"]{
  color: var(--accent) !important;
  -webkit-text-fill-color: var(--accent) !important;
  fill: var(--accent) !important;
  opacity: 1 !important;
}
.stMarkdown .katex,
.stMarkdown .katex *,
.stMarkdown .katex-display,
.stMarkdown .katex-display *,
.msg-ai .katex,
.msg-ai .katex *,
.msg-ai .katex-display,
.msg-ai .katex-display *{
  color: var(--text-main) !important;
  fill: var(--text-main) !important;
  opacity: 1 !important;
}
.stMarkdown mjx-container,
.stMarkdown mjx-container *,
.msg-ai mjx-container,
.msg-ai mjx-container *{
  color: var(--text-main) !important;
  fill: currentColor !important;
  stroke: currentColor !important;
  opacity: 1 !important;
}
.kb-sidebar-logo-wrap{
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  box-sizing: border-box;
  /* Reserve space for the native collapse button on the top-right corner. */
  padding: 0 2.7rem 0 0.35rem;
  margin: 0.38rem 0 0.38rem 0;
  /* No negative top margin so logo stays visible and is not clipped by sidebar scroll. */
  pointer-events: none;
}
.kb-sidebar-logo-img{
  width: 186px;
  max-width: 82%;
  height: auto;
  display: block;
  object-fit: contain;
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
  transform: translateZ(0);
  pointer-events: none;
}
h1, h2, h3, h4, h5{
  color: var(--text-main) !important;
  letter-spacing: -0.01em;
}
h1{
  font-family: var(--font-display);
  font-weight: 800;
}
p, li, td, th{ color: var(--text-main) !important; }
small, .stCaption, .msg-meta, .refbox, .genbox, .chat-empty-state{ color: var(--muted) !important; }
.kb-hero-title{
  margin: 0.18rem 0 0.66rem 0 !important;
  color: var(--text-main) !important;
  letter-spacing: -0.012em !important;
  line-height: 1.04 !important;
  font-size: clamp(2.12rem, 3.2vw, 3.05rem) !important;
  font-family: var(--font-display) !important;
  font-weight: 820 !important;
}
.kb-title-caret{
  display: inline-block;
  color: var(--blue-line);
  margin-left: 0.08rem;
  animation: kb-title-caret-blink 0.72s step-end infinite;
  font-weight: 700;
}
@keyframes kb-title-caret-blink{ 0%,100%{opacity:1;} 50%{opacity:0;} }
@media (prefers-reduced-motion: reduce){ .kb-title-caret{ animation: none !important; } }

div.stButton > button,
button[kind]{
  background: var(--btn-bg) !important;
  border: 1px solid var(--btn-border) !important;
  color: var(--btn-text) !important;
  border-radius: 10px !important;
  padding: 0.38rem 0.72rem !important;
  font-weight: 620 !important;
  font-size: 0.85rem !important;
  box-shadow: 0 1px 0 rgba(16, 24, 40, 0.03);
  transition: all 0.15s ease !important;
}
section[data-testid="stSidebar"] div.stButton > button{ width: 100%; }
div.stButton > button:hover,
button[kind]:hover{
  background: var(--btn-hover) !important;
  border-color: var(--blue-line) !important;
  box-shadow: var(--btn-shadow);
}
div.stButton > button:active,
button[kind]:active{
  background: var(--btn-active) !important;
  border-color: var(--blue-line) !important;
}
div.stButton > button:focus,
div.stButton > button:focus-visible,
button[kind]:focus,
button[kind]:focus-visible{
  outline: none !important;
  box-shadow: 0 0 0 2px var(--blue-weak) !important;
}

textarea,
input,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div{
  background: var(--input-bg) !important;
  color: var(--text-main) !important;
  border: 1px solid var(--input-border) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
}
ul[data-testid="stSelectboxVirtualDropdown"],
div[role="listbox"]{
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
}
li[data-testid="stSelectboxVirtualDropdownOption"],
div[role="option"]{ color: var(--text-main) !important; }

div[data-baseweb="tab-list"]{
  gap: 0.48rem !important;
  border-bottom: 1px solid var(--line) !important;
  padding: 0.04rem 0 0.16rem 0 !important;
}
div[data-baseweb="tab-highlight"]{
  display: none !important;
}
button[data-baseweb="tab"]{
  background: color-mix(in srgb, var(--btn-bg) 84%, transparent) !important;
  border: 1px solid var(--btn-border) !important;
  color: var(--text-soft) !important;
  border-radius: 12px !important;
  padding: 0.42rem 0.9rem !important;
  font-weight: 620 !important;
  transition: background 140ms ease, border-color 140ms ease, color 140ms ease !important;
}
button[data-baseweb="tab"]:hover{
  background: var(--btn-hover) !important;
  border-color: var(--blue-line) !important;
  color: var(--text-main) !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  background: var(--blue-weak) !important;
  border-color: var(--blue-line) !important;
  color: var(--text-main) !important;
}

details[data-testid="stExpander"]{
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
}
details[data-testid="stExpander"] summary,
details[data-testid="stExpander"] summary *{ color: var(--text-main) !important; }

[data-testid="stFileUploaderDropzone"]{
  background: var(--panel) !important;
  border: 1px dashed var(--input-border) !important;
  border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"] *{ color: var(--text-soft) !important; }

pre{
  position: relative;
  border-radius: 12px !important;
  background: var(--code-bg) !important;
  border: 1px solid var(--code-border) !important;
  color: var(--code-text) !important;
  overflow: auto !important;
  box-shadow: none !important;
}
pre code{
  background: transparent !important;
  color: var(--code-text) !important;
  text-decoration: none !important;
  border: 0 !important;
  box-shadow: none !important;
}
pre span{
  background: transparent !important;
  text-decoration: none !important;
  border: 0 !important;
  box-shadow: none !important;
}
pre, pre *{
  text-decoration: none !important;
  background-image: none !important;
}
.kb-plain-code{
  margin: 0 !important;
  white-space: pre !important;
  overflow-x: auto !important;
  border: 0 !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}
.kb-plain-code code{
  display: block !important;
  white-space: pre !important;
  border: 0 !important;
  box-shadow: none !important;
  text-decoration: none !important;
  background-image: none !important;
}
.kb-plain-code,
.kb-plain-code *{
  border-bottom: 0 !important;
  text-decoration: none !important;
  text-decoration-line: none !important;
  box-shadow: none !important;
  background-image: none !important;
}
.kb-plain-code code,
.kb-plain-code code.hljs{
  font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", "Source Code Pro", "Consolas", "SFMono-Regular", monospace !important;
  font-size: 0.93rem !important;
  line-height: 1.62 !important;
  letter-spacing: 0.002em;
  color: var(--code-text) !important;
  background: transparent !important;
}
.kb-plain-code .hljs-comment,
.kb-plain-code .hljs-quote{
  color: var(--code-syn-comment) !important;
  font-style: italic;
}
.kb-plain-code .hljs-keyword,
.kb-plain-code .hljs-selector-tag,
.kb-plain-code .hljs-doctag{
  color: var(--code-syn-keyword) !important;
}
.kb-plain-code .hljs-string,
.kb-plain-code .hljs-regexp,
.kb-plain-code .hljs-attr,
.kb-plain-code .hljs-template-tag{
  color: var(--code-syn-string) !important;
}
.kb-plain-code .hljs-number,
.kb-plain-code .hljs-literal{
  color: var(--code-syn-number) !important;
}
.kb-plain-code .hljs-title,
.kb-plain-code .hljs-title.function_,
.kb-plain-code .hljs-function .hljs-title{
  color: var(--code-syn-func) !important;
}
.kb-plain-code .hljs-type,
.kb-plain-code .hljs-class .hljs-title,
.kb-plain-code .hljs-built_in,
.kb-plain-code .hljs-params{
  color: var(--code-syn-type) !important;
}
.kb-plain-code .hljs-variable,
.kb-plain-code .hljs-symbol,
.kb-plain-code .hljs-bullet{
  color: var(--code-syn-literal) !important;
}
.kb-plain-code .hljs-operator,
.kb-plain-code .hljs-punctuation{
  color: var(--code-syn-operator) !important;
}
.kb-plain-code .kb-syn-comment{ color: var(--code-syn-comment) !important; font-style: italic; }
.kb-plain-code .kb-syn-keyword{ color: var(--code-syn-keyword) !important; }
.kb-plain-code .kb-syn-string{ color: var(--code-syn-string) !important; }
.kb-plain-code .kb-syn-number{ color: var(--code-syn-number) !important; }
.kb-plain-code .kb-syn-func{ color: var(--code-syn-func) !important; }
.kb-plain-code .kb-syn-type{ color: var(--code-syn-type) !important; }
.kb-plain-code .kb-syn-literal{ color: var(--code-syn-literal) !important; }
.kb-plain-code .kb-syn-operator{ color: var(--code-syn-operator) !important; }
div[data-testid="stCodeBlock"],
div[data-testid="stCode"],
.stCodeBlock{
  background: var(--code-bg) !important;
  border: 1px solid var(--code-border) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}
div[data-testid="stCodeBlock"] > div,
div[data-testid="stCodeBlock"] pre,
div[data-testid="stCode"] > div,
div[data-testid="stCode"] pre,
.stCodeBlock > div,
.stCodeBlock pre{
  background: transparent !important;
  border: 0 !important;
  border-radius: 12px !important;
  color: var(--code-text) !important;
  box-shadow: none !important;
}
div[data-testid="stCodeBlock"] code,
div[data-testid="stCodeBlock"] pre code,
div[data-testid="stCode"] code,
div[data-testid="stCode"] pre code,
.stCodeBlock code,
.stCodeBlock pre code,
.stMarkdown div[data-testid="stMarkdownContainer"] pre code,
.stMarkdown pre code,
.msg-ai pre code{
  background: transparent !important;
  color: var(--code-text) !important;
}
div[data-testid="stCodeBlock"] span,
div[data-testid="stCode"] span,
.stCodeBlock span,
.stMarkdown div[data-testid="stMarkdownContainer"] pre span,
.stMarkdown pre span,
.msg-ai pre span{
  background: transparent !important;
  border: 0 !important;
  border-bottom: 0 !important;
  box-shadow: none !important;
  text-decoration: none !important;
}
div[data-testid="stCodeBlock"] pre *,
div[data-testid="stCode"] pre *,
.stCodeBlock pre *{
  border: 0 !important;
  border-bottom: 0 !important;
  outline: 0 !important;
  box-shadow: none !important;
  text-decoration: none !important;
  background-image: none !important;
}
div[data-testid="stCodeBlock"] div,
div[data-testid="stCode"] div,
.stCodeBlock div,
div[data-testid="stCodeBlock"] [class*="line"],
div[data-testid="stCode"] [class*="line"],
.stCodeBlock [class*="line"],
div[data-testid="stCodeBlock"] [style*="border-bottom"],
div[data-testid="stCode"] [style*="border-bottom"],
.stCodeBlock [style*="border-bottom"]{
  border-bottom: 0 !important;
  box-shadow: none !important;
  text-decoration: none !important;
  background-image: none !important;
}
div[data-testid="stCodeBlock"] table,
div[data-testid="stCodeBlock"] tbody,
div[data-testid="stCodeBlock"] tr,
div[data-testid="stCodeBlock"] td,
div[data-testid="stCodeBlock"] th,
div[data-testid="stCode"] table,
div[data-testid="stCode"] tbody,
div[data-testid="stCode"] tr,
div[data-testid="stCode"] td,
div[data-testid="stCode"] th,
.stCodeBlock table,
.stCodeBlock tbody,
.stCodeBlock tr,
.stCodeBlock td,
.stCodeBlock th{
  border: 0 !important;
  border-bottom: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  background-image: none !important;
}
div[data-testid="stCodeBlock"] :where(div, span, td, th, p, code),
div[data-testid="stCode"] :where(div, span, td, th, p, code),
.stCodeBlock :where(div, span, td, th, p, code){
  text-decoration: none !important;
  text-decoration-line: none !important;
  text-decoration-thickness: 0 !important;
  text-underline-offset: 0 !important;
}
div[data-testid="stCodeBlock"] *::before,
div[data-testid="stCodeBlock"] *::after,
div[data-testid="stCode"] *::before,
div[data-testid="stCode"] *::after,
.stCodeBlock *::before,
.stCodeBlock *::after{
  border-bottom: 0 !important;
  box-shadow: none !important;
  background-image: none !important;
}
.stMarkdown :not(pre) > code,
.msg-ai :not(pre) > code{
  background: var(--code-inline-bg) !important;
  color: var(--code-text) !important;
  border: 1px solid var(--code-border) !important;
  border-radius: 6px !important;
  padding: 0.08em 0.32em !important;
}
.refbox code, .meta-kv{ color: var(--text-soft) !important; }
.ref-muted-note{
  font-size: 0.80rem;
  color: var(--muted) !important;
  margin: 0.08rem 0 0.40rem 0;
}
.ref-item{
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.55rem 0.62rem 0.50rem 0.62rem;
  margin: 0.16rem 0 0.34rem 0;
}
.ref-item-compact{
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.48rem 0.58rem;
  margin: 0.12rem 0 0.28rem 0;
}
.ref-item-top{
  display: flex;
  align-items: center;
  gap: 0.42rem;
  min-width: 0;
}
.ref-item-header{
  display: flex;
  align-items: center;
  gap: 0.52rem;
  min-width: 0;
  flex-wrap: wrap;
}
.ref-rank{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 2.0rem;
  height: 1.28rem;
  padding: 0 0.36rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--accent) !important;
  border: 1px solid var(--blue-line);
  background: var(--blue-weak);
  flex-shrink: 0;
}
.ref-source{
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-main) !important;
  font-weight: 620;
  letter-spacing: 0.01em;
}
.ref-source-compact{
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-main) !important;
  font-weight: 620;
  letter-spacing: 0.01em;
  font-size: 0.90rem;
  margin-right: 0.1.5rem;
}
.ref-actions-compact{
  display: inline-flex;
  align-items: center;
  gap: 0.32rem;
  flex-shrink: 0;
}
.ref-chip{
  display: inline-flex;
  align-items: center;
  height: 1.28rem;
  padding: 0 0.42rem;
  border-radius: 999px;
  font-size: 0.70rem;
  font-weight: 640;
  color: var(--text-soft) !important;
  border: 1px solid var(--line);
  background: rgba(148, 163, 184, 0.12);
  flex-shrink: 0;
}
.ref-score{
  display: inline-flex;
  align-items: center;
  height: 1.28rem;
  padding: 0 0.44rem;
  border-radius: 999px;
  font-size: 0.70rem;
  font-weight: 700;
  border: 1px solid transparent;
  flex-shrink: 0;
}
.ref-score-hi{
  color: #22c55e !important;
  border-color: rgba(34, 197, 94, 0.36);
  background: rgba(34, 197, 94, 0.14);
}
.ref-score-mid{
  color: #f59e0b !important;
  border-color: rgba(245, 158, 11, 0.36);
  background: rgba(245, 158, 11, 0.14);
}
.ref-score-low{
  color: var(--text-soft) !important;
  border-color: var(--line);
  background: rgba(148, 163, 184, 0.10);
}
.ref-item-sub{
  margin-top: 0.34rem;
  font-size: 0.82rem;
  color: var(--text-soft) !important;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ref-item-sub-compact{
  margin-top: 0.00rem;
  font-size: 0.80rem;
  color: var(--text-soft) !important;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: normal;
  line-height: 1.35;
  opacity: 0.85;
}
.kb-ref-header-block{
  display: flex;
  flex-direction: column;
  gap: 0.34rem;
  padding: 0.22rem 0.02rem 0.08rem 0.02rem;
}
.kb-ref-title-row{
  display: flex;
  align-items: flex-start;
  gap: 0.58rem;
  min-width: 0;
}
.kb-ref-title-stack{
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.26rem;
}
.kb-ref-title{
  color: var(--text-main) !important;
  font-size: 1.00rem;
  line-height: 1.28;
  font-weight: 680;
  letter-spacing: 0.005em;
  word-break: break-word;
}
.kb-ref-heading-path{
  color: var(--text-soft) !important;
  font-size: 0.78rem;
  line-height: 1.34;
  opacity: 0.92;
  word-break: break-word;
}
.kb-ref-loc-row{
  display: flex;
  flex-wrap: wrap;
  gap: 0.34rem;
  align-items: center;
}
.kb-ref-loc-chip{
  display: inline-flex;
  align-items: center;
  gap: 0.24rem;
  height: 1.36rem;
  padding: 0 0.54rem;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.00));
  color: var(--text-soft) !important;
  font-size: 0.73rem;
  font-weight: 560;
  line-height: 1;
  max-width: 100%;
}
.kb-ref-loc-chip-label{
  opacity: 0.74;
  font-weight: 650;
}
.kb-ref-loc-chip-value{
  color: var(--text-main) !important;
  opacity: 0.92;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: min(38ch, 62vw);
}
.kb-ref-status-stack{
  min-height: 100%;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 0.34rem;
  flex-wrap: wrap;
}
.kb-ref-status-stack .ref-chip{
  height: 1.34rem;
  padding: 0 0.50rem;
}
.kb-ref-status-stack .ref-score{
  height: 1.34rem;
  padding: 0 0.52rem;
}
.kb-ref-insight-grid{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 0.60rem;
  margin: 0.42rem 0 0.20rem 0;
}
.kb-ref-insight-card{
  border: 1px solid var(--line);
  border-radius: 12px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.00) 100%),
    linear-gradient(90deg, rgba(15,108,189,0.03), rgba(15,108,189,0.00));
  padding: 0.68rem 0.82rem 0.74rem 0.82rem;
  min-height: 8.2rem;
  box-shadow: 0 1px 0 rgba(16,24,40,0.02);
}
.kb-ref-insight-head{
  display: inline-flex;
  align-items: center;
  gap: 0.36rem;
  min-width: 0;
}
.kb-ref-insight-tag{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 1.18rem;
  padding: 0 0.40rem;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--blue-line) 68%, var(--line));
  background: color-mix(in srgb, var(--blue-weak) 65%, transparent);
  color: var(--accent) !important;
  font-size: 0.68rem;
  font-weight: 720;
  letter-spacing: 0.02em;
  flex-shrink: 0;
}
.kb-ref-insight-text{
  margin-top: 0.40rem;
  color: var(--text-main) !important;
  font-size: 0.88rem;
  line-height: 1.56;
  opacity: 0.96;
  word-break: break-word;
}
.kb-ref-metrics-row{
  margin: 0.38rem 0 0.08rem 0;
  padding: 0.50rem 0.62rem;
  border: 1px dashed color-mix(in srgb, var(--line) 84%, transparent);
  border-radius: 10px;
  background: color-mix(in srgb, var(--panel) 94%, transparent);
  color: var(--text-main) !important;
  font-size: 0.79rem;
  line-height: 1.45;
  word-break: break-word;
}
.kb-ref-metric-src{
  color: var(--text-soft) !important;
  font-size: 0.72rem;
}
.kb-ref-metric-tag{
  display: inline-flex;
  align-items: center;
  height: 1.06rem;
  padding: 0 0.36rem;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--blue-line) 46%, var(--line));
  background: color-mix(in srgb, var(--blue-weak) 72%, transparent);
  color: var(--text-main) !important;
  font-size: 0.70rem;
  font-weight: 640;
  line-height: 1;
}
.kb-ref-metric-na{
  color: var(--text-soft) !important;
  font-weight: 600;
}
.kb-ref-doi-link{
  color: var(--accent) !important;
  -webkit-text-fill-color: var(--accent) !important;
  text-decoration: underline;
  text-underline-offset: 2px;
  font-weight: 680;
}
.kb-ref-doi-link:hover{
  opacity: 0.88;
}
.kb-ref-guide-details{
  margin: 0.46rem 0 0.18rem 0;
  border: 1px solid var(--line);
  border-radius: 14px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.00) 100%),
    var(--panel);
  overflow: hidden;
}
.kb-ref-guide-details > summary{
  list-style: none;
  cursor: pointer;
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: 0.65rem;
  padding: 0.62rem 0.72rem;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
  color: var(--text-main) !important;
  user-select: none;
}
.kb-ref-guide-details > summary::-webkit-details-marker{
  display: none;
}
.kb-ref-guide-summary-main{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  min-width: 0;
}
.kb-ref-guide-caret{
  width: 1.1rem;
  min-width: 1.1rem;
  height: 1.1rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--text-soft) !important;
  background: rgba(148,163,184,0.08);
  font-size: 0.76rem;
  line-height: 1;
  transition: transform 120ms ease;
}
.kb-ref-guide-details[open] .kb-ref-guide-caret{
  transform: rotate(90deg);
}
.kb-ref-guide-summary-title{
  color: var(--text-main) !important;
  font-size: 0.92rem;
  font-weight: 680;
  line-height: 1.2;
}
.kb-ref-guide-summary-hint{
  color: var(--text-soft) !important;
  font-size: 0.74rem;
  line-height: 1.25;
  text-align: right;
  max-width: min(42ch, 38vw);
  opacity: 0.9;
}
.kb-ref-guide-details[open] > summary{
  border-bottom: 1px solid var(--line);
}
.kb-ref-guide-body{
  padding: 0.72rem 0.78rem 0.80rem 0.78rem;
}
.kb-ref-guide-scoreline{
  margin-bottom: 0.52rem;
  color: var(--text-soft) !important;
  font-size: 0.74rem;
  line-height: 1.25;
}
.kb-ref-guide-grid{
  display: grid;
  gap: 0.58rem;
}
.kb-ref-guide-block{
  border: 1px solid color-mix(in srgb, var(--line) 86%, transparent);
  border-radius: 12px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.018), rgba(255,255,255,0.00)),
    color-mix(in srgb, var(--panel) 92%, transparent);
  padding: 0.56rem 0.62rem 0.60rem 0.62rem;
}
.kb-ref-guide-block-head{
  display: inline-flex;
  align-items: center;
  gap: 0.38rem;
  min-width: 0;
}
.kb-ref-guide-block-tag{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 1.1rem;
  padding: 0 0.38rem;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--blue-line) 62%, var(--line));
  background: color-mix(in srgb, var(--blue-weak) 72%, transparent);
  color: var(--accent) !important;
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  flex-shrink: 0;
}
.kb-ref-guide-block-title{
  color: var(--text-main) !important;
  font-size: 0.82rem;
  line-height: 1.22;
  font-weight: 700;
}
.kb-ref-guide-block-body{
  margin-top: 0.34rem;
  color: var(--text-main) !important;
}
.kb-ref-guide-paragraph{
  margin: 0;
  color: var(--text-main) !important;
  font-size: 0.84rem;
  line-height: 1.48;
  word-break: break-word;
}
.kb-ref-guide-focus{
  margin-top: 0.32rem;
}
.kb-ref-guide-focus-label{
  color: var(--text-soft) !important;
  font-size: 0.76rem;
  line-height: 1.35;
  font-weight: 620;
}
.kb-ref-guide-row{
  display: grid;
  grid-template-columns: 8.0rem minmax(0, 1fr);
  gap: 0.64rem;
  align-items: flex-start;
  padding: 0.12rem 0;
}
.kb-ref-guide-label{
  color: var(--text-soft) !important;
  font-size: 0.76rem;
  line-height: 1.35;
  font-weight: 700;
  letter-spacing: 0.01em;
}
.kb-ref-guide-value{
  color: var(--text-main) !important;
  font-size: 0.84rem;
  line-height: 1.48;
  word-break: break-word;
}
.kb-ref-focus-wrap{
  margin-top: 0.12rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.38rem;
}
.kb-ref-focus-chip{
  display: inline-flex;
  align-items: center;
  height: 1.34rem;
  padding: 0 0.52rem;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--blue-line) 54%, var(--line));
  background: color-mix(in srgb, var(--blue-weak) 78%, transparent);
  color: var(--text-main) !important;
  font-size: 0.76rem;
  font-weight: 560;
  line-height: 1;
}
.kb-ref-rank-wrap{
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding-top: 0.28rem;
}
.kb-ref-inline-label{
  margin: 0 !important;
  color: var(--text-main) !important;
  font-size: 0.78rem;
  line-height: 1.18;
  font-weight: 700;
}
.kb-ref-insight-card{
  position: relative;
  overflow: hidden;
  padding-left: 0.92rem;
  box-shadow:
    0 1px 0 rgba(16,24,40,0.02),
    inset 0 1px 0 rgba(255,255,255,0.025);
}
.kb-ref-insight-card::before{
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 3px;
  background:
    linear-gradient(180deg,
      color-mix(in srgb, var(--accent) 72%, transparent) 0%,
      color-mix(in srgb, var(--blue-line) 58%, transparent) 42%,
      transparent 92%);
  opacity: 0.92;
}
.kb-ref-title{
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  overflow: hidden;
}
.kb-ref-heading-path{
  display: inline-block;
  width: fit-content;
  max-width: 100%;
  font-family: inherit;
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0;
}
.kb-ref-heading-meta-row{
  display: flex;
  align-items: center;
  gap: 0.42rem;
  flex-wrap: wrap;
}
.kb-ref-heading-score-wrap{
  display: inline-flex;
  align-items: center;
}
.kb-ref-heading-score-wrap .ref-score{
  height: 1.20rem;
  padding: 0 0.44rem;
  font-size: 0.68rem;
}
.kb-ref-guide-details{
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
}
.kb-ref-guide-details > summary:hover{
  background:
    linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.005)),
    color-mix(in srgb, var(--blue-weak) 20%, transparent);
}
.kb-ref-guide-details > summary:focus-visible{
  outline: 2px solid color-mix(in srgb, var(--blue-line) 72%, transparent);
  outline-offset: -2px;
}
.kb-ref-guide-summary-title{
  letter-spacing: 0.005em;
}
.kb-ref-guide-summary-hint{
  opacity: 0.95;
}
.kb-ref-guide-value{
  min-width: 0;
}
@media (max-width: 760px){
  .kb-ref-title{ font-size: 0.94rem; }
  .kb-ref-insight-grid{ grid-template-columns: 1fr; }
  .kb-ref-guide-row{ grid-template-columns: 1fr; gap: 0.18rem; }
  .kb-ref-guide-summary-hint{ max-width: 100%; text-align: left; }
  .kb-ref-guide-details > summary{ grid-template-columns: 1fr; }
}
.ref-item-gap{ height: 0.26rem; }
.ref-item-gap-compact{ height: 0.20rem; }
.citation-loading{
  font-size: 0.80rem;
  color: var(--text-soft) !important;
  margin: 0.20rem 0 0.30rem 0;
  opacity: 0.75;
}
.snipbox-compact{
  background: var(--snip-bg);
  border: 1px solid var(--snip-border);
  border-radius: 10px;
  padding: 8px 10px;
  margin: 0.28rem 0 0.40rem 0;
}
.snipbox-compact pre{
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--snip-text) !important;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  font-size: 0.85rem;
  line-height: 1.5;
}
.hr{ height: 1px; background: var(--line); margin: 1rem 0; }
/* Sidebar visual polish: improve spacing, control hierarchy, and tactile feel
   without changing widget behavior. */
section[data-testid="stSidebar"] > div:first-child{
  background:
    linear-gradient(180deg,
      color-mix(in srgb, var(--sidebar-bg) 88%, var(--panel)) 0%,
      var(--sidebar-bg) 22%,
      var(--sidebar-bg) 100%) !important;
  box-shadow:
    inset -1px 0 0 color-mix(in srgb, var(--line) 90%, transparent),
    inset 0 1px 0 color-mix(in srgb, var(--panel) 22%, transparent) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] > div > div{
  padding-bottom: 0.45rem !important;
}
section[data-testid="stSidebar"] > div{
  scrollbar-width: thin;
  scrollbar-color: color-mix(in srgb, var(--sidebar-soft-text) 28%, transparent) transparent;
}
section[data-testid="stSidebar"] *::-webkit-scrollbar{
  width: 10px;
  height: 10px;
}
section[data-testid="stSidebar"] *::-webkit-scrollbar-thumb{
  background: color-mix(in srgb, var(--sidebar-soft-text) 24%, transparent);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
section[data-testid="stSidebar"] *::-webkit-scrollbar-thumb:hover{
  background: color-mix(in srgb, var(--sidebar-soft-text) 40%, transparent);
  border: 2px solid transparent;
  background-clip: padding-box;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4{
  font-family: var(--font-display) !important;
  color: var(--sidebar-strong-text) !important;
  letter-spacing: -0.02em !important;
}
section[data-testid="stSidebar"] h3{
  font-size: 1.26rem !important;
  font-weight: 820 !important;
  line-height: 1.12 !important;
  margin: 0.28rem 0 0.18rem 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{
  margin-top: -0.06rem !important;
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] *{
  font-size: 0.78rem !important;
  line-height: 1.35 !important;
  letter-spacing: 0.01em;
  opacity: 0.92 !important;
}
section[data-testid="stSidebar"] .hr{
  height: 1px !important;
  margin: 0.9rem 0 0.95rem 0 !important;
  border: 0 !important;
  background:
    linear-gradient(90deg,
      transparent 0%,
      color-mix(in srgb, var(--line) 85%, transparent) 12%,
      color-mix(in srgb, var(--line) 100%, transparent) 50%,
      color-mix(in srgb, var(--line) 85%, transparent) 88%,
      transparent 100%) !important;
}
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"]{
  margin-bottom: 0.14rem !important;
}
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] label,
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] span{
  font-weight: 650 !important;
  letter-spacing: 0.01em;
}
section[data-testid="stSidebar"] div[data-testid="stButton"]{
  margin: 0.22rem 0 !important;
}
section[data-testid="stSidebar"] div.stButton > button{
  min-height: 42px !important;
  border-radius: 14px !important;
  border-color: color-mix(in srgb, var(--btn-border) 82%, var(--line)) !important;
  background:
    linear-gradient(180deg,
      color-mix(in srgb, var(--btn-bg) 88%, var(--panel)) 0%,
      color-mix(in srgb, var(--btn-bg) 96%, transparent) 100%) !important;
  box-shadow:
    0 6px 18px rgba(9, 16, 29, 0.08),
    inset 0 1px 0 color-mix(in srgb, var(--panel) 24%, white) !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover{
  transform: translateY(-1px);
  border-color: color-mix(in srgb, var(--blue-line) 82%, var(--btn-border)) !important;
  box-shadow:
    0 10px 22px rgba(9, 16, 29, 0.12),
    inset 0 1px 0 color-mix(in srgb, var(--panel) 28%, white) !important;
}
section[data-testid="stSidebar"] div.stButton > button:active{
  transform: translateY(0);
  box-shadow:
    0 4px 10px rgba(9, 16, 29, 0.10),
    inset 0 1px 0 color-mix(in srgb, var(--panel) 16%, white) !important;
}
/* 对话记录区不应用圆角按钮样式：新建会话 + 会话列表行（无边框小字） */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div div.stButton > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div.stButton > button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) div.stButton > button{
  min-height: unset !important;
  border-radius: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  transform: none !important;
  font-weight: inherit !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div div.stButton > button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div.stButton > button:hover,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) div.stButton > button:hover{
  transform: none !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"],
section[data-testid="stSidebar"] div[data-testid="stTextInput"],
section[data-testid="stSidebar"] div[data-testid="stTextArea"],
section[data-testid="stSidebar"] div[data-testid="stNumberInput"]{
  margin: 0.22rem 0 0.34rem 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input,
section[data-testid="stSidebar"] div[data-testid="stTextArea"] textarea,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input{
  border-radius: 16px !important;
  border-color: color-mix(in srgb, var(--input-border) 84%, var(--line)) !important;
  background:
    linear-gradient(180deg,
      color-mix(in srgb, var(--input-bg) 88%, var(--panel)) 0%,
      color-mix(in srgb, var(--input-bg) 100%, transparent) 100%) !important;
  box-shadow:
    inset 0 1px 0 color-mix(in srgb, var(--panel) 18%, white),
    0 2px 8px rgba(11, 18, 32, 0.05) !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:hover,
section[data-testid="stSidebar"] div[data-testid="stTextArea"] textarea:hover,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input:hover{
  border-color: color-mix(in srgb, var(--blue-line) 52%, var(--input-border)) !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus,
section[data-testid="stSidebar"] div[data-testid="stTextArea"] textarea:focus,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input:focus{
  border-color: color-mix(in srgb, var(--blue-line) 78%, var(--input-border)) !important;
  box-shadow:
    0 0 0 3px color-mix(in srgb, var(--blue-weak) 82%, transparent),
    inset 0 1px 0 color-mix(in srgb, var(--panel) 18%, white) !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] *{
  font-weight: 620 !important;
}
ul[data-testid="stSelectboxVirtualDropdown"],
div[role="listbox"]{
  border-radius: 14px !important;
  box-shadow: 0 14px 28px rgba(8, 14, 26, 0.22) !important;
  overflow: hidden;
}
li[data-testid="stSelectboxVirtualDropdownOption"],
div[role="option"]{
  border-radius: 8px !important;
}
li[data-testid="stSelectboxVirtualDropdownOption"]:hover,
div[role="option"]:hover{
  background: color-mix(in srgb, var(--blue-weak) 68%, transparent) !important;
}
div[role="option"][aria-selected="true"]{
  background: color-mix(in srgb, var(--blue-weak) 90%, transparent) !important;
  color: var(--text-main) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"],
section[data-testid="stSidebar"] div[data-testid="stCheckbox"]{
  margin: 0.22rem 0 0.32rem 0 !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] > label{
  border: 1px solid color-mix(in srgb, var(--line) 78%, transparent);
  background: color-mix(in srgb, var(--panel) 28%, transparent);
  border-radius: 14px;
  padding: 0.35rem 0.55rem !important;
  margin: 0.16rem 0 !important;
  transition: background 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:hover,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] > label:hover{
  border-color: color-mix(in srgb, var(--blue-line) 48%, var(--line));
  background: color-mix(in srgb, var(--blue-weak) 44%, transparent);
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked),
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] > label:has(input:checked){
  border-color: color-mix(in srgb, var(--blue-line) 88%, var(--line));
  background: color-mix(in srgb, var(--blue-weak) 78%, transparent);
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--panel) 20%, white);
}
section[data-testid="stSidebar"] input[type="radio"],
section[data-testid="stSidebar"] input[type="checkbox"]{
  accent-color: var(--accent) !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"]{
  margin: 0.24rem 0 0.48rem 0 !important;
  padding: 0.42rem 0.58rem 0.38rem 0.58rem !important;
  border: 1px solid color-mix(in srgb, var(--line) 75%, transparent);
  border-radius: 16px !important;
  background:
    linear-gradient(180deg,
      color-mix(in srgb, var(--panel) 34%, transparent) 0%,
      color-mix(in srgb, var(--sidebar-bg) 88%, transparent) 100%) !important;
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--panel) 20%, white) !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"]{
  margin-top: 0.12rem !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]{
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--blue-weak) 36%, transparent) !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] > div > div{
  border-radius: 999px !important;
}
section[data-testid="stSidebar"] details[data-testid="stExpander"]{
  border: 1px solid color-mix(in srgb, var(--line) 72%, transparent) !important;
  border-radius: 16px !important;
  background: color-mix(in srgb, var(--panel) 20%, transparent) !important;
  overflow: hidden;
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--panel) 18%, white) !important;
}
section[data-testid="stSidebar"] details[data-testid="stExpander"] summary{
  padding: 0.5rem 0.7rem !important;
}
section[data-testid="stSidebar"] details[data-testid="stExpander"] > div{
  padding: 0.1rem 0.25rem 0.3rem 0.25rem !important;
}
/* 项目行三点菜单 expander：无边框、无内阴影，与通用 sidebar expander 区分 */
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div details[data-testid="stExpander"],
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"],
section.stSidebar div:has(.kb-history-project-expander) + div details[data-testid="stExpander"],
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"]{
  border: none !important;
  border-width: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  overflow: visible !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div details[data-testid="stExpander"] > div,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"] > div,
section.stSidebar div:has(.kb-history-project-expander) + div details[data-testid="stExpander"] > div,
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"] > div{
  border: none !important;
  box-shadow: none !important;
}
/* Minimal sidebar override: flatten the previous decorative treatment. */
section[data-testid="stSidebar"] > div:first-child{
  background: var(--sidebar-bg) !important;
  box-shadow: inset -1px 0 0 color-mix(in srgb, var(--line) 92%, transparent) !important;
}
section[data-testid="stSidebar"] h3{
  font-size: 1.04rem !important;
  font-weight: 760 !important;
  line-height: 1.18 !important;
  letter-spacing: -0.012em !important;
  margin: 0.18rem 0 0.10rem 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{
  margin-top: 0 !important;
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] *{
  font-size: 0.76rem !important;
  line-height: 1.32 !important;
  opacity: 0.84 !important;
  letter-spacing: 0 !important;
}
section[data-testid="stSidebar"] .hr{
  margin: 0.82rem 0 0.88rem 0 !important;
  background: color-mix(in srgb, var(--line) 95%, transparent) !important;
}
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"]{
  margin-bottom: 0.08rem !important;
}
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] label,
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] span{
  font-weight: 600 !important;
  letter-spacing: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"]{
  margin: 0.16rem 0 !important;
}
section[data-testid="stSidebar"] div.stButton > button{
  min-height: 38px !important;
  border-radius: 12px !important;
  padding: 0.34rem 0.68rem !important;
  font-size: 0.82rem !important;
  font-weight: 650 !important;
  letter-spacing: 0 !important;
  background: color-mix(in srgb, var(--btn-bg) 94%, transparent) !important;
  border: 1px solid color-mix(in srgb, var(--btn-border) 88%, var(--line)) !important;
  box-shadow: none !important;
  transform: none !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover{
  background: color-mix(in srgb, var(--btn-hover) 90%, var(--btn-bg)) !important;
  border-color: color-mix(in srgb, var(--blue-line) 58%, var(--btn-border)) !important;
  box-shadow: none !important;
  transform: none !important;
}
section[data-testid="stSidebar"] div.stButton > button:active{
  background: color-mix(in srgb, var(--btn-active) 92%, var(--btn-bg)) !important;
  border-color: color-mix(in srgb, var(--blue-line) 68%, var(--btn-border)) !important;
  box-shadow: none !important;
  transform: none !important;
}
/* 对话记录区不应用圆角按钮样式（同上） */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div div.stButton > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div.stButton > button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) div.stButton > button{
  min-height: unset !important;
  border-radius: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  transform: none !important;
  font-weight: inherit !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"],
section[data-testid="stSidebar"] div[data-testid="stTextInput"],
section[data-testid="stSidebar"] div[data-testid="stTextArea"],
section[data-testid="stSidebar"] div[data-testid="stNumberInput"]{
  margin: 0.16rem 0 0.24rem 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input,
section[data-testid="stSidebar"] div[data-testid="stTextArea"] textarea,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input{
  border-radius: 12px !important;
  background: var(--input-bg) !important;
  border: 1px solid color-mix(in srgb, var(--input-border) 90%, var(--line)) !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:hover,
section[data-testid="stSidebar"] div[data-testid="stTextArea"] textarea:hover,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input:hover{
  border-color: color-mix(in srgb, var(--blue-line) 45%, var(--input-border)) !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus,
section[data-testid="stSidebar"] div[data-testid="stTextArea"] textarea:focus,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input:focus{
  border-color: color-mix(in srgb, var(--blue-line) 65%, var(--input-border)) !important;
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--blue-weak) 65%, transparent) !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] > label{
  border: 0 !important;
  background: transparent !important;
  border-radius: 0 !important;
  padding: 0.14rem 0 !important;
  margin: 0.06rem 0 !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:hover,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] > label:hover{
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked),
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] > label:has(input:checked){
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"],
section[data-testid="stSidebar"] div[data-testid="stCheckbox"]{
  margin: 0.14rem 0 0.18rem 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"]{
  margin: 0.14rem 0 0.36rem 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] details[data-testid="stExpander"]{
  border: 1px solid color-mix(in srgb, var(--line) 82%, transparent) !important;
  border-radius: 12px !important;
  background: transparent !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] details[data-testid="stExpander"] summary{
  padding: 0.42rem 0.58rem !important;
}
section[data-testid="stSidebar"] details[data-testid="stExpander"] > div{
  padding: 0.06rem 0.15rem 0.18rem 0.15rem !important;
}
/* 项目行三点 + 会话行 ⋮：无边框、无下拉框感（覆盖上面 sidebar expander 的 border；含包装 div 与 summary） */
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div details[data-testid="stExpander"],
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div [data-testid="stExpander"],
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"],
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"],
section[data-testid="stSidebar"] div:has(.kb-history-list) details[data-testid="stExpander"],
section[data-testid="stSidebar"] div:has(.kb-history-list) [data-testid="stExpander"],
section.stSidebar div:has(.kb-history-project-expander) + div details[data-testid="stExpander"],
section.stSidebar div:has(.kb-history-project-expander) + div [data-testid="stExpander"],
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"],
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"],
section.stSidebar div:has(.kb-history-list) details[data-testid="stExpander"],
section.stSidebar div:has(.kb-history-list) [data-testid="stExpander"]{
  border: none !important;
  border-width: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
  background-color: transparent !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div details[data-testid="stExpander"] > div,
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div [data-testid="stExpander"] > div,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"] > div,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] > div,
section[data-testid="stSidebar"] div:has(.kb-history-list) details[data-testid="stExpander"] > div,
section[data-testid="stSidebar"] div:has(.kb-history-list) [data-testid="stExpander"] > div,
section.stSidebar div:has(.kb-history-project-expander) + div details[data-testid="stExpander"] > div,
section.stSidebar div:has(.kb-history-project-expander) + div [data-testid="stExpander"] > div,
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"] > div,
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] > div,
section.stSidebar div:has(.kb-history-list) details[data-testid="stExpander"] > div,
section.stSidebar div:has(.kb-history-list) [data-testid="stExpander"] > div{
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
}
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div details[data-testid="stExpander"] summary,
section[data-testid="stSidebar"] div:has(.kb-history-project-expander) + div [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"] summary,
section[data-testid="stSidebar"] div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] div:has(.kb-history-list) details[data-testid="stExpander"] summary,
section[data-testid="stSidebar"] div:has(.kb-history-list) [data-testid="stExpander"] summary,
section.stSidebar div:has(.kb-history-project-expander) + div details[data-testid="stExpander"] summary,
section.stSidebar div:has(.kb-history-project-expander) + div [data-testid="stExpander"] summary,
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) details[data-testid="stExpander"] summary,
section.stSidebar div:has(.kb-history-project-row) + div [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stExpander"] summary,
section.stSidebar div:has(.kb-history-list) details[data-testid="stExpander"] summary,
section.stSidebar div:has(.kb-history-list) [data-testid="stExpander"] summary{
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
  border-radius: 0 !important;
}
section[data-testid="stSidebar"] a{
  color: color-mix(in srgb, var(--sidebar-soft-text) 92%, var(--sidebar-strong-text)) !important;
  text-decoration-color: color-mix(in srgb, var(--sidebar-soft-text) 45%, transparent) !important;
  text-underline-offset: 2px;
}
section[data-testid="stSidebar"] a:hover{
  color: var(--sidebar-strong-text) !important;
  text-decoration-color: color-mix(in srgb, var(--blue-line) 65%, transparent) !important;
}
.pill{
  display: inline-flex;
  align-items: center;
  padding: 0.12rem 0.52rem;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 650;
  border: 1px solid transparent;
}
.pill.ok{
  background: rgba(34, 197, 94, 0.14);
  color: #22c55e !important;
  border-color: rgba(34, 197, 94, 0.34);
}
.pill.warn{
  background: rgba(245, 158, 11, 0.16);
  color: #f59e0b !important;
  border-color: rgba(245, 158, 11, 0.36);
}
.pill.run{
  background: var(--blue-weak);
  color: var(--accent) !important;
  border-color: var(--blue-line);
}

.msg-user{
  background: var(--msg-user-bg);
  border: 1px solid var(--msg-user-border);
  color: var(--msg-user-text) !important;
  border-radius: 16px;
  padding: 10px 14px;
  width: fit-content;
  max-width: min(900px, 94%);
  white-space: pre-wrap;
  word-break: keep-all;
  overflow-wrap: anywhere;
  margin-left: 0;
}
.msg-user-wrap{
  display: flex;
  align-items: flex-start;
  justify-content: flex-end;
  gap: 0.42rem;
  width: fit-content;
  max-width: min(900px, 94%);
  margin-left: auto;
}
.msg-user-wrap .msg-user{
  margin-left: 0 !important;
}
.msg-user-wrap .msg-meta-user{
  margin: 0.14rem 0 0 0;
  text-align: right;
  white-space: nowrap;
  line-height: 1.2;
  font-weight: 560;
}
.msg-ai{ background: transparent; border: none; max-width: min(900px, 94%); }
.msg-ai-stream{
  background: var(--msg-ai-bg);
  border: 1px solid var(--msg-ai-border);
  border-radius: 14px;
  padding: 12px 14px;
}
.msg-refs{
  margin: 0.35rem 0 0.80rem 0;
  padding: 0.06rem 0 0.80rem 0;
  border-left: none !important;
  outline: none !important;
  box-shadow: none !important;
}
.msg-refs::before,
.msg-refs::after{
  display: none !important;
  content: none !important;
}
.msg-refs div[data-testid="stVerticalBlockBorderWrapper"]{
  margin: 0.24rem 0 0.46rem 0 !important;
  border-radius: 16px !important;
  border: 1px solid color-mix(in srgb, var(--line) 76%, var(--blue-line)) !important;
  background:
    radial-gradient(110% 90% at 0% 0%, color-mix(in srgb, var(--blue-weak) 46%, transparent), transparent 62%),
    radial-gradient(90% 100% at 100% 0%, rgba(34,197,94,0.05), transparent 58%),
    linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00)),
    var(--panel) !important;
  box-shadow:
    0 10px 26px rgba(16,24,40,0.045),
    0 2px 6px rgba(16,24,40,0.025),
    inset 0 1px 0 rgba(255,255,255,0.03) !important;
  overflow: hidden !important;
}
.msg-refs div[data-testid="stVerticalBlockBorderWrapper"] > div{
  padding: 0.16rem 0.18rem 0.18rem 0.18rem !important;
}
.msg-refs details[data-testid="stExpander"]{
  background:
    radial-gradient(120% 90% at 0% 0%, color-mix(in srgb, var(--blue-weak) 55%, transparent), transparent 62%),
    var(--panel) !important;
  border: 1px solid color-mix(in srgb, var(--line) 82%, var(--blue-line)) !important;
  border-radius: 14px !important;
  box-shadow: 0 10px 24px rgba(16, 24, 40, 0.04), inset 0 1px 0 rgba(255,255,255,0.03);
}
.msg-refs details[data-testid="stExpander"] summary,
.msg-refs details[data-testid="stExpander"] summary *,
.msg-refs details[data-testid="stExpander"] summary p,
.msg-refs details[data-testid="stExpander"] summary span{
  color: var(--refs-title-text) !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--refs-title-text) !important;
}
.msg-refs [data-testid="stMarkdownContainer"] *,
.msg-refs .refbox,
.msg-refs .refbox *{
  color: var(--refs-body-text) !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--refs-body-text) !important;
}
.msg-refs details[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] *,
.msg-refs details[data-testid="stExpander"] summary p,
.msg-refs details[data-testid="stExpander"] summary span,
.msg-refs details[data-testid="stExpander"] summary div{
  color: var(--refs-title-text) !important;
  -webkit-text-fill-color: var(--refs-title-text) !important;
  opacity: 1 !important;
}
.msg-refs details[data-testid="stExpander"] summary svg,
.msg-refs details[data-testid="stExpander"] summary path{
  fill: var(--refs-title-text) !important;
  stroke: var(--refs-title-text) !important;
}
.msg-refs .ref-rank{
  color: var(--accent) !important;
  -webkit-text-fill-color: var(--accent) !important;
}
.msg-refs .ref-source,
.msg-refs .ref-item-sub{
  color: var(--text-main) !important;
  -webkit-text-fill-color: var(--text-main) !important;
}
.msg-refs .ref-item-sub{
  color: var(--text-soft) !important;
  -webkit-text-fill-color: var(--text-soft) !important;
}
.msg-refs .ref-chip{
  color: var(--text-soft) !important;
  -webkit-text-fill-color: var(--text-soft) !important;
}
.msg-refs .ref-score-hi{
  color: #22c55e !important;
  -webkit-text-fill-color: #22c55e !important;
}
.msg-refs .ref-score-mid{
  color: #f59e0b !important;
  -webkit-text-fill-color: #f59e0b !important;
}
.msg-refs .ref-score-low{
  color: var(--text-soft) !important;
  -webkit-text-fill-color: var(--text-soft) !important;
}
button.kb-ref-action-btn,
.msg-refs div[data-testid="stButton"] > button,
div[data-testid="stHorizontalBlock"]:has(.kb-ref-header-block) div[data-testid="stButton"] > button{
  min-height: 2.02rem !important;
  height: 2.02rem !important;
  border-radius: 6px !important;
  border: none !important;
  border-color: transparent !important;
  border-style: none !important;
  outline: none !important;
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--panel) 95%, var(--blue-weak) 5%),
      color-mix(in srgb, var(--panel) 92%, var(--blue-weak) 8%)
    ) !important;
  backdrop-filter: saturate(120%) blur(8px) !important;
  -webkit-backdrop-filter: saturate(120%) blur(8px) !important;
  color: var(--text-main) !important;
  font-weight: 665 !important;
  box-shadow:
    none !important;
  padding: 0 0.86rem !important;
  letter-spacing: 0.008em !important;
  transition: background 150ms ease, color 150ms ease !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: clip !important;
  min-width: 0 !important;
  font-size: clamp(0.5rem, 1.4vw, 0.875rem) !important;
}
button.kb-ref-action-btn p,
button.kb-ref-action-btn span,
.msg-refs div[data-testid="stButton"] > button p,
.msg-refs div[data-testid="stButton"] > button span,
div[data-testid="stHorizontalBlock"]:has(.kb-ref-header-block) div[data-testid="stButton"] > button p,
div[data-testid="stHorizontalBlock"]:has(.kb-ref-header-block) div[data-testid="stButton"] > button span{
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: clip !important;
  font-size: inherit !important;
}
button.kb-ref-action-btn{
  position: relative !important;
  overflow: hidden !important;
  isolation: isolate !important;
}
button.kb-ref-action-btn::before{
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
}
button.kb-ref-action-btn::after{
  content: "" !important;
  position: absolute !important;
  top: -128% !important;
  left: -42% !important;
  width: 40% !important;
  height: 356% !important;
  pointer-events: none !important;
  z-index: 0 !important;
  background: linear-gradient(
    90deg,
    rgba(120, 176, 255, 0.00) 0%,
    rgba(120, 176, 255, 0.22) 48%,
    rgba(120, 176, 255, 0.00) 100%
  ) !important;
  transform: translate3d(-165%, 0, 0) rotate(24deg) !important;
  opacity: 0 !important;
}
button.kb-ref-action-btn:hover::after{
  opacity: 0.34 !important;
  animation: kb-ref-glass-sheen 760ms cubic-bezier(0.22, 0.61, 0.36, 1.0) 1 !important;
}
button.kb-ref-action-btn > *{
  position: relative !important;
  z-index: 1 !important;
}
@keyframes kb-ref-glass-sheen{
  0%{
    transform: translate3d(-165%, 0, 0) rotate(24deg);
  }
  100%{
    transform: translate3d(290%, 0, 0) rotate(24deg);
  }
}
button.kb-ref-action-btn:hover,
.msg-refs div[data-testid="stButton"] > button:hover,
div[data-testid="stHorizontalBlock"]:has(.kb-ref-header-block) div[data-testid="stButton"] > button:hover{
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--panel) 92%, var(--blue-weak) 8%),
      color-mix(in srgb, var(--panel) 88%, var(--blue-weak) 12%)
    ) !important;
  border: none !important;
  box-shadow: none !important;
  transform: none !important;
}
button.kb-ref-action-btn:focus-visible,
.msg-refs div[data-testid="stButton"] > button:focus-visible,
div[data-testid="stHorizontalBlock"]:has(.kb-ref-header-block) div[data-testid="stButton"] > button:focus-visible{
  outline: none !important;
  border: none !important;
  box-shadow: none !important;
}
button.kb-ref-action-btn:active,
.msg-refs div[data-testid="stButton"] > button:active,
div[data-testid="stHorizontalBlock"]:has(.kb-ref-header-block) div[data-testid="stButton"] > button:active{
  border: none !important;
  transform: none !important;
  box-shadow: none !important;
}
button.kb-ref-action-btn:disabled,
.msg-refs div[data-testid="stButton"] > button:disabled,
div[data-testid="stHorizontalBlock"]:has(.kb-ref-header-block) div[data-testid="stButton"] > button:disabled{
  opacity: 0.54 !important;
  border: none !important;
  transform: none !important;
  box-shadow: none !important;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.00)),
    color-mix(in srgb, var(--panel) 92%, transparent) !important;
}
/* Hard fallback for refs Open/Cite buttons: main area 4-col row, columns 3/4. */
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] > button,
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(4) div[data-testid="stButton"] > button{
  border: none !important;
  border-color: transparent !important;
  border-style: none !important;
  border-radius: 6px !important;
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--panel) 95%, var(--blue-weak) 5%),
      color-mix(in srgb, var(--panel) 92%, var(--blue-weak) 8%)
    ) !important;
  backdrop-filter: saturate(120%) blur(8px) !important;
  -webkit-backdrop-filter: saturate(120%) blur(8px) !important;
  box-shadow: none !important;
  position: relative !important;
  overflow: hidden !important;
  isolation: isolate !important;
  white-space: nowrap !important;
  text-overflow: clip !important;
  min-width: 0 !important;
  font-size: clamp(0.5rem, 1.4vw, 0.875rem) !important;
}
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] > button p,
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] > button span,
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(4) div[data-testid="stButton"] > button p,
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(4) div[data-testid="stButton"] > button span{
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: clip !important;
  font-size: inherit !important;
}
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] > button::after,
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(4) div[data-testid="stButton"] > button::after{
  content: "" !important;
  position: absolute !important;
  top: -128% !important;
  left: -44% !important;
  width: 42% !important;
  height: 356% !important;
  pointer-events: none !important;
  z-index: 0 !important;
  background: linear-gradient(
    90deg,
    rgba(120, 176, 255, 0.00) 0%,
    rgba(120, 176, 255, 0.24) 48%,
    rgba(120, 176, 255, 0.00) 100%
  ) !important;
  transform: translate3d(-175%, 0, 0) rotate(24deg) !important;
  opacity: 0 !important;
}
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] > button:hover::after,
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(4) div[data-testid="stButton"] > button:hover::after{
  opacity: 0.34 !important;
  animation: kb-ref-glass-sheen 760ms cubic-bezier(0.22, 0.61, 0.36, 1.0) 1 !important;
}
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] > button:hover,
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(4) div[data-testid="stButton"] > button:hover{
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--panel) 92%, var(--blue-weak) 8%),
      color-mix(in srgb, var(--panel) 88%, var(--blue-weak) 12%)
    ) !important;
  transform: none !important;
}
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div[data-testid="stButton"] > button > *,
section[data-testid="stAppViewContainer"] div[data-testid="stMainBlockContainer"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(4) div[data-testid="stButton"] > button > *{
  position: relative !important;
  z-index: 1 !important;
}
/* Flatten reference panel visuals for a cleaner, less rounded layout. */
.msg-refs div[data-testid="stVerticalBlockBorderWrapper"],
.msg-refs details[data-testid="stExpander"],
.msg-refs .ref-rank,
.msg-refs .ref-chip,
.msg-refs .ref-score,
.msg-refs .kb-ref-heading-path,
.msg-refs .kb-ref-loc-chip,
.msg-refs .kb-ref-insight-card,
.msg-refs .kb-ref-insight-tag,
.msg-refs .kb-ref-metrics-row,
.msg-refs .kb-ref-metric-tag,
.msg-refs .kb-ref-guide-details,
.msg-refs .kb-ref-guide-caret,
.msg-refs .kb-ref-guide-block,
.msg-refs .kb-ref-guide-block-tag,
.msg-refs .kb-ref-focus-chip{
  border-radius: 0 !important;
}
/* De-emphasize borders in refs panel for a cleaner visual hierarchy. */
.msg-refs div[data-testid="stVerticalBlockBorderWrapper"]{
  border-color: color-mix(in srgb, var(--line) 16%, transparent) !important;
  box-shadow: none !important;
  background: color-mix(in srgb, var(--panel) 98%, transparent) !important;
}
.msg-refs details[data-testid="stExpander"]{
  border-color: color-mix(in srgb, var(--line) 14%, transparent) !important;
  box-shadow: none !important;
  background: color-mix(in srgb, var(--panel) 99%, transparent) !important;
}
.msg-refs .kb-ref-insight-card,
.msg-refs .kb-ref-metrics-row,
.msg-refs .kb-ref-guide-details,
.msg-refs .kb-ref-guide-block{
  border-color: color-mix(in srgb, var(--line) 12%, transparent) !important;
  box-shadow: none !important;
  background: color-mix(in srgb, var(--panel) 99%, transparent) !important;
}
.msg-refs .kb-ref-loc-chip{
  border-color: color-mix(in srgb, var(--line) 15%, transparent) !important;
  background: color-mix(in srgb, var(--panel) 96%, transparent) !important;
}
.msg-refs .kb-ref-heading-path{
  border: none !important;
  background: transparent !important;
  padding: 0 !important;
  font-family: inherit !important;
}
.kb-ref-item-gap{
  height: 0.42rem;
}
.msg-refs .kb-ref-insight-card::before{
  opacity: 0.45 !important;
}
body.kb-cite-dragging{
  user-select: none !important;
  cursor: grabbing !important;
}
.kb-cite-pop{
  position: fixed;
  z-index: 10080;
  max-width: min(500px, calc(100vw - 24px));
  min-width: min(300px, calc(100vw - 24px));
  background: var(--panel);
  color: var(--text-main) !important;
  border: 1px solid var(--line);
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.24);
  padding: 10px 12px;
}
.kb-cite-pop-head{
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding-right: 66px;
  cursor: grab;
}
.kb-cite-pop-head:active{
  cursor: grabbing;
}
.kb-cite-pop-title{
  font-size: 0.84rem;
  font-weight: 700;
  color: var(--accent) !important;
  margin-bottom: 6px;
}
.kb-cite-pop-main{
  font-size: 0.92rem;
  line-height: 1.45;
  color: var(--text-main) !important;
}
.kb-cite-pop-sub{
  font-size: 0.80rem;
  line-height: 1.35;
  color: var(--text-soft) !important;
  margin-top: 6px;
}
.kb-cite-pop-doi{
  margin-top: 7px;
  font-size: 0.84rem;
}
.kb-cite-pop-doi a{
  color: var(--accent) !important;
  text-decoration: underline;
}
.kb-cite-pop-actions{
  margin-top: 10px;
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
.kb-cite-pop-add,
.kb-cite-pop-open-shelf{
  border: 1px solid var(--btn-border);
  border-radius: 8px;
  background: var(--btn-bg);
  color: var(--btn-text);
  height: 30px;
  padding: 0 10px;
  font-size: 0.80rem;
  font-weight: 650;
  cursor: pointer;
}
.kb-cite-pop-add:hover,
.kb-cite-pop-open-shelf:hover{
  background: var(--btn-hover);
}
.kb-cite-pop-add{
  background: var(--accent);
  border-color: transparent;
  color: #ffffff;
}
.kb-cite-pop-add:hover{
  filter: brightness(0.94);
}
.kb-cite-pop-add.kb-added{
  background: color-mix(in srgb, var(--accent) 16%, var(--btn-bg));
  border-color: color-mix(in srgb, var(--accent) 48%, var(--btn-border));
  color: var(--text-main);
}
.kb-cite-pop-add.kb-added:hover{
  filter: none;
  background: color-mix(in srgb, var(--accent) 24%, var(--btn-bg));
}
.kb-cite-pop-close{
  position: absolute;
  top: 0;
  right: 0;
  border: 1px solid var(--btn-border);
  border-radius: 8px;
  background: var(--btn-bg);
  color: var(--btn-text);
  width: 24px;
  height: 24px;
  line-height: 22px;
  text-align: center;
  font-size: 14px;
  cursor: pointer;
}
.kb-cite-shelf{
  position: fixed;
  z-index: 10070;
  top: 66px;
  right: 12px;
  width: min(390px, calc(100vw - 20px));
  max-height: calc(100vh - 88px);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  box-shadow: 0 12px 34px rgba(0,0,0,0.20);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transform: translateX(calc(100% + 20px));
  opacity: 0;
  pointer-events: none;
  transition: transform 0.20s ease, opacity 0.20s ease;
}
.kb-cite-shelf.kb-open{
  transform: translateX(0);
  opacity: 1;
  pointer-events: auto;
}
.kb-cite-shelf-head{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
}
.kb-cite-shelf-title{
  font-size: 0.88rem;
  font-weight: 760;
  color: var(--text-main) !important;
}
.kb-cite-shelf-meta{
  font-size: 0.78rem;
  color: var(--text-soft) !important;
}
.kb-cite-shelf-head-actions{
  display: flex;
  align-items: center;
  gap: 6px;
}
.kb-cite-shelf-btn{
  border: 1px solid var(--btn-border);
  border-radius: 8px;
  background: var(--btn-bg);
  color: var(--btn-text);
  height: 26px;
  padding: 0 8px;
  font-size: 0.76rem;
  font-weight: 650;
  cursor: pointer;
}
.kb-cite-shelf-btn:hover{
  background: var(--btn-hover);
}
.kb-cite-shelf-list{
  padding: 10px 10px 12px;
  overflow: auto;
  min-height: 80px;
  max-height: calc(100vh - 152px);
}
.kb-cite-shelf-empty{
  font-size: 0.82rem;
  color: var(--text-soft) !important;
  padding: 10px 4px;
}
.kb-cite-shelf-item{
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px 10px;
  margin-bottom: 8px;
  background: color-mix(in srgb, var(--panel) 86%, var(--bg));
}
.kb-cite-shelf-item.kb-flash{
  animation: kb-shelf-flash 1.25s ease-out 1;
}
@keyframes kb-shelf-flash{
  0%{
    box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent) 38%, transparent);
    border-color: color-mix(in srgb, var(--accent) 65%, var(--line));
    background: color-mix(in srgb, var(--accent) 16%, var(--panel));
  }
  100%{
    box-shadow: 0 0 0 0 transparent;
    border-color: var(--line);
    background: color-mix(in srgb, var(--panel) 86%, var(--bg));
  }
}
.kb-cite-shelf-item-title{
  font-size: 0.84rem;
  font-weight: 670;
  line-height: 1.35;
  color: var(--text-main) !important;
}
.kb-cite-shelf-item-sub{
  margin-top: 4px;
  font-size: 0.78rem;
  color: var(--text-soft) !important;
  line-height: 1.34;
}
.kb-cite-shelf-item-links{
  margin-top: 6px;
  font-size: 0.78rem;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.kb-cite-shelf-item-links a{
  color: var(--accent) !important;
  text-decoration: underline;
}
.kb-cite-shelf-toggle{
  position: fixed;
  z-index: 10065;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  border: 1px solid var(--btn-border);
  border-radius: 999px;
  background: var(--btn-bg);
  color: var(--btn-text);
  height: 38px;
  padding: 0 12px;
  font-size: 0.80rem;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 8px 20px rgba(0,0,0,0.14);
}
.kb-cite-shelf.kb-open + .kb-cite-shelf-toggle{
  opacity: 0;
  pointer-events: none;
}
.kb-cite-shelf-toggle:hover{
  background: var(--btn-hover);
}
.kb-inpaper-cite{
  display: inline;
  font-size: inherit !important;
  font-weight: 500;
  line-height: inherit;
  vertical-align: baseline !important;
  text-decoration: none !important;
  color: var(--accent) !important;
}
.kb-inpaper-cite:hover{
  text-decoration: underline !important;
}
a[href^="#kb-cite-"]{
  display: inline;
  font-size: inherit !important;
  font-weight: 500;
  line-height: inherit;
  vertical-align: baseline !important;
  text-decoration: none !important;
  color: var(--accent) !important;
}
a[href^="#kb-cite-"]:hover{
  text-decoration: underline !important;
}
a[href^="#kb-cite-"]::before{ content: "["; }
a[href^="#kb-cite-"]::after{ content: "]"; }

.snipbox{
  background: var(--snip-bg);
  border: 1px solid var(--snip-border);
  border-radius: 12px;
  padding: 10px 12px;
  margin: 0.35rem 0 0.55rem 0;
}
.snipbox pre{
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--snip-text) !important;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
}
.snipquote{ border-left: 3px solid var(--snip-quote-border); background: var(--snip-quote-bg); border-radius: 10px; padding: 9px 11px; margin: 0.15rem 0 0.45rem 0; }
.snipquote .snipquote-title{ font-size: 0.78rem; color: var(--muted) !important; margin: 0 0 0.25rem 0; }
.snipquote .snipquote-body{ font-size: 0.88rem; line-height: 1.42; color: var(--text-main) !important; }
.snipquote mark{ background: var(--snip-mark-bg); color: var(--snip-mark-text); }

.kb-notice{
  font-size: 0.84rem;
  color: var(--notice-text) !important;
  background: var(--notice-bg);
  border: 1px solid var(--notice-border);
  border-radius: 10px;
  padding: 0.35rem 0.55rem;
  margin: 0 0 0.55rem 0;
}

.stTextArea{ position: relative; }
.stTextArea::after,
div[data-testid="stTextArea"]::after{
  content: "Ctrl+Enter 发送";
  position: absolute;
  right: 14px;
  bottom: 10px;
  font-size: 12px;
  color: var(--hint-text);
  pointer-events: none;
}

/* Chat input dock: single source. chat_dock_runtime.js adds .kb-input-dock class; only these rules apply. */
.kb-input-dock{
  position: fixed !important;
  bottom: max(10px, env(safe-area-inset-bottom, 0px)) !important;
  z-index: 40 !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  width: min(var(--content-max), calc(100vw - 1.6rem)) !important;
  box-sizing: border-box;
  background: var(--dock-bg) !important;
  border: 1px solid var(--dock-border) !important;
  border-radius: 20px;
  padding: 0.34rem 0.46rem 0.34rem 0.46rem;
  box-shadow: var(--dock-shadow);
  backdrop-filter: saturate(118%) blur(8px) !important;
  -webkit-backdrop-filter: saturate(118%) blur(8px) !important;
  margin: 0 !important;
  height: auto !important;
  min-height: 0 !important;
  max-height: none !important;
  flex: none !important;
  display: block !important;
  overflow: hidden !important;
  isolation: isolate !important;
}
.kb-input-dock > div,
.kb-input-dock form,
.kb-input-dock div[data-testid="stForm"]{
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  border-color: transparent !important;
  height: auto !important;
  min-height: 0 !important;
  max-height: none !important;
  flex: none !important;
}
.kb-input-dock form,
.kb-input-dock div[data-testid="stForm"]{
  position: relative !important;
}
.kb-input-dock textarea,
.kb-input-dock div[data-testid="stTextArea"] textarea{
  border-color: var(--input-border) !important;
}
.kb-input-dock div[data-testid="stTextArea"] [data-baseweb="textarea"],
.kb-input-dock div[data-testid="stTextArea"] [data-baseweb="textarea"] > div,
.kb-input-dock div[data-testid="stTextArea"] [data-baseweb="base-input"],
.kb-input-dock div[data-testid="stTextArea"] [data-baseweb="base-input"] > div{
  outline: none !important;
  box-shadow: none !important;
}
.kb-input-dock div[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within,
.kb-input-dock div[data-testid="stTextArea"] [data-baseweb="textarea"][aria-invalid="true"],
.kb-input-dock div[data-testid="stTextArea"] [data-baseweb="textarea"][data-invalid="true"],
.kb-input-dock div[data-testid="stTextArea"] [data-baseweb="base-input"]:focus-within,
.kb-input-dock div[data-testid="stTextArea"] [data-baseweb="base-input"][aria-invalid="true"],
.kb-input-dock div[data-testid="stTextArea"] [data-baseweb="base-input"][data-invalid="true"]{
  border-color: transparent !important;
  outline: none !important;
  box-shadow: none !important;
}
.kb-input-dock textarea:focus,
.kb-input-dock div[data-testid="stTextArea"] textarea:focus{
  border-color: var(--blue-line) !important;
  box-shadow: 0 0 0 1px var(--blue-weak) !important;
}
.kb-input-dock.kb-dock-drop-active div[data-testid="stTextArea"] textarea,
.kb-input-dock.kb-dock-drop-active textarea{
  border-color: color-mix(in srgb, var(--blue-line) 82%, transparent) !important;
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--blue-weak) 52%, transparent) !important;
  background: color-mix(in srgb, var(--panel) 94%, #eaf3ff 6%) !important;
}
.kb-input-dock.kb-dock-drop-active::before{
  color: color-mix(in srgb, var(--text-soft) 72%, var(--blue-line) 28%) !important;
}
.kb-input-dock.kb-dock-drop-active div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]{
  background: color-mix(in srgb, var(--btn-hover) 58%, var(--panel) 42%) !important;
}
.kb-input-dock textarea:invalid,
.kb-input-dock textarea:user-invalid,
.kb-input-dock div[data-testid="stTextArea"] textarea:invalid,
.kb-input-dock div[data-testid="stTextArea"] textarea:user-invalid{
  border-color: color-mix(in srgb, var(--input-border) 92%, var(--line)) !important;
  box-shadow: none !important;
}
.kb-input-dock textarea:focus:invalid,
.kb-input-dock textarea:focus:user-invalid,
.kb-input-dock div[data-testid="stTextArea"] textarea:focus:invalid,
.kb-input-dock div[data-testid="stTextArea"] textarea:focus:user-invalid{
  border-color: var(--blue-line) !important;
  box-shadow: 0 0 0 1px var(--blue-weak) !important;
}
html[data-theme="dark"] .kb-input-dock,
body[data-theme="dark"] .kb-input-dock{
  background: var(--dock-bg) !important;
  border-color: var(--dock-border) !important;
}
body.kb-resizing .kb-input-dock,
body.kb-resizing .kb-input-dock *{
  border-color: var(--dock-border) !important;
}
.kb-input-dock.kb-dock-positioned{ max-width: none !important; }
.kb-input-dock div[data-testid="stForm"]{ margin-bottom: 0 !important; }
.kb-input-dock .kb-dock-action-layer{
  position: absolute !important;
  inset: 0 !important;
  z-index: 9 !important;
  pointer-events: none !important;
}
.kb-input-dock .kb-dock-send-anchor{
  position: absolute !important;
  right: 0.98rem !important;
  bottom: 1.20rem !important;
  width: 36px !important;
  height: 36px !important;
  pointer-events: none !important;
}
.kb-input-dock .kb-dock-send-anchor::before{
  content: none;
}
.kb-input-dock .kb-dock-stop-anchor{
  position: absolute !important;
  right: 3.55rem !important;
  bottom: 1.23rem !important;
  width: 28px !important;
  height: 28px !important;
  pointer-events: none !important;
}
.kb-input-dock .kb-dock-action-layer .kb-dock-send-wrap,
.kb-input-dock .kb-dock-action-layer .kb-dock-stop-wrap{
  position: static !important;
  left: auto !important;
  right: auto !important;
  top: auto !important;
  bottom: auto !important;
  margin: 0 !important;
  width: 100% !important;
  min-width: 0 !important;
  height: 100% !important;
  min-height: 0 !important;
  display: block !important;
  pointer-events: auto !important;
}
.kb-input-dock::before{
  content: "Ask anything... (searches Markdown first)";
  display: block;
  font-size: 0.75rem;
  color: color-mix(in srgb, var(--text-soft) 86%, var(--text-main) 14%);
  opacity: 0.96;
  margin: 0 0 0.20rem 0.18rem;
}
.kb-input-dock::after{
  content: "π-zaya can make mistakes. check important info";
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  bottom: 0.60rem;
  font-size: 11px;
  color: var(--hint-text);
  opacity: 0.82;
  line-height: 1;
  white-space: nowrap;
  text-align: center;
  pointer-events: none;
  z-index: 4;
}
html[data-theme="light"] .kb-input-dock,
body[data-theme="light"] .kb-input-dock{
  background:
    linear-gradient(180deg,
      rgba(255,255,255,0.70) 0%,
      rgba(247,249,252,0.76) 22%,
      rgba(239,243,248,0.82) 100%) !important;
  border-color: rgba(174, 184, 196, 0.32) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.72),
    inset 0 -1px 0 rgba(255,255,255,0.22),
    0 1px 2px rgba(15,23,42,0.05),
    0 10px 26px rgba(15,23,42,0.07) !important;
}
html[data-theme="light"] .kb-input-dock::before,
body[data-theme="light"] .kb-input-dock::before{
  text-shadow: 0 1px 0 rgba(255,255,255,0.45);
}
html[data-theme="light"] .kb-input-dock::after,
body[data-theme="light"] .kb-input-dock::after{
  color: color-mix(in srgb, var(--hint-text) 74%, #5f6d7c 26%);
  opacity: 0.88;
  text-shadow: 0 1px 0 rgba(255,255,255,0.38);
}
.kb-input-dock div[data-testid="stTextArea"] label{ display: none !important; }
.kb-input-dock div[data-testid="stTextArea"]{ position: relative !important; margin-bottom: 0 !important; }
.kb-input-dock .kb-dock-attach-strip{
  position: absolute !important;
  left: 0.72rem !important;
  right: 4.65rem !important;
  top: 0.58rem !important;
  z-index: 6 !important;
  display: none;
  align-items: center;
  gap: 0.42rem;
  min-height: 34px;
  max-height: 38px;
  overflow-x: auto;
  overflow-y: hidden;
  scrollbar-width: none;
  pointer-events: none;
}
.kb-input-dock .kb-dock-attach-strip::-webkit-scrollbar{ display: none; }
.kb-input-dock .kb-dock-attach-item{
  flex: 0 0 auto;
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  overflow: hidden;
  background: color-mix(in srgb, var(--panel) 88%, white 12%);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--line) 82%, transparent);
  pointer-events: auto;
  user-select: none;
}
.kb-input-dock .kb-dock-attach-item:hover{
  box-shadow:
    inset 0 0 0 1px color-mix(in srgb, var(--line) 92%, transparent),
    0 2px 10px rgba(15,23,42,0.08);
}
.kb-input-dock .kb-dock-attach-item.is-image{
  width: 34px;
  height: 34px;
  cursor: zoom-in;
}
.kb-input-dock .kb-dock-attach-thumb{
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.kb-input-dock .kb-dock-attach-item.is-file{
  max-width: 168px;
  height: 30px;
  padding: 0 1.22rem 0 0.50rem;
  gap: 0.34rem;
  border-radius: 999px;
}
.kb-input-dock .kb-dock-attach-remove{
  position: absolute;
  top: 50%;
  right: 0.18rem;
  transform: translateY(-50%);
  width: 16px;
  height: 16px;
  border: none !important;
  border-radius: 999px;
  margin: 0;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  line-height: 1;
  color: color-mix(in srgb, var(--text-main) 80%, var(--text-soft) 20%);
  background: color-mix(in srgb, var(--panel) 86%, rgba(255,255,255,0.45) 14%);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--line) 82%, transparent);
  opacity: 0;
  pointer-events: auto;
  cursor: pointer;
  transition: opacity .14s ease, background-color .14s ease, color .14s ease;
  z-index: 2;
}
.kb-input-dock .kb-dock-attach-item.is-image .kb-dock-attach-remove{
  top: 0.14rem;
  right: 0.14rem;
  transform: none;
  width: 15px;
  height: 15px;
  font-size: 11px;
  background: rgba(17, 24, 39, 0.70);
  color: rgba(255,255,255,0.94);
  box-shadow: none;
}
.kb-input-dock .kb-dock-attach-item:hover .kb-dock-attach-remove,
.kb-input-dock .kb-dock-attach-item:focus-within .kb-dock-attach-remove{
  opacity: 1;
}
.kb-input-dock .kb-dock-attach-remove:hover{
  background: color-mix(in srgb, var(--btn-hover) 70%, var(--panel) 30%);
  color: var(--text-main);
}
.kb-input-dock .kb-dock-attach-item.is-image .kb-dock-attach-remove:hover{
  background: rgba(17, 24, 39, 0.90);
  color: #fff;
}
.kb-input-dock .kb-dock-attach-fileicon{
  font-size: 0.58rem;
  line-height: 1;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: color-mix(in srgb, var(--text-soft) 56%, var(--text-main) 44%);
}
.kb-input-dock .kb-dock-attach-label{
  max-width: 116px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.66rem;
  line-height: 1;
  color: color-mix(in srgb, var(--text-main) 92%, var(--text-soft) 8%);
}
.kb-input-dock .kb-dock-attach-more{
  flex: 0 0 auto;
  height: 28px;
  min-width: 28px;
  padding: 0 0.45rem;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.66rem;
  line-height: 1;
  color: color-mix(in srgb, var(--text-soft) 74%, var(--text-main) 26%);
  background: color-mix(in srgb, var(--panel) 84%, transparent);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--line) 76%, transparent);
  pointer-events: auto;
}
.kb-dock-img-preview{
  position: fixed;
  inset: 0;
  z-index: 2147482900;
  display: none;
}
.kb-dock-img-preview[aria-hidden="true"]{
  display: none;
}
.kb-dock-img-preview.is-open{
  display: block;
}
.kb-dock-img-preview-backdrop{
  position: absolute;
  inset: 0;
  background: rgba(2, 6, 23, 0.56);
  backdrop-filter: blur(3px);
}
.kb-dock-img-preview-dialog{
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  width: min(92vw, 980px);
  max-height: min(86vh, 920px);
  border-radius: 14px;
  padding: 0.7rem 0.7rem 0.55rem;
  background: color-mix(in srgb, var(--panel) 92%, rgba(255,255,255,0.08) 8%);
  box-shadow:
    0 24px 70px rgba(15,23,42,0.28),
    inset 0 0 0 1px color-mix(in srgb, var(--line) 82%, transparent);
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}
.kb-dock-img-preview-img{
  width: 100%;
  max-height: calc(min(86vh, 920px) - 74px);
  object-fit: contain;
  border-radius: 10px;
  background: rgba(0,0,0,0.10);
}
.kb-dock-img-preview-caption{
  min-height: 1rem;
  padding: 0 0.1rem;
  font-size: 0.78rem;
  line-height: 1.25;
  color: color-mix(in srgb, var(--text-soft) 62%, var(--text-main) 38%);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.kb-dock-img-preview-close{
  position: absolute;
  top: 0.42rem;
  right: 0.42rem;
  width: 28px;
  height: 28px;
  border: none !important;
  border-radius: 999px;
  margin: 0;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  line-height: 1;
  color: rgba(255,255,255,0.92);
  background: rgba(17, 24, 39, 0.54);
  cursor: pointer;
}
.kb-dock-img-preview-close:hover{
  background: rgba(17, 24, 39, 0.82);
}
.kb-input-dock.kb-dock-has-attachments textarea,
.kb-input-dock.kb-dock-has-attachments div[data-testid="stTextArea"] textarea{
  padding-top: 3.20rem !important;
}
.kb-input-dock div[data-testid="stTextArea"]::before{
  content: "Ctrl+Enter";
  position: absolute;
  right: 4.15rem;
  bottom: 0.80rem;
  font-size: 11px;
  line-height: 1;
  color: var(--hint-text);
  opacity: 0.82;
  white-space: nowrap;
  pointer-events: none;
  z-index: 6;
}
.kb-input-dock div[data-testid="stTextArea"],
.kb-input-dock div[data-testid="stTextArea"] *{
  filter: none !important;
  opacity: 1 !important;
}
.kb-input-dock div[data-testid="stTextArea"] [data-testid="InputInstructions"]{ display: none !important; }
.kb-input-dock .stTextArea::after,
.kb-input-dock div[data-testid="stTextArea"]::after{ content: none !important; display: none !important; }
.kb-input-dock textarea{
  min-height: 110px !important;
  border-radius: 14px !important;
  padding-right: 4.85rem !important;
  padding-bottom: 3.05rem !important;
  background: color-mix(in srgb, var(--panel) 97%, white 3%) !important;
  color: var(--text-main) !important;
  border-color: color-mix(in srgb, var(--input-border) 92%, var(--line)) !important;
  opacity: 1 !important;
}
.kb-input-dock textarea::placeholder{
  color: color-mix(in srgb, var(--text-soft) 82%, var(--muted) 18%) !important;
  opacity: 0.88 !important;
}
.kb-input-dock div[data-testid="stFileUploader"]{
  position: absolute !important;
  left: 0.98rem !important;
  bottom: 1.20rem !important;
  z-index: 6 !important;
  width: 104px !important;
  min-width: 104px !important;
  max-width: 104px !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
}
.kb-input-dock div[data-testid="stFileUploader"] > label{
  display: none !important;
}
.kb-input-dock div[data-testid="stFileUploader"] > div{
  margin: 0 !important;
}
.kb-input-dock div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]{
  min-height: 32px !important;
  height: 32px !important;
  padding: 0 !important;
  border-radius: 999px !important;
  border: none !important;
  background: color-mix(in srgb, var(--panel) 94%, transparent) !important;
  box-shadow: none !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  overflow: hidden !important;
}
.kb-input-dock div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover{
  background: color-mix(in srgb, var(--btn-hover) 62%, var(--panel) 38%) !important;
  border-color: transparent !important;
}
.kb-input-dock div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] svg,
.kb-input-dock div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] small,
.kb-input-dock div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] p,
.kb-input-dock div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"]{
  display: none !important;
}
.kb-input-dock div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button{
  all: unset !important;
  box-sizing: border-box !important;
  width: 100% !important;
  height: 100% !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
  color: transparent !important;
  font-size: 0 !important;
  line-height: 0 !important;
  position: relative !important;
}
.kb-input-dock div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button::before{
  content: "+ Add files" !important;
  color: color-mix(in srgb, var(--text-main) 94%, var(--text-soft) 6%) !important;
  font-size: 0.70rem !important;
  line-height: 1 !important;
  font-weight: 560 !important;
  letter-spacing: 0.01em !important;
}
.kb-input-dock div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div{
  width: 100% !important;
  height: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
}
/* Hide uploaded file list rows in the dock to keep composer compact; upload feedback is shown as caption. */
.kb-input-dock div[data-testid="stFileUploader"] ul,
.kb-input-dock div[data-testid="stFileUploader"] [data-testid*="stFileUploaderFile"],
.kb-input-dock div[data-testid="stFileUploader"] [data-testid*="FileUploaderFile"]{
  display: none !important;
}
.kb-input-dock div[data-testid="stFormSubmitButton"]{
  display: flex !important;
  justify-content: flex-end !important;
  margin: 0 !important;
  min-height: 0 !important;
}
.kb-input-dock div[data-testid="stFormSubmitButton"] > button{
  width: 40px !important;
  min-width: 40px !important;
  height: 40px !important;
  min-height: 40px !important;
  border-radius: 999px !important;
  padding: 0 !important;
}
.kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap,
.kb-input-dock .kb-dock-send-wrap{
  position: absolute !important;
  right: 0.98rem !important;
  bottom: 1.20rem !important;
  top: auto !important;
  left: auto !important;
  z-index: 8 !important;
  width: 36px !important;
  min-width: 36px !important;
  height: 36px !important;
  min-height: 36px !important;
  margin: 0 !important;
  display: block !important;
}
.kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap > button.kb-dock-send-btn,
.kb-input-dock .kb-dock-send-wrap button.kb-dock-send-btn{
  all: unset;
  box-sizing: border-box !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 36px !important;
  min-width: 36px !important;
  height: 36px !important;
  min-height: 36px !important;
  border-radius: 999px !important;
  border: none !important;
  background: linear-gradient(180deg, #2d333d 0%, #262b33 100%) !important;
  color: #ffffff !important;
  font-size: 1.02rem !important;
  line-height: 1 !important;
  font-weight: 800 !important;
  cursor: pointer !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.10),
    0 1px 2px rgba(0, 0, 0, 0.18),
    0 6px 14px rgba(0, 0, 0, 0.10) !important;
  transition: transform 120ms ease, background 120ms ease, box-shadow 120ms ease !important;
}
.kb-input-dock button.kb-dock-send-btn{
  position: absolute !important;
  right: 0 !important;
  bottom: 0 !important;
  left: auto !important;
  top: auto !important;
  margin: 0 !important;
}
.kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap > button.kb-dock-send-btn:hover,
.kb-input-dock .kb-dock-send-wrap button.kb-dock-send-btn:hover{
  background: linear-gradient(180deg, #353c47 0%, #2d333d 100%) !important;
  transform: translateY(-1px) !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.12),
    0 2px 4px rgba(0, 0, 0, 0.20),
    0 8px 18px rgba(0, 0, 0, 0.12) !important;
}
.kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap > button.kb-dock-send-btn:active,
.kb-input-dock .kb-dock-send-wrap button.kb-dock-send-btn:active{
  background: #242a32 !important;
  transform: translateY(0) !important;
}
.kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap > button.kb-dock-send-btn,
.kb-input-dock .kb-dock-send-wrap button.kb-dock-send-btn{
  color: #ffffff !important;
}
html[data-theme="light"] .kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap > button.kb-dock-send-btn,
html[data-theme="light"] .kb-input-dock .kb-dock-send-wrap button.kb-dock-send-btn,
body[data-theme="light"] .kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap > button.kb-dock-send-btn,
body[data-theme="light"] .kb-input-dock .kb-dock-send-wrap button.kb-dock-send-btn{
  border: none !important;
  background: linear-gradient(180deg, #f8fafc 0%, #eef2f6 100%) !important;
  color: #28313d !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.95),
    0 1px 2px rgba(16, 24, 40, 0.06),
    0 5px 12px rgba(16, 24, 40, 0.08) !important;
}
html[data-theme="light"] .kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap > button.kb-dock-send-btn:hover,
html[data-theme="light"] .kb-input-dock .kb-dock-send-wrap button.kb-dock-send-btn:hover,
body[data-theme="light"] .kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap > button.kb-dock-send-btn:hover,
body[data-theme="light"] .kb-input-dock .kb-dock-send-wrap button.kb-dock-send-btn:hover{
  background: linear-gradient(180deg, #ffffff 0%, #f1f4f8 100%) !important;
  color: #1f2733 !important;
}
html[data-theme="light"] .kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap > button.kb-dock-send-btn:active,
html[data-theme="light"] .kb-input-dock .kb-dock-send-wrap button.kb-dock-send-btn:active,
body[data-theme="light"] .kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap > button.kb-dock-send-btn:active,
body[data-theme="light"] .kb-input-dock .kb-dock-send-wrap button.kb-dock-send-btn:active{
  background: #e9edf2 !important;
  color: #1f2733 !important;
}
.kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-send-wrap > button.kb-dock-send-btn:disabled,
.kb-input-dock .kb-dock-send-wrap button.kb-dock-send-btn:disabled{
  opacity: 0.56 !important;
  box-shadow: none !important;
  cursor: default !important;
}
.kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-stop-wrap,
.kb-input-dock .kb-dock-stop-wrap{
  position: absolute !important;
  right: 3.55rem !important;
  bottom: 1.23rem !important;
  top: auto !important;
  left: auto !important;
  z-index: 7 !important;
  width: 28px !important;
  min-width: 28px !important;
  height: 28px !important;
  min-height: 28px !important;
  margin: 0 !important;
  display: block !important;
}
.kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-stop-wrap > button.kb-dock-stop-btn,
.kb-input-dock .kb-dock-stop-wrap button.kb-dock-stop-btn{
  all: unset;
  box-sizing: border-box !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 28px !important;
  min-width: 28px !important;
  height: 28px !important;
  min-height: 28px !important;
  border-radius: 999px !important;
  border: none !important;
  background: rgba(255, 255, 255, 0.035) !important;
  color: var(--text-main) !important;
  font-size: 0.78rem !important;
  line-height: 1 !important;
  cursor: pointer !important;
}
.kb-input-dock button.kb-dock-stop-btn{
  position: absolute !important;
  right: 0 !important;
  bottom: 0 !important;
  left: auto !important;
  top: auto !important;
  margin: 0 !important;
}
.kb-input-dock div[data-testid="stFormSubmitButton"].kb-dock-stop-wrap > button.kb-dock-stop-btn:hover,
.kb-input-dock .kb-dock-stop-wrap button.kb-dock-stop-btn:hover{
  background: rgba(255, 255, 255, 0.07) !important;
  border-color: transparent !important;
}

.kb-copybar{
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: nowrap;
  margin: 6px 0 10px 0;
}
.kb-ai-inline-logo{
  height: 32px;
  width: auto;
  max-width: 88px;
  object-fit: contain;
  display: inline-block;
  flex: 0 0 auto;
}
.kb-ai-livebar{
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0.12rem 0 0.42rem 0;
}
.kb-ai-live-logo{
  height: 22px;
  width: auto;
  max-width: 60px;
  object-fit: contain;
  display: inline-block;
}
.kb-ai-live-pill{
  display: inline-flex;
  align-items: center;
  padding: 0.14rem 0.52rem;
  border-radius: 999px;
  border: 1px solid var(--blue-line);
  background: var(--blue-weak);
  color: var(--text-main) !important;
  font-size: 0.76rem;
  font-weight: 650;
}
.kb-ai-live-stage{
  color: var(--text-soft) !important;
  font-size: 0.82rem;
}
.kb-ai-live-dots{
  margin: 0.02rem 0 0.46rem 0.08rem;
  color: var(--muted) !important;
  font-size: 0.88rem;
  letter-spacing: 0.28em;
  user-select: none;
}
.kb-copybtn{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 32px;
  min-height: 32px;
  padding: 0 10px;
  border-radius: 10px;
  border: 1px solid var(--copy-btn-border);
  background: var(--copy-btn-bg);
  color: var(--copy-btn-text);
  font-weight: 600;
  font-size: 12px;
  white-space: nowrap;
  cursor: pointer;
}
.kb-copybtn:hover{ background: var(--btn-hover); border-color: var(--blue-line); }
.kb-copybtn:active{ background: var(--btn-active); border-color: var(--blue-line); }
.kb-codecopy{
  position: absolute;
  top: 10px;
  right: 10px;
  padding: 4px 8px;
  border-radius: 10px;
  border: 1px solid var(--copy-btn-border);
  background: var(--copy-btn-bg);
  color: var(--copy-btn-text);
  font-weight: 600;
  font-size: 12px;
  cursor: pointer;
  z-index: 2;
}
.kb-codecopy:hover{ background: var(--btn-hover); border-color: var(--blue-line); }
div[data-testid="stCodeBlock"]:not([data-kb-normalized="1"]) .kb-codecopy,
div[data-testid="stCode"]:not([data-kb-normalized="1"]) .kb-codecopy,
.stCodeBlock:not([data-kb-normalized="1"]) .kb-codecopy{
  display: none !important;
}
.kb-toast{
  position: fixed;
  right: 18px;
  bottom: 18px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid var(--toast-border);
  background: var(--toast-bg);
  color: var(--toast-text);
  font-weight: 600;
  font-size: 12px;
  opacity: 0;
  transform: translateY(6px);
  transition: opacity 120ms ease, transform 120ms ease;
  z-index: 999999;
  pointer-events: none;
}
.kb-toast.show{ opacity: 1; transform: translateY(0); }
html[data-theme="dark"] small,
html[data-theme="dark"] .stCaption,
html[data-theme="dark"] div[data-testid="stCaptionContainer"] *,
body[data-theme="dark"] small,
body[data-theme="dark"] .stCaption,
body[data-theme="dark"] div[data-testid="stCaptionContainer"] *{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] div[data-testid="stWidgetLabel"],
html[data-theme="dark"] div[data-testid="stWidgetLabel"] *,
body[data-theme="dark"] div[data-testid="stWidgetLabel"],
body[data-theme="dark"] div[data-testid="stWidgetLabel"] *{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] *,
html[data-theme="dark"] section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
html[data-theme="dark"] section[data-testid="stSidebar"] p,
html[data-theme="dark"] section[data-testid="stSidebar"] span,
body[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] *,
body[data-theme="dark"] section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
body[data-theme="dark"] section[data-testid="stSidebar"] p,
body[data-theme="dark"] section[data-testid="stSidebar"] span{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] div[data-testid="stRadio"] label,
html[data-theme="dark"] div[data-testid="stRadio"] label *,
html[data-theme="dark"] div[data-testid="stCheckbox"] label,
html[data-theme="dark"] div[data-testid="stCheckbox"] label *,
body[data-theme="dark"] div[data-testid="stRadio"] label,
body[data-theme="dark"] div[data-testid="stRadio"] label *,
body[data-theme="dark"] div[data-testid="stCheckbox"] label,
body[data-theme="dark"] div[data-testid="stCheckbox"] label *{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] div[data-testid="stSlider"] label,
html[data-theme="dark"] div[data-testid="stSlider"] label *,
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMax"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stThumbValue"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderValue"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-baseweb="slider"] *,
body[data-theme="dark"] div[data-testid="stSlider"] label,
body[data-theme="dark"] div[data-testid="stSlider"] label *,
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMax"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stThumbValue"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderValue"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-baseweb="slider"] *{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] details[data-testid="stExpander"] summary,
html[data-theme="dark"] details[data-testid="stExpander"] summary *,
body[data-theme="dark"] details[data-testid="stExpander"] summary,
body[data-theme="dark"] details[data-testid="stExpander"] summary *{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] [data-testid="stMarkdownContainer"] p,
html[data-theme="dark"] [data-testid="stMarkdownContainer"] li,
html[data-theme="dark"] [data-testid="stMarkdownContainer"] span,
body[data-theme="dark"] [data-testid="stMarkdownContainer"] p,
body[data-theme="dark"] [data-testid="stMarkdownContainer"] li,
body[data-theme="dark"] [data-testid="stMarkdownContainer"] span{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] .msg-meta,
html[data-theme="dark"] .refbox,
html[data-theme="dark"] .genbox,
html[data-theme="dark"] .chat-empty-state,
body[data-theme="dark"] .msg-meta,
body[data-theme="dark"] .refbox,
body[data-theme="dark"] .genbox,
body[data-theme="dark"] .chat-empty-state{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] *,
body[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] *{
  color: var(--text-soft) !important;
  fill: var(--text-soft) !important;
  stroke: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] [style*="color"],
body[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] [style*="color"]{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
/* IDE-like / OpenAI-like sidebar controls: low-contrast surfaces, crisp borders, restrained motion. */
section[data-testid="stSidebar"]{
  --kb-side-btn-bg: color-mix(in srgb, var(--sidebar-bg) 80%, var(--panel));
  --kb-side-btn-hover: color-mix(in srgb, var(--sidebar-bg) 62%, var(--panel));
  --kb-side-btn-active: color-mix(in srgb, var(--sidebar-bg) 54%, var(--panel));
  --kb-side-btn-border: color-mix(in srgb, var(--line) 88%, transparent);
  --kb-side-btn-border-hover: color-mix(in srgb, var(--blue-line) 28%, var(--line));
  --kb-side-ctrl-bg: color-mix(in srgb, var(--sidebar-bg) 76%, var(--panel));
  --kb-side-ctrl-border: color-mix(in srgb, var(--line) 90%, transparent);
  --kb-side-focus-ring: color-mix(in srgb, var(--accent) 22%, transparent);
}
section[data-testid="stSidebar"] div[data-testid="stButton"]{
  margin: 0.12rem 0 !important;
}
section[data-testid="stSidebar"] div.stButton > button{
  min-height: 36px !important;
  height: auto !important;
  padding: 0.36rem 0.72rem !important;
  border-radius: 10px !important;
  border: 1px solid var(--kb-side-btn-border) !important;
  background: var(--kb-side-btn-bg) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.025) !important;
  color: var(--sidebar-strong-text) !important;
  font-size: 0.84rem !important;
  font-weight: 620 !important;
  line-height: 1.2 !important;
  letter-spacing: 0 !important;
  transition:
    background-color 120ms ease,
    border-color 120ms ease,
    box-shadow 120ms ease,
    color 120ms ease !important;
  transform: none !important;
}
section[data-testid="stSidebar"] div.stButton > button *{
  color: inherit !important;
  -webkit-text-fill-color: inherit !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover{
  background: var(--kb-side-btn-hover) !important;
  border-color: var(--kb-side-btn-border-hover) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03) !important;
  transform: none !important;
}
section[data-testid="stSidebar"] div.stButton > button:active{
  background: var(--kb-side-btn-active) !important;
  border-color: color-mix(in srgb, var(--blue-line) 42%, var(--line)) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.02) !important;
  transform: none !important;
}
section[data-testid="stSidebar"] div.stButton > button:focus,
section[data-testid="stSidebar"] div.stButton > button:focus-visible{
  outline: none !important;
  box-shadow:
    0 0 0 2px var(--kb-side-focus-ring) !important;
  border-color: color-mix(in srgb, var(--accent) 38%, var(--line)) !important;
}
section[data-testid="stSidebar"] div.stButton > button[disabled],
section[data-testid="stSidebar"] div.stButton > button:disabled{
  opacity: 0.55 !important;
  cursor: not-allowed !important;
}
/* 对话记录区不应用圆角按钮样式（同上） */
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-actions) + div div.stButton > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(.kb-history-row) + div div.stButton > button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.kb-history-row-btn-slot) div.stButton > button{
  min-height: unset !important;
  border-radius: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  font-weight: inherit !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"],
section[data-testid="stSidebar"] div[data-testid="stTextInput"],
section[data-testid="stSidebar"] div[data-testid="stTextArea"],
section[data-testid="stSidebar"] div[data-testid="stNumberInput"]{
  margin: 0.14rem 0 0.22rem 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input,
section[data-testid="stSidebar"] div[data-testid="stTextArea"] textarea,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input{
  min-height: 38px !important;
  border-radius: 10px !important;
  border: 1px solid var(--kb-side-ctrl-border) !important;
  background: var(--kb-side-ctrl-bg) !important;
  box-shadow: none !important;
  color: var(--sidebar-strong-text) !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:hover,
section[data-testid="stSidebar"] div[data-testid="stTextArea"] textarea:hover,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input:hover{
  border-color: color-mix(in srgb, var(--blue-line) 24%, var(--line)) !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus,
section[data-testid="stSidebar"] div[data-testid="stTextArea"] textarea:focus,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input:focus{
  border-color: color-mix(in srgb, var(--accent) 34%, var(--line)) !important;
  box-shadow: 0 0 0 2px var(--kb-side-focus-ring) !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] *{
  font-weight: 560 !important;
}
ul[data-testid="stSelectboxVirtualDropdown"],
div[role="listbox"]{
  border-radius: 10px !important;
  border: 1px solid color-mix(in srgb, var(--line) 92%, transparent) !important;
  background: color-mix(in srgb, var(--panel) 92%, var(--sidebar-bg)) !important;
  box-shadow: 0 10px 24px rgba(5, 10, 19, 0.18) !important;
}
li[data-testid="stSelectboxVirtualDropdownOption"],
div[role="option"]{
  border-radius: 8px !important;
  font-weight: 520 !important;
}
li[data-testid="stSelectboxVirtualDropdownOption"]:hover,
div[role="option"]:hover{
  background: color-mix(in srgb, var(--panel) 62%, var(--blue-weak)) !important;
}
div[role="option"][aria-selected="true"]{
  background: color-mix(in srgb, var(--panel) 48%, var(--blue-weak)) !important;
}

/* Remove bulky row pills around checkbox/radio controls (Streamlit/BaseWeb variants). */
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-baseweb="checkbox"],
section[data-testid="stSidebar"] [data-baseweb="radio"]{
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label:hover,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover,
section[data-testid="stSidebar"] [data-baseweb="checkbox"]:hover,
section[data-testid="stSidebar"] [data-baseweb="radio"]:hover{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label:has(input:checked),
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked),
section[data-testid="stSidebar"] [data-baseweb="checkbox"]:has(input:checked),
section[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked){
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stCheckbox"],
section[data-testid="stSidebar"] div[data-testid="stRadio"]{
  margin: 0.10rem 0 0.16rem 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] p,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] span,
section[data-testid="stSidebar"] div[data-testid="stRadio"] span{
  font-weight: 560 !important;
}

/* Sidebar section titles: compact, IDE-like. */
section[data-testid="stSidebar"] h3{
  font-size: 1.00rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em !important;
  margin: 0.16rem 0 0.08rem 0 !important;
}
section[data-testid="stSidebar"] .hr{
  margin: 0.72rem 0 0.82rem 0 !important;
  background: color-mix(in srgb, var(--line) 84%, transparent) !important;
}
/* Ultra-minimal sidebar action chips (less "button", more IDE action affordance). */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button{
  width: fit-content !important;
  max-width: 100% !important;
  min-width: 0 !important;
  min-height: 32px !important;
  padding: 0.28rem 0.62rem !important;
  border-radius: 9px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 0.35rem !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 92%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 92%, var(--sidebar-soft-text)) !important;
  font-size: 0.82rem !important;
  font-weight: 560 !important;
  letter-spacing: 0 !important;
  line-height: 1.2 !important;
  transition:
    background-color 120ms ease,
    border-color 120ms ease,
    color 120ms ease,
    box-shadow 120ms ease !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover{
  background: color-mix(in srgb, var(--panel) 22%, transparent) !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:active{
  background: color-mix(in srgb, var(--panel) 28%, transparent) !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus-visible{
  outline: none !important;
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 10%, transparent) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[disabled],
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:disabled{
  opacity: 0.48 !important;
  background: transparent !important;
}

/* Conversation history (row styles shared by popover/expander variants) */
button.kb-conv-picker-trigger,
section[data-testid="stSidebar"] button.kb-conv-picker-trigger{
  width: 100% !important;
  min-height: 36px !important;
  justify-content: flex-start !important;
  text-align: left !important;
  border-radius: 10px !important;
  background: color-mix(in srgb, var(--panel) 24%, var(--sidebar-bg)) !important;
  border: 1px solid color-mix(in srgb, var(--line) 84%, transparent) !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
  font-weight: 560 !important;
}
button.kb-conv-picker-trigger:hover,
section[data-testid="stSidebar"] button.kb-conv-picker-trigger:hover{
  background: color-mix(in srgb, var(--panel) 36%, var(--sidebar-bg)) !important;
  border-color: color-mix(in srgb, var(--line) 98%, var(--blue-line)) !important;
}
.kb-conv-popover-panel{
  width: min(100%, var(--kb-conv-panel-width, 320px)) !important;
  max-width: var(--kb-conv-panel-width, 320px) !important;
  min-width: min(100%, 240px) !important;
  border-radius: 10px !important;
  border: 1px solid color-mix(in srgb, var(--line) 86%, transparent) !important;
  background: color-mix(in srgb, var(--panel) 90%, var(--sidebar-bg)) !important;
  box-shadow: 0 14px 32px rgba(3, 8, 16, 0.24) !important;
  padding: 6px !important;
  overflow: hidden !important;
}
.kb-conv-popover-panel div[data-testid="stButton"]{
  margin: 0 !important;
}
.kb-conv-popover-scroll{
  max-height: min(56vh, 420px) !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  padding-right: 2px !important;
}
.kb-conv-row-wrap{
  position: relative !important;
  border-radius: 8px !important;
  margin: 0 !important;
  padding: 0 !important;
  gap: 0.14rem !important;
  align-items: center !important;
  overflow: clip !important;
}
.kb-conv-row-wrap::after{
  content: none !important;
}
.kb-conv-row-wrap > div[data-testid="column"]{
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
.kb-conv-row-wrap div[data-testid="stButton"]{
  margin: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(button.kb-conv-row-btn),
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(button.kb-conv-row-btn){
  margin: 0 !important;
  padding: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(button.kb-conv-row-btn) > div,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(button.kb-conv-row-btn) > div{
  margin: 0 !important;
  padding-top: 0 !important;
  padding-bottom: 1px !important; /* leave a hairline gap between rows */
}
/* Flat list mode: style rows by structure (row = horizontal block containing a conversation row button) */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-conv-row-btn){
  margin: 0 !important;
  padding: 0 !important;
  gap: 0.02rem !important;
  align-items: center !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-conv-row-btn) > div[data-testid="column"]{
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-conv-row-btn) div[data-testid="stButton"]{
  margin: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-conv-row-btn) > div[data-testid="column"]:first-child div[data-testid="stButton"] > button{
  min-height: 18px !important;
  padding: 0.01rem 0.10rem 0.01rem 0.06rem !important;
  border: 0 !important;
  border-width: 0 !important;
  border-style: none !important;
  border-color: transparent !important;
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
  outline: none !important;
  appearance: none !important;
  -webkit-appearance: none !important;
  border-radius: 5px !important;
  justify-content: flex-start !important;
  text-align: left !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-conv-row-btn) > div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-conv-row-btn) > div[data-testid="column"]:first-child div[data-testid="stButton"] > button span{
  margin: 0 !important;
  font-size: 0.64rem !important;
  line-height: 1 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-conv-row-btn):hover > div[data-testid="column"]:first-child div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-conv-row-btn):focus-within > div[data-testid="column"]:first-child div[data-testid="stButton"] > button{
  background: color-mix(in srgb, var(--panel) 10%, transparent) !important;
  border: 0 !important;
  border-width: 0 !important;
  border-style: none !important;
  border-color: transparent !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-current) > div[data-testid="column"]:first-child div[data-testid="stButton"] > button{
  background: color-mix(in srgb, var(--panel) 14%, transparent) !important;
  border: 0 !important;
  border-width: 0 !important;
  border-style: none !important;
  border-color: transparent !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-conv-row-btn) > div[data-testid="column"]:last-child{
  display: flex !important;
  justify-content: flex-end !important;
  align-items: center !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-conv-row-btn) > div[data-testid="column"]:last-child div[data-testid="stButton"] > button{
  width: 12px !important;
  min-width: 12px !important;
  max-width: 12px !important;
  height: 12px !important;
  min-height: 12px !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  border-width: 0 !important;
  border-style: none !important;
  border-color: transparent !important;
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
}
button.kb-conv-row-btn,
section[data-testid="stSidebar"] button.kb-conv-row-btn{
  width: 100% !important;
  min-height: 20px !important;
  justify-content: flex-start !important;
  text-align: left !important;
  border-radius: 5px !important;
  background: transparent !important;
  background-color: transparent !important;
  border: 0 !important;
  border-width: 0 !important;
  border-style: none !important;
  border-color: transparent !important;
  box-shadow: none !important;
  outline: none !important;
  appearance: none !important;
  -webkit-appearance: none !important;
  padding: 0.02rem 0.12rem 0.02rem 0.07rem !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 84%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 84%, var(--sidebar-soft-text)) !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  line-height: 1.0 !important;
  transition: background-color 100ms ease, color 100ms ease !important;
}
button.kb-conv-row-btn p,
section[data-testid="stSidebar"] button.kb-conv-row-btn p,
button.kb-conv-row-btn span,
section[data-testid="stSidebar"] button.kb-conv-row-btn span{
  margin: 0 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  line-height: 1 !important;
  font-size: 0.72rem !important;
}
button.kb-conv-row-btn:hover,
section[data-testid="stSidebar"] button.kb-conv-row-btn:hover{
  background: color-mix(in srgb, var(--panel) 13%, transparent) !important;
  border: 0 !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
}
button.kb-conv-row-btn.kb-current,
section[data-testid="stSidebar"] button.kb-conv-row-btn.kb-current{
  background: color-mix(in srgb, var(--panel) 16%, transparent) !important;
  border: 0 !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
  font-weight: 540 !important;
}
button.kb-conv-trash-btn,
section[data-testid="stSidebar"] button.kb-conv-trash-btn{
  width: 12px !important;
  min-width: 12px !important;
  max-width: 12px !important;
  height: 12px !important;
  min-height: 12px !important;
  padding: 0 !important;
  margin: 0 !important;
  border-radius: 3px !important;
  border: 0 !important;
  border-width: 0 !important;
  border-style: none !important;
  border-color: transparent !important;
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  color: color-mix(in srgb, var(--sidebar-soft-text) 76%, transparent) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-soft-text) 76%, transparent) !important;
  opacity: 0.16 !important;
  pointer-events: auto !important;
  transition: none !important;
}
button.kb-conv-trash-btn:hover,
section[data-testid="stSidebar"] button.kb-conv-trash-btn:hover{
  background: color-mix(in srgb, #ef4444 10%, transparent) !important;
  border: 0 !important;
  color: #fca5a5 !important;
  -webkit-text-fill-color: #fca5a5 !important;
  opacity: 1 !important;
}
.kb-conv-row-wrap:hover button.kb-conv-trash-btn,
.kb-conv-row-wrap:focus-within button.kb-conv-trash-btn{
  opacity: 0.55 !important;
  pointer-events: auto !important;
}
.kb-conv-row-wrap:hover button.kb-conv-row-btn,
.kb-conv-row-wrap:focus-within button.kb-conv-row-btn{
  background: color-mix(in srgb, var(--panel) 13%, transparent) !important;
}
.kb-conv-row-wrap button.kb-conv-row-btn.kb-current + *,
.kb-conv-row-wrap:has(button.kb-current) button.kb-conv-trash-btn{
  opacity: 0.34 !important;
}
button.kb-conv-trash-btn:focus,
button.kb-conv-trash-btn:focus-visible,
section[data-testid="stSidebar"] button.kb-conv-trash-btn:focus,
section[data-testid="stSidebar"] button.kb-conv-trash-btn:focus-visible{
  box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.10) !important;
  border: 0 !important;
  opacity: 1 !important;
  pointer-events: auto !important;
}

/* Conversation history expander: base + trash (compact layout defined later in this file). */
section[data-testid="stSidebar"] details.kb-conv-history-expander,
section[data-testid="stSidebar"] div[data-testid="stExpander"].kb-conv-history-expander{
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary:focus,
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary:focus-visible{
  outline: none !important;
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--line) 88%, transparent) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary svg{
  opacity: 0.7 !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander button.kb-conv-trash-btn{
  min-width: 28px !important;
  min-height: 28px !important;
  border-radius: 7px !important;
  opacity: 0 !important;
  pointer-events: none !important;
  color: color-mix(in srgb, var(--sidebar-soft-text) 50%, transparent) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-soft-text) 50%, transparent) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap:hover button.kb-conv-trash-btn,
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap:focus-within button.kb-conv-trash-btn{
  opacity: 0.8 !important;
  pointer-events: auto !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap:has(button.kb-current) button.kb-conv-trash-btn{
  opacity: 0.35 !important;
  pointer-events: auto !important;
}

/* Conversation history compact layout + action buttons */
button.kb-history-action-btn,
section[data-testid="stSidebar"] button.kb-history-action-btn{
  min-height: 24px !important;
  padding: 0.05rem 0.16rem !important;
  border: 0 !important;
  border-radius: 7px !important;
  background: transparent !important;
  box-shadow: none !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 80%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 80%, var(--sidebar-soft-text)) !important;
  font-size: 0.68rem !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  justify-content: flex-start !important;
  text-align: left !important;
}
button.kb-history-action-btn:hover,
section[data-testid="stSidebar"] button.kb-history-action-btn:hover{
  background: color-mix(in srgb, var(--panel) 12%, transparent) !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
}
button.kb-history-action-btn:active,
section[data-testid="stSidebar"] button.kb-history-action-btn:active,
button.kb-history-action-btn:focus,
button.kb-history-action-btn:focus-visible,
section[data-testid="stSidebar"] button.kb-history-action-btn:focus,
section[data-testid="stSidebar"] button.kb-history-action-btn:focus-visible{
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(button.kb-history-toggle-btn),
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(button.kb-history-toggle-btn){
  margin: 0 !important;
  padding: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="element-container"]:has(button.kb-history-toggle-btn) > div,
section[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(button.kb-history-toggle-btn) > div{
  margin: 0 !important;
  padding-top: 0 !important;
  padding-bottom: 1px !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"]:has(> button.kb-history-toggle-btn){
  margin: 0 !important;
  padding: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"]:has(> button.kb-history-toggle-btn) > button{
  border: 0 !important;
  border-width: 0 !important;
  border-style: none !important;
  border-color: transparent !important;
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  outline: none !important;
}

/* Older-history toggle: compact link-like (same as expand state) */
button.kb-history-toggle-btn,
section[data-testid="stSidebar"] button.kb-history-toggle-btn{
  width: auto !important;
  min-height: 18px !important;
  padding: 0.01rem 0.08rem 0.01rem 0.06rem !important;
  border: 0 !important;
  border-width: 0 !important;
  border-style: none !important;
  border-color: transparent !important;
  border-radius: 5px !important;
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
  outline: none !important;
  appearance: none !important;
  -webkit-appearance: none !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 76%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 76%, var(--sidebar-soft-text)) !important;
  font-size: 0.58rem !important;
  font-weight: 500 !important;
  justify-content: center !important;
  text-align: center !important;
}
button.kb-history-toggle-btn p,
section[data-testid="stSidebar"] button.kb-history-toggle-btn p,
button.kb-history-toggle-btn span,
section[data-testid="stSidebar"] button.kb-history-toggle-btn span{
  margin: 0 !important;
  font-size: 0.58rem !important;
  line-height: 1.0 !important;
}
button.kb-history-toggle-btn:hover,
section[data-testid="stSidebar"] button.kb-history-toggle-btn:hover,
button.kb-history-toggle-btn:focus,
button.kb-history-toggle-btn:focus-visible,
section[data-testid="stSidebar"] button.kb-history-toggle-btn:focus,
section[data-testid="stSidebar"] button.kb-history-toggle-btn:focus-visible{
  background: color-mix(in srgb, var(--panel) 7%, transparent) !important;
  border: 0 !important;
  border-color: transparent !important;
  box-shadow: none !important;
  outline: none !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
}

/* Final hard override: defeat generic sidebar chip style for history rows/toggles */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn{
  all: unset !important;
  box-sizing: border-box !important;
  display: flex !important;
  width: 100% !important;
  min-height: 17px !important;
  padding: 0.00rem 0.08rem 0.00rem 0.05rem !important;
  border: 0 !important;
  border-radius: 5px !important;
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
  outline: none !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 84%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 84%, var(--sidebar-soft-text)) !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  line-height: 1 !important;
  justify-content: flex-start !important;
  align-items: center !important;
  text-align: left !important;
  cursor: pointer !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn::before,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn::after,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn::before,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn::after{
  content: none !important;
  display: none !important;
  border: 0 !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn p,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn span,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn p,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn span{
  margin: 0 !important;
  font-size: 0.72rem !important;
  line-height: 1 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn:hover,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn:focus,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn:focus-visible,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn:hover,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn:focus,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn:focus-visible{
  background: color-mix(in srgb, var(--panel) 4%, transparent) !important;
  background-color: color-mix(in srgb, var(--panel) 4%, transparent) !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn.kb-current,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn.kb-current{
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--blue-weak) 22%, var(--panel) 16%),
    color-mix(in srgb, var(--panel) 14%, transparent) 28%,
    color-mix(in srgb, var(--panel) 10%, transparent)
  ) !important;
  background-color: color-mix(in srgb, var(--panel) 14%, var(--blue-weak) 12%) !important;
  border: 0 !important;
  box-shadow:
    inset 3px 0 0 color-mix(in srgb, var(--blue-line) 88%, var(--accent)),
    inset 0 0 0 1px color-mix(in srgb, var(--blue-line) 20%, transparent) !important;
  outline: none !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
  font-weight: 700 !important;
  padding-left: 0.28rem !important;
  gap: 0.22rem !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn.kb-current p,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn.kb-current span,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn.kb-current p,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn.kb-current span{
  font-weight: 700 !important;
  letter-spacing: 0.005em !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-conv-row-btn.kb-current::before,
section[data-testid="stSidebar"] div.stButton > button.kb-conv-row-btn.kb-current::before{
  content: "" !important;
  display: inline-block !important;
  width: 5px !important;
  min-width: 5px !important;
  height: 5px !important;
  border-radius: 999px !important;
  background: color-mix(in srgb, var(--blue-line) 92%, var(--accent)) !important;
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--blue-weak) 30%, transparent) !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
  flex: 0 0 auto !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(button.kb-current) > div[data-testid="column"]:last-child button.kb-conv-trash-btn{
  opacity: 0.52 !important;
  color: color-mix(in srgb, var(--sidebar-soft-text) 88%, var(--sidebar-strong-text) 12%) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-soft-text) 88%, var(--sidebar-strong-text) 12%) !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-toggle-btn,
section[data-testid="stSidebar"] div.stButton > button.kb-history-toggle-btn{
  all: unset !important;
  box-sizing: border-box !important;
  display: inline-flex !important;
  width: auto !important;
  min-height: 18px !important;
  padding: 0.04rem 0.16rem !important;
  border: 0 !important;
  border-radius: 4px !important;
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
  outline: none !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 68%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 68%, var(--sidebar-soft-text)) !important;
  font-size: 0.58rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.02em !important;
  line-height: 1.2 !important;
  justify-content: center !important;
  align-items: center !important;
  text-align: center !important;
  cursor: pointer !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-toggle-btn::before,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-toggle-btn::after,
section[data-testid="stSidebar"] div.stButton > button.kb-history-toggle-btn::before,
section[data-testid="stSidebar"] div.stButton > button.kb-history-toggle-btn::after{
  content: none !important;
  display: none !important;
  border: 0 !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-toggle-btn p,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-toggle-btn span,
section[data-testid="stSidebar"] div.stButton > button.kb-history-toggle-btn p,
section[data-testid="stSidebar"] div.stButton > button.kb-history-toggle-btn span{
  margin: 0 !important;
  font-size: 0.58rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.02em !important;
  line-height: 1.2 !important;
  white-space: nowrap !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-toggle-btn:hover,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-toggle-btn:focus,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-toggle-btn:focus-visible,
section[data-testid="stSidebar"] div.stButton > button.kb-history-toggle-btn:hover,
section[data-testid="stSidebar"] div.stButton > button.kb-history-toggle-btn:focus,
section[data-testid="stSidebar"] div.stButton > button.kb-history-toggle-btn:focus-visible{
  background: color-mix(in srgb, var(--panel) 5%, transparent) !important;
  background-color: color-mix(in srgb, var(--panel) 5%, transparent) !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
}

/* Final hard override: top history action buttons ("New chat" / "Delete current") */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn),
section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn){
  all: unset !important;
  box-sizing: border-box !important;
  display: inline-flex !important;
  width: 100% !important;
  min-width: 0 !important;
  min-height: 34px !important;
  padding: 0.28rem 0.70rem !important;
  border-radius: 12px !important;
  border: 1px solid color-mix(in srgb, var(--line) 70%, transparent) !important;
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--panel) 22%, transparent),
      color-mix(in srgb, var(--panel) 12%, transparent)
    ) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.030),
    0 1px 0 rgba(0,0,0,0.18) !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 92%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 92%, var(--sidebar-soft-text)) !important;
  font-size: 0.80rem !important;
  font-weight: 600 !important;
  line-height: 1.05 !important;
  justify-content: center !important;
  align-items: center !important;
  text-align: center !important;
  cursor: pointer !important;
  overflow: hidden !important;
  transition:
    background-color 120ms ease,
    border-color 120ms ease,
    color 120ms ease,
    box-shadow 120ms ease,
    transform 90ms ease !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn) p,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn) span,
section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn) p,
section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn) span{
  margin: 0 !important;
  font-size: 0.80rem !important;
  line-height: 1.05 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  width: 100% !important;
  min-width: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn):hover,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn):focus,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn):focus-visible,
section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn):hover,
section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn):focus,
section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn):focus-visible{
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--panel) 28%, transparent),
      color-mix(in srgb, var(--panel) 16%, transparent)
    ) !important;
  border-color: color-mix(in srgb, var(--line) 92%, var(--blue-line)) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.036),
    0 0 0 1px color-mix(in srgb, var(--line) 22%, transparent) !important;
  outline: none !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn):active,
section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn):active{
  transform: translateY(0.5px) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.020),
    0 0 0 1px color-mix(in srgb, var(--line) 18%, transparent) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn)[disabled],
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn):disabled,
section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn)[disabled],
section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn):disabled{
  opacity: 0.42 !important;
  cursor: not-allowed !important;
  border-color: color-mix(in srgb, var(--line) 50%, transparent) !important;
  box-shadow: none !important;
}

/* New chat: slightly brighter, accent-tinted */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-new-btn,
section[data-testid="stSidebar"] div.stButton > button.kb-history-new-btn{
  border-color: color-mix(in srgb, var(--blue-line) 34%, var(--line)) !important;
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--blue-weak) 18%, var(--panel) 18%),
      color-mix(in srgb, var(--blue-weak) 10%, transparent)
    ) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-new-btn:hover,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-new-btn:focus,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-new-btn:focus-visible,
section[data-testid="stSidebar"] div.stButton > button.kb-history-new-btn:hover,
section[data-testid="stSidebar"] div.stButton > button.kb-history-new-btn:focus,
section[data-testid="stSidebar"] div.stButton > button.kb-history-new-btn:focus-visible{
  border-color: color-mix(in srgb, var(--blue-line) 56%, var(--line)) !important;
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--blue-weak) 24%, var(--panel) 22%),
      color-mix(in srgb, var(--blue-weak) 13%, transparent)
    ) !important;
}

/* Delete current: subtle danger tone, not aggressive */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-danger-btn,
section[data-testid="stSidebar"] div.stButton > button.kb-history-danger-btn{
  border-color: color-mix(in srgb, #ef4444 20%, var(--line)) !important;
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, #ef4444 8%, var(--panel) 18%),
      color-mix(in srgb, #ef4444 4%, transparent)
    ) !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 88%, #fecaca 12%) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 88%, #fecaca 12%) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-danger-btn:hover,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-danger-btn:focus,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-danger-btn:focus-visible,
section[data-testid="stSidebar"] div.stButton > button.kb-history-danger-btn:hover,
section[data-testid="stSidebar"] div.stButton > button.kb-history-danger-btn:focus,
section[data-testid="stSidebar"] div.stButton > button.kb-history-danger-btn:focus-visible{
  border-color: color-mix(in srgb, #ef4444 36%, var(--line)) !important;
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, #ef4444 12%, var(--panel) 22%),
      color-mix(in srgb, #ef4444 6%, transparent)
    ) !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 80%, #fecaca 20%) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 80%, #fecaca 20%) !important;
}

/* Responsive squeeze for top history action buttons when sidebar is dragged narrow.
   Gradually shrink font/padding to keep Chinese labels inside the pill. */
@supports (width: 1cqw){
  section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn),
  section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn){
    font-size: clamp(0.66rem, 4.2cqw, 0.80rem) !important;
    padding-left: clamp(0.34rem, 2.0cqw, 0.70rem) !important;
    padding-right: clamp(0.34rem, 2.0cqw, 0.70rem) !important;
    min-height: clamp(31px, 10cqw, 34px) !important;
  }
  section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn) p,
  section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-history-action-btn:not(.kb-history-toggle-btn) span,
  section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn) p,
  section[data-testid="stSidebar"] div.stButton > button.kb-history-action-btn:not(.kb-history-toggle-btn) span{
    font-size: clamp(0.66rem, 4.2cqw, 0.80rem) !important;
  }
}

/* Model connectivity test button: more premium technical pill */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-model-test-btn,
section[data-testid="stSidebar"] div.stButton > button.kb-model-test-btn{
  all: unset !important;
  box-sizing: border-box !important;
  display: inline-flex !important;
  width: fit-content !important;
  max-width: 100% !important;
  min-height: 35px !important;
  padding: 0.30rem 0.78rem !important;
  border-radius: 12px !important;
  border: 1px solid color-mix(in srgb, var(--blue-line) 30%, var(--line)) !important;
  background:
    radial-gradient(120% 160% at 12% 0%, color-mix(in srgb, var(--blue-weak) 22%, transparent) 0%, transparent 55%),
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--panel) 22%, transparent),
      color-mix(in srgb, var(--panel) 10%, transparent)
    ) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.040),
    inset 0 0 0 1px color-mix(in srgb, var(--blue-line) 10%, transparent),
    0 1px 0 rgba(0,0,0,0.18) !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 94%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 94%, var(--sidebar-soft-text)) !important;
  font-size: 0.81rem !important;
  font-weight: 620 !important;
  line-height: 1.06 !important;
  letter-spacing: 0.01em !important;
  justify-content: center !important;
  align-items: center !important;
  text-align: center !important;
  cursor: pointer !important;
  transition:
    transform 100ms ease,
    border-color 130ms ease,
    box-shadow 130ms ease,
    background-color 130ms ease,
    color 130ms ease !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-model-test-btn::before,
section[data-testid="stSidebar"] div.stButton > button.kb-model-test-btn::before{
  content: "" !important;
  width: 7px !important;
  min-width: 7px !important;
  height: 7px !important;
  border-radius: 999px !important;
  margin-right: 0.45rem !important;
  background:
    radial-gradient(circle at 35% 35%, #ffffff 0 14%, color-mix(in srgb, var(--accent) 88%, var(--blue-line)) 18% 100%) !important;
  box-shadow:
    0 0 0 2px color-mix(in srgb, var(--blue-weak) 20%, transparent),
    0 0 10px color-mix(in srgb, var(--blue-line) 18%, transparent) !important;
  flex: 0 0 auto !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-model-test-btn p,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-model-test-btn span,
section[data-testid="stSidebar"] div.stButton > button.kb-model-test-btn p,
section[data-testid="stSidebar"] div.stButton > button.kb-model-test-btn span{
  margin: 0 !important;
  font-size: 0.81rem !important;
  font-weight: 620 !important;
  line-height: 1.06 !important;
  letter-spacing: 0.01em !important;
  white-space: nowrap !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-model-test-btn:hover,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-model-test-btn:focus,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-model-test-btn:focus-visible,
section[data-testid="stSidebar"] div.stButton > button.kb-model-test-btn:hover,
section[data-testid="stSidebar"] div.stButton > button.kb-model-test-btn:focus,
section[data-testid="stSidebar"] div.stButton > button.kb-model-test-btn:focus-visible{
  transform: translateY(-0.5px) !important;
  border-color: color-mix(in srgb, var(--blue-line) 54%, var(--line)) !important;
  background:
    radial-gradient(120% 160% at 12% 0%, color-mix(in srgb, var(--blue-weak) 28%, transparent) 0%, transparent 58%),
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--panel) 28%, transparent),
      color-mix(in srgb, var(--panel) 14%, transparent)
    ) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.055),
    inset 0 0 0 1px color-mix(in srgb, var(--blue-line) 18%, transparent),
    0 0 0 1px color-mix(in srgb, var(--line) 20%, transparent),
    0 4px 16px color-mix(in srgb, var(--blue-line) 10%, transparent) !important;
  outline: none !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button.kb-model-test-btn:active,
section[data-testid="stSidebar"] div.stButton > button.kb-model-test-btn:active{
  transform: translateY(0) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.025),
    inset 0 0 0 1px color-mix(in srgb, var(--blue-line) 12%, transparent),
    0 0 0 1px color-mix(in srgb, var(--line) 18%, transparent) !important;
}

section[data-testid="stSidebar"] details.kb-conv-history-expander > summary{
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  min-height: 30px !important;
  padding: 0.08rem 0.18rem !important;
  border-radius: 6px !important;
  margin: 0 !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 96%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 96%, var(--sidebar-soft-text)) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary:hover{
  background: color-mix(in srgb, var(--panel) 11%, transparent) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander[open] > summary{
  background: color-mix(in srgb, var(--panel) 22%, transparent) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary p{
  font-size: 0.75rem !important;
  font-weight: 540 !important;
  line-height: 1.08 !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div{
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding-top: 0.06rem !important;
  padding-bottom: 0.02rem !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap{
  margin: 0 !important;
  border-radius: 0 !important;
  padding: 0.01rem 0 !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap::after{
  content: "";
  position: absolute;
  left: 0.45rem;
  right: 0.10rem;
  bottom: -1px;
  height: 1px;
  background: color-mix(in srgb, var(--line) 34%, transparent);
  opacity: 0.78;
  pointer-events: none;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap:last-child::after{
  opacity: 0;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap::before{
  content: none !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander button.kb-conv-row-btn{
  width: 100% !important;
  min-height: 18px !important;
  padding: 0.02rem 0.20rem 0.02rem 0.12rem !important;
  border-radius: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 85%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 85%, var(--sidebar-soft-text)) !important;
  font-size: 0.70rem !important;
  font-weight: 500 !important;
  line-height: 1.0 !important;
  justify-content: flex-start !important;
  text-align: left !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander button.kb-conv-row-btn p,
section[data-testid="stSidebar"] details.kb-conv-history-expander button.kb-conv-row-btn span{
  font-size: 0.70rem !important;
  line-height: 1.0 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap:hover button.kb-conv-row-btn,
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap:focus-within button.kb-conv-row-btn{
  background: transparent !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander button.kb-conv-row-btn.kb-current{
  background: transparent !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
  font-weight: 550 !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap [data-testid="stPopover"] > button{
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  outline: none !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap [data-testid="stPopover"] > button:hover,
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap [data-testid="stPopover"] > button:focus,
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap div[data-testid="stButton"] > button:focus-visible,
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap [data-testid="stPopover"] > button:focus-visible{
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  outline: none !important;
}

button.kb-conv-menu-trigger,
section[data-testid="stSidebar"] button.kb-conv-menu-trigger{
  width: 16px !important;
  min-width: 16px !important;
  max-width: 16px !important;
  height: 16px !important;
  min-height: 16px !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  color: color-mix(in srgb, var(--sidebar-soft-text) 70%, transparent) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-soft-text) 70%, transparent) !important;
  font-size: 0.60rem !important;
  font-weight: 600 !important;
  line-height: 1 !important;
  letter-spacing: 0 !important;
  white-space: nowrap !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  opacity: 0 !important;
  pointer-events: none !important;
  transition: opacity 90ms ease, color 90ms ease, background-color 90ms ease !important;
}
button.kb-conv-menu-trigger svg,
section[data-testid="stSidebar"] button.kb-conv-menu-trigger svg,
button.kb-conv-menu-trigger [data-testid="stIconMaterial"],
section[data-testid="stSidebar"] button.kb-conv-menu-trigger [data-testid="stIconMaterial"]{
  display: none !important;
}
button.kb-conv-menu-trigger p,
section[data-testid="stSidebar"] button.kb-conv-menu-trigger p,
button.kb-conv-menu-trigger span,
section[data-testid="stSidebar"] button.kb-conv-menu-trigger span{
  font-size: 0.60rem !important;
  line-height: 1 !important;
  margin: 0 !important;
  letter-spacing: 0 !important;
  white-space: nowrap !important;
}
section[data-testid="stSidebar"] button.kb-conv-menu-trigger [data-testid="stPopoverChevron"],
section[data-testid="stSidebar"] button.kb-conv-menu-trigger svg,
section[data-testid="stSidebar"] button.kb-conv-menu-trigger [data-testid="stIcon"],
section[data-testid="stSidebar"] button.kb-conv-menu-trigger [data-testid="stIconMaterial"]{
  display: none !important;
}
section[data-testid="stSidebar"] button.kb-conv-menu-trigger > div,
section[data-testid="stSidebar"] button.kb-conv-menu-trigger > span{
  display: inline-flex !important;
  flex-direction: row !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 0 !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap:hover button.kb-conv-menu-trigger,
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap:focus-within button.kb-conv-menu-trigger{
  opacity: 0.55 !important;
  pointer-events: auto !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap:has(button.kb-current) button.kb-conv-menu-trigger{
  opacity: 0.26 !important;
  pointer-events: auto !important;
}
button.kb-conv-menu-trigger:hover,
section[data-testid="stSidebar"] button.kb-conv-menu-trigger:hover{
  background: transparent !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
  opacity: 1 !important;
}
button.kb-conv-menu-trigger:focus,
button.kb-conv-menu-trigger:focus-visible,
section[data-testid="stSidebar"] button.kb-conv-menu-trigger:focus,
section[data-testid="stSidebar"] button.kb-conv-menu-trigger:focus-visible{
  border: 0 !important;
  box-shadow: none !important;
  opacity: 1 !important;
  pointer-events: auto !important;
}
/* Flatten the internal expander trigger button if Streamlit renders one. */
section[data-testid="stSidebar"] details.kb-conv-history-expander button.kb-conv-picker-trigger{
  min-height: 22px !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  justify-content: flex-start !important;
  text-align: left !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander button.kb-conv-picker-trigger:hover{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
/* Hard override: remove button chrome inside conversation dropdown rows (text list only) */
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="stHorizontalBlock"]{
  position: relative !important;
  margin: 0 !important;
  padding: 0 !important;
  gap: 0 !important;
  align-items: center !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="stHorizontalBlock"]::after{
  content: "" !important;
  position: absolute !important;
  left: 0.25rem !important;
  right: 0.05rem !important;
  bottom: -1px !important;
  height: 1px !important;
  background: color-mix(in srgb, var(--line) 40%, transparent) !important;
  opacity: 0.9 !important;
  pointer-events: none !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="stHorizontalBlock"]:last-child::after{
  opacity: 0 !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]{
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"] div[data-testid="stButton"],
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"] [data-testid="stPopover"]{
  margin: 0 !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"] div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"] [data-testid="stPopover"] > button{
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  outline: none !important;
  min-height: 18px !important;
  padding-top: 0.04rem !important;
  padding-bottom: 0.04rem !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"] div[data-testid="stButton"] > button:hover,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"] [data-testid="stPopover"] > button:hover,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"] div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"] [data-testid="stPopover"] > button:focus,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"] div[data-testid="stButton"] > button:focus-visible,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"] [data-testid="stPopover"] > button:focus-visible{
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  outline: none !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:first-child div[data-testid="stButton"] > button{
  justify-content: flex-start !important;
  text-align: left !important;
  padding-left: 0.12rem !important;
  padding-right: 0.18rem !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:first-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:first-child div[data-testid="stButton"] > button span{
  font-size: 0.70rem !important;
  line-height: 1.0 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child{
  display: flex !important;
  justify-content: flex-end !important;
  align-items: center !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child [data-testid="stPopover"] > button,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child div[data-testid="stButton"] > button{
  width: 16px !important;
  min-width: 16px !important;
  max-width: 16px !important;
  height: 16px !important;
  min-height: 16px !important;
  padding: 0 !important;
  margin: 0 !important;
  justify-content: center !important;
  text-align: center !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child [data-testid="stPopover"] > button p,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child [data-testid="stPopover"] > button span,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child div[data-testid="stButton"] > button p,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child div[data-testid="stButton"] > button span{
  font-size: 0.60rem !important;
  line-height: 1 !important;
  margin: 0 !important;
  white-space: nowrap !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child [data-testid="stPopover"] > button svg,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child [data-testid="stPopover"] > button [data-testid="stPopoverChevron"],
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child [data-testid="stPopover"] > button [data-testid="stIcon"],
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child [data-testid="stPopover"] > button [data-testid="stIconMaterial"]{
  display: none !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child [data-testid="stPopover"] > button > div,
section[data-testid="stSidebar"] details.kb-conv-history-expander > div div[data-testid="column"]:last-child [data-testid="stPopover"] > button > span{
  display: inline-flex !important;
  flex-direction: row !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 0 !important;
}
</style>
<script>
(function () {
  const host = window.parent || window;
  const doc = host.document || document;
  const mode = "__MODE__";
  try {
    doc.documentElement.setAttribute("data-theme", mode);
    if (doc.body) doc.body.setAttribute("data-theme", mode);
  } catch (e) {}
})();
</script>
"""
    st.markdown(
        css.replace("__TOKENS__", tokens)
        .replace("__SCHEME__", color_scheme)
        .replace("__MODE__", mode)
        .replace("__SIDEBAR_CORE__", _build_sidebar_core_css()),
        unsafe_allow_html=True,
    )


def _sync_theme_with_browser_preference() -> None:
    """
    Follow browser/system color scheme without adding an in-app theme toggle.
    We keep `_init_theme_css()` as the base stylesheet (light mode baseline),
    then override CSS variables + data-theme in the browser when dark mode is active.
    """
    # 1) Inject theme CSS overrides (safe in markdown HTML)
    st.markdown(
        """
<style>
html[data-theme="light"], body[data-theme="light"]{
  color-scheme: light !important;
}
html[data-theme="dark"], body[data-theme="dark"]{
  """
        + _DARK_THEME_TOKENS
        + """
  color-scheme: dark !important;
}
</style>
<script>
(function () {
  try {
    const candidates = [];
    try { if (window.parent) candidates.push(window.parent); } catch (e) {}
    try { if (window.top && window.top !== window.parent) candidates.push(window.top); } catch (e) {}
    candidates.push(window);

    let host = window;
    let doc = document;
    for (const c of candidates) {
      try {
        const d = c.document;
        if (!d) continue;
        if (d.querySelector('[data-testid="stAppViewContainer"], [data-testid="stMain"]')) {
          host = c;
          doc = d;
          break;
        }
      } catch (e) {}
    }
    if (!doc || !doc.documentElement) return;
    const mq = (host.matchMedia && host.matchMedia("(prefers-color-scheme: dark)")) || null;
    function apply() {
      try {
        const dark = !!(mq && mq.matches);
        const mode = dark ? "dark" : "light";
        doc.documentElement.setAttribute("data-theme", mode);
        if (doc.body) doc.body.setAttribute("data-theme", mode);
      } catch (e) {}
    }
    apply();
    if (mq) {
      try { mq.addEventListener("change", apply); }
      catch (e) { try { mq.addListener(apply); } catch (e2) {} }
    }
  } catch (e) {}
})();
</script>
        """,
        unsafe_allow_html=True,
    )

    # 2) Inject executable JS via components iframe (Streamlit markdown may strip/ignore <script>)
    components.html(
        """
<script>
(function () {
  try {
    const candidates = [];
    try { if (window.parent) candidates.push(window.parent); } catch (e) {}
    try { if (window.top && window.top !== window.parent) candidates.push(window.top); } catch (e) {}
    candidates.push(window);

    let host = window;
    let doc = document;
    for (const c of candidates) {
      try {
        const d = c.document;
        if (!d) continue;
        if (d.querySelector('[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stApp"]')) {
          host = c;
          doc = d;
          break;
        }
      } catch (e) {}
    }
    if (!doc || !doc.documentElement) return;

    const KEY = "__kbBrowserThemeSyncV1";
    try {
      if (host[KEY] && typeof host[KEY].teardown === "function") host[KEY].teardown();
    } catch (e) {}

    const mq = (host.matchMedia && host.matchMedia("(prefers-color-scheme: dark)")) || null;
    let listener = null;

    function apply() {
      try {
        const dark = !!(mq && mq.matches);
        const mode = dark ? "dark" : "light";
        doc.documentElement.setAttribute("data-theme", mode);
        if (doc.body) doc.body.setAttribute("data-theme", mode);
      } catch (e) {}
    }

    apply();

    if (mq) {
      listener = function () { apply(); };
      try { mq.addEventListener("change", listener); }
      catch (e) { try { mq.addListener(listener); } catch (e2) {} }
    }

    host[KEY] = {
      teardown: function () {
        try {
          if (mq && listener) {
            try { mq.removeEventListener("change", listener); }
            catch (e) { try { mq.removeListener(listener); } catch (e2) {} }
          }
        } catch (e) {}
      }
    };
  } catch (e) {}
})();
</script>
        """,
        height=0,
    )

