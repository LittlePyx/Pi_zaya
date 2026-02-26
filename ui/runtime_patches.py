from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_CHAT_DOCK_JS_PATH = Path(__file__).resolve().parent / "assets" / "chat_dock_runtime.js"
_CHAT_DOCK_JS_CACHE: str | None = None

def _init_theme_css(theme_mode: str = "dark") -> None:
    mode = "dark" if str(theme_mode or "").lower() == "dark" else "light"
    color_scheme = "dark" if mode == "dark" else "light"

    if mode == "dark":
        tokens = """
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
  --font-display: "LittleP", "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  --font-body: "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
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
  --font-display: "LittleP", "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  --font-body: "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
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
section[data-testid="stSidebar"] > div:first-child{
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"]{
  background: var(--sidebar-bg) !important;
  --text-color: var(--sidebar-strong-text) !important;
  --secondary-text-color: var(--sidebar-soft-text) !important;
  --body-text-color: var(--sidebar-strong-text) !important;
  container-type: inline-size;
  container-name: kb-sidebar;
}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div{
  background: var(--sidebar-bg) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="鍏抽棴"]{
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
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"]:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="鍏抽棴"]:hover{
  background: var(--btn-hover) !important;
  border-color: var(--blue-line) !important;
  transform: none !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:active,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"]:active,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="鍏抽棴"]:active{
  background: var(--btn-active) !important;
  border-color: var(--blue-line) !important;
  transform: none !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button svg,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"] svg,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="鍏抽棴"] svg,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button [data-testid="stIcon"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"] [data-testid="stIcon"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="鍏抽棴"] [data-testid="stIcon"]{
  display: none !important;
}
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
  margin: -3.55rem 0 0.38rem 0;
  /* Logo is pulled into the sidebar header area; do not block the native collapse button. */
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
.msg-refs details[data-testid="stExpander"]{
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
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
.msg-refs div[data-testid="stButton"] > button{
  min-height: 2.06rem !important;
  height: 2.06rem !important;
  border-radius: 10px !important;
  border-color: var(--btn-border) !important;
  background: var(--btn-bg) !important;
  color: var(--btn-text) !important;
  font-weight: 650 !important;
  box-shadow: none !important;
  padding: 0 0.74rem !important;
}
.msg-refs div[data-testid="stButton"] > button:hover{
  background: var(--btn-hover) !important;
}
.msg-refs div[data-testid="stButton"] > button:disabled{
  opacity: 0.50 !important;
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
  content: "Ctrl+Enter 鍙戦€?;
  position: absolute;
  right: 14px;
  bottom: 10px;
  font-size: 12px;
  color: var(--hint-text);
  pointer-events: none;
}

.kb-input-dock{
  position: fixed !important;
  bottom: max(10px, env(safe-area-inset-bottom, 0px));
  z-index: 40;
  left: 50%;
  transform: translateX(-50%);
  width: min(var(--content-max), calc(100vw - 1.6rem));
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
  border: 1px solid color-mix(in srgb, var(--line) 78%, transparent) !important;
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
  background: color-mix(in srgb, var(--panel) 26%, transparent) !important;
  border-color: color-mix(in srgb, var(--line) 96%, var(--blue-line)) !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.018) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:active{
  background: color-mix(in srgb, var(--panel) 34%, transparent) !important;
  border-color: color-mix(in srgb, var(--blue-line) 32%, var(--line)) !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus,
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus-visible{
  outline: none !important;
  border-color: color-mix(in srgb, var(--accent) 34%, var(--line)) !important;
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 12%, transparent) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[disabled],
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:disabled{
  opacity: 0.48 !important;
  border-color: color-mix(in srgb, var(--line) 62%, transparent) !important;
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
  opacity: 0.26 !important;
}
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

/* Conversation history expander (ChatGPT-like, low-button-feel) */
section[data-testid="stSidebar"] details.kb-conv-history-expander,
section[data-testid="stSidebar"] div[data-testid="stExpander"].kb-conv-history-expander{
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary{
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  border-radius: 9px !important;
  min-height: 34px !important;
  padding: 0.28rem 0.5rem !important;
  margin: 0 !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 96%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 96%, var(--sidebar-soft-text)) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary:hover{
  background: color-mix(in srgb, var(--panel) 28%, transparent) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander[open] > summary{
  background: color-mix(in srgb, var(--panel) 22%, transparent) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary:focus,
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary:focus-visible{
  outline: none !important;
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--line) 88%, transparent) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary p{
  margin: 0 !important;
  font-size: 0.86rem !important;
  font-weight: 560 !important;
  line-height: 1.2 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary svg{
  opacity: 0.7 !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div{
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding-top: 0.18rem !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap{
  margin: 0 !important;
  border-radius: 8px !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander button.kb-conv-row-btn{
  min-height: 31px !important;
  border-radius: 8px !important;
  padding: 0.16rem 0.42rem !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
  background: transparent !important;
  border: 0 !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap:hover button.kb-conv-row-btn,
section[data-testid="stSidebar"] details.kb-conv-history-expander .kb-conv-row-wrap:focus-within button.kb-conv-row-btn{
  background: color-mix(in srgb, var(--panel) 30%, transparent) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander button.kb-conv-row-btn.kb-current{
  background: color-mix(in srgb, var(--panel) 38%, transparent) !important;
  color: var(--sidebar-strong-text) !important;
  -webkit-text-fill-color: var(--sidebar-strong-text) !important;
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

/* Conversation history compact override (text-list, GPT-like) */
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

/* Older-history toggle should look like plain text until hover */
button.kb-history-toggle-btn,
section[data-testid="stSidebar"] button.kb-history-toggle-btn{
  width: 100% !important;
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
  font-size: 0.63rem !important;
  font-weight: 500 !important;
  justify-content: flex-start !important;
  text-align: left !important;
}
button.kb-history-toggle-btn p,
section[data-testid="stSidebar"] button.kb-history-toggle-btn p,
button.kb-history-toggle-btn span,
section[data-testid="stSidebar"] button.kb-history-toggle-btn span{
  margin: 0 !important;
  font-size: 0.63rem !important;
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
  display: flex !important;
  width: 100% !important;
  min-height: 18px !important;
  padding: 0.02rem 0.08rem !important;
  border: 0 !important;
  border-radius: 5px !important;
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
  outline: none !important;
  color: color-mix(in srgb, var(--sidebar-strong-text) 68%, var(--sidebar-soft-text)) !important;
  -webkit-text-fill-color: color-mix(in srgb, var(--sidebar-strong-text) 68%, var(--sidebar-soft-text)) !important;
  font-size: 0.61rem !important;
  font-weight: 560 !important;
  letter-spacing: 0.02em !important;
  line-height: 1 !important;
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
  font-size: 0.61rem !important;
  font-weight: 560 !important;
  letter-spacing: 0.02em !important;
  line-height: 1 !important;
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
  min-height: 30px !important;
  padding: 0.08rem 0.18rem !important;
  border-radius: 6px !important;
  border: 0 !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary:hover{
  background: color-mix(in srgb, var(--panel) 11%, transparent) !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > summary p{
  font-size: 0.75rem !important;
  font-weight: 540 !important;
  line-height: 1.08 !important;
}
section[data-testid="stSidebar"] details.kb-conv-history-expander > div{
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
        css.replace("__TOKENS__", tokens).replace("__SCHEME__", color_scheme).replace("__MODE__", mode),
        unsafe_allow_html=True,
    )


def _sync_theme_with_browser_preference() -> None:
    """
    Follow browser/system color scheme without adding an in-app theme toggle.
    We keep `_init_theme_css()` as the base stylesheet (light mode baseline),
    then override CSS variables + data-theme in the browser when dark mode is active.
    """
    dark_tokens = """
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
  --font-display: "LittleP", "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  --font-body: "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
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
    # 1) Inject CSS overrides (safe in markdown HTML)
    st.markdown(
        """
<style>
html[data-theme="light"], body[data-theme="light"]{
  color-scheme: light !important;
}
html[data-theme="dark"], body[data-theme="dark"]{
  """
        + dark_tokens
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

def _inject_copy_js() -> None:
    """
    Attach clipboard behaviors to:
    - Answer-level copy buttons (text / markdown)
    - Per-code-block copy buttons
    - Click-to-copy for LaTeX formulas rendered by KaTeX/MathJax (best effort)
    """
    components.html(
        r"""
<script>
(function () {
  const host = window.parent || window;
  const root = host.document || document;
  const TOAST_ID = "kb_toast";
  const HLJS_KEY = "__kbHljsReady";
  const HLJS_LOADING_KEY = "__kbHljsLoading";

  function ensureToast() {
    let t = root.getElementById(TOAST_ID);
    if (!t) {
      t = root.createElement("div");
      t.id = TOAST_ID;
      t.className = "kb-toast";
      t.textContent = "\u5df2\u590d\u5236";
      root.body.appendChild(t);
    }
    return t;
  }

  function toast(msg) {
    const t = ensureToast();
    t.textContent = msg || "\u5df2\u590d\u5236";
    t.classList.add("show");
    clearTimeout(t._kbTimer);
    t._kbTimer = setTimeout(() => t.classList.remove("show"), 900);
  }

  function ensureHighlightJs() {
    try {
      if (host[HLJS_KEY] && host.hljs && typeof host.hljs.highlightElement === "function") {
        return Promise.resolve(host.hljs);
      }
      if (host[HLJS_LOADING_KEY]) {
        return host[HLJS_LOADING_KEY];
      }
      host[HLJS_LOADING_KEY] = new Promise((resolve, reject) => {
        const existing = root.querySelector('script[data-kb-hljs="1"]');
        if (existing && host.hljs && typeof host.hljs.highlightElement === "function") {
          host[HLJS_KEY] = true;
          resolve(host.hljs);
          return;
        }
        const script = existing || root.createElement("script");
        if (!existing) {
          script.src = "https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/common.min.js";
          script.async = true;
          script.defer = true;
          script.dataset.kbHljs = "1";
          (root.head || root.body || root.documentElement).appendChild(script);
        }
        script.addEventListener("load", () => {
          if (host.hljs && typeof host.hljs.highlightElement === "function") {
            host[HLJS_KEY] = true;
            resolve(host.hljs);
          } else {
            reject(new Error("hljs unavailable"));
          }
        }, { once: true });
        script.addEventListener("error", () => reject(new Error("hljs load failed")), { once: true });
      }).catch(() => null);
      return host[HLJS_LOADING_KEY];
    } catch (e) {
      return Promise.resolve(null);
    }
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      toast("\u5df2\u590d\u5236");
      return true;
    } catch (e) {
      // Fallback: execCommand
      try {
        const ta = root.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        root.body.appendChild(ta);
        ta.select();
        root.execCommand("copy");
        root.body.removeChild(ta);
        toast("\u5df2\u590d\u5236");
        return true;
      } catch (e2) {
        toast("\u590d\u5236\u5931\u8d25");
        return false;
      }
    }
  }

  function hookCopyButtons() {
    const btns = root.querySelectorAll("button.kb-copybtn");
    for (const b of btns) {
      if (b.dataset.kbHooked === "1") continue;
      b.dataset.kbHooked = "1";
      b.addEventListener("click", async (e) => {
        e.preventDefault();
        const targetId = b.getAttribute("data-target");
        if (!targetId) return;
        const ta = root.getElementById(targetId);
        if (!ta) return;
        await copyText(ta.value || "");
      });
    }
  }

  function hookCodeBlocks() {
    function normalizeNativeCodeBlocks() {
      const blocks = root.querySelectorAll('div[data-testid="stCodeBlock"], div[data-testid="stCode"], .stCodeBlock');
      for (const block of blocks) {
        if (!block || !block.dataset) continue;
        if (block.dataset.kbNormalized === "1") continue;
        let codeNode = null;
        try {
          codeNode = block.querySelector("pre code, code");
        } catch (e) {
          codeNode = null;
        }
        if (!codeNode) continue;
        const txt = String(codeNode.innerText || codeNode.textContent || "");
        const codeClass = String(codeNode.className || "");
        if (!txt.trim()) continue;
        try {
          block.innerHTML = "";
          const pre = root.createElement("pre");
          pre.className = "kb-plain-code";
          const code = root.createElement("code");
          if (codeClass) code.className = codeClass;
          code.textContent = txt.replace(/\r\n/g, "\n");
          pre.appendChild(code);
          block.appendChild(pre);
          block.dataset.kbNormalized = "1";
        } catch (e) {}
      }
    }

    function escapeHtml(s) {
      return String(s || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function inferLang(raw, cls) {
      const c = String(cls || "").toLowerCase();
      const t = String(raw || "");
      if (c.includes("python") || c.includes("language-py") || c.includes("lang-py")) return "python";
      if (c.includes("javascript") || c.includes("language-js") || c.includes("lang-js") || c.includes("typescript")) return "javascript";
      if (/\b(def|import|from|return|lambda|None|True|False|async|await)\b/.test(t)) return "python";
      if (/\b(function|const|let|var|return|=>|async|await)\b/.test(t)) return "javascript";
      return "plain";
    }

    function simpleHighlight(raw, lang) {
      let s = escapeHtml(String(raw || "").replace(/\r\n/g, "\n"));
      const stash = [];
      function keep(regex, cls) {
        s = s.replace(regex, (m) => {
          const id = stash.length;
          stash.push(`<span class="${cls}">${m}</span>`);
          return `@@KBH${id}@@`;
        });
      }

      keep(/("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/g, "kb-syn-string");
      keep(/(#[^\n]*|\/\/[^\n]*)/g, "kb-syn-comment");

      let kw = [];
      if (lang === "python") {
        kw = ["and","as","assert","async","await","break","class","continue","def","del","elif","else","except","False","finally","for","from","global","if","import","in","is","lambda","None","nonlocal","not","or","pass","raise","return","True","try","while","with","yield"];
      } else if (lang === "javascript") {
        kw = ["await","break","case","catch","class","const","continue","debugger","default","delete","do","else","export","extends","finally","for","function","if","import","in","instanceof","let","new","return","super","switch","this","throw","try","typeof","var","void","while","with","yield"];
      }
      if (kw.length) {
        const kwRe = new RegExp("\\\\b(" + kw.join("|") + ")\\\\b", "g");
        s = s.replace(kwRe, '<span class="kb-syn-keyword">$1</span>');
      }
      s = s.replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="kb-syn-number">$1</span>');
      s = s.replace(/\b([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()/g, '<span class="kb-syn-func">$1</span>');

      s = s.replace(/@@KBH(\d+)@@/g, (_, idx) => stash[Number(idx)] || "");
      return s;
    }

    function applyTyporaHighlight() {
      const codes = root.querySelectorAll("pre.kb-plain-code > code, .msg-ai pre code, .stMarkdown pre code");
      ensureHighlightJs().then((hljs) => {
        for (const code of codes) {
          if (!code) continue;
          const raw = String(code.textContent || "");
          if (!raw.trim()) continue;
          const lang = inferLang(raw, code.className || "");
          const sig = String(raw.length) + ":" + raw.slice(0, 120) + ":" + lang;
          if (code.dataset.kbHlSig === sig) continue;
          code.dataset.kbHlSig = sig;
          try {
            code.removeAttribute("data-highlighted");
            code.classList.remove("hljs");
            code.textContent = raw;
            if (hljs && typeof hljs.highlightElement === "function") {
              hljs.highlightElement(code);
              code.classList.add("hljs");
            } else {
              code.innerHTML = simpleHighlight(raw, lang);
              code.classList.add("hljs");
            }
          } catch (e) {
            try {
              code.innerHTML = simpleHighlight(raw, lang);
              code.classList.add("hljs");
            } catch (e2) {}
          }
        }
      });
    }

    function hasNativeCopy(pre) {
      if (!pre) return false;
      try {
        const hostBlock = pre.closest('div[data-testid="stCodeBlock"], div[data-testid="stCode"], .stCodeBlock');
        if (hostBlock) {
          return String(hostBlock.dataset && hostBlock.dataset.kbNormalized || "") !== "1";
        }
        const host = pre.parentElement;
        if (!host) return false;
        const nativeBtn = host.querySelector(
          'button[aria-label*="copy" i], button[title*="copy" i], button[aria-label*="澶嶅埗"], button[title*="澶嶅埗"], [data-testid*="copy" i]'
        );
        return !!nativeBtn;
      } catch (e) {
        return false;
      }
    }

    normalizeNativeCodeBlocks();
    applyTyporaHighlight();

    const pres = root.querySelectorAll("pre");
    for (const pre of pres) {
      if (hasNativeCopy(pre)) {
        const oldBtn = pre.querySelector(".kb-codecopy");
        if (oldBtn) {
          try { oldBtn.remove(); } catch (e) {}
        }
        pre.dataset.kbCodeHooked = "1";
        continue;
      }
      if (pre.dataset.kbCodeHooked === "1") continue;
      const code = pre.querySelector("code");
      if (!code) continue;
      pre.dataset.kbCodeHooked = "1";
      const btn = root.createElement("button");
      btn.className = "kb-codecopy";
      btn.type = "button";
      btn.textContent = "\u590d\u5236\u4ee3\u7801";
      btn.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        await copyText(code.innerText || "");
      });
      pre.appendChild(btn);
    }
  }

  function extractTexFromKaTeX(node) {
    try {
      const ann = node.querySelector('annotation[encoding="application/x-tex"]');
      if (ann && ann.textContent) return ann.textContent;
    } catch (e) {}
    return null;
  }

  function hookMathClickToCopy() {
    const mathNodes = root.querySelectorAll(".katex, .MathJax, mjx-container");
    for (const n of mathNodes) {
      if (n.dataset && n.dataset.kbMathHooked === "1") continue;
      if (n.dataset) n.dataset.kbMathHooked = "1";
      n.style.cursor = "copy";
      n.addEventListener("click", async (e) => {
        // Prefer KaTeX annotation if available.
        const tex = extractTexFromKaTeX(n) || (n.innerText || "").trim();
        if (!tex) return;
        await copyText(tex);
        toast("\u5df2\u590d\u5236 LaTeX");
      });
    }
  }

  function tick() {
    try { if (doc && doc.hidden) return; } catch (e) {}
    hookCopyButtons();
    hookCodeBlocks();
    hookMathClickToCopy();
  }

  tick();
  setInterval(tick, 2200);
})();
</script>
        """,
        height=0,
    )

def _inject_runtime_ui_fixes(theme_mode: str, conv_id: str = "") -> None:
    raw_mode = str(theme_mode or "").lower().strip()
    mode = "auto" if raw_mode in {"", "auto", "system", "browser"} else ("dark" if raw_mode == "dark" else "light")
    conv_js = json.dumps(str(conv_id or ""))
    components.html(
        f"""
<script>
(function () {{
  const host = window.parent || window;
  const doc = host.document || document;
  const KEY = "__kbUiRuntimeFixV2";
  const modeHint = "{mode}";
  const ACTIVE_CONV_INPUT = {conv_js};
  try {{
    if (host[KEY] && typeof host[KEY].destroy === "function") {{
      host[KEY].destroy();
    }}
  }} catch (e) {{}}

  const SHELF_ROOT_KEY = "__kbCiteShelfRootV2";
  const SHELF_STORAGE_KEY = "__kb_cite_shelf_by_conv_v2";

  function normalizeConvId(v) {{
    const s = String(v || "").trim();
    if (!s) return "default";
    return s.replace(/[^a-zA-Z0-9_.:\\-]/g, "_").slice(0, 160) || "default";
  }}

  const ACTIVE_CONV_ID = normalizeConvId(ACTIVE_CONV_INPUT);

  function _readShelfStore() {{
    try {{
      const ls = host.localStorage;
      if (!ls) return {{}};
      const raw = String(ls.getItem(SHELF_STORAGE_KEY) || "");
      if (!raw) return {{}};
      const obj = JSON.parse(raw);
      if (obj && typeof obj === "object" && !Array.isArray(obj)) {{
        return obj;
      }}
    }} catch (e) {{}}
    return {{}};
  }}

  function _ensureShelfRoot() {{
    let root = null;
    try {{
      root = host[SHELF_ROOT_KEY];
    }} catch (e) {{}}
    if (!root || typeof root !== "object" || Array.isArray(root)) {{
      root = {{ byConv: {{}} }};
    }}
    const badByConv = (!root.byConv) || (typeof root.byConv !== "object") || Array.isArray(root.byConv);
    if (badByConv || Object.keys(root.byConv).length === 0) {{
      const loaded = _readShelfStore();
      if (loaded && typeof loaded === "object") {{
        root.byConv = loaded;
      }}
    }}
    if (!root.byConv || typeof root.byConv !== "object" || Array.isArray(root.byConv)) {{
      root.byConv = {{}};
    }}
    try {{
      host[SHELF_ROOT_KEY] = root;
    }} catch (e) {{}}
    return root;
  }}

  function _persistShelfRoot() {{
    try {{
      const root = _ensureShelfRoot();
      const ls = host.localStorage;
      if (!ls) return;
      ls.setItem(SHELF_STORAGE_KEY, JSON.stringify(root.byConv || {{}}));
    }} catch (e) {{}}
  }}

  function paint(el, color) {{
    if (!el || !el.style) return;
    try {{
      el.style.setProperty("color", color, "important");
      el.style.setProperty("-webkit-text-fill-color", color, "important");
      el.style.setProperty("fill", color, "important");
      el.style.setProperty("stroke", color, "important");
      el.style.setProperty("opacity", "1", "important");
      el.style.setProperty("filter", "none", "important");
    }} catch (e) {{}}
  }}

  function clearInlineThemeForRefs() {{
    try {{
      const nodes = doc.querySelectorAll(".msg-refs, .msg-refs *");
      for (const n of nodes) {{
        if (!n || !n.style) continue;
        n.style.removeProperty("color");
        n.style.removeProperty("-webkit-text-fill-color");
        n.style.removeProperty("fill");
        n.style.removeProperty("stroke");
        n.style.removeProperty("opacity");
        n.style.removeProperty("filter");
      }}
    }} catch (e) {{}}
  }}

  function resolveThemeMode() {{
    try {{
      if (modeHint === "dark" || modeHint === "light") return modeHint;
      const attrMode = String((doc.documentElement && doc.documentElement.getAttribute("data-theme")) || "").toLowerCase();
      if (attrMode === "dark" || attrMode === "light") return attrMode;
      if (host.matchMedia && host.matchMedia("(prefers-color-scheme: dark)").matches) return "dark";
    }} catch (e) {{}}
    return "light";
  }}

  function normalizeSidebarCloseIcon() {{
    try {{
      const isStale = function (el) {{
        try {{
          return !!(el && el.closest && el.closest('[data-stale="true"], .stale-element, [data-testid="staleElementOverlay"], [data-testid="stale-overlay"]'));
        }} catch (e) {{
          return false;
        }}
      }};

      // Compatibility mode on newer Streamlit: do not rewrite sidebar collapse button DOM.
      // Only clean up old injected class/inline styles from previous runs.
      const patchedBtns = Array.from(doc.querySelectorAll('button.kb-sidebar-close-btn'));
      for (const b of patchedBtns) {{
        if (!b || isStale(b)) continue;
        try {{ b.classList.remove("kb-sidebar-close-btn"); }} catch (e) {{}}
        try {{
          const props = [
            "width","height","min-width","min-height","padding","display",
            "align-items","justify-content","font-size","line-height","font-weight",
            "font-family","color","-webkit-text-fill-color","text-shadow",
            "border","border-radius","background"
          ];
          for (const p of props) b.style.removeProperty(p);
        }} catch (e) {{}}
      }}
    }} catch (e) {{}}
  }}

  function clearCodeLineArtifacts() {{
    try {{
      const blocks = doc.querySelectorAll('div[data-testid="stCodeBlock"], div[data-testid="stCode"], .stCodeBlock');
      for (const block of blocks) {{
        const hrs = block.querySelectorAll("hr");
        for (const h of hrs) {{
          if (!h || !h.style) continue;
          h.style.setProperty("display", "none", "important");
          h.style.setProperty("border", "0", "important");
          h.style.setProperty("border-bottom", "0", "important");
          h.style.setProperty("height", "0", "important");
          h.style.setProperty("margin", "0", "important");
          h.style.setProperty("padding", "0", "important");
        }}

        const nodes = block.querySelectorAll("*");
        for (const n of nodes) {{
          if (!n || !n.style) continue;
          const tag = String(n.tagName || "").toLowerCase();
          if (tag === "button" || n.closest("button")) continue;
          if (n.classList && n.classList.contains("kb-codecopy")) continue;
          n.style.setProperty("border-bottom", "0", "important");
          n.style.setProperty("box-shadow", "none", "important");
          n.style.setProperty("outline", "0", "important");
          n.style.setProperty("text-decoration", "none", "important");
          n.style.setProperty("text-decoration-line", "none", "important");
          n.style.setProperty("text-decoration-thickness", "0", "important");
          n.style.setProperty("text-underline-offset", "0", "important");
          n.style.setProperty("background-image", "none", "important");
        }}
      }}
    }} catch (e) {{}}
  }}

  function decorateConversationHistoryButtons() {{
    try {{
      const normText = (v) => String(v || "").replace(/\s+/g, " ").trim();
      try {{
        const oldWraps = doc.querySelectorAll(".kb-conv-row-wrap");
        for (const n of oldWraps) n.classList.remove("kb-conv-row-wrap");
      }} catch (e) {{}}
      try {{
        const oldPanels = doc.querySelectorAll(".kb-conv-popover-panel, .kb-conv-popover-scroll");
        for (const n of oldPanels) {{
          n.classList.remove("kb-conv-popover-panel", "kb-conv-popover-scroll");
          try {{
            n.style.removeProperty("--kb-conv-panel-width");
            n.style.removeProperty("width");
            n.style.removeProperty("max-width");
            n.style.removeProperty("min-width");
          }} catch (e) {{}}
        }}
      }} catch (e) {{}}

      const taggedBtns = doc.querySelectorAll(
        "button.kb-conv-picker-trigger, button.kb-conv-row-btn, button.kb-conv-trash-btn, button.kb-conv-menu-trigger, button.kb-history-action-btn, button.kb-history-toggle-btn, button.kb-history-new-btn, button.kb-history-danger-btn, button.kb-model-test-btn, button.kb-current"
      );
      for (const b of taggedBtns) {{
        if (!b) continue;
        try {{
          b.classList.remove("kb-conv-picker-trigger", "kb-conv-row-btn", "kb-conv-trash-btn", "kb-conv-menu-trigger", "kb-history-action-btn", "kb-history-toggle-btn", "kb-history-new-btn", "kb-history-danger-btn", "kb-model-test-btn", "kb-current");
        }} catch (e) {{}}
      }}

      const buttons = doc.querySelectorAll("button");
      for (const b of buttons) {{
        if (!b || !b.closest) continue;
        const txt = normText(b.innerText || b.textContent || "");
        if (!txt) continue;
        const inSidebar = !!b.closest('section[data-testid="stSidebar"]');
        const inSummary = !!b.closest("summary");

        const isHistoryAction = (
          txt === "新建对话" || txt === "删除本会话" || txt === "New chat" ||
          txt.includes("更早会话") || txt.includes("older chat")
        );
        const isHistoryToggle = (txt.includes("更早会话") || txt.includes("older chat"));
        const isHistoryNew = (txt === "新建对话" || txt === "New chat");
        const isHistoryDelete = (txt === "删除本会话");
        const isModelTestBtn = (txt === "测试模型连接" || txt === "Test model connection");
        if (inSidebar && isHistoryAction) {{
          try {{ b.classList.add("kb-history-action-btn"); }} catch (e) {{}}
          if (isHistoryToggle) {{
            try {{ b.classList.add("kb-history-toggle-btn"); }} catch (e) {{}}
          }}
          if (isHistoryNew) {{
            try {{ b.classList.add("kb-history-new-btn"); }} catch (e) {{}}
          }}
          if (isHistoryDelete) {{
            try {{ b.classList.add("kb-history-danger-btn"); }} catch (e) {{}}
          }}
        }}
        if (inSidebar && isModelTestBtn) {{
          try {{ b.classList.add("kb-model-test-btn"); }} catch (e) {{}}
        }}

        const looksConvLabel = /\\d{{2}}-\\d{{2}}\\s+\\d{{2}}:\\d{{2}}\\s+\\|/.test(txt);
        const isTrash = (
          txt === "🗑" || txt === "🗑️" ||
          txt === "Del" || txt === "Delete" || txt === "删除"
        );
        const isMenuTrigger = (!looksConvLabel) && (
          txt.includes("...") || txt.includes("…") || txt.includes("⋯") || txt.includes("⋮")
        );
        if (!(inSidebar || isTrash || isMenuTrigger || looksConvLabel)) continue;
        if (isTrash) {{
          try {{ b.classList.add("kb-conv-trash-btn"); }} catch (e) {{}}
          continue;
        }}
        if (isMenuTrigger) {{
          try {{ b.classList.add("kb-conv-menu-trigger"); }} catch (e) {{}}
          continue;
        }}

        if (!looksConvLabel) continue;

        if (inSummary) {{
          try {{ b.classList.add("kb-conv-picker-trigger"); }} catch (e) {{}}
          try {{
            const d = b.closest("details");
            const x = b.closest('div[data-testid="stExpander"]');
            if (d) d.classList.add("kb-conv-history-expander");
            if (x) x.classList.add("kb-conv-history-expander");
          }} catch (e) {{}}
          continue;
        }}

        // Timestamp-like buttons in sidebar are conversation rows (flat list or expander rows).
        try {{
          if (inSidebar) {{
            b.classList.add("kb-conv-row-btn");
          }}
        }} catch (e) {{}}

        // Distinguish row item by checking whether a nearby row also has a delete/menu action button.
        let hasNearbyAction = false;
        let cur = b;
        for (let k = 0; k < 4 && cur; k += 1) {{
          cur = cur.parentElement;
          if (!cur) break;
          try {{
            const rowBtns = cur.querySelectorAll ? cur.querySelectorAll("button") : [];
            for (const rb of rowBtns) {{
              const t2 = normText(rb.innerText || rb.textContent || "");
              const t2LooksConv = /\\d{{2}}-\\d{{2}}\\s+\\d{{2}}:\\d{{2}}\\s+\\|/.test(t2);
              const t2IsMenu = (!t2LooksConv) && (
                t2.includes("...") || t2.includes("…") || t2.includes("⋯") || t2.includes("⋮")
              );
              const t2IsTrash = (t2 === "🗑" || t2 === "🗑️" || t2 === "Del" || t2 === "Delete" || t2 === "删除");
              if (t2IsTrash || t2IsMenu) {{
                hasNearbyAction = true;
                break;
              }}
            }}
          }} catch (e) {{}}
          if (hasNearbyAction) break;
        }}

        try {{
          if (hasNearbyAction) {{
            b.classList.add("kb-conv-row-btn");
          }}
        }} catch (e) {{}}
      }}

      // Mark expander containers by summary text (works even if summary has no <button>).
      try {{
        const summaries = doc.querySelectorAll('section[data-testid="stSidebar"] details summary');
        for (const sm of summaries) {{
          const txt = normText(sm.innerText || sm.textContent || "");
          if (!/\\d{{2}}-\\d{{2}}\\s+\\d{{2}}:\\d{{2}}\\s+\\|/.test(txt)) continue;
          try {{
            const d = sm.closest("details");
            const x = sm.closest('div[data-testid="stExpander"]');
            if (d) d.classList.add("kb-conv-history-expander");
            if (x) x.classList.add("kb-conv-history-expander");
          }} catch (e) {{}}
        }}
      }} catch (e) {{}}

      // Mark row wrappers (common ancestor of one conversation row button + one trash button).
      try {{
        const rowBtns = doc.querySelectorAll("button.kb-conv-row-btn");
        for (const rb of rowBtns) {{
          let cur = rb;
          for (let k = 0; k < 6 && cur; k += 1) {{
            cur = cur.parentElement;
            if (!cur) break;
            let hasTrash = false;
            let hasMenu = false;
            let hasRow = false;
            try {{
              hasTrash = !!cur.querySelector("button.kb-conv-trash-btn");
              hasMenu = !!cur.querySelector("button.kb-conv-menu-trigger");
              hasRow = !!cur.querySelector("button.kb-conv-row-btn");
            }} catch (e) {{}}
            if ((hasTrash || hasMenu) && hasRow) {{
              try {{ cur.classList.add("kb-conv-row-wrap"); }} catch (e) {{}}
              break;
            }}
          }}
        }}
      }} catch (e) {{}}

      // Flat-list fallback: mark the horizontal row container even if trash/menu detection misses.
      try {{
        const rowBtns2 = doc.querySelectorAll('section[data-testid="stSidebar"] button.kb-conv-row-btn');
        for (const rb of rowBtns2) {{
          let cur = rb;
          for (let k = 0; k < 8 && cur; k += 1) {{
            cur = cur.parentElement;
            if (!cur) break;
            try {{
              const isHBlock = cur.matches && cur.matches('div[data-testid="stHorizontalBlock"]');
              if (isHBlock) {{
                cur.classList.add("kb-conv-row-wrap");
                break;
              }}
            }} catch (e) {{}}
          }}
        }}
      }} catch (e) {{}}

      // Mark the current conversation row by matching expander summary label text.
      try {{
        const expanders = doc.querySelectorAll('section[data-testid="stSidebar"] details.kb-conv-history-expander');
        for (const d of expanders) {{
          const summary = d.querySelector("summary");
          const curLabel = normText(summary ? (summary.innerText || summary.textContent || "") : "");
          if (!curLabel) continue;
          const rows = d.querySelectorAll("button.kb-conv-row-btn");
          for (const rb of rows) {{
            const rowTxt = normText(rb.innerText || rb.textContent || "");
            if (rowTxt && rowTxt === curLabel) {{
              try {{ rb.classList.add("kb-current"); }} catch (e) {{}}
              break;
            }}
          }}
        }}
      }} catch (e) {{}}

      // Fallback for flat list mode (no expander): active conversation is rendered first.
      try {{
        const curRows = doc.querySelectorAll('section[data-testid="stSidebar"] button.kb-conv-row-btn.kb-current');
        if ((!curRows) || (curRows.length === 0)) {{
          const flatRows = doc.querySelectorAll('section[data-testid="stSidebar"] button.kb-conv-row-btn');
          if (flatRows && flatRows.length > 0) {{
            try {{ flatRows[0].classList.add("kb-current"); }} catch (e) {{}}
          }}
        }}
      }} catch (e) {{}}

      // Disabled risky popover/panel width mutations: they can affect unrelated layout containers
      // on some Streamlit builds. Keep only row-level styling hooks.
    }} catch (e) {{}}
  }}

  let _kbLastHistoryDecorTs = 0;
  function applyNow() {{
    try {{
      clearInlineThemeForRefs();
      normalizeSidebarCloseIcon();
      clearCodeLineArtifacts();
      const nowTs = Date.now ? Date.now() : (+new Date());
      if ((!_kbLastHistoryDecorTs) || ((nowTs - _kbLastHistoryDecorTs) > 220)) {{
        decorateConversationHistoryButtons();
        _kbLastHistoryDecorTs = nowTs;
      }}
    }} catch (e) {{}}
  }}

  let citePopup = null;
  let citeClickBound = false;
  let citeDocClick = null;
  let citeDocKey = null;
  let citeLinkClick = null;
  let citeDragMove = null;
  let citeDragUp = null;
  let citeShelfEl = null;
  let citeShelfToggleEl = null;

  function escapeHtml(s) {{
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }}

  function safeUrl(s) {{
    const u = String(s || "").trim();
    if (!u) return "";
    const low = u.toLowerCase();
    if (low.startsWith("http://") || low.startsWith("https://")) return u;
    return "";
  }}

  function shelfState() {{
    const root = _ensureShelfRoot();
    const cid = String(ACTIVE_CONV_ID || "default");
    let cur = root.byConv[cid];
    if (!cur || typeof cur !== "object" || Array.isArray(cur)) {{
      cur = {{}};
    }}
    if (!Array.isArray(cur.items)) cur.items = [];
    if (typeof cur.open !== "boolean") cur.open = false;
    if (typeof cur.focusKey !== "string") cur.focusKey = "";
    root.byConv[cid] = cur;
    return cur;
  }}

  function saveShelfState() {{
    _persistShelfRoot();
  }}

  function shelfItemKey(rec) {{
    if (!rec || typeof rec !== "object") return "";
    const doiUrl = String(rec.doi_url || "").trim().toLowerCase();
    const doi = String(rec.doi || "").trim().toLowerCase();
    const src = String(rec.source_name || "").trim().toLowerCase();
    const num = String(rec.num || "").trim();
    const title = String(rec.title || rec.raw || "").trim().toLowerCase();
    if (doiUrl) return "durl:" + doiUrl;
    if (doi) return "doi:" + doi;
    return "ref:" + src + "|" + num + "|" + title;
  }}

  function normalizeShelfItem(payload) {{
    if (!payload || typeof payload !== "object") return null;
    const num0 = Number(payload.num || 0);
    const num = isFinite(num0) && num0 > 0 ? Math.floor(num0) : 0;
    const sourceName = String(payload.source_name || "").trim();
    const citeFmt = String(payload.cite_fmt || "").trim();
    const title = String(payload.title || "").trim();
    const authors = String(payload.authors || "").trim();
    const raw = String(payload.raw || "").trim();
    const venue = String(payload.venue || "").trim();
    const year = String(payload.year || "").trim();
    const volume = String(payload.volume || "").trim();
    const issue = String(payload.issue || "").trim();
    const pages = String(payload.pages || "").trim();
    const doi = String(payload.doi || "").trim();
    const doiUrl0 = safeUrl(String(payload.doi_url || "").trim());
    const doiUrl = doiUrl0 || (doi ? ("https://doi.org/" + doi) : "");
    function stripLeadLabel(s) {{
      let t = String(s || "").trim();
      if (!t) return "";
      for (let i = 0; i < 3; i += 1) {{
        const t2 = t.replace(/^\s*(?:\[\s*\d{1,4}\s*\]\s*){1,3}/, "").replace(/^\s*\d{1,4}\s*[.)]\s*/, "").trim();
        if (t2 === t) break;
        t = t2;
      }}
      return t;
    }}
    const main = stripLeadLabel(citeFmt || title || raw || "(no reference text)");
    const rec = {{
      num: num,
      source_name: sourceName,
      cite_fmt: citeFmt,
      title: title,
      authors: authors,
      raw: raw,
      venue: venue,
      year: year,
      volume: volume,
      issue: issue,
      pages: pages,
      doi: doi,
      doi_url: doiUrl,
      main: main,
    }};
    rec.key = shelfItemKey(rec);
    return rec;
  }}

  function isShelfItemPresent(itemKey) {{
    const k = String(itemKey || "");
    if (!k) return false;
    const st0 = shelfState();
    const items = Array.isArray(st0.items) ? st0.items : [];
    for (const it of items) {{
      if (!it || typeof it !== "object") continue;
      if (String(shelfItemKey(it) || "") === k) return true;
    }}
    return false;
  }}

  function clearCiteDrag() {{
    try {{
      if (citeDragMove) doc.removeEventListener("mousemove", citeDragMove, true);
      if (citeDragUp) doc.removeEventListener("mouseup", citeDragUp, true);
    }} catch (e) {{}}
    citeDragMove = null;
    citeDragUp = null;
    try {{
      if (doc.body) doc.body.classList.remove("kb-cite-dragging");
    }} catch (e) {{}}
  }}

  function closeCitePopup() {{
    clearCiteDrag();
    try {{
      if (citePopup && citePopup.parentNode) {{
        citePopup.parentNode.removeChild(citePopup);
      }}
    }} catch (e) {{}}
    citePopup = null;
  }}

  function removeCiteShelfDom() {{
    try {{
      if (citeShelfEl && citeShelfEl.parentNode) citeShelfEl.parentNode.removeChild(citeShelfEl);
    }} catch (e) {{}}
    try {{
      if (citeShelfToggleEl && citeShelfToggleEl.parentNode) citeShelfToggleEl.parentNode.removeChild(citeShelfToggleEl);
    }} catch (e) {{}}
    citeShelfEl = null;
    citeShelfToggleEl = null;
  }}

  function ensureCiteShelfDom() {{
    if (citeShelfEl && citeShelfToggleEl && citeShelfEl.isConnected && citeShelfToggleEl.isConnected) return;
    removeCiteShelfDom();
    try {{
      const panel = doc.createElement("aside");
      panel.className = "kb-cite-shelf";
      panel.innerHTML =
        '<div class="kb-cite-shelf-head">' +
          '<div>' +
            '<div class="kb-cite-shelf-title">文献篮</div>' +
            '<div class="kb-cite-shelf-meta">已收藏 <span class="kb-cite-shelf-count">0</span> 条</div>' +
          '</div>' +
          '<div class="kb-cite-shelf-head-actions">' +
            '<button type="button" class="kb-cite-shelf-btn kb-cite-shelf-clear">清空</button>' +
            '<button type="button" class="kb-cite-shelf-btn kb-cite-shelf-close" aria-label="Close">×</button>' +
          '</div>' +
        '</div>' +
        '<div class="kb-cite-shelf-list"></div>';

      const toggle = doc.createElement("button");
      toggle.type = "button";
      toggle.className = "kb-cite-shelf-toggle";
      toggle.textContent = "文献篮";

      doc.body.appendChild(panel);
      doc.body.appendChild(toggle);
      citeShelfEl = panel;
      citeShelfToggleEl = toggle;

      const closeBtn = panel.querySelector(".kb-cite-shelf-close");
      if (closeBtn) {{
        closeBtn.addEventListener("click", function (e) {{
          e.preventDefault();
          e.stopPropagation();
          const st0 = shelfState();
          st0.open = false;
          saveShelfState();
          renderCiteShelf();
        }});
      }}
      const clearBtn = panel.querySelector(".kb-cite-shelf-clear");
      if (clearBtn) {{
        clearBtn.addEventListener("click", function (e) {{
          e.preventDefault();
          e.stopPropagation();
          const st0 = shelfState();
          st0.items = [];
          st0.focusKey = "";
          saveShelfState();
          renderCiteShelf();
        }});
      }}
      toggle.addEventListener("click", function (e) {{
        e.preventDefault();
        e.stopPropagation();
        const st0 = shelfState();
        st0.open = !Boolean(st0.open);
        saveShelfState();
        renderCiteShelf();
      }});
    }} catch (e) {{}}
  }}

  function renderCiteShelf() {{
    ensureCiteShelfDom();
    if (!citeShelfEl) return;
    const st0 = shelfState();
    const items = Array.isArray(st0.items) ? st0.items : [];
    try {{
      citeShelfEl.classList.toggle("kb-open", Boolean(st0.open));
      if (citeShelfToggleEl) citeShelfToggleEl.classList.toggle("kb-open", Boolean(st0.open));
    }} catch (e) {{}}
    try {{
      const countEl = citeShelfEl.querySelector(".kb-cite-shelf-count");
      if (countEl) countEl.textContent = String(items.length);
      const listEl = citeShelfEl.querySelector(".kb-cite-shelf-list");
      if (!listEl) return;
      if (!items.length) {{
        listEl.innerHTML = '<div class="kb-cite-shelf-empty">从文内引用弹窗点击“加入文献篮”，这里会保存文献摘要和 DOI 链接。</div>';
        return;
      }}
      let htmlParts = "";
      for (const it of items) {{
        if (!it || typeof it !== "object") continue;
        const main = escapeHtml(String(it.main || it.title || it.raw || ""));
        const sourceName = escapeHtml(String(it.source_name || ""));
        const venue = escapeHtml(String(it.venue || ""));
        const year = escapeHtml(String(it.year || ""));
        const doi = escapeHtml(String(it.doi || ""));
        const doiUrl = safeUrl(String(it.doi_url || ""));
        const num = Number(it.num || 0);
        // `num` is numeric, so no escaping is required here.
        const leadRe = (isFinite(num) && num > 0) ? new RegExp("^\\s*\\[" + String(num) + "\\]\\s*") : null;
        const main1 = (leadRe ? main.replace(leadRe, "") : main).trim();
        const head = (isFinite(num) && num > 0 && main1 === main) ? ("[" + String(num) + "] ") : "";
        let sub = "";
        if (sourceName) sub += "source: " + sourceName;
        if (venue) sub += (sub ? " | " : "") + venue;
        if (year) sub += (sub ? " | " : "") + year;
        htmlParts += '<div class="kb-cite-shelf-item" data-kb-shelf-key="' + escapeHtml(String(it.key || "")) + '">';
        htmlParts += '<div class="kb-cite-shelf-item-title">' + head + main1 + '</div>';
        if (sub) htmlParts += '<div class="kb-cite-shelf-item-sub">' + sub + '</div>';
        htmlParts += '<div class="kb-cite-shelf-item-links">';
        if (doiUrl) {{
          htmlParts += 'DOI: <a href="' + escapeHtml(doiUrl) + '" target="_blank" rel="noopener noreferrer">' + (doi || escapeHtml(doiUrl)) + '</a>';
        }} else {{
          htmlParts += '<span>无 DOI 链接</span>';
        }}
        htmlParts += '</div></div>';
      }}
      listEl.innerHTML = htmlParts;
      const focusKey = String(st0.focusKey || "").trim();
      if (focusKey) {{
        st0.focusKey = "";
        saveShelfState();
        let target = null;
        try {{
          const nodes = listEl.querySelectorAll(".kb-cite-shelf-item[data-kb-shelf-key]");
          for (const nd of nodes) {{
            if (String(nd.getAttribute("data-kb-shelf-key") || "") === focusKey) {{
              target = nd;
              break;
            }}
          }}
        }} catch (e) {{}}
        if (target) {{
          try {{ target.scrollIntoView({{ behavior: "smooth", block: "nearest" }}); }} catch (e) {{}}
          try {{
            target.classList.remove("kb-flash");
            void target.offsetWidth;
            target.classList.add("kb-flash");
            host.setTimeout(function () {{
              try {{ target.classList.remove("kb-flash"); }} catch (e) {{}}
            }}, 1400);
          }} catch (e) {{}}
        }}
      }}
    }} catch (e) {{}}
  }}

  function openCiteShelf() {{
    const st0 = shelfState();
    st0.open = true;
    saveShelfState();
    renderCiteShelf();
  }}

  function addToCiteShelf(payload) {{
    const item = normalizeShelfItem(payload);
    if (!item) {{
      openCiteShelf();
      return {{ added: false, key: "" }};
    }}
    const st0 = shelfState();
    const cur = Array.isArray(st0.items) ? st0.items.slice() : [];
    const key = String(item.key || "");
    let exists = false;
    const next = [];
    for (const x of cur) {{
      if (!x || typeof x !== "object") continue;
      if (String(shelfItemKey(x) || "") === key) {{
        exists = true;
        next.push(x);
      }} else {{
        next.push(x);
      }}
    }}
    if (!exists) next.unshift(item);
    st0.items = next.slice(0, 120);
    st0.open = true;
    st0.focusKey = key;
    saveShelfState();
    renderCiteShelf();
    return {{ added: !exists, key: key }};
  }}

  function findCitePayload(anchorId) {{
    if (!anchorId) return null;
    try {{
      const nodes = doc.querySelectorAll(".kb-cite-data[data-kb-cite]");
      for (const n of nodes) {{
        if (!n) continue;
        if (String(n.getAttribute("data-kb-cite") || "") !== String(anchorId || "")) continue;
        const raw = String(n.getAttribute("data-kb-payload") || "");
        if (!raw) return null;
        try {{
          return JSON.parse(raw);
        }} catch (e) {{
          return null;
        }}
      }}
    }} catch (e) {{}}
    return null;
  }}

  function payloadFromAnchorFallback(a, anchorId) {{
    try {{
      if (!a) return null;
      const t = String(a.getAttribute("title") || "").trim();
      if (!t) return null;
      const txt = t.replace(/\s+/g, " ").trim();
      // Typical title format:
      // "source: X | ref [12] | Title ... | DOI: 10.xxxx/..."
      const parts = txt.split("|").map(function (x) {{ return String(x || "").trim(); }}).filter(Boolean);
      let sourceName = "";
      let num = 0;
      let doi = "";
      const mainParts = [];
      for (const p of parts) {{
        const low = p.toLowerCase();
        if (low.startsWith("source:")) {{
          sourceName = p.slice(7).trim();
          continue;
        }}
        const mRef = p.match(/^ref\s*\[(\d{{1,4}})\]$/i);
        if (mRef) {{
          const n0 = Number(mRef[1] || 0);
          if (isFinite(n0) && n0 > 0) num = Math.floor(n0);
          continue;
        }}
        if (low.startsWith("doi:")) {{
          doi = p.slice(4).trim();
          continue;
        }}
        mainParts.push(p);
      }}
      const main = mainParts.join(" | ").trim();
      const doiUrl = doi ? ("https://doi.org/" + doi.replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, "")) : "";
      return {{
        num: num,
        source_name: sourceName,
        title: "",
        raw: main || txt,
        venue: "",
        year: "",
        doi: doi,
        doi_url: doiUrl,
        anchor: String(anchorId || ""),
      }};
    }} catch (e) {{}}
    return null;
  }}

  function renderCitePopup(payload, x, y) {{
    if (!payload || typeof payload !== "object") return;
    closeCitePopup();
    renderCiteShelf();

    const num = Number(payload.num || 0);
    const citeFmt = String(payload.cite_fmt || "");
    const sourceName = String(payload.source_name || "");
    const title = String(payload.title || "");
    const authors = String(payload.authors || "");
    const raw = String(payload.raw || "");
    const venue = String(payload.venue || "");
    const year = String(payload.year || "");
    const volume = String(payload.volume || "");
    const issue = String(payload.issue || "");
    const pages = String(payload.pages || "");
    const doi = String(payload.doi || "");
    const doiUrl = String(payload.doi_url || "");

    function trimLine(s, maxLen) {{
      const t = String(s || "").replace(/\s+/g, " ").trim();
      if (t.length <= maxLen) return t;
      return t.slice(0, Math.max(0, maxLen - 3)).trimEnd() + "...";
    }}
    function stripLeadLabel(s) {{
      let t = String(s || "").trim();
      if (!t) return "";
      for (let i = 0; i < 3; i += 1) {{
        const t2 = t.replace(/^\s*(?:\[\s*\d{1,4}\s*\]\s*){1,3}/, "").replace(/^\s*\d{1,4}\s*[.)]\s*/, "").trim();
        if (t2 === t) break;
        t = t2;
      }}
      return t;
    }}

    let main = stripLeadLabel(String(citeFmt || "").trim());
    if (!main) {{
      const segs = [];
      if (authors) segs.push(stripLeadLabel(authors));
      if (title) segs.push(stripLeadLabel(title));
      let venueSeg = venue;
      if (volume) {{
        venueSeg += (venueSeg ? ", " : "") + volume;
        if (issue) venueSeg += "(" + issue + ")";
      }}
      if (pages) {{
        if (volume) venueSeg += ":" + pages;
        else venueSeg += (venueSeg ? ", " : "") + pages;
      }}
      if (year) venueSeg += (venueSeg ? " (" + year + ")" : year);
      if (venueSeg) segs.push(venueSeg);
      main = segs.join(". ").trim();
      if (main && !/[.!?]$/.test(main)) main += ".";
    }}
    if (!main) main = trimLine(stripLeadLabel(raw), 420) || "(no reference text)";
    const subParts = [];
    if (sourceName) subParts.push("source: " + sourceName);
    if (venue) subParts.push(venue);
    if (year) subParts.push(year);
    const sub = subParts.join(" | ");

    const itemForState = normalizeShelfItem(payload);
    const alreadyInShelf = Boolean(itemForState && isShelfItemPresent(itemForState.key));
    const addBtnLabel = alreadyInShelf ? "已加入文献篮" : "加入文献篮";
    const addBtnClass = alreadyInShelf ? "kb-cite-pop-add kb-added" : "kb-cite-pop-add";

    const pop = doc.createElement("div");
    pop.className = "kb-cite-pop";
    pop.innerHTML =
      '<div class="kb-cite-pop-head">' +
      '<div class="kb-cite-pop-title">[' + (isFinite(num) && num > 0 ? String(num) : "?") + "] 文内参考</div>" +
      '<button type="button" class="kb-cite-pop-close" aria-label="Close">×</button>' +
      '</div>' +
      '<div class="kb-cite-pop-main">' + escapeHtml(main) + "</div>" +
      (sub ? ('<div class="kb-cite-pop-sub">' + escapeHtml(sub) + "</div>") : "") +
      (doiUrl
        ? ('<div class="kb-cite-pop-doi">DOI: <a href="' + escapeHtml(doiUrl) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(doi || doiUrl) + "</a></div>")
        : "") +
      '<div class="kb-cite-pop-actions">' +
      '<button type="button" class="kb-cite-pop-open-shelf">打开文献篮</button>' +
      '<button type="button" class="' + addBtnClass + '">' + addBtnLabel + '</button>' +
      '</div>';

    doc.body.appendChild(pop);
    citePopup = pop;

    function clampPopup(left0, top0) {{
      const vvW = Math.max(0, host.innerWidth || doc.documentElement.clientWidth || 0);
      const vvH = Math.max(0, host.innerHeight || doc.documentElement.clientHeight || 0);
      const rr = pop.getBoundingClientRect();
      let leftX = Math.max(8, Number(left0 || 0));
      let topY = Math.max(8, Number(top0 || 0));
      if (leftX + rr.width > vvW - 8) leftX = Math.max(8, vvW - rr.width - 8);
      if (topY + rr.height > vvH - 8) topY = Math.max(8, vvH - rr.height - 8);
      return {{ left: leftX, top: topY }};
    }}

    const pos0 = clampPopup(Math.max(8, Number(x || 0) + 12), Math.max(8, Number(y || 0) + 12));
    pop.style.left = Math.round(pos0.left) + "px";
    pop.style.top = Math.round(pos0.top) + "px";

    try {{
      const head = pop.querySelector(".kb-cite-pop-head");
      if (head) {{
        head.addEventListener("mousedown", function (ev) {{
          if (Number(ev.button || 0) !== 0) return;
          const target = ev.target;
          if (target && target.closest && target.closest("button, a, input, textarea, select, label")) return;
          ev.preventDefault();
          ev.stopPropagation();
          const rr = pop.getBoundingClientRect();
          const dx = Number(ev.clientX || 0) - rr.left;
          const dy = Number(ev.clientY || 0) - rr.top;
          clearCiteDrag();
          citeDragMove = function (mv) {{
            const p1 = clampPopup(Number(mv.clientX || 0) - dx, Number(mv.clientY || 0) - dy);
            pop.style.left = Math.round(p1.left) + "px";
            pop.style.top = Math.round(p1.top) + "px";
          }};
          citeDragUp = function () {{
            clearCiteDrag();
          }};
          doc.addEventListener("mousemove", citeDragMove, true);
          doc.addEventListener("mouseup", citeDragUp, true);
          try {{
            if (doc.body) doc.body.classList.add("kb-cite-dragging");
          }} catch (e) {{}}
        }}, true);
      }}
    }} catch (e) {{}}

    try {{
      const closeBtn = pop.querySelector(".kb-cite-pop-close");
      if (closeBtn) {{
        closeBtn.addEventListener("click", function (e) {{
          e.preventDefault();
          e.stopPropagation();
          closeCitePopup();
        }});
      }}
      const openBtn = pop.querySelector(".kb-cite-pop-open-shelf");
      if (openBtn) {{
        openBtn.addEventListener("click", function (e) {{
          e.preventDefault();
          e.stopPropagation();
          openCiteShelf();
        }});
      }}
      const addBtn = pop.querySelector(".kb-cite-pop-add");
      if (addBtn) {{
        addBtn.addEventListener("click", function (e) {{
          e.preventDefault();
          e.stopPropagation();
          addToCiteShelf(payload);
          closeCitePopup();
        }});
      }}
    }} catch (e) {{}}
  }}

  function bindCitationPopover() {{
    if (citeClickBound) return;
    citeClickBound = true;

    citeLinkClick = function (e) {{
      try {{
        const target = e.target;
        const a = target && target.closest ? target.closest("a[href^='#kb-cite-']") : null;
        if (!a) return;
        const href = String(a.getAttribute("href") || "");
        const id = href.startsWith("#") ? href.slice(1) : href;
        if (!id) return;
        const payload = findCitePayload(id) || payloadFromAnchorFallback(a, id);
        if (!payload) return;
        e.preventDefault();
        e.stopPropagation();
        const x = Number(e.clientX || 0);
        const y = Number(e.clientY || 0);
        renderCitePopup(payload, x, y);
      }} catch (err) {{}}
    }};
    doc.addEventListener("click", citeLinkClick, true);

    citeDocClick = function (e) {{
      if (!citePopup) return;
      const t = e.target;
      if (t && t.closest && t.closest("a[href^='#kb-cite-']")) return;
      if (t && citePopup.contains(t)) return;
      closeCitePopup();
    }};
    citeDocKey = function (e) {{
      if (!citePopup) return;
      if (String(e.key || "").toLowerCase() === "escape") closeCitePopup();
    }};
    doc.addEventListener("click", citeDocClick, true);
    doc.addEventListener("keydown", citeDocKey, true);
  }}

  let raf = 0;
  function schedule() {{
    if (raf) return;
    raf = host.requestAnimationFrame(function () {{
      raf = 0;
      try {{ applyNow(); }} catch (e) {{}}
    }});
  }}

  let mo = null;
  function observe() {{
    if (typeof MutationObserver === "undefined") return;
    try {{
      mo = new MutationObserver(function () {{ schedule(); }});
      mo.observe(doc.body, {{ childList: true, subtree: true }});
    }} catch (e) {{}}
  }}

  function destroy() {{
    closeCitePopup();
    clearCiteDrag();
    removeCiteShelfDom();
    try {{ if (mo) mo.disconnect(); }} catch (e) {{}}
    try {{ if (raf) host.cancelAnimationFrame(raf); }} catch (e) {{}}
    try {{ if (citeLinkClick) doc.removeEventListener("click", citeLinkClick, true); }} catch (e) {{}}
    try {{ if (citeDocClick) doc.removeEventListener("click", citeDocClick, true); }} catch (e) {{}}
    try {{ if (citeDocKey) doc.removeEventListener("keydown", citeDocKey, true); }} catch (e) {{}}
    citeLinkClick = null;
    citeDocClick = null;
    citeDocKey = null;
    citeClickBound = false;
  }}

  host[KEY] = {{ destroy }};

  try {{ schedule(); }} catch (e) {{}}
  try {{ observe(); }} catch (e) {{}}
  try {{ bindCitationPopover(); }} catch (e) {{}}
  try {{ renderCiteShelf(); }} catch (e) {{}}
}})();
</script>
        """,
        height=0,
    )

def _teardown_chat_dock_runtime() -> None:
    components.html(
        """
<script>
(function () {
  try {
    const host = window.parent || window;
    const root = host.document;
    if (!root || !root.body) return;

    const NS = "__kbDockManagerStableV3";
    try {
      if (host[NS] && typeof host[NS].destroy === "function") host[NS].destroy();
      delete host[NS];
    } catch (e) {}

    try {
      root.body.classList.remove("kb-resizing");
      root.body.classList.remove("kb-live-streaming");
      root.body.classList.remove("kb-hide-stale-rerun");
    } catch (e) {}

    const docks = root.querySelectorAll(".kb-input-dock, .kb-dock-positioned");
    docks.forEach(function (el) {
      try {
        el.classList.remove("kb-input-dock", "kb-dock-positioned");
        el.style.left = "";
        el.style.right = "";
        el.style.width = "";
        el.style.transform = "";
      } catch (e) {}
    });
  } catch (e) {}
})();
</script>
        """,
        height=0,
    )

def _set_live_streaming_mode(active: bool, hide_stale: bool = False) -> None:
    on_flag = "true" if bool(active) else "false"
    hide_stale_flag = "true" if bool(hide_stale) else "false"
    components.html(
        f"""
<script>
(function () {{
  try {{
    const host = window.parent || window;
    const root = host.document;
    if (!root || !root.body) return;
    const on = {on_flag};
    const hideStale = {hide_stale_flag};
    if (on) root.body.classList.add("kb-live-streaming");
    else root.body.classList.remove("kb-live-streaming");
    if (on && hideStale) root.body.classList.add("kb-hide-stale-rerun");
    else root.body.classList.remove("kb-hide-stale-rerun");
  }} catch (e) {{}}
}})();
</script>
        """,
        height=0,
    )


def _remember_scroll_for_next_rerun(*, nonce: str = "", anchor_id: str = "") -> None:
    nonce_js = json.dumps(str(nonce or ""))
    anchor_js = json.dumps(str(anchor_id or ""))
    html_js = """
<script>
(function () {
  try {
    const host = window.parent || window;
    const doc = host.document || document;
    const store = host.sessionStorage || window.sessionStorage;
    if (!store) return;

    const NONCE = __NONCE__;
    const ANCHOR_ID = __ANCHOR_ID__;
    const sels = [
      '[data-testid="stAppViewContainer"]',
      'section.main',
      '[data-testid="stMain"]',
      '[data-testid="stMainBlockContainer"]',
      'main',
      'body',
      'html'
    ];
    let bestSel = "";
    let bestY = 0;
    for (const sel of sels) {
      let el = null;
      try { el = doc.querySelector(sel); } catch (e) {}
      if (!el) continue;
      let y0 = 0;
      try { y0 = Number(el.scrollTop || 0); } catch (e) { y0 = 0; }
      if (isFinite(y0) && y0 >= bestY) {
        bestY = y0;
        bestSel = sel;
      }
    }
    let winY = 0;
    try {
      winY = Number(
        host.scrollY ||
        host.pageYOffset ||
        (doc && doc.documentElement && doc.documentElement.scrollTop) ||
        (doc && doc.body && doc.body.scrollTop) ||
        0
      );
    } catch (e) {}
    if (!isFinite(winY)) winY = 0;
    if (!isFinite(bestY)) bestY = 0;

    store.setItem("__kb_scroll_restore_v1", JSON.stringify({
      nonce: String(NONCE || ""),
      anchorId: String(ANCHOR_ID || ""),
      winY: Math.max(0, Math.floor(winY)),
      sel: String(bestSel || ""),
      y: Math.max(0, Math.floor(bestY)),
      ts: Date.now(),
      src: "chat-finish-rerun"
    }));
  } catch (e) {}
})();
</script>
    """.replace("__NONCE__", nonce_js).replace("__ANCHOR_ID__", anchor_js)
    components.html(html_js, height=0)


def _restore_scroll_after_rerun_if_needed(*, max_age_ms: int = 10000) -> None:
    max_age = max(1000, int(max_age_ms))
    html_js = """
<script>
(function () {
  try {
    const host = window.parent || window;
    const doc = host.document || document;
    const store = host.sessionStorage || window.sessionStorage;
    if (!store) return;
    const raw = store.getItem("__kb_scroll_restore_v1");
    if (!raw) return;

    let rec = null;
    try { rec = JSON.parse(raw); } catch (e) { rec = null; }
    if (!rec || typeof rec !== "object") {
      try { store.removeItem("__kb_scroll_restore_v1"); } catch (e) {}
      return;
    }

    const y = Number(rec.y);
    const winY = Number(rec.winY);
    const sel = String(rec.sel || "");
    const anchorId = String(rec.anchorId || "");
    const ts = Number(rec.ts || 0);
    if (!isFinite(y)) {
      try { store.removeItem("__kb_scroll_restore_v1"); } catch (e) {}
      return;
    }
    if (ts > 0 && (Date.now() - ts) > __MAX_AGE__) {
      try { store.removeItem("__kb_scroll_restore_v1"); } catch (e) {}
      return;
    }

    function allTargets() {
      const targets = [];
      const sels = [
        sel,
        '[data-testid="stAppViewContainer"]',
        'section.main',
        '[data-testid="stMain"]',
        '[data-testid="stMainBlockContainer"]',
        'main',
        'body',
        'html'
      ];
      for (const s of sels) {
        if (!s) continue;
        let el = null;
        try { el = doc.querySelector(s); } catch (e) {}
        if (el && targets.indexOf(el) < 0) targets.push(el);
      }
      return targets;
    }

    let tries = 0;
    function apply() {
      try {
        const targetWinY = isFinite(winY) ? winY : y;
        if (typeof host.scrollTo === "function") host.scrollTo(0, targetWinY);
        if (doc && doc.documentElement) doc.documentElement.scrollTop = targetWinY;
        if (doc && doc.body) doc.body.scrollTop = targetWinY;
        const targets = allTargets();
        for (const el of targets) {
          try {
            if ("scrollTop" in el) el.scrollTop = y;
          } catch (e) {}
        }
        if (anchorId) {
          let anchor = null;
          try { anchor = doc.getElementById(anchorId); } catch (e) {}
          if (anchor && typeof anchor.scrollIntoView === "function") {
            try { anchor.scrollIntoView({ block: "nearest", inline: "nearest" }); } catch (e) {}
          }
        }
      } catch (e) {}
      tries += 1;
      if (tries < 42) {
        try { host.requestAnimationFrame(apply); }
        catch (e) { setTimeout(apply, 40); }
      } else {
        try { store.removeItem("__kb_scroll_restore_v1"); } catch (e) {}
      }
    }

    try { host.requestAnimationFrame(apply); }
    catch (e) { setTimeout(apply, 0); }
  } catch (e) {}
})();
</script>
    """.replace("__MAX_AGE__", str(max_age))
    components.html(html_js, height=0)


def _inject_chat_dock_runtime() -> None:
    global _CHAT_DOCK_JS_CACHE
    if _CHAT_DOCK_JS_CACHE is None:
        try:
            _CHAT_DOCK_JS_CACHE = _CHAT_DOCK_JS_PATH.read_text(encoding="utf-8")
        except Exception:
            _CHAT_DOCK_JS_CACHE = ""
    js = str(_CHAT_DOCK_JS_CACHE or "").strip()
    if not js:
        return
    components.html("<script>\n" + js + "\n</script>", height=0)


def _inject_auto_rerun_once(*, delay_ms: int = 3500, pulse_button_label: str = "", nonce: str = "") -> None:
    delay = max(300, int(delay_ms))
    pulse_label_js = json.dumps(str(pulse_button_label or ""))
    nonce_js = json.dumps(str(nonce or ""))
    components.html(
        f"""
<script>
(function () {{
  try {{
    const delay = {delay};
    const pulseLabel = {pulse_label_js};
    const nonce = {nonce_js}; // force srcdoc changes across reruns
    const hostCandidates = [];
    try {{ if (window.parent) hostCandidates.push(window.parent); }} catch (e) {{}}
    try {{ if (window.top && window.top !== window.parent) hostCandidates.push(window.top); }} catch (e) {{}}
    hostCandidates.push(window);

    const hostDocs = [];
    for (const h of hostCandidates) {{
      try {{
        if (h && h.document && hostDocs.indexOf(h.document) < 0) hostDocs.push(h.document);
      }} catch (e) {{}}
    }}

    function findPulseButton(doc) {{
      try {{
        if (!doc || !pulseLabel) return null;
        const buttons = doc.querySelectorAll("button");
        for (const btn of buttons) {{
          try {{
            const txt = String(btn.textContent || "").replace(/\\s+/g, " ").trim();
            if (txt === pulseLabel) return btn;
          }} catch (e) {{}}
        }}
      }} catch (e) {{}}
      return null;
    }}

    function hidePulseButton(btn) {{
      try {{
        if (!btn) return;
        btn.setAttribute("data-kb-auto-rerun-pulse", "1");
        btn.tabIndex = -1;
        btn.style.pointerEvents = "none";
        btn.style.opacity = "0";
        btn.style.width = "1px";
        btn.style.minWidth = "1px";
        btn.style.height = "1px";
        btn.style.minHeight = "1px";
        btn.style.padding = "0";
        btn.style.margin = "0";
        const wrap = btn.closest('[data-testid="stButton"]') || btn.parentElement;
        if (wrap) {{
          wrap.style.height = "0";
          wrap.style.minHeight = "0";
          wrap.style.overflow = "hidden";
          wrap.style.margin = "0";
          wrap.style.padding = "0";
        }}
      }} catch (e) {{}}
    }}

    function hidePulseButtons() {{
      for (const d of hostDocs) {{
        const btn = findPulseButton(d);
        if (btn) hidePulseButton(btn);
      }}
    }}

    function clickPulseButton() {{
      for (const d of hostDocs) {{
        const btn = findPulseButton(d);
        if (!btn) continue;
        hidePulseButton(btn);
        try {{
          if (!btn.disabled) {{
            btn.click();
            return true;
          }}
        }} catch (e) {{}}
      }}
      return false;
    }}

    // Hide the pulse button immediately to avoid visual pollution near page bottom.
    try {{ hidePulseButtons(); }} catch (e) {{}}

    // Keep exactly one one-shot timer per delay (best-effort).
    // Do NOT store timers only on window.top; some environments expose a WindowProxy
    // that rejects property writes, which silently kills auto-refresh.
    const timerKey = "_kbAutoRefreshTimer_" + delay;
    try {{
      if (window[timerKey]) {{
        clearTimeout(window[timerKey]);
        window[timerKey] = null;
      }}
    }} catch (e) {{}}

    window[timerKey] = window.setTimeout(function () {{
      try {{
        try {{ window[timerKey] = null; }} catch (e) {{}}
        // Preferred path for Streamlit<=1.12 in this app: click a hidden Streamlit button.
        if (clickPulseButton()) return;

        // Fallback path: try internal Streamlit postMessage rerun hooks (works only on some versions/builds).
        const msgs = [
          {{ isStreamlitMessage: true, type: "streamlit:rerunScript" }},
          {{ type: "streamlit:rerunScript" }},
          {{ type: "streamlit:rerun" }},
          {{ isStreamlitMessage: true, type: "streamlit:rerun" }},
          {{ type: "streamlit:scriptRequestRerun" }},
        ];
        for (const m of msgs) {{
          for (const t of hostCandidates) {{
            try {{
              if (t && typeof t.postMessage === "function") t.postMessage(m, "*");
            }} catch (e) {{}}
          }}
        }}
        // Some Streamlit builds only react when the iframe is focused.
        try {{ window.focus(); }} catch (e) {{}}
      }} catch (e) {{}}
    }}, delay);
  }} catch (e) {{
    // Frontend JS errors should never break the Streamlit app.
  }}
}})();
</script>
        """,
        height=0,
    )
