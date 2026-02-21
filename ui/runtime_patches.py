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
  --dock-bg: linear-gradient(180deg, rgba(252, 252, 253, 0.76) 0%, rgba(252, 252, 253, 0.96) 18%, rgba(252, 252, 253, 0.99) 100%);
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
}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div{
  background: var(--sidebar-bg) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="鍏抽棴"]{
  width: 34px !important;
  min-width: 34px !important;
  height: 34px !important;
  min-height: 34px !important;
  padding: 0 !important;
  border-radius: 10px !important;
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
  width: 34px !important;
  min-width: 34px !important;
  height: 34px !important;
  min-height: 34px !important;
  padding: 0 !important;
  border-radius: 10px !important;
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
  margin: -4.2rem 0 0.28rem 0;
}
.kb-sidebar-logo-img{
  width: 220px;
  max-width: 86%;
  height: auto;
  display: block;
  object-fit: contain;
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
  transform: translateZ(0);
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
  padding: 0.48rem 0.56rem 0.18rem 0.56rem;
  box-shadow: var(--dock-shadow);
  backdrop-filter: blur(5px);
  margin: 0 !important;
}
.kb-input-dock,
.kb-input-dock > div,
.kb-input-dock div[data-testid="stForm"]{
  background: var(--dock-bg) !important;
  border-color: var(--dock-border) !important;
}
.kb-input-dock textarea,
.kb-input-dock div[data-testid="stTextArea"] textarea{
  border-color: var(--input-border) !important;
}
.kb-input-dock textarea:focus,
.kb-input-dock div[data-testid="stTextArea"] textarea:focus{
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
.kb-input-dock::before{
  content: "Ask anything... (searches Markdown first)";
  display: block;
  font-size: 0.75rem;
  color: var(--muted);
  margin: 0 0 0.32rem 0.16rem;
}
.kb-input-dock div[data-testid="stTextArea"] label{ display: none !important; }
.kb-input-dock textarea{ min-height: 92px !important; border-radius: 14px !important; }
.kb-input-dock div[data-testid="stFormSubmitButton"]{ display: flex !important; justify-content: flex-end !important; margin-top: 0.38rem !important; }
.kb-input-dock div[data-testid="stFormSubmitButton"] > button{
  width: 40px !important;
  min-width: 40px !important;
  height: 40px !important;
  min-height: 40px !important;
  border-radius: 999px !important;
  padding: 0 !important;
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
    hookCopyButtons();
    hookCodeBlocks();
    hookMathClickToCopy();
  }

  tick();
  setInterval(tick, 900);
})();
</script>
        """,
        height=0,
    )

def _inject_runtime_ui_fixes(theme_mode: str, conv_id: str = "") -> None:
    mode = "dark" if str(theme_mode or "").lower() == "dark" else "light"
    conv_js = json.dumps(str(conv_id or ""))
    components.html(
        f"""
<script>
(function () {{
  const host = window.parent || window;
  const doc = host.document || document;
  const KEY = "__kbUiRuntimeFixV2";
  const mode = "{mode}";
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

  function normalizeSidebarCloseIcon() {{
    try {{
      const mainText = mode === "dark" ? "#e7eaef" : "#1f2329";
      const collapseGlyph = "\\u2039"; // left chevron

      function walkButtonsDeep(rootNode, out) {{
        try {{
          if (!rootNode) return;
          const nt = Number(rootNode.nodeType || 0);
          if (nt === 1 && String(rootNode.tagName || "").toUpperCase() === "BUTTON") {{
            out.push(rootNode);
          }}
          const sr = rootNode.shadowRoot;
          if (sr) walkButtonsDeep(sr, out);
          const children = rootNode.children || [];
          for (const ch of children) walkButtonsDeep(ch, out);
        }} catch (e) {{}}
      }}

      const allBtns = [];
      walkButtonsDeep(doc.documentElement || doc, allBtns);
      if (!allBtns.length) return;

      let sidebarRect = null;
      try {{
        const sb = doc.querySelector('section[data-testid="stSidebar"]');
        if (sb) sidebarRect = sb.getBoundingClientRect();
      }} catch (e) {{}}

      function maybeSidebarCloseBtn(b) {{
        try {{
          if (!(b instanceof Element)) return false;
          const r = b.getBoundingClientRect();
          const w = Number(r.width || 0);
          const h = Number(r.height || 0);
          if (!(w >= 24 && w <= 52 && h >= 24 && h <= 52)) return false;
          const txt = String((b.textContent || "")).replace(/\s+/g, "");
          const aria = String(b.getAttribute("aria-label") || "").toLowerCase();
          const title = String(b.getAttribute("title") || "").toLowerCase();
          const hasCloseSem = /close|collapse|hide|关闭|收起/.test(aria + " " + title);
          const hasCloseTxt = /^(?:×|x|✕|✖)$/.test(txt);
          const semHit = hasCloseSem || hasCloseTxt;

          // Geometry guard: top-left band near sidebar/header area.
          if (sidebarRect) {{
            const nearBand =
              Number(r.top || 0) <= (Number(sidebarRect.top || 0) + 140) &&
              Number(r.left || 0) <= (Number(sidebarRect.right || 0) + 180);
            if (!nearBand) return false;
            // Source-level fallback: if semantics unavailable, still patch the very top small square.
            if (semHit) return true;
            const veryTop = Number(r.top || 0) <= (Number(sidebarRect.top || 0) + 88);
            return veryTop;
          }} else {{
            const nearBand = (Number(r.top || 0) <= 150 && Number(r.left || 0) <= 560);
            if (!nearBand) return false;
            return semHit;
          }}
        }} catch (e) {{
          return false;
        }}
      }}

      for (const b of allBtns) {{
        if (!maybeSidebarCloseBtn(b)) continue;
        try {{ b.classList.add("kb-sidebar-close-btn"); }} catch (e) {{}}
        try {{
          // Source-level replacement: replace content text with left chevron directly.
          while (b.firstChild) {{
            b.removeChild(b.firstChild);
          }}
        }} catch (e) {{}}
        try {{
          b.textContent = collapseGlyph;
          b.style.setProperty("width", "34px", "important");
          b.style.setProperty("height", "34px", "important");
          b.style.setProperty("min-width", "34px", "important");
          b.style.setProperty("min-height", "34px", "important");
          b.style.setProperty("padding", "0", "important");
          b.style.setProperty("display", "inline-flex", "important");
          b.style.setProperty("align-items", "center", "important");
          b.style.setProperty("justify-content", "center", "important");
          b.style.setProperty("font-size", "26px", "important");
          b.style.setProperty("line-height", "1", "important");
          b.style.setProperty("font-weight", "600", "important");
          b.style.setProperty("font-family", "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial", "important");
          b.style.setProperty("color", mainText, "important");
          b.style.setProperty("-webkit-text-fill-color", mainText, "important");
          b.style.setProperty("text-shadow", "none", "important");
          b.style.setProperty("border", "1px solid var(--btn-border)", "important");
          b.style.setProperty("border-radius", "10px", "important");
          b.style.setProperty("background", "color-mix(in srgb, var(--sidebar-bg) 76%, var(--panel))", "important");
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

  function applyNow() {{
    clearInlineThemeForRefs();
    normalizeSidebarCloseIcon();
    clearCodeLineArtifacts();
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
      applyNow();
    }});
  }}

  let mo = null;
  function observe() {{
    if (typeof MutationObserver === "undefined") return;
    try {{
      mo = new MutationObserver(function () {{ schedule(); }});
      mo.observe(doc.body, {{ childList: true, subtree: true, attributes: true }});
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

  schedule();
  observe();
  bindCitationPopover();
  renderCiteShelf();
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

def _set_live_streaming_mode(active: bool) -> None:
    on_flag = "true" if bool(active) else "false"
    components.html(
        f"""
<script>
(function () {{
  try {{
    const host = window.parent || window;
    const root = host.document;
    if (!root || !root.body) return;
    const on = {on_flag};
    if (on) root.body.classList.add("kb-live-streaming");
    else root.body.classList.remove("kb-live-streaming");
  }} catch (e) {{}}
}})();
</script>
        """,
        height=0,
    )


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


def _inject_auto_rerun_once(*, delay_ms: int = 3500) -> None:
    delay = max(300, int(delay_ms))
    components.html(
        f"""
<script>
(function () {{
  try {{
    const root = window.top || window.parent || window;
    if (!root) return;
    
    // Keep exactly one one-shot timer per delay.
    // Streamlit rerun recreates component iframes; recursive timers can break.
    // IMPORTANT: do not use location.reload() here (it refreshes the whole page).
    const timerKey = "_kbAutoRefreshTimer_" + {delay};
    
    // Clear existing timer before creating a new one.
    if (root[timerKey]) {{
      try {{
        clearTimeout(root[timerKey]);
        root[timerKey] = null;
      }} catch (e) {{}}
    }}
    
    root[timerKey] = setTimeout(function () {{
      try {{
        root[timerKey] = null;
        const msgs = [
          {{ isStreamlitMessage: true, type: "streamlit:rerunScript" }},
          {{ type: "streamlit:rerun" }},
        ];
        const targets = [window, window.parent, root];
        for (const m of msgs) {{
          for (const t of targets) {{
            try {{ t.postMessage(m, "*"); }} catch (e) {{}}
          }}
        }}
      }} catch (e) {{}}
    }}, {delay});
  }} catch (e) {{
    // Frontend JS errors should never break the Streamlit app.
  }}
}})();
</script>
        """,
        height=0,
    )
