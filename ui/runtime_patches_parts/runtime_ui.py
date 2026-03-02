from __future__ import annotations

import json

import streamlit.components.v1 as components


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

  function decorateReferenceActionButtons() {{
    try {{
      const normText = (v) => String(v || "").replace(/\s+/g, " ").trim().toLowerCase();
      const bindRefGlassHover = (btn) => {{
        if (!btn || !btn.addEventListener || !btn.dataset) return;
        if (String(btn.dataset.kbRefGlassBound || "") === "1") return;
        btn.dataset.kbRefGlassBound = "1";

        const baseBg = "linear-gradient(180deg, color-mix(in srgb, var(--panel) 95%, var(--blue-weak) 5%), color-mix(in srgb, var(--panel) 92%, var(--blue-weak) 8%))";
        const hoverBg = "linear-gradient(180deg, color-mix(in srgb, var(--panel) 92%, var(--blue-weak) 8%), color-mix(in srgb, var(--panel) 88%, var(--blue-weak) 12%))";

        const ensureHostLayer = () => {{
          try {{
            btn.style.setProperty("position", "relative", "important");
            btn.style.setProperty("overflow", "hidden", "important");
            btn.style.setProperty("isolation", "isolate", "important");
            const kids = btn.querySelectorAll(":scope > *");
            for (const k of kids) {{
              if (!k || !k.style) continue;
              k.style.setProperty("position", "relative", "important");
              k.style.setProperty("z-index", "1", "important");
            }}
          }} catch (e) {{}}
        }};

        const runSheen = () => {{
          try {{
            ensureHostLayer();
            let old = null;
            try {{
              old = btn.querySelector(":scope > .kb-ref-glass-sheen-inline");
            }} catch (e) {{}}
            if (old && old.parentNode) {{
              try {{ old.parentNode.removeChild(old); }} catch (e) {{}}
            }}

            const sheen = doc.createElement("span");
            sheen.className = "kb-ref-glass-sheen-inline";
            sheen.setAttribute("aria-hidden", "true");
            sheen.style.position = "absolute";
            sheen.style.left = "-44%";
            sheen.style.top = "-138%";
            sheen.style.width = "42%";
            sheen.style.height = "376%";
            sheen.style.zIndex = "0";
            sheen.style.pointerEvents = "none";
            sheen.style.opacity = "0.0";
            sheen.style.background = "linear-gradient(90deg, rgba(120,176,255,0.00) 0%, rgba(120,176,255,0.26) 48%, rgba(120,176,255,0.00) 100%)";
            sheen.style.transform = "translate3d(-180%,0,0) rotate(24deg)";
            sheen.style.transition = "transform 760ms cubic-bezier(0.22,0.61,0.36,1.0), opacity 180ms ease";
            btn.appendChild(sheen);

            const raf = host.requestAnimationFrame || window.requestAnimationFrame || function (f) {{ return setTimeout(f, 16); }};
            raf(function () {{
              raf(function () {{
                try {{
                  sheen.style.opacity = "0.34";
                  sheen.style.transform = "translate3d(295%,0,0) rotate(24deg)";
                }} catch (e) {{}}
              }});
            }});

            (host.setTimeout || window.setTimeout)(function () {{
              try {{
                if (sheen && sheen.parentNode) sheen.parentNode.removeChild(sheen);
              }} catch (e) {{}}
            }}, 860);
          }} catch (e) {{}}
        }};

        try {{
          btn.addEventListener("mouseenter", function () {{
            try {{
              btn.style.setProperty("background", hoverBg, "important");
            }} catch (e) {{}}
            runSheen();
          }}, {{ passive: true }});
          btn.addEventListener("mouseleave", function () {{
            try {{
              btn.style.setProperty("background", baseBg, "important");
            }} catch (e) {{}}
          }}, {{ passive: true }});
          btn.addEventListener("focus", function () {{
            runSheen();
          }}, {{ passive: true }});
        }} catch (e) {{}}
      }};
      const applyRefBtnStyle = (btn) => {{
        if (!btn || !btn.style) return;
        try {{
          btn.style.setProperty("border", "none", "important");
          btn.style.setProperty("border-color", "transparent", "important");
          btn.style.setProperty("border-width", "0", "important");
          btn.style.setProperty("border-style", "none", "important");
          btn.style.setProperty("border-radius", "6px", "important");
          btn.style.setProperty("box-shadow", "none", "important");
          btn.style.setProperty(
            "background",
            "linear-gradient(180deg, color-mix(in srgb, var(--panel) 82%, transparent), color-mix(in srgb, var(--panel) 68%, transparent))",
            "important",
          );
          btn.style.setProperty("backdrop-filter", "saturate(120%) blur(8px)", "important");
          btn.style.setProperty("-webkit-backdrop-filter", "saturate(120%) blur(8px)", "important");
          btn.style.setProperty("color", "var(--text-main)", "important");
          btn.style.setProperty("-webkit-text-fill-color", "var(--text-main)", "important");
          btn.style.setProperty("min-height", "2.02rem", "important");
          btn.style.setProperty("height", "2.02rem", "important");
          btn.style.setProperty("font-weight", "665", "important");
          btn.style.setProperty("letter-spacing", "0.008em", "important");
          btn.style.setProperty("outline", "none", "important");
        }} catch (e) {{}}
      }};
      const applyRefBtnWrapStyle = (btn) => {{
        if (!btn || !btn.closest) return;
        try {{
          const wrap = btn.closest('div[data-testid="stButton"], div.stButton');
          if (!wrap || !wrap.style) return;
          wrap.style.setProperty("border", "none", "important");
          wrap.style.setProperty("border-color", "transparent", "important");
          wrap.style.setProperty("box-shadow", "none", "important");
          wrap.style.setProperty("background", "transparent", "important");
          wrap.style.setProperty("outline", "none", "important");
        }} catch (e) {{}}
      }};
      try {{
        const oldTagged = doc.querySelectorAll("button.kb-ref-action-btn, button.kb-ref-open-btn, button.kb-ref-cite-btn");
        for (const b of oldTagged) {{
          if (!b || !b.classList) continue;
          b.classList.remove("kb-ref-action-btn", "kb-ref-open-btn", "kb-ref-cite-btn");
        }}
      }} catch (e) {{}}

      const headers = doc.querySelectorAll(".kb-ref-header-block");
      for (const h of headers) {{
        if (!h || !h.closest) continue;
        const row = h.closest('div[data-testid="stHorizontalBlock"]');
        if (!row) continue;
        const btns = row.querySelectorAll('div[data-testid="stButton"] > button, div.stButton > button');
        for (const b of btns) {{
          if (!b || !b.classList) continue;
          const txt = normText(b.innerText || b.textContent || "");
          if (!txt) continue;
          b.classList.add("kb-ref-action-btn");
          applyRefBtnStyle(b);
          applyRefBtnWrapStyle(b);
          bindRefGlassHover(b);
          if (txt === "open" || txt === "打开") {{
            b.classList.add("kb-ref-open-btn");
          }} else if (txt === "cite" || txt === "close" || txt === "引用" || txt === "关闭") {{
            b.classList.add("kb-ref-cite-btn");
          }}
        }}
      }}

      // Fallback: match by visible label in main area even if header marker is not found.
      const allBtns = doc.querySelectorAll("button");
      for (const b of allBtns) {{
        if (!b || !b.classList || !b.closest) continue;
        if (b.closest('section[data-testid="stSidebar"]')) continue;
        const txt = normText(b.innerText || b.textContent || "");
        if (!(txt === "open" || txt === "cite" || txt === "close" || txt === "打开" || txt === "引用" || txt === "关闭")) continue;
        b.classList.add("kb-ref-action-btn");
        if (txt === "open" || txt === "打开") {{
          b.classList.add("kb-ref-open-btn");
        }} else {{
          b.classList.add("kb-ref-cite-btn");
        }}
        applyRefBtnStyle(b);
        applyRefBtnWrapStyle(b);
        bindRefGlassHover(b);
      }}
    }} catch (e) {{}}
  }}

  function injectTrashNoBorderCSS() {{
    try {{
      const targetDoc = doc;
      const id = "kb-trash-noborder-css";
      if (targetDoc.getElementById(id)) return;
      const style = targetDoc.createElement("style");
      style.id = id;
      style.textContent = (
        "section[data-testid=\\"stSidebar\\"] .kb-trash-cell, section[data-testid=\\"stSidebar\\"] .kb-trash-cell *, section[data-testid=\\"stSidebar\\"] .kb-trash-wrap, section[data-testid=\\"stSidebar\\"] .kb-trash-wrap *, section[data-testid=\\"stSidebar\\"] button.kb-history-trash-btn {{ border: none !important; border-width: 0 !important; box-shadow: none !important; outline: none !important; background: transparent !important; background-color: transparent !important; }} "
        + "section[data-testid=\\"stSidebar\\"] button.kb-history-trash-btn:hover, section[data-testid=\\"stSidebar\\"] .kb-trash-wrap:hover button, "
        + "section.stSidebar button.kb-history-trash-btn:hover, section.stSidebar .kb-trash-wrap:hover button {{ "
        + "color: #6baf6b !important; -webkit-text-fill-color: #6baf6b !important; background: transparent !important; background-color: transparent !important; border: none !important; "
        + "filter: brightness(0) saturate(100%) invert(58%) sepia(35%) saturate(1200%) hue-rotate(95deg) !important; }}"
      );
      (targetDoc.head || targetDoc.documentElement).appendChild(style);
    }} catch (e) {{}}
  }}

  function injectActionButtonsCSS() {{
    try {{
      const targetDoc = doc;
      const id = "kb-action-buttons-css";
      const old = targetDoc.getElementById(id);
      if (old) old.remove();
      const style = targetDoc.createElement("style");
      style.id = id;
      style.textContent = (
        "/* 新建会话样式由 theme_history_overrides 统一（仿 open/cite），此处仅保留兜底 */ "
        + "section[data-testid=\\"stSidebar\\"] div:has(.kb-history-actions) + div div[data-testid=\\"stHorizontalBlock\\"] div[data-testid=\\"stButton\\"] > button {{ "
        + "border: none !important; box-shadow: none !important; outline: none !important; }} "
        + "section[data-testid=\\"stSidebar\\"] div:has(.kb-history-actions) + div div[data-testid=\\"stHorizontalBlock\\"] div[data-testid=\\"stButton\\"] > button:hover {{ "
        + "border: none !important; color: var(--blue-line) !important; -webkit-text-fill-color: var(--blue-line) !important; }}"
      );
      (targetDoc.head || targetDoc.documentElement).appendChild(style);
    }} catch (e) {{}}
  }}

  function decorateConversationHistoryButtons() {{
    try {{
      injectTrashNoBorderCSS();
      injectActionButtonsCSS();
      const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
      if (!sidebar) return;

      try {{
        const staleTags = sidebar.querySelectorAll(
          ".kb-history-actions-block, .kb-history-row-block, .kb-history-row-block-current, .kb-history-toggle-block, .kb-history-older-block, .kb-history-action-btn-inline, .kb-history-row-btn, .kb-history-trash-btn, .kb-history-toggle-btn"
        );
        for (const n of staleTags) {{
          if (!n || !n.classList) continue;
          n.classList.remove(
            "kb-history-actions-block",
            "kb-history-row-block",
            "kb-history-row-block-current",
            "kb-history-toggle-block",
            "kb-history-older-block",
            "kb-history-action-btn-inline",
            "kb-history-row-btn",
            "kb-history-trash-btn",
            "kb-history-toggle-btn"
          );
        }}
      }} catch (e) {{}}

      const findElementContainer = (marker) => {{
        let cur = marker;
        for (let i = 0; i < 10 && cur; i += 1) {{
          try {{
            const isElemContainer =
              !!(cur.matches && (
                cur.matches('div[data-testid="stElementContainer"]') ||
                cur.matches('div[data-testid="element-container"]') ||
                (cur.classList && cur.classList.contains("stElementContainer"))
              ));
            if (isElemContainer) return cur;
          }} catch (e) {{}}
          cur = cur.parentElement;
        }}
        return (marker && marker.parentElement) ? marker.parentElement : null;
      }};

      const findNextHorizontalBlock = (container) => {{
        let sib = container ? container.nextElementSibling : null;
        for (let i = 0; i < 8 && sib; i += 1) {{
          try {{
            if (sib.matches && sib.matches('div[data-testid="stHorizontalBlock"]')) return sib;
            const nested = sib.querySelector ? sib.querySelector('div[data-testid="stHorizontalBlock"]') : null;
            if (nested) return nested;
          }} catch (e) {{}}
          sib = sib.nextElementSibling;
        }}
        const isAfterNode = (base, candidate) => {{
          try {{
            if (!base || !candidate || base === candidate) return false;
            const rel = base.compareDocumentPosition(candidate);
            return !!(rel & Node.DOCUMENT_POSITION_FOLLOWING);
          }} catch (e) {{}}
          return false;
        }};
        const scopes = [];
        try {{
          if (container && container.parentElement) scopes.push(container.parentElement);
        }} catch (e) {{}}
        try {{
          const vb = container && container.closest ? container.closest('div[data-testid="stVerticalBlock"]') : null;
          if (vb) scopes.push(vb);
        }} catch (e) {{}}
        scopes.push(sidebar);
        for (const scope of scopes) {{
          if (!scope || !scope.querySelectorAll) continue;
          try {{
            const blocks = scope.querySelectorAll('div[data-testid="stHorizontalBlock"]');
            for (const cand of blocks) {{
              if (isAfterNode(container, cand)) return cand;
            }}
          }} catch (e) {{}}
        }}
        return null;
      }};

      const findNextBlockForToggle = (container) => {{
        let block = findNextHorizontalBlock(container);
        if (block) return block;
        let sib = container ? container.nextElementSibling : null;
        for (let i = 0; i < 8 && sib; i += 1) {{
          try {{
            const hasBtn = sib.querySelector && sib.querySelector('div[data-testid="stButton"] > button, div.stButton > button');
            if (hasBtn) return sib;
          }} catch (e) {{}}
          sib = sib ? sib.nextElementSibling : null;
        }}
        return null;
      }};

      const bindMarkerToBlock = (markerSelector, blockClass, onBound) => {{
        const markers = sidebar.querySelectorAll(markerSelector);
        for (const marker of markers) {{
          const container = findElementContainer(marker);
          const block = findNextHorizontalBlock(container);
          if (!block || !block.classList) continue;
          try {{ block.classList.add(blockClass); }} catch (e) {{}}
          if (typeof onBound === "function") {{
            try {{ onBound(marker, block); }} catch (e) {{}}
          }}
        }}
      }};

      const bindToggleMarkerToBlock = (markerSelector, blockClass, onBound) => {{
        const markers = sidebar.querySelectorAll(markerSelector);
        for (const marker of markers) {{
          const container = findElementContainer(marker);
          const block = findNextBlockForToggle(container) || findNextHorizontalBlock(container);
          if (block && block.classList) {{
            try {{ block.classList.add(blockClass); }} catch (e) {{}}
            if (typeof onBound === "function") {{
              try {{ onBound(marker, block); }} catch (e) {{}}
            }}
          }}
        }}
      }};

      bindToggleMarkerToBlock(".kb-history-actions", "kb-history-actions-block", function (_marker, block) {{
        if (!block) return;
        const btns = block.querySelectorAll('div[data-testid="stButton"] > button, button');
        for (const b of btns) {{
          try {{ b.classList.add("kb-history-action-btn-inline"); }} catch (e) {{}}
        }}
        /* 强制新建会话行及按钮容器占满侧边栏宽度（覆盖 Streamlit 内联样式） */
        try {{
          if (block.style) {{
            block.style.setProperty("width", "100%", "important");
            block.style.setProperty("maxWidth", "100%", "important");
            block.style.setProperty("boxSizing", "border-box", "important");
          }}
          const stBtn = block.querySelector('div[data-testid="stButton"], div.stButton');
          if (stBtn && stBtn.style) {{
            stBtn.style.setProperty("width", "100%", "important");
            stBtn.style.setProperty("maxWidth", "100%", "important");
          }}
          let parent = block.parentElement;
          for (let i = 0; i < 8 && parent && parent !== sidebar; i++) {{
            if (parent.style) {{
              parent.style.setProperty("width", "100%", "important");
              parent.style.setProperty("maxWidth", "100%", "important");
            }}
            parent = parent.parentElement;
          }}
        }} catch (e) {{}}
      }});

      bindMarkerToBlock(".kb-history-row", "kb-history-row-block", function (marker, block) {{
        try {{
          if (marker && marker.classList && marker.classList.contains("kb-history-row-current")) {{
            block.classList.add("kb-history-row-block-current");
          }}
        }} catch (e) {{}}
        let rowBtn = null;
        let trashBtn = null;
        try {{
          rowBtn = block.querySelector(
            'div[data-testid="column"]:first-child div[data-testid="stButton"] > button, div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button'
          );
        }} catch (e) {{}}
        try {{
          trashBtn = block.querySelector(
            'div[data-testid="column"]:last-child div[data-testid="stButton"] > button, div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button'
          );
        }} catch (e) {{}}
        if (!rowBtn || !trashBtn) {{
          try {{
            const btnNodes = block.querySelectorAll('div[data-testid="stButton"] > button, button');
            const btns = [];
            for (const b of btnNodes) {{
              if (!b || !b.classList) continue;
              if (btns.indexOf(b) >= 0) continue;
              btns.push(b);
            }}
            if (!rowBtn && btns.length) rowBtn = btns[0];
            if (!trashBtn && btns.length > 1) trashBtn = btns[btns.length - 1];
          }} catch (e) {{}}
        }}
        try {{
          if (rowBtn && rowBtn.classList) rowBtn.classList.add("kb-history-row-btn");
        }} catch (e) {{}}
        try {{
          if (trashBtn && trashBtn.classList) {{
            trashBtn.classList.add("kb-history-trash-btn");
          }}
          const stBtnWrap = trashBtn ? (trashBtn.closest('div[data-testid="stButton"]') || trashBtn.closest('div.stButton') || trashBtn.parentElement) : null;
          if (stBtnWrap && stBtnWrap.classList) stBtnWrap.classList.add("kb-trash-wrap");
          const lastCol = block ? block.querySelector('div[data-testid="column"]:last-child, div[data-testid="stColumn"]:last-child') : null;
          if (lastCol && lastCol.classList) lastCol.classList.add("kb-trash-cell");
          const clearBorder = (el) => {{
            if (!el || !el.style) return;
            el.style.setProperty("border", "none", "important");
            el.style.setProperty("border-width", "0", "important");
            el.style.setProperty("border-style", "none", "important");
            el.style.setProperty("box-shadow", "none", "important");
            el.style.setProperty("outline", "none", "important");
            el.style.setProperty("background", "transparent", "important");
            el.style.setProperty("background-color", "transparent", "important");
          }};
          if (trashBtn) clearBorder(trashBtn);
          let p = trashBtn ? trashBtn.parentElement : null;
          while (p && p !== block) {{
            clearBorder(p);
            if (p.querySelectorAll) {{
              try {{ p.querySelectorAll("*").forEach(clearBorder); }} catch (e) {{}}
            }}
            p = p.parentElement;
          }}
          if (lastCol) {{ clearBorder(lastCol); try {{ lastCol.querySelectorAll("*").forEach(clearBorder); }} catch (e) {{}} }}
        }} catch (e) {{}}
      }});
      try {{
        sidebar.querySelectorAll("button").forEach(function (btn) {{
          if (btn.textContent && btn.textContent.indexOf(String.fromCodePoint(0x1F5D1)) >= 0) {{
            btn.classList.add("kb-history-trash-btn");
            const wrap = btn.closest('div[data-testid="stButton"], div.stButton');
            if (wrap) wrap.classList.add("kb-trash-wrap");
            const cell = btn.closest('div[data-testid="column"]:last-child, div[data-testid="stColumn"]:last-child');
            if (cell) cell.classList.add("kb-trash-cell");
          }}
        }});
      }} catch (e) {{}}

      bindToggleMarkerToBlock(".kb-history-toggle-marker", "kb-history-toggle-block", function (_marker, block) {{
        const btn = block ? block.querySelector('div[data-testid="stButton"] > button, div.stButton > button, button') : null;
        if (btn && btn.classList) try {{ btn.classList.add("kb-history-toggle-btn"); }} catch (e) {{}}
      }});
      /* Decorate ALL "更早会话" rows by structure only: no marker binding, so no row is missed. */
      try {{
        const olderListEl = sidebar.querySelector(".kb-history-older-list");
        if (!olderListEl) {{}} else {{
          let root = olderListEl;
          for (let i = 0; i < 25 && root; i += 1) {{
            root = root.parentElement;
            if (!root || !root.querySelectorAll) continue;
            const allBlocks = root.querySelectorAll('div[data-testid="stHorizontalBlock"]');
            const candidates = [];
            for (const block of allBlocks) {{
              if (!(olderListEl.compareDocumentPosition(block) & Node.DOCUMENT_POSITION_FOLLOWING)) continue;
              const cols = block.querySelectorAll('div[data-testid="column"], div[data-testid="stColumn"]');
              if (!cols || cols.length < 2) continue;
              candidates.push(block);
            }}
            if (candidates.length) {{
              const applyRowBlockStyles = (block) => {{
                if (!block || !block.classList) return;
                const cols = block.querySelectorAll('div[data-testid="column"], div[data-testid="stColumn"]');
                if (!cols || cols.length < 2) return;
                block.classList.add("kb-history-row-block");
                if (block.style) {{
                  block.style.setProperty("display", "flex", "important");
                  block.style.setProperty("width", "100%", "important");
                  block.style.setProperty("box-sizing", "border-box", "important");
                }}
                const firstCol = block.querySelector('div[data-testid="column"]:first-child, div[data-testid="stColumn"]:first-child');
                if (firstCol && firstCol.style) {{
                  firstCol.style.setProperty("flex", "1 1 0%", "important");
                  firstCol.style.setProperty("min-width", "0", "important");
                }}
                const rowBtn = block.querySelector('div[data-testid="column"]:first-child div[data-testid="stButton"] > button, div[data-testid="stColumn"]:first-child div[data-testid="stButton"] > button');
                const trashBtn = block.querySelector('div[data-testid="column"]:last-child div[data-testid="stButton"] > button, div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button');
                const btns = block.querySelectorAll ? block.querySelectorAll('div[data-testid="stButton"] > button, div.stButton > button, button') : [];
                const r = rowBtn || (btns[0] || null);
                const t = trashBtn || (btns.length >= 2 ? btns[btns.length - 1] : null);
                if (r && r.classList) r.classList.add("kb-history-row-btn");
                if (t && t.classList) {{
                  t.classList.add("kb-history-trash-btn");
                  const wrap = t.closest('div[data-testid="stButton"]') || t.closest('div.stButton');
                  if (wrap && wrap.classList) wrap.classList.add("kb-trash-wrap");
                  const lastCol = block.querySelector('div[data-testid="column"]:last-child, div[data-testid="stColumn"]:last-child');
                  if (lastCol && lastCol.classList) {{
                    lastCol.classList.add("kb-trash-cell");
                    if (lastCol.style) {{
                      lastCol.style.setProperty("display", "flex", "important");
                      lastCol.style.setProperty("justify-content", "flex-end", "important");
                      lastCol.style.setProperty("align-items", "center", "important");
                      lastCol.style.setProperty("flex", "0 0 auto", "important");
                      lastCol.style.setProperty("margin-left", "auto", "important");
                    }}
                  }}
                }}
              }};
              for (const block of candidates) applyRowBlockStyles(block);
              break;
            }}
            if (root === sidebar) break;
          }}
        }}
      }} catch (e) {{}}
      /* Fallback: by trash-emoji button, add classes only (no inline styles). */
      try {{
        const trashChar = String.fromCodePoint(0x1F5D1);
        const allTrashBtns = sidebar.querySelectorAll("button");
        const seenBlocks = new Set();
        for (const btn of allTrashBtns) {{
          if ((btn.textContent || "").trim().indexOf(trashChar) < 0) continue;
          let block = btn.closest('div[data-testid="stHorizontalBlock"]') || btn.closest('div[data-testid="horizontalBlock"]');
          if (!block) {{
            let p = btn.parentElement;
            for (let i = 0; i < 10 && p && p !== sidebar; i += 1) {{
              const cols = p.querySelectorAll('div[data-testid="column"], div[data-testid="stColumn"], [data-testid="column"]');
              if (cols && cols.length >= 2) {{ block = p; break; }}
              p = p.parentElement;
            }}
          }}
          if (!block || seenBlocks.has(block)) continue;
          seenBlocks.add(block);
          if (block.classList) block.classList.add("kb-history-row-block");
          const rowBtn = block.querySelector('div[data-testid="column"]:first-child button, div[data-testid="stColumn"]:first-child button');
          if (rowBtn && rowBtn !== btn && rowBtn.classList) rowBtn.classList.add("kb-history-row-btn");
          if (btn.classList) btn.classList.add("kb-history-trash-btn");
          const wrap = btn.closest('div[data-testid="stButton"]') || btn.closest('div.stButton');
          if (wrap && wrap.classList) wrap.classList.add("kb-trash-wrap");
          const lastCol = block.querySelector('div[data-testid="column"]:last-child, div[data-testid="stColumn"]:last-child') || btn.closest('div[data-testid="column"], div[data-testid="stColumn"]');
          if (lastCol && lastCol.classList) lastCol.classList.add("kb-trash-cell");
        }}
      }} catch (e) {{}}
      /* Fallback: match toggle by text so both states (展开/收起) get consistent styling */
      try {{
        const toggleTexts = ["\u25b8 \u5c55\u5f00\u66f4\u65e9\u4f1a\u8bdd", "\u25be \u6536\u8d77\u66f4\u65e9\u4f1a\u8bdd"];
        sidebar.querySelectorAll("button").forEach(function (btn) {{
          if (!btn || !btn.classList) return;
          const txt = (btn.textContent || "").trim();
          if (toggleTexts.indexOf(txt) < 0) return;
          btn.classList.add("kb-history-toggle-btn");
          const block = btn.closest('div[data-testid="stHorizontalBlock"]');
          if (block && block.classList) block.classList.add("kb-history-toggle-block");
        }});
      }} catch (e) {{}}
    }} catch (e) {{}}
  }}

  function applyNow() {{
    try {{
      clearInlineThemeForRefs();
      normalizeSidebarCloseIcon();
      clearCodeLineArtifacts();
      decorateReferenceActionButtons();
      decorateConversationHistoryButtons();
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
  try {{ host.setTimeout(function () {{ schedule(); }}, 80); }} catch (e) {{}}
  try {{ host.setTimeout(function () {{ schedule(); }}, 250); }} catch (e) {{}}
  try {{ host.setTimeout(function () {{ schedule(); }}, 500); }} catch (e) {{}}
  try {{ bindCitationPopover(); }} catch (e) {{}}
  try {{ renderCiteShelf(); }} catch (e) {{}}
}})();
</script>
        """,
        height=0,
    )

