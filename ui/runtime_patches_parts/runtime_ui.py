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

