from __future__ import annotations

import json
from pathlib import Path

import streamlit.components.v1 as components

_CHAT_DOCK_JS_PATH = Path(__file__).resolve().parent.parent / "assets" / "chat_dock_runtime.js"
_CHAT_DOCK_JS_CACHE: str | None = None
_CHAT_DOCK_JS_MTIME_NS_CACHE: int | None = None


def _inject_script(js_body: str, *, height: int = 0) -> None:
    """Inject JS into an iframe via components.html. js_body is the content inside <script> (no tags)."""
    components.html("<script>\n" + js_body.strip() + "\n</script>", height=height)


def _teardown_chat_dock_runtime() -> None:
    _inject_script("""
(function () {
  try {
    const host = window.parent || window;
    const root = host.document;
    if (!root || !root.body) return;

    const NS_LIST = ["__kbDockManagerStableV12", "__kbDockManagerStableV11", "__kbDockManagerStableV10", "__kbDockManagerStableV9", "__kbDockManagerStableV8", "__kbDockManagerStableV7", "__kbDockManagerStableV6", "__kbDockManagerStableV5", "__kbDockManagerStableV4", "__kbDockManagerStableV3"];
    for (const NS of NS_LIST) {
      try {
        if (host[NS] && typeof host[NS].destroy === "function") host[NS].destroy();
        delete host[NS];
      } catch (e) {}
    }

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
""")

def _set_live_streaming_mode(active: bool, hide_stale: bool = False) -> None:
    on_flag = "true" if bool(active) else "false"
    hide_stale_flag = "true" if bool(hide_stale) else "false"
    _inject_script(f"""
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
""")

def _remember_scroll_for_next_rerun(*, nonce: str = "", anchor_id: str = "") -> None:
    nonce_js = json.dumps(str(nonce or ""))
    anchor_js = json.dumps(str(anchor_id or ""))
    body = """
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
""".replace("__NONCE__", nonce_js).replace("__ANCHOR_ID__", anchor_js)
    _inject_script(body)

def _restore_scroll_after_rerun_if_needed(*, max_age_ms: int = 10000) -> None:
    max_age = max(1000, int(max_age_ms))
    body = """
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
""".replace("__MAX_AGE__", str(max_age))
    _inject_script(body)

def _inject_chat_dock_runtime() -> None:
    global _CHAT_DOCK_JS_CACHE, _CHAT_DOCK_JS_MTIME_NS_CACHE
    # Single source: chat_dock_runtime.js adds .kb-input-dock class + inline position/size.
    # theme_legacy.css provides .kb-input-dock styles (var(--dock-bg), etc). No duplicate CSS/script injection.
    cur_mtime_ns: int | None = None
    try:
        cur_mtime_ns = int(_CHAT_DOCK_JS_PATH.stat().st_mtime_ns)
    except Exception:
        cur_mtime_ns = None
    if (_CHAT_DOCK_JS_CACHE is None) or (_CHAT_DOCK_JS_MTIME_NS_CACHE != cur_mtime_ns):
        try:
            _CHAT_DOCK_JS_CACHE = _CHAT_DOCK_JS_PATH.read_text(encoding="utf-8")
        except Exception:
            _CHAT_DOCK_JS_CACHE = ""
        _CHAT_DOCK_JS_MTIME_NS_CACHE = cur_mtime_ns
    js = str(_CHAT_DOCK_JS_CACHE or "").strip()
    if not js:
        return
    _inject_script(js)

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

    // Keep exactly one active timer per pulse-label across reruns.
    // Use a writable host window when possible (component iframes are ephemeral and can accumulate stale timers).
    function pickTimerHost() {{
      const candidates = [];
      for (const h of hostCandidates) {{
        if (!h) continue;
        if (candidates.indexOf(h) < 0) candidates.push(h);
      }}
      if (candidates.indexOf(window) < 0) candidates.push(window);
      for (const h of candidates) {{
        try {{
          const probe = "__kbAutoTimerProbe__";
          h[probe] = 1;
          delete h[probe];
          return h;
        }} catch (e) {{}}
      }}
      return window;
    }}
    const timerHost = pickTimerHost();
    const timerBucketKey = "__kbAutoRefreshTimers__";
    let timerBucket = null;
    try {{
      timerBucket = timerHost[timerBucketKey];
      if (!timerBucket || typeof timerBucket !== "object") {{
        timerBucket = {{}};
        timerHost[timerBucketKey] = timerBucket;
      }}
    }} catch (e) {{
      timerBucket = {{}};
    }}
    const timerKey = "_kbAutoRefreshPulse_" + (pulseLabel || "default");
    try {{
      const prevTimer = timerBucket[timerKey];
      if (prevTimer) {{
        timerHost.clearTimeout(prevTimer);
        timerBucket[timerKey] = null;
      }}
    }} catch (e) {{}}

    timerBucket[timerKey] = timerHost.setTimeout(function () {{
      try {{
        try {{ timerBucket[timerKey] = null; }} catch (e) {{}}
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
        try {{ timerHost.focus(); }} catch (e) {{}}
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

