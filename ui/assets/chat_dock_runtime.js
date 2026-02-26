(function () {
  const host = window.parent || window;
  const root = host.document;
  if (!root || !root.body) return;

  const PREV_NS_LIST = ["__kbDockManagerStableV11", "__kbDockManagerStableV10", "__kbDockManagerStableV9", "__kbDockManagerStableV8", "__kbDockManagerStableV7", "__kbDockManagerStableV6", "__kbDockManagerStableV5", "__kbDockManagerStableV4", "__kbDockManagerStableV3"];
  for (const k of PREV_NS_LIST) {
    try {
      if (host[k] && typeof host[k].destroy === "function") {
        host[k].destroy();
      }
      try { delete host[k]; } catch (e0) {}
    } catch (e) {}
  }

  const NS = "__kbDockManagerStableV12";
  if (host[NS] && typeof host[NS].schedule === "function") {
    try { host[NS].schedule(); } catch (e) {}
    return;
  }

  const RESIZE_CLASS = "kb-resizing";
  const DOCK_SIDE_GAP = 35;
  const DOCK_RIGHT_GAP = 35;
  const MIN_WIDTH = 320;
  // Sidebar-edge drag support conflicts with Streamlit's native sidebar collapse button
  // on newer builds. Prefer native behavior reliability over this optional enhancement.
  const DISABLE_SIDEBAR_EDGE_DRAG = true;
  // Temporary: enabled to diagnose post-submit dock degradation.
  // Disable via `window.__KB_DOCK_DEBUG = false` in DevTools if needed.
  const DEBUG_DOCK = true;
  const state = {
    raf: 0,
    delayTimer: 0,
    recoverTimer: 0,
    recoverBackoff: 0,
    timer: 0,
    ro: null,
    roSizeMap: null,
    mo: null,
    dragging: false,
    lastHookTs: 0,
    form: null,
    ta: null,
    onMouseDown: null,
    onPointerDown: null,
    onTouchStart: null,
    onMouseMove: null,
    onPointerMove: null,
    onMouseUp: null,
    onPointerUp: null,
    onPointerCancel: null,
    onTouchEnd: null,
    onKeyDown: null,
    onBlur: null,
    onResize: null,
    disableDockCompat: false,
    dbgSeq: 0,
    dbgLastSig: "",
    dbgLastTs: 0,
  };

  function _dbgConsole() {
    try { return host.console || console; } catch (e) { return null; }
  }
  function _dbgEnabled() {
    if (!DEBUG_DOCK) return false;
    try {
      if (host && host.__KB_DOCK_DEBUG === false) return false;
      if (host && host.localStorage) {
        const v = String(host.localStorage.getItem("__kbDockDebug") || "").trim();
        if (v === "0" || v.toLowerCase() === "false") return false;
      }
    } catch (e) {}
    return true;
  }
  function _dbgElBrief(el) {
    if (!el) return null;
    const out = {};
    try { out.tag = String(el.tagName || "").toLowerCase(); } catch (e) {}
    try { out.testid = String(el.getAttribute("data-testid") || ""); } catch (e) {}
    try { out.id = String(el.id || ""); } catch (e) {}
    try {
      const cls = String(el.className || "").trim().replace(/\s+/g, " ");
      out.cls = cls.length > 180 ? (cls.slice(0, 180) + "...") : cls;
    } catch (e) {}
    try { out.stale = isInsideStaleNode(el); } catch (e) {}
    try { out.connected = (el.isConnected !== false); } catch (e) {}
    try {
      if (el.classList) {
        out.kbDock = el.classList.contains("kb-input-dock");
        out.kbPos = el.classList.contains("kb-dock-positioned");
      }
    } catch (e) {}
    try {
      if (el.dataset) {
        out.kbPromptRoot = String(el.dataset.kbPromptDockRoot || "");
        out.kbGeomSig = String(el.dataset.kbDockGeomSig || "");
      }
    } catch (e) {}
    try {
      if (el.getBoundingClientRect) {
        const r = el.getBoundingClientRect();
        out.rect = {
          x: Math.round(Number(r.left || 0)),
          y: Math.round(Number(r.top || 0)),
          w: Math.round(Number(r.width || 0)),
          h: Math.round(Number(r.height || 0)),
        };
      }
    } catch (e) {}
    return out;
  }
  function _dbgBtnList(scope) {
    const out = [];
    if (!scope || !scope.querySelectorAll) return out;
    let idx = 0;
    try {
      const btns = Array.from(scope.querySelectorAll("button"));
      for (const b of btns) {
        if (!b) continue;
        idx += 1;
        const item = { i: idx };
        try { item.txt = _btnText(b); } catch (e) { item.txt = ""; }
        try { item.type = String(b.getAttribute("type") || "").toLowerCase(); } catch (e) { item.type = ""; }
        try { item.disabled = !!b.disabled; } catch (e) {}
        try { item.stale = isInsideStaleNode(b); } catch (e) {}
        try { item.inUploader = !!(b.closest && b.closest('div[data-testid="stFileUploader"]')); } catch (e) {}
        try { item.inSubmitWrap = !!(b.closest && (b.closest('div[data-testid="stFormSubmitButton"]') || b.closest('div[data-testid*="FormSubmit"]'))); } catch (e) {}
        try { item.aria = String(b.getAttribute("aria-label") || ""); } catch (e) {}
        out.push(item);
        if (out.length >= 12) break;
      }
    } catch (e) {}
    return out;
  }
  function _dbgCount(scope, sel) {
    if (!scope || !scope.querySelectorAll) return 0;
    try { return Number((scope.querySelectorAll(sel) || []).length || 0); } catch (e) { return 0; }
  }
  function _dbgActionLayerSummary(scope) {
    const out = {};
    if (!scope || !scope.querySelector) return out;
    try {
      const layer = scope.querySelector(':scope > .kb-dock-action-layer');
      out.layer = !!layer;
      if (!layer) return out;
      out.sendAnchor = !!layer.querySelector(".kb-dock-send-anchor");
      out.stopAnchor = !!layer.querySelector(".kb-dock-stop-anchor");
      out.sendWrapInLayer = !!layer.querySelector(".kb-dock-send-wrap");
      out.stopWrapInLayer = !!layer.querySelector(".kb-dock-stop-wrap");
    } catch (e) {}
    return out;
  }
  function _dbgBodyFlags() {
    const out = {};
    try {
      const cl = root.body && root.body.classList ? root.body.classList : null;
      if (!cl) return out;
      out.live = cl.contains("kb-live-streaming");
      out.hideStale = cl.contains("kb-hide-stale-rerun");
      out.resizing = cl.contains("kb-resizing");
    } catch (e) {}
    return out;
  }
  function _dbgCandidateScan(limit) {
    const out = [];
    const maxN = Math.max(1, Number(limit || 8));
    let n = 0;
    try {
      const nodes = [
        ...root.querySelectorAll("form"),
        ...root.querySelectorAll('div[data-testid="stForm"]')
      ];
      for (const node of nodes) {
        if (!node) continue;
        n += 1;
        const rec = { n, node: _dbgElBrief(node) };
        try { rec.inMain = !!(node.closest && node.closest("section.main")); } catch (e) {}
        try {
          rec.hasTextarea = !!node.querySelector("textarea");
          rec.hasUploader = !!node.querySelector('div[data-testid="stFileUploader"]');
          rec.hasSubmitWrap = !!node.querySelector('div[data-testid="stFormSubmitButton"], div[data-testid*="FormSubmit"]');
          rec.hasActionLayer = !!node.querySelector(":scope > .kb-dock-action-layer");
        } catch (e) {}
        try { rec.hasSendButton = hasSendButton(node); } catch (e) {}
        rec.buttons = _dbgBtnList(node);
        out.push(rec);
        if (out.length >= maxN) break;
      }
    } catch (e) {}
    return out;
  }
  function _dbgSigFromSnapshot(snap) {
    try {
      const root0 = snap && snap.root ? snap.root : {};
      const counts0 = snap && snap.counts ? snap.counts : {};
      const action0 = snap && snap.action ? snap.action : {};
      const body0 = snap && snap.body ? snap.body : {};
      const btns0 = Array.isArray(snap && snap.buttons) ? snap.buttons : [];
      const btnSig = btns0.map(function (b) {
        return [
          String(b.txt || ""),
          b.inUploader ? "u" : "",
          b.inSubmitWrap ? "s" : "",
          b.stale ? "x" : "",
          b.disabled ? "d" : ""
        ].join("");
      }).join("|");
      return JSON.stringify({
        e: String(snap.event || ""),
        rk: String(root0.tag || "") + "/" + String(root0.testid || "") + "/" + String(root0.id || ""),
        k: [!!root0.kbDock, !!root0.kbPos, String(root0.kbGeomSig || "")],
        c: counts0,
        a: action0,
        b: body0,
        bs: btnSig,
        r: snap && snap.reason ? String(snap.reason) : "",
      });
    } catch (e) {
      return "";
    }
  }
  function _dbgDock(eventName, form, ta, extra, force) {
    if (!_dbgEnabled()) return;
    const c = _dbgConsole();
    if (!c || typeof c.log !== "function") return;
    const snap = {
      event: String(eventName || ""),
      ts: new Date().toISOString(),
      body: _dbgBodyFlags(),
      root: _dbgElBrief(form || null),
      textarea: _dbgElBrief(ta || null),
      counts: {
        textareas: _dbgCount(form, "textarea"),
        uploaders: _dbgCount(form, 'div[data-testid="stFileUploader"]'),
        submitWraps: _dbgCount(form, 'div[data-testid="stFormSubmitButton"], div[data-testid*="FormSubmit"]'),
        buttons: _dbgCount(form, "button"),
      },
      action: _dbgActionLayerSummary(form),
      buttons: _dbgBtnList(form),
    };
    if (extra && typeof extra === "object") {
      try {
        for (const k of Object.keys(extra)) snap[k] = extra[k];
      } catch (e) {}
    }
    const sig = _dbgSigFromSnapshot(snap);
    const now = Date.now ? Date.now() : (+new Date());
    const sameSig = (sig && sig === state.dbgLastSig);
    if (!force && sameSig && (now - Number(state.dbgLastTs || 0) < 900)) return;
    state.dbgLastSig = sig;
    state.dbgLastTs = now;
    state.dbgSeq = Number(state.dbgSeq || 0) + 1;
    const head = "[kb-dock][" + String(state.dbgSeq) + "] " + String(eventName || "");
    try {
      if (typeof c.groupCollapsed === "function") c.groupCollapsed(head);
      else c.log(head);
      c.log(snap);
      if (typeof c.groupEnd === "function") c.groupEnd();
    } catch (e) {}
  }

  function isInsideStaleNode(el) {
    try {
      return !!(
        el &&
        el.closest &&
        el.closest('[data-stale="true"], .stale-element, [data-testid="staleElementOverlay"], [data-testid="stale-overlay"]')
      );
    } catch (e) { return false; }
  }
  function pickFresh(nodes) {
    for (const n of (nodes || [])) {
      if (!n) continue;
      try { if (n.isConnected === false) continue; } catch (e) {}
      if (!isInsideStaleNode(n)) return n;
    }
    return (nodes && nodes.length) ? nodes[0] : null;
  }
  function _isVisibleForDock(el) {
    if (!el) return false;
    try { if (el.isConnected === false) return false; } catch (e0) {}
    try {
      const cs = host.getComputedStyle ? host.getComputedStyle(el) : null;
      if (cs) {
        if (cs.display === "none" || cs.visibility === "hidden") return false;
        if (Number(cs.opacity || 1) <= 0.001) return false;
      }
    } catch (e1) {}
    try {
      const r = el.getBoundingClientRect ? el.getBoundingClientRect() : null;
      if (!r || !isFinite(r.width) || !isFinite(r.height)) return false;
      if (r.width < 4 || r.height < 4) return false;
    } catch (e2) { return false; }
    return true;
  }
  function _pickPromptTextareaIn(form) {
    if (!form || !form.querySelectorAll) return null;
    try {
      return pickFresh([
        ...form.querySelectorAll('div[data-testid="stTextArea"] textarea'),
        ...form.querySelectorAll('.stTextArea textarea'),
        ...form.querySelectorAll('textarea'),
      ]);
    } catch (e) {
      return null;
    }
  }
  function _pickPromptSubmitWrapIn(form) {
    if (!form || !form.querySelectorAll) return null;
    try {
      return pickFresh([
        ...form.querySelectorAll('div[data-testid="stFormSubmitButton"]'),
        ...form.querySelectorAll('div[data-testid*="FormSubmit"]'),
        ...form.querySelectorAll('.stButton'),
      ]);
    } catch (e) {
      return null;
    }
  }
  function _pickPromptUploaderIn(form) {
    if (!form || !form.querySelectorAll) return null;
    try {
      return pickFresh(form.querySelectorAll('div[data-testid="stFileUploader"]'));
    } catch (e) {
      return null;
    }
  }
  function _scorePromptRootCandidate(dockRoot, ta, submitWrap, uploader, mainRegion) {
    let score = 0;
    if (!dockRoot || !ta) return -999999;
    try { score += (dockRoot === state.form) ? 240 : 0; } catch (e) {}
    try { score += isInsideStaleNode(dockRoot) ? -900 : 220; } catch (e) {}
    try { score += isInsideStaleNode(ta) ? -700 : 260; } catch (e) {}
    try { score += _isVisibleForDock(dockRoot) ? 280 : -360; } catch (e) {}
    try { score += _isVisibleForDock(ta) ? 180 : -220; } catch (e) {}
    try { if (mainRegion && dockRoot.closest && dockRoot.closest('section.main')) score += 160; } catch (e) {}
    try { if (submitWrap) score += 120; } catch (e) {}
    try { if (uploader) score += 80; } catch (e) {}
    try { score += hasSendButton(dockRoot) ? 220 : -180; } catch (e) {}
    try {
      const r = dockRoot.getBoundingClientRect ? dockRoot.getBoundingClientRect() : null;
      if (r && isFinite(r.top) && isFinite(r.bottom)) {
        const vpH = Math.max(0, host.innerHeight || root.documentElement.clientHeight || 0);
        const bottomDist = Math.abs(vpH - Number(r.bottom || 0));
        score += Math.max(0, 140 - Math.min(140, Math.round(bottomDist / 4)));
        score += Math.max(0, Math.min(100, Math.round(Number(r.top || 0) / 8)));
      }
    } catch (e) {}
    return score;
  }
  function findMainRegion() {
    return pickFresh(root.querySelectorAll('section.main'));
  }
  function findMainContainer() {
    return pickFresh([
      ...root.querySelectorAll('section.main .block-container'),
      ...root.querySelectorAll('[data-testid="stMainBlockContainer"]'),
      ...root.querySelectorAll('.block-container')
    ]);
  }
  function findSidebar() {
    return pickFresh(root.querySelectorAll('section[data-testid="stSidebar"]'));
  }
  function _btnText(btn) {
    return String((btn && (btn.innerText || btn.textContent)) || "")
      .replace(/[\u200B-\u200D\uFEFF]/g, "")
      .replace(/\s+/g, " ")
      .trim()
      .toLowerCase();
  }
  function _btnAuxText(btn) {
    try {
      return String(
        (btn && (
          btn.getAttribute("aria-label") ||
          btn.getAttribute("title") ||
          btn.getAttribute("data-testid")
        )) || ""
      )
        .replace(/[\u200B-\u200D\uFEFF]/g, "")
        .replace(/\s+/g, " ")
        .trim()
        .toLowerCase();
    } catch (e) {
      return "";
    }
  }
  function isStopBtnText(t) {
    const s = String(t || "")
      .replace(/[\u200B-\u200D\uFEFF]/g, "")
      .replace(/\s+/g, " ")
      .trim()
      .toLowerCase();
    return s === "■" || s === "停止" || s === "stop" || s.includes("■") || s.includes("停止") || s.includes("stop");
  }
  function isSendBtnText(t) {
    const s = String(t || "")
      .replace(/[\u200B-\u200D\uFEFF]/g, "")
      .replace(/\s+/g, " ")
      .trim()
      .toLowerCase();
    return (
      s === "发送" || s === "↑" || s === "send" || s === "submit" ||
      s.includes("↑") || s.includes("发送") || s.includes("send") || s.includes("submit")
    );
  }
  function isStopBtnLike(btn, txt) {
    if (isStopBtnText(txt == null ? _btnText(btn) : txt)) return true;
    const aux = _btnAuxText(btn);
    return !!(aux && (aux.includes("stop") || aux.includes("停止")));
  }
  function isSendBtnLike(btn, txt) {
    if (isSendBtnText(txt == null ? _btnText(btn) : txt)) return true;
    const aux = _btnAuxText(btn);
    return !!(aux && (aux.includes("send") || aux.includes("submit") || aux.includes("发送")));
  }
  function hasSendButton(form) {
    if (!form) return false;
    const btns = form.querySelectorAll('button');
    let hasSubmitLike = false;
    for (const b of btns) {
      if (!b || isInsideStaleNode(b)) continue;
      try {
        if (b.closest && b.closest('div[data-testid="stFileUploader"]')) continue;
      } catch (e0) {}
      const txt = _btnText(b);
      if (isSendBtnText(txt) || isStopBtnText(txt)) return true;
      try {
        const typ = String(b.getAttribute("type") || "").toLowerCase();
        const aria = String(b.getAttribute("aria-label") || "").toLowerCase();
        if (typ === "submit") hasSubmitLike = true;
        if (b.closest && (b.closest('div[data-testid="stFormSubmitButton"]') || b.closest('div[data-testid*="FormSubmit"]'))) {
          hasSubmitLike = true;
        }
        if (
          aria &&
          (aria.includes("send") || aria.includes("submit") || aria.includes("stop") || aria.includes("发送") || aria.includes("停止"))
        ) return true;
      } catch (e) {}
    }
    return hasSubmitLike;
  }
  function clickSendButton(form) {
    if (!form) return false;
    try { _syncPromptUploaderFilesFromCurrent(form); } catch (e0) {}
    const btns = Array.from(form.querySelectorAll("button"));
    let fallback = null;
    for (const b of btns) {
      if (!b || isInsideStaleNode(b) || b.disabled) continue;
      try {
        if (b.closest && b.closest('div[data-testid="stFileUploader"]')) continue;
      } catch (e0) {}
      const txt = _btnText(b);
      if (isStopBtnText(txt)) continue;
      if (isSendBtnText(txt)) {
        b.click();
        return true;
      }
      if (!fallback) {
        const typ = String(b.getAttribute("type") || "").toLowerCase();
        if (typ === "submit" || b.closest('div[data-testid="stFormSubmitButton"]')) {
          fallback = b;
        }
      }
    }
    if (fallback) {
      fallback.click();
      return true;
    }
    return false;
  }
  function findSubmitButtonWrap(btn) {
    if (!btn || !btn.closest) return null;
    try {
      if (btn.closest('div[data-testid="stFileUploader"]')) return null;
    } catch (e0) {}
    const sels = [
      'div[data-testid="stFormSubmitButton"]',
      'div[data-testid*="FormSubmit"]',
      'div[data-testid="stButton"]',
      '.stButton'
    ];
    for (const sel of sels) {
      try {
        const hit = btn.closest(sel);
        if (hit) return hit;
      } catch (e) {}
    }
    return null;
  }
  function decoratePromptButtons(form) {
    if (!form) return;
    const btns = Array.from(form.querySelectorAll("button"));
    const submitCandidates = [];
    let hasSendClass = false;
    let hasStopClass = false;
    for (const b of btns) {
      if (!b) continue;
      try {
        if (b.closest && b.closest('div[data-testid="stFileUploader"]')) continue;
      } catch (e0) {}
      if (isInsideStaleNode(b)) continue;
      try {
        b.classList.remove("kb-dock-send-btn", "kb-dock-stop-btn");
      } catch (e) {}
      let wrap = null;
      try { wrap = findSubmitButtonWrap(b); } catch (e) {}
      if (wrap) {
        try {
          wrap.classList.remove("kb-dock-send-wrap", "kb-dock-stop-wrap");
        } catch (e) {}
      }
      const txt = _btnText(b);
      const typ = String(b.getAttribute("type") || "").toLowerCase();
      if ((typ === "submit") || wrap) submitCandidates.push({ b, wrap, txt });
      if (isStopBtnLike(b, txt)) {
        try { b.classList.add("kb-dock-stop-btn"); } catch (e) {}
        if (wrap) {
          try { wrap.classList.add("kb-dock-stop-wrap"); } catch (e) {}
        }
        hasStopClass = true;
        continue;
      }
      if (isSendBtnLike(b, txt)) {
        try { b.classList.add("kb-dock-send-btn"); } catch (e) {}
        if (wrap) {
          try { wrap.classList.add("kb-dock-send-wrap"); } catch (e) {}
        }
        hasSendClass = true;
      }
    }
    if (!hasSendClass && !hasStopClass && submitCandidates.length) {
      let picked = null;
      let preferStop = false;
      try {
        preferStop = !!(root.body && root.body.classList && root.body.classList.contains("kb-live-streaming"));
      } catch (e) {}
      if (preferStop) {
        picked = submitCandidates[0];
      } else {
        for (const item of submitCandidates) {
          if (!isStopBtnLike(item.b, item.txt)) { picked = item; break; }
        }
        if (!picked) picked = submitCandidates[0];
      }
      if (picked && picked.b) {
        if (preferStop) {
          try { picked.b.classList.add("kb-dock-stop-btn"); } catch (e) {}
          if (picked.wrap) {
            try { picked.wrap.classList.add("kb-dock-stop-wrap"); } catch (e) {}
          }
        } else {
          try { picked.b.classList.add("kb-dock-send-btn"); } catch (e) {}
          if (picked.wrap) {
            try { picked.wrap.classList.add("kb-dock-send-wrap"); } catch (e) {}
          }
        }
      }
    }
  }
  function mountPromptActionWrappers(form) {
    if (!form || !form.querySelector) return;
    let layer = null;
    try { layer = form.querySelector(':scope > .kb-dock-action-layer'); } catch (e) {}
    if (!layer) {
      try {
        layer = root.createElement("div");
        layer.className = "kb-dock-action-layer";
        form.appendChild(layer);
      } catch (e) {
        layer = null;
      }
    }
    if (!layer) return;

    let sendAnchor = null;
    let stopAnchor = null;
    try { sendAnchor = layer.querySelector('.kb-dock-send-anchor'); } catch (e) {}
    try { stopAnchor = layer.querySelector('.kb-dock-stop-anchor'); } catch (e) {}
    if (!sendAnchor) {
      try {
        sendAnchor = root.createElement("div");
        sendAnchor.className = "kb-dock-send-anchor";
        layer.appendChild(sendAnchor);
      } catch (e) { sendAnchor = null; }
    }
    if (!stopAnchor) {
      try {
        stopAnchor = root.createElement("div");
        stopAnchor.className = "kb-dock-stop-anchor";
        layer.appendChild(stopAnchor);
      } catch (e) { stopAnchor = null; }
    }

    let sendWrap = null;
    let stopWrap = null;
    try {
      const sendWraps = Array.from(form.querySelectorAll(".kb-dock-send-wrap"));
      sendWrap = pickFresh(sendWraps);
    } catch (e) {}
    try {
      const stopWraps = Array.from(form.querySelectorAll(".kb-dock-stop-wrap"));
      stopWrap = pickFresh(stopWraps);
    } catch (e) {}

    let movedSend = false;
    let movedStop = false;
    if (sendAnchor && sendWrap && sendWrap.parentElement !== sendAnchor) {
      try {
        sendAnchor.appendChild(sendWrap);
        movedSend = true;
      } catch (e) {}
    } else if (sendAnchor && sendWrap && sendWrap.parentElement === sendAnchor) {
      movedSend = true;
    }
    if (stopAnchor && stopWrap && stopWrap.parentElement !== stopAnchor) {
      try {
        stopAnchor.appendChild(stopWrap);
        movedStop = true;
      } catch (e) {}
    } else if (stopAnchor && stopWrap && stopWrap.parentElement === stopAnchor) {
      movedStop = true;
    }
    return {
      sendAnchor: !!sendAnchor,
      stopAnchor: !!stopAnchor,
      hasSendWrap: !!sendWrap,
      hasStopWrap: !!stopWrap,
      movedSend: !!movedSend,
      movedStop: !!movedStop,
    };
  }
  function findPromptFormAndTextarea() {
    const seenRoots = (typeof Set !== "undefined") ? new Set() : null;
    const forms = [
      ...root.querySelectorAll('form'),
      ...root.querySelectorAll('div[data-testid="stForm"]')
    ];
    const mainRegion = findMainRegion();
    let best = null;
    let bestScore = -999999;
    for (const form of forms) {
      if (!form) continue;
      let dockRoot = form;
      try {
        if (String(form.tagName || "").toUpperCase() !== "FORM") {
          dockRoot = form;
        } else {
          const wrap = form.closest ? form.closest('div[data-testid="stForm"]') : null;
          if (wrap && !isInsideStaleNode(wrap)) dockRoot = wrap;
        }
      } catch (e) {}
      if (!dockRoot) continue;
      if (seenRoots) {
        try {
          if (seenRoots.has(dockRoot)) continue;
          seenRoots.add(dockRoot);
        } catch (e) {}
      }
      if (isInsideStaleNode(dockRoot)) continue;
      if (mainRegion && dockRoot.closest && !dockRoot.closest('section.main')) continue;
      const ta = _pickPromptTextareaIn(dockRoot);
      if (!ta) continue;
      const submitWrap = _pickPromptSubmitWrapIn(dockRoot);
      const uploader = _pickPromptUploaderIn(dockRoot);
      const score = _scorePromptRootCandidate(dockRoot, ta, submitWrap, uploader, mainRegion);
      if (score > bestScore) {
        bestScore = score;
        best = { form: dockRoot, ta };
      }
    }
    return best || { form: null, ta: null };
  }
  function resetDockStyles(form) {
    if (!form) return;
    try {
      form.classList.remove('kb-input-dock', 'kb-dock-positioned');
      try {
        if (form.dataset) {
          delete form.dataset.kbDockGeomSig;
          delete form.dataset.kbDockBaseApplied;
        }
      } catch (e0) {}
      form.style.left = '';
      form.style.right = '';
      form.style.width = '';
      form.style.transform = '';
      form.style.position = '';
      form.style.bottom = '';
      form.style.height = '';
      form.style.maxHeight = '';
      form.style.minHeight = '';
      form.style.flex = '';
      form.style.display = '';
    } catch (e) {}
  }
  function clearRecoveryRetry() {
    try { if (state.recoverTimer) host.clearTimeout(state.recoverTimer); } catch (e) {}
    state.recoverTimer = 0;
    state.recoverBackoff = 0;
  }
  function scheduleRecoveryRetry(msHint) {
    if (state.recoverTimer) return;
    let ms = Number(msHint || 0);
    if (!isFinite(ms) || ms <= 0) {
      const prev = Number(state.recoverBackoff || 0);
      ms = prev > 0 ? Math.min(900, Math.max(70, prev)) : 70;
    }
    ms = Math.max(36, Math.floor(ms));
    const prev0 = Number(state.recoverBackoff || 0);
    const seed = prev0 > 0 ? prev0 : ms;
    state.recoverBackoff = Math.min(1000, Math.max(70, Math.floor(seed * 1.6)));
    state.recoverTimer = host.setTimeout(function () {
      state.recoverTimer = 0;
      scheduleHook(true);
    }, ms);
  }
  function scheduleSubmitRecoveryPulse() {
    _dbgDock("submit-pulse", state.form, state.ta, null, true);
    try { scheduleHook(true); } catch (e) {}
    const steps = [50, 120, 240, 420, 700];
    for (const ms of steps) {
      try {
        host.setTimeout(function () { scheduleHook(true); }, ms);
      } catch (e) {}
    }
  }
  function setResizing(on) {
    if (on) root.body.classList.add(RESIZE_CLASS);
    else root.body.classList.remove(RESIZE_CLASS);
  }
  function isInteractiveTarget(target) {
    try {
      if (!target || !target.closest) return false;
      return !!target.closest([
        'button',
        'a[href]',
        'input',
        'textarea',
        'select',
        'label',
        '[role="button"]',
        '[contenteditable="true"]',
        '[data-testid="stSidebarCollapseButton"]',
        '[data-testid="stSidebarNav"]'
      ].join(', '));
    } catch (e) {
      return false;
    }
  }
  function placeDock(form) {
    if (!form) return false;
    const mainContainer = findMainContainer();
    const mainRegion = findMainRegion();
    const sidebar = findSidebar();
    const anchor = mainContainer || mainRegion;

    const viewportW = Math.max(0, (host.innerWidth || root.documentElement.clientWidth || 0));
    const anchorRect = (anchor && anchor.getBoundingClientRect) ? anchor.getBoundingClientRect() : null;
    const sidebarRect = (sidebar && sidebar.getBoundingClientRect) ? sidebar.getBoundingClientRect() : null;

    let leftBound = DOCK_SIDE_GAP;
    let rightBound = Math.max(leftBound + MIN_WIDTH, viewportW - DOCK_RIGHT_GAP);

    if (anchorRect && isFinite(anchorRect.left) && isFinite(anchorRect.right) && anchorRect.width > 10) {
      leftBound = Math.max(leftBound, Math.floor(anchorRect.left) + DOCK_SIDE_GAP);
      rightBound = Math.min(rightBound, Math.floor(anchorRect.right) - DOCK_SIDE_GAP);
    }
    if (sidebarRect && isFinite(sidebarRect.right) && sidebarRect.width > 10) {
      leftBound = Math.max(leftBound, Math.floor(sidebarRect.right) + DOCK_SIDE_GAP);
    }
    if (!isFinite(leftBound) || !isFinite(rightBound)) return false;

    rightBound = Math.min(rightBound, viewportW - DOCK_RIGHT_GAP);
    if (rightBound - leftBound < MIN_WIDTH) rightBound = leftBound + MIN_WIDTH;
    if (rightBound > viewportW - DOCK_RIGHT_GAP) {
      rightBound = viewportW - DOCK_RIGHT_GAP;
      leftBound = Math.max(DOCK_SIDE_GAP, rightBound - MIN_WIDTH);
    }

    const dockLeft = Math.max(DOCK_SIDE_GAP, Math.floor(leftBound));
    const dockWidth = Math.max(MIN_WIDTH, Math.floor(rightBound - dockLeft));
    const leftPx = dockLeft + 'px';
    const widthPx = dockWidth + 'px';
    const geomSig = `${dockLeft}|${dockWidth}`;

    form.classList.add('kb-input-dock', 'kb-dock-positioned');
    try {
      if (!form.dataset || form.dataset.kbDockBaseApplied !== '1') {
        form.style.setProperty('height', 'auto', 'important');
        form.style.setProperty('min-height', '0', 'important');
        form.style.setProperty('max-height', 'none', 'important');
        form.style.setProperty('flex', 'none', 'important');
        form.style.setProperty('display', 'block', 'important');
        if (form.dataset) form.dataset.kbDockBaseApplied = '1';
      }
    } catch (e) {}
    let sameGeom = false;
    try {
      sameGeom = !!(
        form.dataset &&
        form.dataset.kbDockGeomSig === geomSig &&
        form.style.left === leftPx &&
        form.style.width === widthPx &&
        form.style.right === 'auto' &&
        form.style.transform === 'none'
      );
    } catch (e) {}
    if (sameGeom) return true;
    try {
      if (form.dataset) form.dataset.kbDockGeomSig = geomSig;
    } catch (e) {}
    form.style.left = leftPx;
    form.style.right = 'auto';
    form.style.width = widthPx;
    form.style.transform = 'none';
    return true;
  }
  function markPromptRoot(node) {
    if (!node || !node.setAttribute) return;
    try { node.setAttribute("data-kb-prompt-dock-root", "1"); } catch (e) {}
  }
  function ensurePromptRootMarkers(form) {
    if (!form) return;
    try { markPromptRoot(form); } catch (e) {}
    try {
      const outer = form.closest ? form.closest('div[data-testid="stForm"]') : null;
      if (outer) markPromptRoot(outer);
    } catch (e) {}
  }
  function hasSubmitWrapperNodes(form) {
    if (!form || !form.querySelector) return false;
    try {
      return !!form.querySelector('div[data-testid="stFormSubmitButton"], div[data-testid*="FormSubmit"], .stButton');
    } catch (e) {
      return false;
    }
  }
  function isActionLayerReady(form, actionInfo) {
    const info = actionInfo || {};
    const hasSubmitNodes = hasSubmitWrapperNodes(form);
    if (!hasSubmitNodes) return true;
    if (info.movedSend || info.movedStop) return true;
    // Accept send-wrap present inside action layer even if no move happened in this tick.
    try {
      const layer = form.querySelector(':scope > .kb-dock-action-layer');
      if (!layer) return false;
      if (layer.querySelector('.kb-dock-send-wrap, .kb-dock-stop-wrap')) return true;
    } catch (e) {}
    return false;
  }
  function bindFormSubmitRecovery(form) {
    if (!form || !form.dataset) return;
    if (form.dataset.kbDockSubmitPulseHooked === "1") return;
    form.dataset.kbDockSubmitPulseHooked = "1";

    const pulseIfPromptAction = function (target) {
      let btn = null;
      try { btn = target && target.closest ? target.closest("button") : null; } catch (e) {}
      if (!btn || isInsideStaleNode(btn)) return;
      let isPromptAction = false;
      try {
        const txt = _btnText(btn);
        const typ = String(btn.getAttribute("type") || "").toLowerCase();
        isPromptAction =
          isSendBtnText(txt) ||
          isStopBtnText(txt) ||
          typ === "submit" ||
          !!(btn.closest && (btn.closest('div[data-testid="stFormSubmitButton"]') || btn.closest('div[data-testid*="FormSubmit"]')));
      } catch (e) {}
      if (!isPromptAction) return;
      try { _syncPromptUploaderFilesFromCurrent(form); } catch (e1) {}
      scheduleSubmitRecoveryPulse();
    };

    try {
      form.addEventListener("submit", function () {
        try { _syncPromptUploaderFilesFromCurrent(form); } catch (e0) {}
        _dbgDock("form-submit", form, null, { trigger: "submit" }, true);
        scheduleSubmitRecoveryPulse();
      }, true);
    } catch (e) {}
    try {
      form.addEventListener("click", function (e) {
        try {
          const btn = e.target && e.target.closest ? e.target.closest("button") : null;
          if (btn) _dbgDock("form-click", form, null, { trigger: "click", button: { txt: _btnText(btn), aria: String(btn.getAttribute("aria-label") || "") } }, true);
        } catch (e0) {}
        pulseIfPromptAction(e.target || null);
      }, true);
    } catch (e) {}
    try {
      form.addEventListener("keydown", function (e) {
        const isEnter = String(e.key || "").toLowerCase() === "enter";
        if (!isEnter) return;
        pulseIfPromptAction(e.target || null);
      }, true);
    } catch (e) {}
  }
  function bindCtrlEnter(ta, form) {
    if (!ta || ta.dataset.kbCtrlEnterHooked === "1") return;
    ta.dataset.kbCtrlEnterHooked = "1";
    ta.addEventListener("keydown", function (e) {
      const isCtrlEnter = (e.ctrlKey || e.metaKey) && e.key === "Enter";
      if (!isCtrlEnter) return;
      if (e.isComposing) return;
      const ok = clickSendButton(form || root);
      if (!ok) return;
      e.preventDefault();
      e.stopPropagation();
    }, { capture: true });
  }
  function _uploadExtFromMime(mime) {
    const m = String(mime || "").toLowerCase();
    if (m === "image/png") return ".png";
    if (m === "image/jpeg" || m === "image/jpg") return ".jpg";
    if (m === "image/webp") return ".webp";
    if (m === "image/gif") return ".gif";
    if (m === "image/bmp") return ".bmp";
    if (m === "application/pdf") return ".pdf";
    return "";
  }
  function _isPromptUploadFile(file) {
    if (!file) return false;
    const type = String(file.type || "").toLowerCase();
    const name = String(file.name || "").toLowerCase();
    if (type === "application/pdf" || name.endsWith(".pdf")) return true;
    if (type.startsWith("image/")) return true;
    return (
      name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") ||
      name.endsWith(".webp") || name.endsWith(".gif") || name.endsWith(".bmp")
    );
  }
  function _ensurePromptUploadFileName(file, sourceTag) {
    if (!file) return null;
    const src = String(sourceTag || "upload").trim() || "upload";
    const name = String(file.name || "").trim();
    if (name) return file;
    const ext = _uploadExtFromMime(file.type) || ".png";
    const ts = Date.now();
    try {
      return new File([file], `${src}-${ts}${ext}`, {
        type: String(file.type || ""),
        lastModified: Number(file.lastModified || ts) || ts,
      });
    } catch (e) {
      return file;
    }
  }
  function _findPromptUploaderInput(form) {
    if (!form || !form.querySelectorAll) return null;
    try {
      const inputs = Array.from(form.querySelectorAll('input[type="file"]'));
      const freshInputs = [];
      for (const inp of inputs) {
        if (!inp) continue;
        if (inp.disabled) continue;
        if (isInsideStaleNode(inp)) continue;
        freshInputs.push(inp);
      }
      if (freshInputs.length) return pickFresh(freshInputs);
      for (const inp of inputs) {
        if (!inp || inp.disabled) continue;
        return inp;
      }
    } catch (e) {}
    return null;
  }
  function _listPromptUploaderInputs(form) {
    const out = [];
    const seen = (typeof Set !== "undefined") ? new Set() : null;
    const pushInput = function (inp) {
      if (!inp || inp.disabled) return;
      try { if (inp.type && String(inp.type).toLowerCase() !== "file") return; } catch (e0) {}
      if (seen) {
        try {
          if (seen.has(inp)) return;
          seen.add(inp);
        } catch (e1) {}
      }
      out.push(inp);
    };
    try {
      if (form && form.querySelectorAll) {
        for (const inp of Array.from(form.querySelectorAll('input[type="file"]'))) pushInput(inp);
      }
    } catch (e) {}
    try {
      const nodes = [
        ...root.querySelectorAll('section.main div[data-testid="stForm"]'),
        ...root.querySelectorAll('section.main form'),
      ];
      for (const node of nodes) {
        if (!node || !node.querySelector) continue;
        try {
          if (!node.querySelector('textarea')) continue;
          if (!node.querySelector('div[data-testid="stFileUploader"]')) continue;
        } catch (e0) { continue; }
        try {
          for (const inp of Array.from(node.querySelectorAll('input[type="file"]'))) pushInput(inp);
        } catch (e1) {}
      }
    } catch (e) {}
    return out;
  }
  function _setPromptUploaderFilesAllClones(form, filesLike) {
    if (typeof host.DataTransfer === "undefined") return false;
    const targetFiles = [];
    try {
      for (const f of Array.from(filesLike || [])) {
        if (f) targetFiles.push(f);
      }
    } catch (e) {}
    const targetSig = _fileListSig(targetFiles);
    const inputs = _listPromptUploaderInputs(form);
    if (!inputs.length) return false;
    let changedAny = false;
    for (const inp of inputs) {
      if (!inp) continue;
      try {
        const currSig = _fileListSig(Array.from(inp.files || []));
        if (currSig === targetSig) continue;
      } catch (e0) {}
      try {
        const dt = new host.DataTransfer();
        for (const f of targetFiles) dt.items.add(f);
        inp.files = dt.files;
        try { inp.dispatchEvent(new Event("input", { bubbles: true })); } catch (e1) {}
        try { inp.dispatchEvent(new Event("change", { bubbles: true })); } catch (e2) {}
        changedAny = true;
      } catch (e3) {}
    }
    return changedAny;
  }
  function _syncPromptUploaderFilesFromCurrent(form) {
    if (!form) return false;
    const input = _findPromptUploaderInput(form);
    if (!input) return false;
    try {
      return _setPromptUploaderFilesAllClones(form, Array.from(input.files || []));
    } catch (e) {
      return false;
    }
  }
  function _clickNativePromptUploaderRemove(form, sigToRemove, fileHint) {
    if (!form || !sigToRemove) return false;
    let input = null;
    try { input = _findPromptUploaderInput(form); } catch (e) {}
    if (!input) return false;
    let uploader = null;
    try { uploader = input.closest ? input.closest('div[data-testid="stFileUploader"]') : null; } catch (e) {}
    if (!uploader) {
      try { uploader = form.querySelector('div[data-testid="stFileUploader"]'); } catch (e2) {}
    }
    if (!uploader || !uploader.querySelectorAll) return false;

    const targetName = String((fileHint && fileHint.name) || "").trim().toLowerCase();
    let candidates = [];
    try {
      candidates = Array.from(uploader.querySelectorAll("button"));
    } catch (e) {
      candidates = [];
    }
    let bestBtn = null;
    let bestScore = -9999;
    let nonDropBtns = 0;
    for (const b of candidates) {
      if (!b) continue;
      try { if (b.disabled) continue; } catch (e0) {}
      try {
        if (b.closest && b.closest('[data-testid="stFileUploaderDropzone"]')) continue;
      } catch (e1) {}
      nonDropBtns += 1;
      let score = 0;
      let row = null;
      try {
        row = b.closest && b.closest('[data-testid*="FileUploaderFile"], [data-testid*="stFileUploaderFile"], li, [role="listitem"]');
      } catch (e2) {}
      let txt = "";
      try { txt = String(((row || b).innerText || (row || b).textContent) || "").toLowerCase(); } catch (e3) {}
      let aria = "";
      try {
        aria = String(b.getAttribute("aria-label") || b.getAttribute("title") || "").toLowerCase();
      } catch (e4) {}
      if (targetName && txt && txt.includes(targetName)) score += 100;
      if (aria && (aria.includes("remove") || aria.includes("delete") || aria.includes("clear") || aria.includes("移除") || aria.includes("删除"))) score += 30;
      if (row) score += 10;
      if (score > bestScore) {
        bestScore = score;
        bestBtn = b;
      }
    }
    if (!bestBtn) return false;
    if ((bestScore < 20) && !(nonDropBtns === 1)) return false;
    try {
      bestBtn.click();
      return true;
    } catch (e) {
      return false;
    }
  }
  function _fileSig(file) {
    if (!file) return "";
    return `${String(file.name || "")}::${Number(file.size || 0)}::${Number(file.lastModified || 0)}::${String(file.type || "")}`;
  }
  function _fileListSig(files) {
    try { return (files || []).map(_fileSig).join("|"); } catch (e) { return ""; }
  }
  function _findPromptTextAreaWrap(form) {
    if (!form || !form.querySelector) return null;
    return pickFresh([
      ...form.querySelectorAll('div[data-testid="stTextArea"]'),
      ...form.querySelectorAll('.stTextArea'),
    ]) || null;
  }
  function _ensureAttachStrip(form) {
    const taWrap = _findPromptTextAreaWrap(form);
    if (!taWrap) return null;
    let strip = null;
    try { strip = taWrap.querySelector(':scope > .kb-dock-attach-strip'); } catch (e) {}
    if (!strip) {
      try {
        strip = root.createElement("div");
        strip.className = "kb-dock-attach-strip";
        taWrap.appendChild(strip);
      } catch (e) {
        strip = null;
      }
    }
    return strip;
  }
  function _ensureImagePreviewModal() {
    let modal = null;
    try { modal = root.body.querySelector(':scope > .kb-dock-img-preview'); } catch (e) {}
    if (modal) return modal;
    try {
      modal = root.createElement("div");
      modal.className = "kb-dock-img-preview";
      modal.setAttribute("aria-hidden", "true");
      modal.innerHTML = [
        '<div class="kb-dock-img-preview-backdrop" data-kb-close="1"></div>',
        '<div class="kb-dock-img-preview-dialog" role="dialog" aria-modal="true" aria-label="Image preview">',
        '  <button type="button" class="kb-dock-img-preview-close" data-kb-close="1" aria-label="Close preview">×</button>',
        '  <img class="kb-dock-img-preview-img" alt="" />',
        '  <div class="kb-dock-img-preview-caption"></div>',
        '</div>'
      ].join("");
      root.body.appendChild(modal);
      modal.addEventListener("click", function (e) {
        try {
          const t = e.target;
          if (t && t.closest && t.closest('[data-kb-close="1"]')) {
            modal.classList.remove("is-open");
            modal.setAttribute("aria-hidden", "true");
          }
        } catch (err) {}
      }, true);
      root.addEventListener("keydown", function (e) {
        try {
          if (!modal.classList.contains("is-open")) return;
          if (e.key !== "Escape") return;
          modal.classList.remove("is-open");
          modal.setAttribute("aria-hidden", "true");
        } catch (err) {}
      }, true);
    } catch (e) {
      modal = null;
    }
    return modal;
  }
  function _openAttachImagePreview(item) {
    if (!item) return;
    const modal = _ensureImagePreviewModal();
    if (!modal) return;
    let imgEl = null;
    try { imgEl = item.querySelector(".kb-dock-attach-thumb"); } catch (e) {}
    if (!imgEl) return;
    const src = String(imgEl.getAttribute("src") || "").trim();
    if (!src) return;
    try {
      const modalImg = modal.querySelector(".kb-dock-img-preview-img");
      const modalCap = modal.querySelector(".kb-dock-img-preview-caption");
      if (modalImg) {
        modalImg.setAttribute("src", src);
        modalImg.setAttribute("alt", String(imgEl.getAttribute("alt") || "image"));
      }
      if (modalCap) {
        modalCap.textContent = String(item.getAttribute("title") || imgEl.getAttribute("alt") || "");
      }
      modal.classList.add("is-open");
      modal.setAttribute("aria-hidden", "false");
    } catch (e) {}
  }
  function _clearAttachStrip(strip) {
    if (!strip) return;
    try {
      const imgs = strip.querySelectorAll("img[data-kb-object-url]");
      for (const img of imgs) {
        try {
          const url = img.getAttribute("data-kb-object-url");
          if (url && host.URL && host.URL.revokeObjectURL) host.URL.revokeObjectURL(url);
        } catch (e) {}
      }
    } catch (e) {}
    try { strip.innerHTML = ""; } catch (e) {}
  }
  function _renderAttachStrip(form, files) {
    if (!form) return;
    const strip = _ensureAttachStrip(form);
    if (!strip) return;
    const filtered = Array.from(files || []).filter(_isPromptUploadFile);
    const sig = _fileListSig(filtered);
    if ((form.dataset.kbAttachPreviewSig || "") === sig) return;
    form.dataset.kbAttachPreviewSig = sig;

    _clearAttachStrip(strip);
    if (!filtered.length) {
      try { strip.style.display = "none"; } catch (e) {}
      try { form.classList.remove("kb-dock-has-attachments"); } catch (e) {}
      return;
    }

    try { strip.style.display = "flex"; } catch (e) {}
    try { form.classList.add("kb-dock-has-attachments"); } catch (e) {}

    const maxShow = 6;
    let shown = 0;
    for (const f of filtered) {
      if (shown >= maxShow) break;
      shown += 1;
      const type = String(f.type || "").toLowerCase();
      const isImg = type.startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp)$/i.test(String(f.name || ""));

      let item = null;
      try {
        item = root.createElement("div");
        item.className = "kb-dock-attach-item" + (isImg ? " is-image" : " is-file");
        item.setAttribute("title", String(f.name || ""));
        item.setAttribute("data-kb-file-sig", _fileSig(f));
      } catch (e) {
        item = null;
      }
      if (!item) continue;

      if (isImg) {
        try {
          const img = root.createElement("img");
          img.className = "kb-dock-attach-thumb";
          if (host.URL && host.URL.createObjectURL) {
            const objUrl = host.URL.createObjectURL(f);
            img.src = objUrl;
            img.setAttribute("data-kb-object-url", objUrl);
          }
          img.alt = String(f.name || "image");
          item.appendChild(img);
        } catch (e) {}
      } else {
        try {
          const icon = root.createElement("span");
          icon.className = "kb-dock-attach-fileicon";
          icon.textContent = "PDF";
          item.appendChild(icon);
          const name = root.createElement("span");
          name.className = "kb-dock-attach-label";
          name.textContent = String(f.name || "file.pdf");
          item.appendChild(name);
        } catch (e) {}
      }
      try {
        const removeBtn = root.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "kb-dock-attach-remove";
        removeBtn.setAttribute("aria-label", "Remove attachment");
        removeBtn.setAttribute("title", "Remove");
        removeBtn.textContent = "×";
        item.appendChild(removeBtn);
      } catch (e) {}
      try { strip.appendChild(item); } catch (e) {}
    }

    if (filtered.length > shown) {
      try {
        const more = root.createElement("div");
        more.className = "kb-dock-attach-more";
        more.textContent = `+${filtered.length - shown}`;
        strip.appendChild(more);
      } catch (e) {}
    }
  }
  function bindPromptUploadPreview(form) {
    if (!form) return;
    const input = _findPromptUploaderInput(form);
    _renderAttachStrip(form, input ? Array.from(input.files || []) : []);
    bindAttachStripInteractions(form);
    if (!input) return;
    if (input.dataset.kbAttachPreviewHooked === "1") return;
    input.dataset.kbAttachPreviewHooked = "1";
    try {
      input.addEventListener("change", function () {
        try {
          _renderAttachStrip(form, Array.from(input.files || []));
        } catch (e) {}
      }, true);
    } catch (e) {}
  }
  function _removePromptUploadBySig(form, sigToRemove) {
    if (!form || !sigToRemove) return false;
    const input = _findPromptUploaderInput(form);
    if (!input) return false;
    if (typeof host.DataTransfer === "undefined") return false;
    try {
      const prev = Array.from(input.files || []);
      if (!prev.length) return false;
      const kept = [];
      let removedFile = null;
      let removed = false;
      for (const f of prev) {
        const sig = _fileSig(f);
        if ((!removed) && sig === sigToRemove) {
          removedFile = f;
          removed = true;
          continue;
        }
        kept.push(f);
      }
      if (!removed) return false;
      let nativeRemoved = false;
      try { nativeRemoved = _clickNativePromptUploaderRemove(form, sigToRemove, removedFile); } catch (e0) {}
      if (!_setPromptUploaderFilesAllClones(form, kept)) {
        const dt = new host.DataTransfer();
        for (const f of kept) dt.items.add(f);
        input.files = dt.files;
        try { input.dispatchEvent(new Event("input", { bubbles: true })); } catch (e) {}
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }
      try { _dbgDock("attach-remove", form, state.ta, { sig: sigToRemove, nativeRemoved: !!nativeRemoved, keptCount: kept.length }, true); } catch (e1) {}
      return true;
    } catch (e) {
      return false;
    }
  }
  function bindAttachStripInteractions(form) {
    if (!form) return;
    const strip = _ensureAttachStrip(form);
    if (!strip) return;
    if (strip.dataset.kbAttachStripHooked === "1") return;
    strip.dataset.kbAttachStripHooked = "1";
    try {
      strip.addEventListener("click", function (e) {
        try {
          const target = e.target;
          if (!target || !target.closest) return;
          const removeBtn = target.closest(".kb-dock-attach-remove");
          if (removeBtn) {
            const item = removeBtn.closest(".kb-dock-attach-item");
            const sig = item ? String(item.getAttribute("data-kb-file-sig") || "") : "";
            if (!sig) return;
            e.preventDefault();
            e.stopPropagation();
            _removePromptUploadBySig(form, sig);
            return;
          }
          const item = target.closest(".kb-dock-attach-item.is-image");
          if (!item) return;
          e.preventDefault();
          e.stopPropagation();
          _openAttachImagePreview(item);
        } catch (err) {}
      }, true);
    } catch (e) {}
  }
  function _pushFilesToPromptUploader(form, rawFiles) {
    if (!form || !rawFiles || !rawFiles.length) return false;
    const input = _findPromptUploaderInput(form);
    if (!input) return false;
    if (typeof host.DataTransfer === "undefined") return false;

    const files = [];
    for (const f0 of rawFiles) {
      if (!_isPromptUploadFile(f0)) continue;
      const f = _ensurePromptUploadFileName(f0, "pasted");
      if (!f) continue;
      files.push(f);
    }
    if (!files.length) return false;

    try {
      const dt = new host.DataTransfer();
      const seen = new Set();
      try {
        const prev = Array.from(input.files || []);
        for (const pf of prev) {
          const key = `${pf.name}::${pf.size}::${pf.lastModified}`;
          if (seen.has(key)) continue;
          seen.add(key);
          dt.items.add(pf);
        }
      } catch (e) {}
      for (const f of files) {
        const key = `${f.name}::${f.size}::${f.lastModified}`;
        if (seen.has(key)) continue;
        seen.add(key);
        dt.items.add(f);
      }
      const merged = Array.from(dt.files || []);
      if (!_setPromptUploaderFilesAllClones(form, merged)) {
        input.files = dt.files;
        try { input.dispatchEvent(new Event("input", { bubbles: true })); } catch (e) {}
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }
      try { bindPromptUploadPreview(form); } catch (e) {}
      return true;
    } catch (e) {
      return false;
    }
  }
  function _collectClipboardFiles(e) {
    const out = [];
    if (!e) return out;
    const cd = e.clipboardData || null;
    if (!cd) return out;
    try {
      for (const f of Array.from(cd.files || [])) {
        if (f) out.push(f);
      }
    } catch (e0) {}
    if (out.length) return out;
    try {
      for (const item of Array.from(cd.items || [])) {
        if (!item || String(item.kind || "").toLowerCase() !== "file") continue;
        const f = item.getAsFile ? item.getAsFile() : null;
        if (f) out.push(f);
      }
    } catch (e1) {}
    return out;
  }
  function _hasFileDrag(dataTransfer) {
    try {
      if (!dataTransfer) return false;
      const types = Array.from(dataTransfer.types || []);
      return types.includes("Files");
    } catch (e) {
      return false;
    }
  }
  function bindPasteAndDrop(ta, form) {
    if (!ta || !form) return;
    if (ta.dataset.kbPasteDropHooked === "1") return;
    ta.dataset.kbPasteDropHooked = "1";

    const zone = (ta.closest && ta.closest('div[data-testid="stTextArea"]')) || ta;
    let dragDepth = 0;

    const setDropActive = function (on) {
      try {
        if (form && form.classList) form.classList.toggle("kb-dock-drop-active", !!on);
      } catch (e) {}
    };
    const clearDragState = function () {
      dragDepth = 0;
      setDropActive(false);
    };

    try {
      ta.addEventListener("paste", function (e) {
        try {
          const files = _collectClipboardFiles(e);
          if (!files.length) return;
          const ok = _pushFilesToPromptUploader(form, files);
          if (!ok) return;
          e.preventDefault();
          e.stopPropagation();
        } catch (err) {}
      }, { capture: true });
    } catch (e) {}

    const dragTarget = zone || ta;
    if (!dragTarget || dragTarget.dataset.kbDropZoneHooked === "1") return;
    dragTarget.dataset.kbDropZoneHooked = "1";

    const onDragEnter = function (e) {
      if (!_hasFileDrag(e.dataTransfer)) return;
      dragDepth += 1;
      setDropActive(true);
      try { e.preventDefault(); } catch (err) {}
    };
    const onDragOver = function (e) {
      if (!_hasFileDrag(e.dataTransfer)) return;
      setDropActive(true);
      try { e.preventDefault(); } catch (err) {}
      try { e.stopPropagation(); } catch (err) {}
      try { e.dataTransfer.dropEffect = "copy"; } catch (err) {}
    };
    const onDragLeave = function (e) {
      if (!_hasFileDrag(e.dataTransfer)) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth <= 0) clearDragState();
    };
    const onDrop = function (e) {
      if (!_hasFileDrag(e.dataTransfer)) return;
      try { e.preventDefault(); } catch (err) {}
      try { e.stopPropagation(); } catch (err) {}
      clearDragState();
      let files = [];
      try { files = Array.from((e.dataTransfer && e.dataTransfer.files) || []); } catch (err) {}
      if (!files.length) return;
      _pushFilesToPromptUploader(form, files);
    };

    try { dragTarget.addEventListener("dragenter", onDragEnter, true); } catch (e) {}
    try { dragTarget.addEventListener("dragover", onDragOver, true); } catch (e) {}
    try { dragTarget.addEventListener("dragleave", onDragLeave, true); } catch (e) {}
    try { dragTarget.addEventListener("drop", onDrop, true); } catch (e) {}
    try { form.addEventListener("drop", clearDragState, true); } catch (e) {}
    try { form.addEventListener("dragend", clearDragState, true); } catch (e) {}
  }
  function hook() {
    const hit = findPromptFormAndTextarea();
    if (!hit.form || !hit.ta) {
      _dbgDock("hook:no-hit", null, null, { reason: "findPromptFormAndTextarea returned null", candidates: _dbgCandidateScan(8) });
      scheduleRecoveryRetry(70);
      return;
    }
    const prevForm = (state.form && state.form !== hit.form) ? state.form : null;
    state.form = hit.form;
    state.ta = hit.ta;
    ensurePromptRootMarkers(state.form);
    // Decorate/mount action wrappers first so the height sanity check measures the compact dock layout,
    // not Streamlit's temporary stacked form rows (textarea + uploader + submit).
    decoratePromptButtons(state.form);
    const actionInfo = mountPromptActionWrappers(state.form) || {};
    bindFormSubmitRecovery(state.form);
    if (!isActionLayerReady(state.form, actionInfo)) {
      let keptDock = false;
      try { keptDock = !!placeDock(state.form); } catch (e) {}
      _dbgDock("hook:action-not-ready", state.form, state.ta, {
        reason: "isActionLayerReady=false",
        actionInfo: actionInfo,
        keptDock: !!keptDock,
        candidates: _dbgCandidateScan(6)
      }, true);
      scheduleRecoveryRetry(70);
      return;
    }
    if (!placeDock(state.form)) {
      const alreadyDocked = !!(state.form && state.form.classList && state.form.classList.contains("kb-input-dock"));
      _dbgDock("hook:place-failed", state.form, state.ta, {
        reason: "placeDock=false",
        actionInfo: actionInfo,
        alreadyDocked: !!alreadyDocked,
        candidates: _dbgCandidateScan(6)
      }, true);
      if (!alreadyDocked) resetDockStyles(state.form);
      scheduleRecoveryRetry(70);
      return;
    }
    try {
      const rect = state.form.getBoundingClientRect();
      const viewportH = Math.max(0, host.innerHeight || root.documentElement.clientHeight || 0);
      // Only trip the fail-safe for truly absurd heights (likely wrong wrapper / stale node).
      // Do not permanently disable docking; transient stale DOM during Streamlit reruns should self-recover.
      const maxDockH = Math.max(760, Math.floor(viewportH * 0.92));
      if (rect && isFinite(rect.height) && rect.height > maxDockH) {
        const keepWhileStreaming = !!(root.body && root.body.classList && root.body.classList.contains("kb-live-streaming"));
        _dbgDock("hook:height-failsafe", state.form, state.ta, {
          reason: "rect.height too large",
          rectHeight: Number(rect.height || 0),
          maxDockH: maxDockH,
          actionInfo: actionInfo,
          keepWhileStreaming: !!keepWhileStreaming
        }, true);
        if (!keepWhileStreaming) resetDockStyles(state.form);
        scheduleRecoveryRetry(90);
        return;
      }
    } catch (e) {}
    clearRecoveryRetry();
    if (prevForm && prevForm !== state.form) {
      _dbgDock("hook:handoff-reset-prev", state.form, state.ta, { prevForm: _dbgElBrief(prevForm), actionInfo: actionInfo }, true);
      try { resetDockStyles(prevForm); } catch (e) {}
    }
    _dbgDock("hook:ok", state.form, state.ta, { prevChanged: !!prevForm, actionInfo: actionInfo });
    bindCtrlEnter(state.ta, state.form);
    bindPasteAndDrop(state.ta, state.form);
    bindPromptUploadPreview(state.form);
  }
  function scheduleHook(force) {
    if (state.raf) return;
    const now = Date.now ? Date.now() : (+new Date());
    const minGapMs = force ? 0 : 70;
    const elapsed = now - Number(state.lastHookTs || 0);
    if (elapsed < minGapMs) {
      if (state.delayTimer) return;
      state.delayTimer = host.setTimeout(function () {
        state.delayTimer = 0;
        scheduleHook(true);
      }, Math.max(8, minGapMs - elapsed));
      return;
    }
    state.raf = host.requestAnimationFrame(function () {
      state.raf = 0;
      state.lastHookTs = Date.now ? Date.now() : (+new Date());
      hook();
    });
  }
  function nodeMaybeAffectsDock(node) {
    if (!node) return false;
    let el = null;
    if (node.nodeType === 1) el = node;
    else if (node.nodeType === 3) el = node.parentElement || null;
    if (!el || !el.querySelector) return false;
    try {
      if (el.classList) {
        if (
          el.classList.contains("kb-dock-action-layer") ||
          el.classList.contains("kb-dock-send-anchor") ||
          el.classList.contains("kb-dock-stop-anchor") ||
          el.classList.contains("kb-dock-attach-strip")
        ) return false;
      }
    } catch (e) {}
    try {
      if (el.matches && el.matches('section[data-testid="stSidebar"]')) return true;
    } catch (e) {}
    try {
      if (el.closest && el.closest('section[data-testid="stSidebar"]')) return true;
    } catch (e) {}
    try {
      if (el.matches && el.matches('form, div[data-testid="stForm"], textarea, div[data-testid="stTextArea"], .stTextArea, input[type="file"]')) {
        return true;
      }
    } catch (e) {}
    try {
      if (el.closest && el.closest('form, div[data-testid="stForm"]')) return true;
    } catch (e) {}
    try {
      return !!el.querySelector(
        'form textarea, div[data-testid="stForm"] textarea, form input[type="file"], form div[data-testid="stTextArea"], form .stTextArea'
      );
    } catch (e) {
      return false;
    }
  }
  function mutationsMayAffectDock(records) {
    for (const rec of (records || [])) {
      if (!rec) continue;
      if (rec.type === "childList") {
        if (nodeMaybeAffectsDock(rec.target)) return true;
        try {
          for (const n of Array.from(rec.addedNodes || [])) {
            if (nodeMaybeAffectsDock(n)) return true;
          }
        } catch (e) {}
        try {
          for (const n of Array.from(rec.removedNodes || [])) {
            if (nodeMaybeAffectsDock(n)) return true;
          }
        } catch (e) {}
        continue;
      }
      if (rec.type === "attributes") {
        if (nodeMaybeAffectsDock(rec.target)) return true;
      }
    }
    return false;
  }
  function startDragIfNearSidebarEdge(e) {
    if (DISABLE_SIDEBAR_EDGE_DRAG) return;
    const sidebar = findSidebar();
    if (!sidebar || !e) return;
    try {
      if (isInteractiveTarget(e.target || null)) return;
    } catch (err) {}
    const clientX = Number(e.clientX);
    if (!isFinite(clientX)) return;
    const rect = sidebar.getBoundingClientRect();
    if (!rect || !isFinite(rect.right)) return;
    const nearEdge = Math.abs(rect.right - clientX) <= 24;
    if (!nearEdge) return;
    state.dragging = true;
    setResizing(true);
    scheduleHook();
  }
  function onDragMove() {
    if (!state.dragging) return;
    scheduleHook();
  }
  function stopDrag() {
    if (!state.dragging) return;
    state.dragging = false;
    setResizing(false);
    scheduleHook();
  }
  function installListeners() {
    state.onMouseDown = DISABLE_SIDEBAR_EDGE_DRAG ? null : startDragIfNearSidebarEdge;
    state.onPointerDown = DISABLE_SIDEBAR_EDGE_DRAG ? null : startDragIfNearSidebarEdge;
    state.onTouchStart = DISABLE_SIDEBAR_EDGE_DRAG ? null : function (e) {
      const t = (e.touches && e.touches[0]) ? e.touches[0] : null;
      if (t) startDragIfNearSidebarEdge(t);
    };
    state.onMouseMove = DISABLE_SIDEBAR_EDGE_DRAG ? null : onDragMove;
    state.onPointerMove = DISABLE_SIDEBAR_EDGE_DRAG ? null : onDragMove;
    state.onMouseUp = DISABLE_SIDEBAR_EDGE_DRAG ? null : stopDrag;
    state.onPointerUp = DISABLE_SIDEBAR_EDGE_DRAG ? null : stopDrag;
    state.onPointerCancel = DISABLE_SIDEBAR_EDGE_DRAG ? null : stopDrag;
    state.onTouchEnd = DISABLE_SIDEBAR_EDGE_DRAG ? null : stopDrag;
    state.onKeyDown = function (e) {
      try {
        const isCtrlEnter = (e.ctrlKey || e.metaKey) && e.key === "Enter";
        if (!isCtrlEnter || e.isComposing) return;
        const target = e.target;
        if (!target || isInsideStaleNode(target)) return;
        const ta = (target.tagName === "TEXTAREA") ? target : (target.closest ? target.closest("textarea") : null);
        if (!ta) return;
        const hit = findPromptFormAndTextarea();
        if (!hit.form || !hit.ta) return;
        if (ta !== hit.ta && !hit.form.contains(ta)) return;
        const ok = clickSendButton(hit.form);
        if (!ok) return;
        e.preventDefault();
        e.stopPropagation();
      } catch (err) {}
    };
    state.onBlur = stopDrag;
    state.onResize = scheduleHook;

    if (state.onMouseDown) root.addEventListener("mousedown", state.onMouseDown, true);
    if (state.onPointerDown) root.addEventListener("pointerdown", state.onPointerDown, true);
    if (state.onTouchStart) root.addEventListener("touchstart", state.onTouchStart, true);
    if (state.onMouseMove) root.addEventListener("mousemove", state.onMouseMove, true);
    if (state.onPointerMove) root.addEventListener("pointermove", state.onPointerMove, true);
    if (state.onMouseUp) root.addEventListener("mouseup", state.onMouseUp, true);
    if (state.onPointerUp) root.addEventListener("pointerup", state.onPointerUp, true);
    if (state.onPointerCancel) root.addEventListener("pointercancel", state.onPointerCancel, true);
    if (state.onTouchEnd) root.addEventListener("touchend", state.onTouchEnd, true);
    root.addEventListener("keydown", state.onKeyDown, true);
    host.addEventListener("blur", state.onBlur, true);
    host.addEventListener("resize", state.onResize, { passive: true });
  }
  function installObservers() {
    let installedAny = false;
    if (typeof ResizeObserver !== "undefined") {
      try {
        state.roSizeMap = (typeof WeakMap !== "undefined") ? new WeakMap() : null;
        state.ro = new ResizeObserver(function (entries) {
          let widthChanged = false;
          for (const ent of (entries || [])) {
            const t = ent && ent.target;
            if (!t) continue;
            let w = 0;
            try {
              if (ent.contentRect && isFinite(ent.contentRect.width)) w = Number(ent.contentRect.width || 0);
              else if (t.getBoundingClientRect) w = Number((t.getBoundingClientRect() || {}).width || 0);
            } catch (e) { w = 0; }
            if (!isFinite(w)) w = 0;
            if (!state.roSizeMap) {
              widthChanged = true;
              break;
            }
            const prev = state.roSizeMap.get(t);
            state.roSizeMap.set(t, w);
            if (!isFinite(prev) || Math.abs(prev - w) >= 1) {
              widthChanged = true;
              break;
            }
          }
          if (widthChanged) scheduleHook();
        });
        const candidates = [root.documentElement, findSidebar(), findMainContainer(), findMainRegion()];
        for (const c of candidates) {
          if (!c) continue;
          state.ro.observe(c);
          installedAny = true;
        }
      } catch (e) {}
    }
    if (typeof MutationObserver !== "undefined") {
      try {
        state.mo = new MutationObserver(function (records) {
          if (!mutationsMayAffectDock(records)) return;
          scheduleHook();
        });
        state.mo.observe(root.body, {
          childList: true,
          subtree: true,
          attributes: true,
          attributeFilter: ["class", "data-stale", "style", "aria-label"]
        });
        installedAny = true;
      } catch (e) {}
    }
    return installedAny;
  }
  function destroy() {
    _dbgDock("destroy", state.form, state.ta, null, true);
    try { if (state.timer) host.clearInterval(state.timer); } catch (e) {}
    try { if (state.delayTimer) host.clearTimeout(state.delayTimer); } catch (e) {}
    try { if (state.recoverTimer) host.clearTimeout(state.recoverTimer); } catch (e) {}
    try { if (state.raf) host.cancelAnimationFrame(state.raf); } catch (e) {}
    try { if (state.ro) state.ro.disconnect(); } catch (e) {}
    try { if (state.mo) state.mo.disconnect(); } catch (e) {}
    try { root.removeEventListener("mousedown", state.onMouseDown, true); } catch (e) {}
    try { root.removeEventListener("pointerdown", state.onPointerDown, true); } catch (e) {}
    try { root.removeEventListener("touchstart", state.onTouchStart, true); } catch (e) {}
    try { root.removeEventListener("mousemove", state.onMouseMove, true); } catch (e) {}
    try { root.removeEventListener("pointermove", state.onPointerMove, true); } catch (e) {}
    try { root.removeEventListener("mouseup", state.onMouseUp, true); } catch (e) {}
    try { root.removeEventListener("pointerup", state.onPointerUp, true); } catch (e) {}
    try { root.removeEventListener("pointercancel", state.onPointerCancel, true); } catch (e) {}
    try { root.removeEventListener("touchend", state.onTouchEnd, true); } catch (e) {}
    try { root.removeEventListener("keydown", state.onKeyDown, true); } catch (e) {}
    try { host.removeEventListener("blur", state.onBlur, true); } catch (e) {}
    try { host.removeEventListener("resize", state.onResize, false); } catch (e) {}
    setResizing(false);
    resetDockStyles(state.form);
  }

  try {
    host.__kbDockDebugDump = function (label) {
      try {
        _dbgDock(String(label || "manual-dump"), state.form, state.ta, { candidates: _dbgCandidateScan(10) }, true);
      } catch (e) {}
    };
  } catch (e) {}

  host[NS] = { destroy, schedule: scheduleHook };
  installListeners();
  const hasObservers = installObservers();
  state.timer = host.setInterval(function () {
    let should = false;
    try {
      should = !!(
        root.body &&
        root.body.classList &&
        root.body.classList.contains("kb-live-streaming")
      );
    } catch (e) {}
    if (!should) {
      try {
        should = !(
          state.form &&
          state.form.isConnected &&
          state.form.classList &&
          state.form.classList.contains("kb-dock-positioned")
        );
      } catch (e) {
        should = true;
      }
    }
    if (!should) {
      try {
        should = !!(state.form && !_isVisibleForDock(state.form));
      } catch (e) {
        should = true;
      }
    }
    if (!should) {
      try {
        should = !!(
          !state.ta ||
          state.ta.isConnected === false ||
          isInsideStaleNode(state.ta) ||
          !_isVisibleForDock(state.ta)
        );
      } catch (e) {
        should = true;
      }
    }
    if (should || !hasObservers) scheduleHook();
  }, 700);
  _dbgDock("init", state.form, state.ta, { ns: NS }, true);
  scheduleHook(true);
})();
