(function () {
  const host = window.parent || window;
  const root = host.document;
  if (!root || !root.body) return;

  const NS = "__kbDockManagerStableV3";
  if (host[NS] && typeof host[NS].destroy === "function") {
    try { host[NS].destroy(); } catch (e) {}
  }

  const RESIZE_CLASS = "kb-resizing";
  const DOCK_SIDE_GAP = 35;
  const DOCK_RIGHT_GAP = 35;
  const MIN_WIDTH = 320;
  // Sidebar-edge drag support conflicts with Streamlit's native sidebar collapse button
  // on newer builds. Prefer native behavior reliability over this optional enhancement.
  const DISABLE_SIDEBAR_EDGE_DRAG = true;
  const state = {
    raf: 0,
    timer: 0,
    ro: null,
    mo: null,
    dragging: false,
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
  };

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
  function hasSendButton(form) {
    if (!form) return false;
    const btns = form.querySelectorAll('button');
    for (const b of btns) {
      if (isInsideStaleNode(b)) continue;
      const txt = _btnText(b);
      if (isStopBtnText(txt)) continue;
      if (isSendBtnText(txt)) return true;
    }
    return false;
  }
  function clickSendButton(form) {
    if (!form) return false;
    const btns = Array.from(form.querySelectorAll("button"));
    let fallback = null;
    for (const b of btns) {
      if (!b || isInsideStaleNode(b) || b.disabled) continue;
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
    try { return btn.parentElement || null; } catch (e) {}
    return null;
  }
  function decoratePromptButtons(form) {
    if (!form) return;
    const btns = Array.from(form.querySelectorAll("button"));
    const submitCandidates = [];
    let hasSendClass = false;
    for (const b of btns) {
      if (!b || isInsideStaleNode(b)) continue;
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
      if (isStopBtnText(txt)) {
        try { b.classList.add("kb-dock-stop-btn"); } catch (e) {}
        if (wrap) {
          try { wrap.classList.add("kb-dock-stop-wrap"); } catch (e) {}
        }
        continue;
      }
      if (isSendBtnText(txt)) {
        try { b.classList.add("kb-dock-send-btn"); } catch (e) {}
        if (wrap) {
          try { wrap.classList.add("kb-dock-send-wrap"); } catch (e) {}
        }
        hasSendClass = true;
      }
    }
    if (!hasSendClass && submitCandidates.length) {
      let picked = null;
      for (const item of submitCandidates) {
        if (!isStopBtnText(item.txt)) { picked = item; break; }
      }
      if (!picked) picked = submitCandidates[0];
      if (picked && picked.b) {
        try { picked.b.classList.add("kb-dock-send-btn"); } catch (e) {}
        if (picked.wrap) {
          try { picked.wrap.classList.add("kb-dock-send-wrap"); } catch (e) {}
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
      for (const el of sendWraps) {
        if (!el || isInsideStaleNode(el)) continue;
        sendWrap = el;
        break;
      }
    } catch (e) {}
    try {
      const stopWraps = Array.from(form.querySelectorAll(".kb-dock-stop-wrap"));
      for (const el of stopWraps) {
        if (!el || isInsideStaleNode(el)) continue;
        stopWrap = el;
        break;
      }
    } catch (e) {}

    if (sendAnchor && sendWrap && sendWrap.parentElement !== sendAnchor) {
      try { sendAnchor.appendChild(sendWrap); } catch (e) {}
    }
    if (stopAnchor && stopWrap && stopWrap.parentElement !== stopAnchor) {
      try { stopAnchor.appendChild(stopWrap); } catch (e) {}
    }
  }
  function findPromptFormAndTextarea() {
    const forms = [
      ...root.querySelectorAll('form'),
      ...root.querySelectorAll('div[data-testid="stForm"]')
    ];
    const mainRegion = findMainRegion();
    for (const form of forms) {
      if (!form) continue;
      let targetForm = form;
      try {
        if (String(form.tagName || "").toUpperCase() !== "FORM") {
          const innerForm = form.querySelector ? form.querySelector("form") : null;
          if (innerForm) targetForm = innerForm;
        }
      } catch (e) {}
      if (!targetForm) continue;
      if (isInsideStaleNode(targetForm)) continue;
      if (mainRegion && targetForm.closest && !targetForm.closest('section.main')) continue;
      const ta =
        targetForm.querySelector('div[data-testid="stTextArea"] textarea') ||
        targetForm.querySelector('.stTextArea textarea') ||
        targetForm.querySelector('textarea');
      if (!ta || isInsideStaleNode(ta)) continue;
      if (hasSendButton(targetForm)) return { form: targetForm, ta };
    }
    return { form: null, ta: null };
  }
  function resetDockStyles(form) {
    if (!form) return;
    try {
      form.classList.remove('kb-input-dock', 'kb-dock-positioned');
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
    if (!form) return;
    const mainContainer = findMainContainer();
    const mainRegion = findMainRegion();
    const sidebar = findSidebar();
    const anchor = mainContainer || mainRegion;
    if (!anchor && !sidebar) return;

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
    if (!isFinite(leftBound) || !isFinite(rightBound)) return;

    rightBound = Math.min(rightBound, viewportW - DOCK_RIGHT_GAP);
    if (rightBound - leftBound < MIN_WIDTH) rightBound = leftBound + MIN_WIDTH;
    if (rightBound > viewportW - DOCK_RIGHT_GAP) {
      rightBound = viewportW - DOCK_RIGHT_GAP;
      leftBound = Math.max(DOCK_SIDE_GAP, rightBound - MIN_WIDTH);
    }

    const dockLeft = Math.max(DOCK_SIDE_GAP, Math.floor(leftBound));
    const dockWidth = Math.max(MIN_WIDTH, Math.floor(rightBound - dockLeft));

    form.classList.add('kb-input-dock', 'kb-dock-positioned');
    try {
      form.style.setProperty('height', 'auto', 'important');
      form.style.setProperty('min-height', '0', 'important');
      form.style.setProperty('max-height', 'none', 'important');
      form.style.setProperty('flex', 'none', 'important');
      form.style.setProperty('display', 'block', 'important');
    } catch (e) {}
    form.style.left = dockLeft + 'px';
    form.style.right = 'auto';
    form.style.width = dockWidth + 'px';
    form.style.transform = 'none';
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
  function hook() {
    const hit = findPromptFormAndTextarea();
    if (!hit.form || !hit.ta) return;
    state.form = hit.form;
    state.ta = hit.ta;
    // Decorate/mount action wrappers first so the height sanity check measures the compact dock layout,
    // not Streamlit's temporary stacked form rows (textarea + uploader + submit).
    decoratePromptButtons(state.form);
    mountPromptActionWrappers(state.form);
    if (!state.disableDockCompat) {
      state.form.classList.add("kb-input-dock");
      placeDock(state.form);
      try {
        const rect = state.form.getBoundingClientRect();
        const viewportH = Math.max(0, host.innerHeight || root.documentElement.clientHeight || 0);
        const maxDockH = Math.max(360, Math.floor(viewportH * 0.62));
        if (rect && isFinite(rect.height) && rect.height > maxDockH) {
          // Streamlit DOM shape changed: docking the chosen wrapper can explode to full-height.
          // Fail safe to native layout instead of breaking the whole chat page.
          state.disableDockCompat = true;
          resetDockStyles(state.form);
        }
      } catch (e) {}
    }
    bindCtrlEnter(state.ta, state.form);
  }
  function scheduleHook() {
    if (state.raf) return;
    state.raf = host.requestAnimationFrame(function () {
      state.raf = 0;
      hook();
    });
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
    if (typeof ResizeObserver !== "undefined") {
      try {
        state.ro = new ResizeObserver(function () { scheduleHook(); });
        const candidates = [root.documentElement, root.body, findSidebar(), findMainContainer(), findMainRegion()];
        for (const c of candidates) {
          if (c) state.ro.observe(c);
        }
      } catch (e) {}
    }
    if (typeof MutationObserver !== "undefined") {
      try {
        state.mo = new MutationObserver(function () { scheduleHook(); });
        state.mo.observe(root.body, { childList: true, subtree: true, attributes: true });
      } catch (e) {}
    }
  }
  function destroy() {
    try { if (state.timer) host.clearInterval(state.timer); } catch (e) {}
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

  host[NS] = { destroy, schedule: scheduleHook };
  installListeners();
  installObservers();
  state.timer = host.setInterval(scheduleHook, 120);
  scheduleHook();
})();
