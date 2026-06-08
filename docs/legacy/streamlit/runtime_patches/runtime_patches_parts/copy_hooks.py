from __future__ import annotations

import streamlit.components.v1 as components

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

