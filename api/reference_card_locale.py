from __future__ import annotations

import os
import re

from api.deps import load_prefs
from kb.answer_contract import _prefer_zh_locale


def _refs_card_locale_pref() -> str:
    raw = str(os.environ.get("KB_REFS_CARD_LOCALE") or "").strip().lower()
    if raw in {"zh", "en", "auto"}:
        return raw
    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    raw = str((prefs or {}).get("refs_card_locale") or "").strip().lower()
    if raw in {"zh", "en", "auto"}:
        return raw
    return "auto"


def _refs_card_ui_locale_pref() -> str:
    try:
        prefs = load_prefs()
    except Exception:
        prefs = {}
    raw = str((prefs or {}).get("ui_locale") or "").strip().lower()
    return raw if raw in {"zh", "en"} else ""


def _ref_card_user_locale(prompt: str = "", *fallback_texts: str) -> str:
    pref = _refs_card_locale_pref()
    if pref in {"zh", "en"}:
        return pref

    # "auto" is a user-interface preference: cards should stay in the
    # language selected for the product even when a prompt or a paper happens
    # to use another language.  Prompt inference is only a fallback for older
    # installations that have no saved UI locale.
    ui_pref = _refs_card_ui_locale_pref()
    if ui_pref in {"zh", "en"}:
        return ui_pref

    prompt_text = str(prompt or "").strip()
    if prompt_text:
        if _prefer_zh_locale(prompt_text):
            return "zh"
        if _prompt_strongly_prefers_english(prompt_text):
            return "en"

    fallback_parts = [str(text or "").strip() for text in fallback_texts if str(text or "").strip()]
    if fallback_parts:
        return "zh" if _prefer_zh_locale(*fallback_parts) else "en"
    return "en"


def _prefer_zh_ref_card_locale(*texts: str) -> bool:
    prompt = str(texts[0] or "") if texts else ""
    fallback_texts = tuple(str(text or "") for text in texts[1:]) if len(texts) >= 2 else ()
    return _ref_card_user_locale(prompt, *fallback_texts) == "zh"


def _prompt_strongly_prefers_english(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return cjk == 0 and latin >= 4
