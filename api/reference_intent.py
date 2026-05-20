from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SectionIntentLexicon:
    intent: str
    prompt_patterns: tuple[str, ...]
    base_terms: tuple[str, ...]
    heading_positive: tuple[str, ...]
    heading_negative: tuple[str, ...] = ()


SECTION_INTENT_LEXICONS: tuple[SectionIntentLexicon, ...] = (
    SectionIntentLexicon(
        intent="problem",
        prompt_patterns=(
            r"\b(?:problem|challenge|motivation|contribution|novelty|what .* solve|why .* paper|why .* study|research question)\b",
            r"(?:解决.{0,8}什么问题|想解决|要解决|为什么要做|研究动机|动机|核心问题|主要问题|问题是什么|挑战|瓶颈|贡献|创新点|新意|有什么用|这篇文章.{0,8}做什么)",
        ),
        base_terms=(
            "problem",
            "challenge",
            "motivation",
            "contribution",
            "novelty",
            "research question",
            "问题",
            "挑战",
            "动机",
            "贡献",
            "创新点",
            "瓶颈",
            "目标",
        ),
        heading_positive=(
            "abstract",
            "introduction",
            "motivation",
            "problem",
            "challenge",
            "contribution",
            "overview",
            "摘要",
            "引言",
            "介绍",
            "动机",
            "问题",
            "挑战",
            "贡献",
        ),
        heading_negative=("references", "bibliography", "related work", "experiment", "result", "参考文献", "相关工作", "实验", "结果"),
    ),
    SectionIntentLexicon(
        intent="related",
        prompt_patterns=(
            r"\b(?:related work|prior work|older methods?|traditional methods?|background literature|literature review|cited works?|where .* come from|who did .* before)\b",
            r"(?:相关工作|前人工作|已有工作|已有方法|传统方法|背景文献|文献综述|引用|文内参考|参考编号|借鉴|以前|之前|前作|来源|脉络|谁做|谁先做|谁提出|怎么来的|从哪来|自己想出来|自己发明|参考了谁|沿着哪条线)",
        ),
        base_terms=(
            "related work",
            "prior work",
            "previous work",
            "traditional method",
            "literature",
            "background",
            "相关工作",
            "前人工作",
            "已有方法",
            "传统方法",
            "背景文献",
        ),
        heading_positive=(
            "related work",
            "prior work",
            "previous work",
            "literature",
            "background",
            "相关工作",
            "背景",
        ),
        heading_negative=("method", "experiment", "result", "方法", "实验", "结果"),
    ),
    SectionIntentLexicon(
        intent="method",
        prompt_patterns=(
            r"\b(?:method|methods|methodology|reproduce|replicate|implementation|pipeline|framework|model|training loss|core idea)\b",
            r"\b(?:how can .*recover|why it is plausible|how .* work|how .* works|how .* implement|how .* reproduce)\b",
            r"(?:方法|实现|复现|模块|流程|框架|模型|物理模型|训练损失|核心想法|为什么.*恢复|怎么.*恢复|如何.*工作|怎么做到|怎么实现|怎么跑起来|原理|关键步骤|看懂算法|复现代码)",
        ),
        base_terms=(
            "method",
            "methodology",
            "model",
            "framework",
            "pipeline",
            "implementation",
            "training objective",
            "training loss",
            "方法",
            "模型",
            "框架",
            "流程",
            "实现",
            "训练目标",
        ),
        heading_positive=(
            "method",
            "methodology",
            "model",
            "framework",
            "pipeline",
            "implementation",
            "approach",
            "方法",
            "模型",
            "框架",
            "流程",
        ),
        heading_negative=("experiment", "result", "conclusion", "实验", "结果", "结论"),
    ),
    SectionIntentLexicon(
        intent="experiments",
        prompt_patterns=(
            r"\b(?:experiment|experimental|results?|evidence|strongest evidence|ablation|limitations?|follow-up|future work|baseline|metrics?|dataset)\b",
            r"(?:实验|结果|指标|基线|消融|数据集|真实数据|合成数据|可信度|可靠|靠谱|支撑结论|证据不足|不公平|局限|不足|后续工作|够不够|充分|对比公平|有没有消融|能不能证明|结论站得住)",
        ),
        base_terms=(
            "experiment",
            "experimental setup",
            "results",
            "evaluation",
            "metric",
            "baseline",
            "dataset",
            "ablation",
            "limitation",
            "实验",
            "实验设置",
            "结果",
            "评价指标",
            "基线",
            "数据集",
            "消融",
            "局限",
        ),
        heading_positive=(
            "experiment",
            "experimental",
            "result",
            "evaluation",
            "ablation",
            "dataset",
            "limitation",
            "future",
            "实验",
            "结果",
            "评价",
            "消融",
            "数据集",
            "局限",
        ),
        heading_negative=("method", "methodology", "方法"),
    ),
)

_LEXICON_BY_INTENT = {item.intent: item for item in SECTION_INTENT_LEXICONS}


def normalize_intent_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def refs_prompt_section_intent(prompt: str) -> str:
    low = normalize_intent_text(prompt)
    if not low:
        return ""
    best_intent = ""
    best_score = 0.0
    for lexicon in SECTION_INTENT_LEXICONS:
        score = _lexicon_prompt_score(low, lexicon)
        if score > best_score:
            best_score = score
            best_intent = lexicon.intent
    return best_intent


def _lexicon_prompt_score(normalized_prompt: str, lexicon: SectionIntentLexicon) -> float:
    score = 0.0
    for pattern in lexicon.prompt_patterns:
        matches = list(re.finditer(pattern, normalized_prompt, flags=re.I))
        if not matches:
            continue
        score += min(3.0, float(len(matches)))
    return score


def _quoted_terms(prompt: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"[\"'“”‘’「」『』《》]([^\"'“”‘’「」『』《》]{2,80})[\"'“”‘’「」『』《》]", str(prompt or "")):
        term = re.sub(r"\s+", " ", str(m.group(1) or "").strip())
        if term:
            out.append(term)
    return out


def _technical_phrases(prompt: str) -> list[str]:
    raw = str(prompt or "")
    out: list[str] = []
    heads = (
        "imaging",
        "sensing",
        "field",
        "fields",
        "reconstruction",
        "rendering",
        "network",
        "transformer",
        "diffusion",
        "attention",
        "optimization",
        "model",
        "framework",
        "compression",
        "measurement",
    )
    head_re = "|".join(re.escape(head) for head in heads)
    pattern = re.compile(
        rf"\b([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){{0,4}}\s+(?:{head_re}))\b",
        flags=re.I,
    )
    for match in pattern.finditer(raw):
        term = re.sub(r"\s+", " ", str(match.group(1) or "").strip())
        if term and len(term) >= 5:
            out.append(term)
    return out


def refs_prompt_topic_terms(prompt: str) -> list[str]:
    raw = str(prompt or "")
    terms: list[str] = []
    terms.extend(_quoted_terms(raw))
    for token in re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", raw):
        terms.append(token)
    terms.extend(_technical_phrases(raw.lower()))
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        norm = normalize_intent_text(term)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(str(term).strip())
    return out


def refs_section_intent_terms(prompt: str, intent: str) -> tuple[str, ...]:
    intent_norm = str(intent or "").strip().lower()
    lexicon = _LEXICON_BY_INTENT.get(intent_norm)
    terms: list[str] = []
    terms.extend(refs_prompt_topic_terms(prompt))
    if lexicon:
        terms.extend(lexicon.base_terms)
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        norm = normalize_intent_text(term)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(str(term).strip())
    return tuple(out)


def refs_section_intent_heading_score(prompt: str, heading: str) -> float:
    intent = refs_prompt_section_intent(prompt)
    if not intent:
        return 0.0
    lexicon = _LEXICON_BY_INTENT.get(intent)
    h = normalize_intent_text(heading)
    if not h or not lexicon:
        return 0.0
    score = 0.0
    if re.search(r"\b(references?|bibliography|works cited)\b|参考文献", h):
        score -= 6.0
    if re.search(r"\babstract\b|摘要", h):
        score -= 1.1
    for term in lexicon.heading_positive:
        t = normalize_intent_text(term)
        if t and t in h:
            score += 5.5 if (" " in t or len(t) >= 6) else 4.2
    for term in lexicon.heading_negative:
        t = normalize_intent_text(term)
        if t and t in h:
            score -= 1.4
    if intent == "experiments" and re.search(r"\b(future|limitation|conclusion)\b|局限|不足|后续", h):
        score += 1.3
    return score
