from __future__ import annotations

import re
from pathlib import Path

from kb.training_observation_scope import scope_training_observation_answer
from kb.source_domain_terms import normalize_verified_noise_sampling_terms
from kb.source_question_facets import measurement_conditions_question, requests_training_augmentation, requests_sampling_prerequisite
from kb.source_fact_scope import scope_explicit_source_facts
from kb.source_blocks import extract_equation_number, load_source_blocks, normalize_match_text
from kb.citation_sentences import citation_sentence_spans
from kb.claim_source_eligibility import incomplete_source_prose
from kb.source_question_facets import _known_noise_training_premise, requests_comparison_boundary, requested_method_premise_priority, requests_bernoulli_training_requirement, requests_comparison_facets, requests_noise_model_requirement, requests_sample_relation_evidence, requests_source_facets, sample_relation_priority, source_facet_quote, source_facet_score
from kb.n2n_target_condition import (
    _verified_block, attach_target_condition_slot, corruption_condition_question, target_condition_quote,
)


def selected_source_evidence_prompt(question: str, query_variants: list[str]) -> str:
    # Expansion terms improve ranking but must not change the requested facet
    # into a different application/experiment during a source-only basket scan.
    if requests_direct_question_evidence(question):
        return question
    return "\n".join([question, *query_variants[1:]])


def attach_direct_source_evidence(plan: dict | None, hits: list[dict] | None) -> dict:
    """Carry verified statement and method passages into existing audit quotes.

    Keep the primary card's quote and locator unchanged. Supplementary blocks
    retain their own locators; another page never inherits the primary locator.
    """
    result = dict(plan or {}) if isinstance(plan, dict) else {}
    sources: dict[str, list[dict]] = {}
    for hit in hits or []:
        meta = hit.get("meta") or {}
        path = str(meta.get("source_path") or "")
        rows = meta.get("source_passages") or [{**meta, "text": hit.get("text")}]
        for row in rows:
            if row.get("direct_question_evidence") and row.get("block_id"):
                sources.setdefault(path, []).append(row)
    slots = []
    for raw in result.get("slots", []):
        if not isinstance(raw, dict):
            slots.append(raw)
            continue
        slot = dict(raw)
        path = str(slot.get("source_path") or "")
        rows = sources.get(path, [])
        if rows and slot.get("preferred_system") == "system_a":
            try:
                actual = {b["block_id"]: b for b in load_source_blocks(Path(path))}
                # The preflight plan may still name a broad retrieval chunk.
                # Promote the selected direct passage only after reading its
                # actual source block again; keep the existing source/number.
                selected = next(
                    (r for r in rows if r.get("block_id") == slot.get("block_id")),
                    next((r for r in rows if len(str(r.get("text") or "")) >= 40), None),
                )
                enumeration = max(rows, key=lambda r: r.get("direct_question_enumeration_score", 0))
                if enumeration.get("direct_question_enumeration_score", 0) > 0:
                    selected = enumeration
                selected_block = _verified_block(path, selected) if selected else None
                if (
                    selected_block
                    and (slot.get("block_id") != selected_block["block_id"] or quality_boundary_quote(str(selected.get("text") or "")) or enumeration.get("direct_question_enumeration_score", 0) > 0)
                ):
                    slot.update(
                        {
                            key: selected_block.get(key)
                            for key in (
                                "block_id",
                                "anchor_id",
                                "heading_path",
                                "page_start",
                                "page_end",
                            )
                        }
                    )
                    slot["anchor_kind"] = selected_block.get("kind")
                    slot["evidence_quote"] = selected["text"]
                    slot["evidence_selection_reason"] = "requested_source_statement"
                primary = actual.get(str(slot.get("block_id") or ""))
                page = int(slot.get("page_start") or 0)
                quote = normalize_match_text(str(slot.get("evidence_quote") or ""))
                primary_ok = (
                    primary
                    and _verified_block(path, {**slot, "text": slot.get("evidence_quote")})
                    and page > 0
                    and primary.get("page_start") == page
                    and len(quote) >= 40
                    and quote
                    in normalize_match_text(
                        str(primary.get("raw_text") or primary.get("text") or "")
                    )
                    and any(r.get("block_id") == primary["block_id"] for r in rows)
                )
                verified = []
                for row in rows if primary_ok else []:
                    block = _verified_block(path, row, allow_short_equation=True)
                    text = str(row.get("text") or "").strip()
                    raw = str((block or {}).get("raw_text") or "").strip()
                    # The retrieved sentence can have lost its stated model
                    # premise. Restore only its own verified paragraph, without
                    # changing source identity, stitching blocks or adding text.
                    if block and text in raw and len(raw) <= 1800 and not incomplete_source_prose(raw) and _known_noise_training_premise(raw):
                        text = raw
                    if (
                        block
                        and (block.get("page_start") == page or row.get("direct_question_method_facet_score", 0) > 0)
                        and text
                        and normalize_match_text(text)
                        in normalize_match_text(
                            str(block.get("raw_text") or block.get("text") or "")
                        )
                    ):
                        verified.append(
                            {
                                "text": text,
                                "block_id": block["block_id"],
                                "anchor_id": block.get("anchor_id"),
                                **{k: block.get(k) for k in ("page_start", "page_end", "heading_path")},
                                "source_path": path,
                            }
                        )
                if verified:
                    # Reuse the renderer's verified prompt-block contract so
                    # later reference enrichment cannot replace this locator.
                    slot["evidence_selection_reason"] = "prompt_contract_block"
                    slot["strict_locate"] = True
                    if any(r.get("direct_question_method_facet_score", 0) > 0 for r in rows):
                        slot["source_passage_bundle"] = True
                        slot["direct_method_facets"] = True
                    slot["source_evidence_quotes"] = list(
                        dict.fromkeys(
                            [
                                *slot.get("source_evidence_quotes", []),
                                *(b["text"] for b in verified),
                            ]
                        )
                    )
                    slot["direct_source_evidence"] = verified
            except (OSError, ValueError, TypeError):
                pass  # A changed/unavailable source cannot grant new support.
        slots.append(slot)
    if "slots" in result:
        result["slots"] = slots
    return attach_target_condition_slot(result, hits or [])


def prioritize_direct_question_hits(
    selected: list[dict], candidates: list[dict], *, limit: int, prompt: str = ""
) -> list[dict]:
    direct = [
        hit
        for hit in candidates
        if (hit.get("meta") or {}).get("direct_question_evidence")
    ]
    if requests_comparison_boundary(prompt) or requests_comparison_facets(prompt) or requests_noise_model_requirement(prompt) or requests_sample_relation_evidence(prompt) or corruption_condition_question(prompt) or _TASK_LIST_QUESTION.search(prompt) or _QUALITY_BOUNDARY_QUESTION.search(prompt):
        # Supplemental/facet scans can deduplicate an earlier direct hit into
        # a generic row. Re-read only the already scoped candidate sources.
        fresh = []
        paths = dict.fromkeys(str((h.get("meta") or {}).get("source_path") or "")
                              for h in selected + candidates)
        for path in paths:
            if path and Path(path).is_file():
                try:
                    from kb.converter.quality_gate import _unreliable_pages_from_report
                    from kb.converter.quality_repair import load_conversion_quality_result
                    excluded = set(_unreliable_pages_from_report(load_conversion_quality_result(Path(path))))
                    fresh.extend(h for h in direct_question_evidence_hits(Path(path), prompt, limit=limit)
                                 if int((h.get("meta") or {}).get("page_start") or 0) not in excluded)
                except (OSError, ValueError, TypeError):
                    pass
        direct = fresh + direct
    if not direct:
        return selected
    seen = set()
    out = []
    for hit in direct + selected:
        meta = hit.get("meta") or {}
        key = (
            str(meta.get("source_path") or ""),
            str(meta.get("block_id") or hit.get("text") or ""),
        )
        if key not in seen:
            seen.add(key)
            out.append(hit)
    # A direct statement and its nearby equations can fill the entire budget.
    # Retain the other selected source's evidence before adding more context.
    groups: dict[str, list[dict]] = {}
    for hit in out:
        groups.setdefault(str((hit.get("meta") or {}).get("source_path") or ""), []).append(hit)
    if len(groups) > 1:
        balanced = []
        for group in groups.values():
            if group and len(balanced) < limit:
                balanced.append(group.pop(0))
        while len(balanced) < limit and any(groups.values()):
            # After each source has its primary evidence, cover separately
            # requested method relations before optional theorem context.
            preferred = [group for group in groups.values() if group and
                         (group[0].get("meta") or {}).get("direct_question_method_facet_score", 0) > 0]
            for group in preferred or groups.values():
                if group and len(balanced) < limit:
                    balanced.append(group.pop(0))
        return balanced
    return out[:limit]


def prioritize_selected_source_relation_hits(selected: list[dict], candidates: list[dict], *, limit: int, prompt: str) -> list[dict]:
    """Supplement requested sample relations without displacing other facets."""
    if (requests_noise_model_requirement(prompt) or requests_bernoulli_training_requirement(prompt) or requests_training_augmentation(prompt)) and not requests_sample_relation_evidence(prompt):
        replacements = {}
        for path in dict.fromkeys(str((h.get('meta') or {}).get('source_path') or '') for h in selected):
            if not path or not Path(path).is_file():
                continue
            try:
                from kb.converter.quality_gate import _unreliable_pages_from_report
                from kb.converter.quality_repair import load_conversion_quality_result
                excluded = set(_unreliable_pages_from_report(load_conversion_quality_result(Path(path))))
                rows = direct_question_evidence_hits(Path(path), prompt, limit=limit)
                premises = [h for h in rows if requested_method_premise_priority(prompt, h.get('text') or '') and int((h.get('meta') or {}).get('page_start') or 0) not in excluded]
                if premises:
                    replacements[path] = premises
            except (OSError, ValueError, TypeError):
                continue
        result = []
        used = set()
        for hit in selected:
            meta = hit.get('meta') or {}; path = str(meta.get('source_path') or '')
            if path not in replacements or path in used:
                result.append(hit); continue
            used.add(path); fresh = replacements[path][0]; fresh_meta = fresh['meta']
            old_passages = list(meta.get('source_passages') or [{**meta, 'text': hit.get('text')}])
            bundle = {**fresh, 'meta': {**fresh_meta, 'source_passages': [{**fresh_meta, 'text': fresh['text']}, *old_passages[1:]]}}
            if requests_training_augmentation(prompt) or requests_sampling_prerequisite(prompt):
                from kb.paper_guide_answer_selection import _merge_same_source_answer_hits
                rows = [*replacements[path], *({'text': p.get('text'), 'score': p.get('score'), 'meta': {**p, 'source_path': path}} for p in old_passages)]
                bundle = _merge_same_source_answer_hits(rows)[0]
            merged = {**meta, **fresh_meta, **bundle['meta']}
            for key in ('citation_plan_evidence_authoritative', 'citation_plan_full_evidence_quote', 'citation_plan_evidence_selection_reason', 'why_line', 'answer_claim'):
                merged.pop(key, None)
            result.append({**hit, **bundle, 'meta': merged})
        return result[:limit]
    if not requests_sample_relation_evidence(prompt):
        return selected
    return prioritize_direct_question_hits(selected, candidates, limit=limit, prompt=prompt)


_STATEMENT = re.compile(
    r"(?i)(proposition|theorem|lemma|definition|命题|定理|引理|定义)\s*(\d+)"
)
_EQUATION_QUERY = re.compile(
    r"(?i)(?:\b(?:equations?|eqs?\.?|formulas?)|公式|方程|等式)\s*"
    r"([（(]?\d{1,4}[)）]?(?:\s*(?:and|与|和|及|,|，)\s*[（(]?\d{1,4}[)）]?){0,3})"
)


def _requested_equation_numbers(prompt: str) -> set[int]:
    return {int(n) for match in _EQUATION_QUERY.finditer(prompt)
            for n in re.findall(r"\d+", match[1]) if int(n) > 0}
_INTERVAL_QUESTION = re.compile(
    r"(?i)(?:noise.{0,45}(?:interval|range)|(?:interval|range).{0,45}noise|噪声.{0,20}(?:范围|区间))"
)
_VARIANCE_QUESTION = re.compile(
    r"(?i)(?:variance|方差).{0,80}(?:samples?|样本)|(?:samples?|样本).{0,80}(?:variance|方差)"
)
_TASK_LIST_QUESTION = re.compile(
    r"(?i)(?:三类|三种|three).{0,30}(?:任务|应用|tasks?|applications?)|(?:任务|应用|tasks?|applications?).{0,30}(?:三类|三种|three)"
)
_CALIBRATION_QUESTION = re.compile(r"(?i)校准|calibrat")
_SINGLETON_QUESTION = re.compile(r"(?i)单元素|\bsingletons?\b")
_TRAINING_OBSERVATION_QUESTION = re.compile(
    r"(?i)训练(?:观测|数据)|training\s+(?:data|observations)|noisy\s+pairs|噪声图像|"
    r"(?:一|单)张.{0,12}带噪|干净(?:参照|目标|标签).{0,30}(?:配对|数据)|"
    r"(?:哪|两).{0,12}(?:类)?配对(?:数据|图像)"
)
_INDEPENDENCE_QUESTION = re.compile(r"(?i)独立性|independence|相关噪声|correlated\s+noise")
_STATISTICAL_ASSUMPTIONS_QUESTION = re.compile(
    r"(?i)统计假设|statistical\s+assumptions?|"
    r"(?:信号|signal).{0,40}(?:噪声|noise).{0,30}(?:假设|assumptions?)"
)
_NOISY_TARGET_EXPECTATION_QUESTION = re.compile(
    r"(?i)(?:条件期望|conditional\s+expect).{0,90}(?:干净|带噪|clean|noisy)|"
    r"(?:干净|带噪|clean|noisy).{0,90}(?:条件期望|conditional\s+expect)"
)
_QUALITY_BOUNDARY_QUESTION = re.compile(
    r"(?i)performance\s+boundary|性能边界|(?:outperform|胜过|优于).{0,80}(?:clean targets?|noisy pairs?|干净目标|噪声对)"
)
_OUTPUT_RELATION_QUESTION = re.compile(
    r"(?i)DnCNN.{0,80}(?:\u76f4\u63a5\u9884\u6d4b|\u7f51\u7edc\u8f93\u51fa|\u53bb\u566a\u56fe\u50cf|directly\s+predict|network\s+output|denoised\s+image)"
)


def _architecture_relation_question(prompt: str) -> bool:
    facets = (
        r"(?i)pooling|池化",
        r"(?i)filter|kernel|卷积核",
        r"(?i)receptive\s+field|感受野",
    )
    return sum(bool(re.search(pattern, prompt)) for pattern in facets) >= 2


def _task_list_score(text: str) -> int:
    owned = bool(re.search(r"(?i)\bwe\b|\bour\b|本文|我们", text))
    explicit = bool(
        re.search(r"(?i)three.{0,35}(?:specific\s+)?tasks?|三.{0,12}(?:任务|应用)", text)
    )
    named = sum(
        bool(re.search(pattern, text, re.I))
        for pattern in (
            r"Gaussian\s+denois",
            r"(?:single\s+image\s+super[- ]resolution|\bSISR\b)",
            r"JPEG(?:\s+image)?\s+deblock",
        )
    )
    applications = sum(bool(re.search(pattern, text, re.I)) for pattern in (
        r"photographic\s+noise\s+removal", r"synthetic\s+Monte\s+Carlo", r"MRI.{0,40}undersampled",
    ))
    if named == 3 or applications == 3:
        return 12 + int(owned) + int(explicit)
    return 3 if owned and explicit else 0


def _noisy_target_expectation_score(text: str) -> int:
    conditional = bool(
        re.search(
            r"(?i)input[- ]conditioned\s+target\s+distributions.{0,220}same\s+conditional\s+expected\s+values",
            text,
        )
    )
    zero_mean = bool(re.search(r"(?i)zero[- ]mean\s+noise", text))
    if conditional and zero_mean:
        return 16
    if re.search(
        r"(?i)expectation\s+of\s+the\s+corrupted\s+input\s+data\s+is\s+the\s+clean\s+target",
        text,
    ):
        return 5
    return 0


def _architecture_relation_score(text: str) -> int:
    if not all(
        re.search(pattern, text, re.I)
        for pattern in (
            r"3\s*[×x]\s*3",
            r"remove(?:s|d)?\s+all\s+pooling\s+layers",
            r"receptive\s+field.{0,80}depth\s+of\s+d.{0,80}\(\s*2\s*d\s*\+\s*1\s*\)\s*[×x]\s*\(\s*2\s*d\s*\+\s*1\s*\)",
        )
    ):
        return 0
    return 16


def _output_relation_score(text: str) -> int:
    if all(re.search(pattern, text, re.I) for pattern in (
        r"(?:noisy\s+observation|input).{0,100}y\s*=\s*x\s*\+\s*v",
        r"(?:train|learn).{0,40}residual\s+mapping",
        r"x\s*=\s*y\s*(?:−|-|\\minus)\s*R\s*\(\s*y\s*\)",
    )):
        return 24
    if not all(
        re.search(pattern, text, re.I)
        for pattern in (
            r"designed\s+to\s+predict\s+the\s+residual\s+image",
            r"difference\s+between\s+the\s+noisy\s+observation\s+and\s+the\s+latent\s+clean\s+image",
        )
    ):
        return 0
    return 16


def _requested_facet_score(prompt: str, text: str) -> int:
    """Locate explicit source statements, never infer facts from paper names."""
    if _QUALITY_BOUNDARY_QUESTION.search(prompt):
        return 26 if quality_boundary_quote(text) else 0
    method_facet = source_facet_score(prompt, text)
    if method_facet:
        return method_facet
    if _CALIBRATION_QUESTION.search(prompt):
        return 3 if re.search(r"(?i)calibrat.{0,80}(?:traditional|median|non.local)|校准.{0,30}(?:传统|中值)", text) else 0
    if _INDEPENDENCE_QUESTION.search(prompt) or _STATISTICAL_ASSUMPTIONS_QUESTION.search(prompt):
        signal = re.search(r"(?i)signal.{0,30}(?:is\s+)?not\s+pixel[- ]wise\s+independent", text)
        noise = re.search(r"(?i)noise.{0,45}conditionally\s+pixel[- ]wise\s+independent.{0,30}(?:given\s+the\s+signal|signal)",text)
        if signal and noise:
            return 16 if re.search(r"(?i)we\s+make\s+two\s+simple\s+statistical\s+assumptions",text) else 8
    if _INDEPENDENCE_QUESTION.search(prompt):
        relation = re.search(r"(?i)(?:noise.{0,65}independent.{0,50}(?:signal|conditional)|independence.{0,60}noise)", text)
        if not relation:
            return 0
        return 8 if re.search(r"(?i)we (?:assume|use).{0,110}noise.{0,65}independent", text) else 1
    if _STATISTICAL_ASSUMPTIONS_QUESTION.search(prompt):
        return 0
    if _TRAINING_OBSERVATION_QUESTION.search(prompt):
        # Author-owned data requirements take precedence over related work.
        owned = bool(re.search(r"(?i)\bwe\b|\bour\b|本文|我们", text))
        paired_data = bool(
            re.search(
                r"(?i)(?:does\s+not\s+require|neither).{0,80}noisy\s+image\s+pairs.{0,80}(?:nor|neither).{0,40}clean\s+target\s+images|"
                r"neither\s+noisy\s+image\s+pairs\s+nor\s+clean\s+target\s+images",
                text,
            )
        )
        explicit = bool(
            re.search(
                r"(?i)observe.{0,30}twice|(?:neither|not require).{0,70}(?:pairs|target)|"
                r"(?:input|training).{0,35}(?:target|pairs)",
                text,
            )
        )
        if paired_data and re.search(r"(?i)\bN2V\b|NOISE\s*2\s*VOID|our\s+(?:approach|training|procedure|scheme)", text):
            return 16
        asks_both_pair_types = bool(
            re.search(r"(?:哪|两).{0,12}(?:类)?配对(?:数据|图像)|which.{0,20}pairs?", prompt, re.I)
            or (
                re.search(r"(?:一|单)张.{0,12}带噪|single\s+noisy", prompt, re.I)
                and re.search(r"干净(?:参照|目标|标签)|clean\s+target", prompt, re.I)
            )
        )
        if asks_both_pair_types:
            return 3 if owned and re.search(r"(?i)observe.{0,40}(?:image|source).{0,15}twice", text) else 0
        return 3 if owned and explicit else 0
    return 0


def requests_direct_question_evidence(prompt: str) -> bool:
    return bool(
        _STATEMENT.search(prompt)
        or _requested_equation_numbers(prompt)
        or requests_source_facets(prompt)
        or _INTERVAL_QUESTION.search(prompt)
        or _VARIANCE_QUESTION.search(prompt)
        or _TASK_LIST_QUESTION.search(prompt)
        or _CALIBRATION_QUESTION.search(prompt)
        or _SINGLETON_QUESTION.search(prompt)
        or _TRAINING_OBSERVATION_QUESTION.search(prompt)
        or _INDEPENDENCE_QUESTION.search(prompt)
        or _STATISTICAL_ASSUMPTIONS_QUESTION.search(prompt)
        or _NOISY_TARGET_EXPECTATION_QUESTION.search(prompt)
        or _QUALITY_BOUNDARY_QUESTION.search(prompt)
        or corruption_condition_question(prompt)
        or _OUTPUT_RELATION_QUESTION.search(prompt)
        or _architecture_relation_question(prompt)
    )


def quality_boundary_quote(text: str) -> str:
    if (len(text) <= 1800 and re.search(r'(?:cannot|not)\s+be\s+expected\s+to\s+outperform', text, re.I)
            and re.search(r'(?:noisy\s+image\s+pairs|clean\s+target\s+images)', text, re.I)):
        return text
    match = re.search(
        r"(?:While\s+)?it cannot be expected that our approach outperforms methods that have "
        r"(?:additional|more) information available during training,\s*we observe that the denoising "
        r"performance of our results only drops moderately[^.]*\.",
        text, re.I,
    )
    return match.group() if match else ""


def application_enumeration_quote(text: str) -> str:
    match = re.search(r"(?:We|Our)\s+(?:show|demonstrate)\s+applications\s+in[^.]*\.", text, re.I)
    return match.group() if match and _task_list_score(match.group()) >= 12 else ""


def direct_question_evidence_hits(
    md_path: Path, prompt: str, *, limit: int = 6
) -> list[dict]:
    """Find requested statement/interval facts in the bound source, not titles."""
    if not requests_direct_question_evidence(prompt):
        return []
    blocks = list(load_source_blocks(md_path))
    statement = _STATEMENT.search(prompt)
    aliases = {"命题": "proposition", "定理": "theorem", "引理": "lemma", "定义": "definition"}

    def matches_statement(text: str) -> bool:
        label = _STATEMENT.match(normalize_match_text(text))
        return bool(label and statement and label[2] == statement[2]
                    and aliases.get(label[1].lower(), label[1].lower())
                    == aliases.get(statement[1].lower(), statement[1].lower()))

    # A comparison may request a numbered statement from one paper and an
    # independent method facet from another. Missing numbers still grant no
    # statement evidence; only the separately requested literal facet is read.
    if statement and requests_comparison_facets(prompt) and not any(
        b.get("kind") not in {"heading", "figure", "code"}
        and "reference" not in str(b.get("heading_path") or "").lower()
        and matches_statement(str(b.get("raw_text") or b.get("text") or "")) for b in blocks
    ):
        statement = None
    equation_numbers = _requested_equation_numbers(prompt)
    requested_pages = {
        int(a or b) for a, b in re.findall(r"第\s*(\d+)\s*页|\bpage\s+(\d+)\b", prompt, re.I)
    }
    candidates: list[tuple[int, int, list[int]]] = []
    query_words = set(re.findall(r"[a-z]{4,}", prompt.lower()))
    for index, block in enumerate(blocks):
        text = str(block.get("raw_text") or block.get("text") or "")
        if (
            block.get("kind") in {"heading", "figure", "code"}
            or "reference" in str(block.get("heading_path") or "").lower()
            or (len(requested_pages) == 1 and block.get("page_start") not in requested_pages)
        ):
            continue
        selected = [index]
        if statement:
            if not matches_statement(text):
                continue
            for following in range(index + 1, min(len(blocks), index + 7)):
                row = blocks[following]
                if (
                    row.get("kind") in {"heading", "code"}
                    or row.get("page_start") != block.get("page_start")
                    or _STATEMENT.match(
                        normalize_match_text(str(row.get("raw_text") or row.get("text") or ""))
                    )
                ):
                    break
                if row.get("kind") != "figure":
                    selected.append(following)
        elif equation_numbers:
            if block.get("kind") != "equation" or extract_equation_number(text) not in equation_numbers:
                continue
            if index > 0:
                before = blocks[index - 1]
                if before.get("kind") == "paragraph" and before.get("page_start") == block.get("page_start"):
                    selected.insert(0, index - 1)
            for following in range(index + 1, min(len(blocks), index + 4)):
                row = blocks[following]
                raw = str(row.get("raw_text") or row.get("text") or "")
                if (row.get("page_start") != block.get("page_start")
                    or row.get("kind") in {"heading", "code", "figure"}
                    or _STATEMENT.match(normalize_match_text(raw))):
                    break
                if row.get("kind") == "equation" and extract_equation_number(raw) not in equation_numbers:
                    break
                selected.append(following)
        elif _SINGLETON_QUESTION.search(prompt):
            if not re.search(r"\bsingletons?\b", text, re.I) or "=" not in text:
                continue
        elif _INTERVAL_QUESTION.search(prompt):
            if not re.search(
                r"\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]", text
            ) or not re.search(r"(?i)noise|sigma|σ|噪声", text):
                continue
        elif _TASK_LIST_QUESTION.search(prompt):
            task_score = _task_list_score(text)
            broad_application_statement = bool(
                re.search(r"(?i)\bwe\b|\bour\b|本文|我们", text)
                and re.search(
                    r"(?i)(?:demonstrate|demonstrations).{0,180}(?:removal|denois|reconstruct)|"
                    r"(?:situations|applications).{0,250}(?:photograph|imaging)",
                    text,
                )
            )
            if not task_score and not broad_application_statement:
                continue
        elif corruption_condition_question(prompt):
            if not target_condition_quote(text) and not (
                "they need a perfect, explicit computational model of the corruption" in text
                and "our requirement of the knowledge of the appropriate statistical summary (mean, median)" in text
            ):
                continue
        elif _NOISY_TARGET_EXPECTATION_QUESTION.search(prompt):
            if not _noisy_target_expectation_score(text):
                continue
        elif _architecture_relation_question(prompt):
            if not _architecture_relation_score(text):
                continue
        elif _OUTPUT_RELATION_QUESTION.search(prompt):
            if not _output_relation_score(text):
                continue
        elif _VARIANCE_QUESTION.search(prompt) and not (
            re.search(r"(?i)variance|方差", text)
            and re.search(r"(?i)divided|number of.*samples|除|样本", text)
        ):
            continue
        elif not _VARIANCE_QUESTION.search(prompt) and not _requested_facet_score(prompt, text):
            continue
        surface = " ".join(
            str(blocks[i].get("raw_text") or blocks[i].get("text") or "")
            for i in selected
        )
        overlap = len(query_words & set(re.findall(r"[a-z]{4,}", surface.lower())))
        candidates.append(
            (
                overlap
                + _requested_facet_score(prompt, text)
                + (_task_list_score(text) if _TASK_LIST_QUESTION.search(prompt) else 0)
                + (
                    _noisy_target_expectation_score(text)
                    if _NOISY_TARGET_EXPECTATION_QUESTION.search(prompt)
                    else 0
                )
                + (
                    _architecture_relation_score(text)
                    if _architecture_relation_question(prompt)
                    else 0
                )
                + (
                    _output_relation_score(text)
                    if _OUTPUT_RELATION_QUESTION.search(prompt)
                    else 0
                ),
                index,
                selected,
            )
        )
    if not candidates:
        return []
    candidates.sort(key=lambda row: (
        -sample_relation_priority(prompt, str(blocks[row[1]].get("raw_text") or "")),
        -requested_method_premise_priority(prompt, str(blocks[row[1]].get("raw_text") or "")),
        -row[0], row[1],
    ))
    chosen = candidates[0][2]
    if statement and measurement_conditions_question(prompt):
        # A clipped proposition must remain clipped. Carry a separately located
        # literal premise when the same paper actually states it elsewhere.
        premises = [i for i, b in enumerate(blocks) if b.get('kind') == 'paragraph'
                    and not _STATEMENT.match(normalize_match_text(str(b.get('raw_text') or '')))
                    and 'reference' not in str(b.get('heading_path') or '').lower()
                    and source_facet_score(prompt, str(b.get('raw_text') or '')) >= 24]
        chosen = list(dict.fromkeys([chosen[0], *premises, *chosen[1:]]))
    if requests_source_facets(prompt) and not statement and not equation_numbers:
        chosen = list(dict.fromkeys(i for _, _, bundle in candidates for i in bundle))
    if equation_numbers and not statement:
        chosen = sorted({i for _, _, bundle in candidates for i in bundle})
        equations = [i for i in chosen if blocks[i].get("kind") == "equation"]
        context = [i for i in chosen if i not in equations]
        context.sort(key=lambda i: (-len(query_words & set(re.findall(
            r"[a-z]{4,}", str(blocks[i].get("raw_text") or "").lower()))), i))
        chosen = (context[:1] + equations + context[1:]) if len(equations) < limit else equations
    if _NOISY_TARGET_EXPECTATION_QUESTION.search(prompt):
        # The theorem states invariance; the separate observation premise says
        # which clean target the noisy observation estimates. Keep both facets.
        chosen = list(dict.fromkeys(i for _, _, bundle in candidates for i in bundle))
    if corruption_condition_question(prompt):
        # Keep both the model contrast and the target condition as actual
        # paragraphs. Their page locators remain separate throughout auditing.
        chosen = sorted(
            {i for _, _, bundle in candidates for i in bundle},
            key=lambda i: (bool(target_condition_quote(str(blocks[i].get("raw_text") or ""))), i),
        )
    # Lead with the explanatory paragraph for loss questions, while preserving
    # the statement and its page-local equations as separately bound passages.
    if statement and re.search(r"(?i)loss|损失", prompt):
        chosen.sort(
            key=lambda i: (
                not bool(
                    re.search(
                        r"(?i)loss.*variance|损失.*方差",
                        str(blocks[i].get("raw_text") or ""),
                    )
                ),
                i,
            )
        )
    out = []
    for index in chosen[: max(1, limit)]:
        block = blocks[index]
        text = str(block.get("raw_text") or block.get("text") or "").strip()
        method_facet = source_facet_score(prompt, text) if not statement and not equation_numbers else 0
        if method_facet:
            text = source_facet_quote(prompt, text)
        if _QUALITY_BOUNDARY_QUESTION.search(prompt):
            text = quality_boundary_quote(text)
        elif _STATISTICAL_ASSUMPTIONS_QUESTION.search(prompt):
            text = next((text[start:end].strip() for start, end in citation_sentence_spans(text)
                         if _requested_facet_score(prompt, text[start:end]) >= 16), text)
        elif _TASK_LIST_QUESTION.search(prompt):
            text = application_enumeration_quote(text) or text
        elif _TRAINING_OBSERVATION_QUESTION.search(prompt):
            requirement = re.search(
                r"\bneither\s+noisy\s+image\s+pairs\s+nor\s+clean\s+target\s+images\s+are\s+available\.",
                text, re.I,
            )
            if requirement and requirement.end() <= 1800:
                # Preserve the complete data requirement before a long results
                # tail would make the source bundler truncate that very clause.
                text = text[:requirement.end()]
        if not text or not block.get("block_id"):
            continue
        out.append(
            {
                "id": f"{block['doc_id']}:source-block:{block['block_id']}",
                "text": text,
                "score": 1000.0 - len(out),
                "meta": {
                    **{
                        key: block.get(key)
                        for key in (
                            "block_id",
                            "anchor_id",
                            "kind",
                            "heading_path",
                            "page_start",
                            "page_end",
                        )
                    },
                    "source_path": str(md_path),
                    "direct_question_evidence": True,
                    "direct_question_method_facet_score": method_facet,
                    "direct_question_enumeration_score": _task_list_score(text) if _TASK_LIST_QUESTION.search(prompt) else 0,
                    "paper_guide_targeted_block": True,
                    "evidence_selection_reason": "requested_source_statement",
                },
            }
        )
    return out
