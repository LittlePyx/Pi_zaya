# Paper Guide Smoke (SCINeRF CVPR 2024 v1)

- Time: `2026-03-26T14:27:26`
- Base URL: `http://127.0.0.1:8005`
- Suite: `paper_guide_smoke_scinerf2024_v1`

## Summary
- total_cases: `6`
- pass_cases: `3`
- failed_cases: `3`
- overall: **FAIL**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| SCINERF2024_ABSTRACT_SUMMARY_ZH | cvpr_2024_scinerf | abstract | FAIL | PASS | n/a | FAIL | n/a | n/a | n/a |
| SCINERF2024_CITATION_SCI_OVERVIEW | cvpr_2024_scinerf | citation_lookup | FAIL | FAIL | FAIL | FAIL | n/a | n/a | n/a |
| SCINERF2024_CITATION_ADMM | cvpr_2024_scinerf | citation_lookup | FAIL | FAIL | FAIL | FAIL | n/a | n/a | n/a |
| SCINERF2024_METHOD_IMPL_ITER_BATCH | cvpr_2024_scinerf | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| SCINERF2024_CITATION_SPLINE | cvpr_2024_scinerf | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| SCINERF2024_FIG3_CAPTION_COMPONENTS_ZH | cvpr_2024_scinerf | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | n/a | PASS |

## Findings
### SCINERF2024_ABSTRACT_SUMMARY_ZH Abstract summary with exact anchor (Chinese)
- paper: `cvpr_2024_scinerf`
- family: `abstract`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `FAIL`
- gate_reason[locate]: `locate_missing_matching_segment`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `locate:locate_missing_matching_segment`
- answer_preview:

```text
摘要原文:
*Figure 1.** Given a single snapshot compressed image, our method is able to recover the underlying 3D scene representation. Leveraging the strong novel-view image synthesis capabilities of NeRF, we can render multi-view consistent images in high quality from *the single measurement*.*

中文翻译:
**图1。** 给定单帧压缩图像，我们的方法能够恢复其底层的3D场景表示。借助NeRF强大的新视角图像合成能力，我们可仅凭*单次测量*高质量地渲染出多视角一致的图像。
```

### SCINERF2024_CITATION_SCI_OVERVIEW SCI overview citation
- paper: `cvpr_2024_scinerf`
- family: `citation_lookup`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `FAIL`
- gate_reason[answer]: `answer_missing_contains_any`
- gate[citation]: `FAIL`
- gate_reason[citation]: `missing_citation_ref_num`
- gate_reason[citation]: `expected_ref_nums_missing=[50]`
- gate[locate]: `FAIL`
- gate_reason[locate]: `locate_missing_matching_segment`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `answer:answer_missing_contains_any`
- reason: `citation:missing_citation_ref_num`
- reason: `citation:expected_ref_nums_missing=[50]`
- reason: `locate:locate_missing_matching_segment`
- answer_preview:

```text
Abstract text:
*Figure 1.** Given a single snapshot compressed image, our method is able to recover the underlying 3D scene representation. Leveraging the strong novel-view image synthesis capabilities of NeRF, we can render multi-view consistent images in high quality from *the single measurement*.*
```

### SCINERF2024_CITATION_ADMM ADMM citation in SCI decoding
- paper: `cvpr_2024_scinerf`
- family: `citation_lookup`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `2`
- raw_structured_cite_count: `0`
- gate[answer]: `FAIL`
- gate_reason[answer]: `answer_missing_contains_any`
- gate[citation]: `FAIL`
- gate_reason[citation]: `expected_ref_nums_missing=[4]`
- gate_reason[citation]: `unexpected_ref_nums=[47, 49]`
- citation_matched_ref_nums: `[47, 49]`
- gate[locate]: `FAIL`
- gate_reason[locate]: `locate_missing_matching_segment`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `answer:answer_missing_contains_any`
- reason: `citation:expected_ref_nums_missing=[4]`
- reason: `citation:unexpected_ref_nums=[47, 49]`
- reason: `locate:locate_missing_matching_segment`
- answer_preview:

```text
The paper cites [18], [20], [47], [49] for this point.
This is stated in SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 2. Related Work:
> Early SCI image decoding/reconstruction methods focus on regularized optimization-based approach $^{[18,20,47,49]}$.
```

### SCINERF2024_METHOD_IMPL_ITER_BATCH Implementation details: iterations and batch size (rays)
- paper: `cvpr_2024_scinerf`
- family: `method`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup']`
- locate_matched_anchors: `['We train our model for 100–200K iterations, with 5000 rays as batch size.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
They train for **100–200K iterations** with a **batch size of 5000 rays**.

The exact supporting sentence is:

> “We train our model for 100–200K iterations, with 5000 rays as batch size.”
> —
```

### SCINERF2024_CITATION_SPLINE Higher-order spline citation for complex motion
- paper: `cvpr_2024_scinerf`
- family: `citation_lookup`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `1`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `PASS`
- citation_matched_ref_nums: `[17]`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 3. Method / 3.3. Proposed Framework']`
- locate_matched_anchors: `['For more complex motions, we can exploit higher-order spline $^{ }$ or directly optimize individual poses without the loss of generality.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The exact supporting sentence is:

> “For more complex motions, we can exploit higher-order spline $^{[17]}$ or directly optimize individual poses without the loss of generality.”

This appears in Section 3.3 *Proposed Framework* of the paper, and the cited reference is **[17]**.
```

### SCINERF2024_FIG3_CAPTION_COMPONENTS_ZH Figure 3 caption clause jump (Chinese)
- paper: `cvpr_2024_scinerf`
- family: `figure_walkthrough`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup / Figure 3']`
- locate_matched_anchors: `['This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.']`
- gate[jump]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
Figure 3 的实验装置包含以下关键部件（依据图注原文）：

- CCD camera（用于记录快照测量）
- primary and relay lens（主透镜与中继透镜）
- DMD（用于调制输入帧）

对应图注中的完整句子为：
**“This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.”**

该句即 *Figure 3* 的 caption 全文，可直接定位至 Figure 3 的图注。

Caption anchor: *Figure 3. Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.*
```
