# Paper Guide Smoke (SCINeRF CVPR 2024 Natural v1)

- Time: `2026-03-26T15:26:42`
- Base URL: `http://127.0.0.1:8015`
- Suite: `paper_guide_smoke_scinerf2024_natural_v1`

## Summary
- total_cases: `6`
- pass_cases: `4`
- failed_cases: `2`
- overall: **FAIL**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| NATURAL_SCINERF_CITE_SCI | cvpr_2024_scinerf | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| NATURAL_SCINERF_CITE_ADMM | cvpr_2024_scinerf | citation_lookup | FAIL | PASS | FAIL | PASS | n/a | n/a | n/a |
| NATURAL_SCINERF_CITE_COLMAP | cvpr_2024_scinerf | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| NATURAL_SCINERF_METHOD_IMPL | cvpr_2024_scinerf | method | FAIL | PASS | n/a | FAIL | n/a | n/a | n/a |
| NATURAL_SCINERF_CITE_SPLINE | cvpr_2024_scinerf | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| NATURAL_SCINERF_FIG3_SETUP | cvpr_2024_scinerf | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | n/a | PASS |

## Findings
### NATURAL_SCINERF_CITE_SCI SCI cited ref number
- paper: `cvpr_2024_scinerf`
- family: `citation_lookup`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `1`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `PASS`
- citation_matched_ref_nums: `[50]`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / Abstract']`
- locate_matched_anchors: `['Drawing inspiration from Compressed Sensing (CS) $^{[5,8]}$, video Snapshot Compressive Imaging (SCI) $^{[50]}$ system has emerged to address these limitations.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper cites [50] for this point.
This is stated in SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / Abstract:
> Drawing inspiration from Compressed Sensing (CS) $^{[5,8]}$, video Snapshot Compressive Imaging (SCI) $^{[50]}$ system has emerged to address these limitations.
```

### NATURAL_SCINERF_CITE_ADMM ADMM cited ref number
- paper: `cvpr_2024_scinerf`
- family: `citation_lookup`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `FAIL`
- gate_reason[citation]: `missing_citation_ref_num`
- gate_reason[citation]: `expected_ref_nums_missing=[4]`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 2. Related Work']`
- locate_matched_anchors: `['ADMM-Net $^{[21]}$ modeled the decoding process as a tensor rec']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `citation:missing_citation_ref_num`
- reason: `citation:expected_ref_nums_missing=[4]`
- answer_preview:

```text
The paper cites [21] for this point.
This is stated in SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 2. Related Work:
> ADMM-Net $^{[21]}$ modeled the decoding process as a tensor rec
```

### NATURAL_SCINERF_CITE_COLMAP COLMAP cited ref number
- paper: `cvpr_2024_scinerf`
- family: `citation_lookup`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `1`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `PASS`
- citation_matched_ref_nums: `[32]`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 1. Introduction']`
- locate_matched_anchors: `['Since it is impossible to recover the camera poses from a single compressed image via COLMAP $^{[32]}$, we conduct a joint optimization on both the camera poses and NeRF, via minimizing the difference between the synthesized compressed imag']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper cites [32] for this point.
This is stated in SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 1. Introduction:
> Since it is impossible to recover the camera poses from a single compressed image via COLMAP $^{[32]}$, we conduct a joint optimization on both the camera poses and NeRF, via minimizing the difference between the synthesized compressed image and real measurement from the encoder.
```

### NATURAL_SCINERF_METHOD_IMPL Implementation details (iterations and rays batch size)
- paper: `cvpr_2024_scinerf`
- family: `method`
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
训练迭代次数和 batch size（按 rays 设置）的具体数值在提供的检索内容中**未明确说明**。当前片段仅包含论文的摘要与引言部分，未涉及实验设置或实现细节（如优化超参数、迭代轮数、ray batch size 等）。

如需该信息，建议查阅论文的 **“Implementation Details”** 或 **“Experiments”** 章节（通常在附录或正文后半部分）。
```

### NATURAL_SCINERF_CITE_SPLINE Higher-order spline cited ref number
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
- locate_matched_anchors: `['For more complex motions, we can exploit higher-order spline $^{[17]}$ or directly optimize individual poses without the loss of generality.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper cites [17] for this point.
This is stated in SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 3. Method / 3.3. Proposed Framework:
> For more complex motions, we can exploit higher-order spline $^{[17]}$ or directly optimize individual poses without the loss of generality.
```

### NATURAL_SCINERF_FIG3_SETUP Figure 3: setup components
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
Figure 3 的实验装置主要由以下关键部件组成：

- CCD 相机（用于记录快照压缩测量）
- 主镜头（primary lens）与中继镜头（relay lens）
- 数字微镜器件（DMD，用于调制输入帧）

该描述直接来自图 3 的图注：*“This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.”*

Caption anchor: *Figure 3. Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.*
```
