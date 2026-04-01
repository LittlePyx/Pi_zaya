# Paper Guide Smoke (SCINeRF CVPR 2024 v1)

- Time: `2026-03-26T15:20:11`
- Base URL: `http://127.0.0.1:8015`
- Suite: `paper_guide_smoke_scinerf2024_v1`

## Summary
- total_cases: `6`
- pass_cases: `6`
- failed_cases: `0`
- overall: **PASS**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| SCINERF2024_ABSTRACT_SUMMARY_ZH | cvpr_2024_scinerf | abstract | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| SCINERF2024_CITATION_SCI_OVERVIEW | cvpr_2024_scinerf | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| SCINERF2024_CITATION_ADMM | cvpr_2024_scinerf | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| SCINERF2024_METHOD_IMPL_ITER_BATCH | cvpr_2024_scinerf | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| SCINERF2024_CITATION_SPLINE | cvpr_2024_scinerf | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| SCINERF2024_FIG3_CAPTION_COMPONENTS_ZH | cvpr_2024_scinerf | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | n/a | PASS |

## Findings
### SCINERF2024_ABSTRACT_SUMMARY_ZH Abstract summary with exact anchor (Chinese)
- paper: `cvpr_2024_scinerf`
- family: `abstract`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / Abstract']`
- locate_matched_anchors: `['Drawing inspiration from Compressed Sensing (CS) $^{[5,8]}$, video Snapshot Compressive Imaging (SCI) $^{[50]}$ system has emerged to address these limitations. A conventional video SCI system usually contains a hardware encoder and a softw']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
摘要原文:
Drawing inspiration from Compressed Sensing (CS) $^{[5,8]}$, video Snapshot Compressive Imaging (SCI) $^{[50]}$ system has emerged to address these limitations. A conventional video SCI system usually contains a hardware encoder and a software decoder. The hardware encoder employs a series of specially designed 2D masks to modulate the incoming photons across exposure time into a single compressed image. It enriches the low-cost cameras the ability to capture high-speed scenes, which further reduces the storage requirement. The whole encoding process can also be achieved via software implementation on pre-captured images, which can reduce the storage/transmission requirement and offer additional privacy protection. On the other hand, the software decoder restores the high frame-rate images using the compressed measurement and corresponding binary masks.

In recent years, several im
```

### SCINERF2024_CITATION_SCI_OVERVIEW SCI overview citation
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

### SCINERF2024_CITATION_ADMM ADMM citation in SCI decoding
- paper: `cvpr_2024_scinerf`
- family: `citation_lookup`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `1`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `PASS`
- citation_matched_ref_nums: `[4]`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 2. Related Work']`
- locate_matched_anchors: `['most of the existing methods employ alternating direction method of multipliers (ADMM) $^{[4]}$, which leads to good results and easier to adapt to different systems.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper cites [4] for this point.
This is stated in SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 2. Related Work:
> most of the existing methods employ alternating direction method of multipliers (ADMM) $^{[4]}$, which leads to good results and easier to adapt to different systems.
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
The paper states this explicitly in SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup:
> We train our model for 100–200K iterations, with 5000 rays as batch size.
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
