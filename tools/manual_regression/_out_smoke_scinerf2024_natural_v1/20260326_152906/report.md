# Paper Guide Smoke (SCINeRF CVPR 2024 Natural v1)

- Time: `2026-03-26T15:29:06`
- Base URL: `http://127.0.0.1:8015`
- Suite: `paper_guide_smoke_scinerf2024_natural_v1`

## Summary
- total_cases: `6`
- pass_cases: `6`
- failed_cases: `0`
- overall: **PASS**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| NATURAL_SCINERF_CITE_SCI | cvpr_2024_scinerf | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| NATURAL_SCINERF_CITE_ADMM | cvpr_2024_scinerf | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| NATURAL_SCINERF_CITE_COLMAP | cvpr_2024_scinerf | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| NATURAL_SCINERF_METHOD_IMPL | cvpr_2024_scinerf | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
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
