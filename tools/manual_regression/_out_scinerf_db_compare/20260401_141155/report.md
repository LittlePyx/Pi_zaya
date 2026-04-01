# Paper Guide Baseline (P0 v1)

- Time: `2026-04-01T14:11:55`
- Base URL: `http://127.0.0.1:8000`
- Suite: `paper_guide_baseline_p0_v1`

## Summary
- total_cases: `3`
- pass_cases: `2`
- failed_cases: `1`
- overall: **FAIL**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| SCINERF2024_CITATION_ADMM | cvpr_2024_scinerf | citation_lookup | FAIL | PASS | FAIL | PASS | n/a | n/a | n/a |
| SCINERF2024_METHOD_IMPL_ITER_BATCH | cvpr_2024_scinerf | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| SCINERF2024_FIG3_CAPTION_COMPONENTS | cvpr_2024_scinerf | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | n/a | PASS |

## Findings
### SCINERF2024_CITATION_ADMM ADMM citation in SCI decoding
- paper: `cvpr_2024_scinerf`
- family: `citation_lookup`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `2`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `FAIL`
- gate_reason[citation]: `unexpected_ref_nums=[35]`
- citation_matched_ref_nums: `[4, 35]`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 2. Related Work']`
- locate_matched_anchors: `['[26,35] pro optimization problems, instead of using conventional gradient descent algorithm, most of the existing methods employ alternating direction method of multipliers (ADMM) [4], posed NeRF, an epochal novel scene representation metho']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `citation:unexpected_ref_nums=[35]`
- answer_preview:

```text
The paper cites [4] for this point.
This is stated in SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 2. Related Work:
> [26,35] pro optimization problems, instead of using conventional gradient descent algorithm, most of the existing methods employ alternating direction method of multipliers (ADMM) [4], posed NeRF, an epochal novel scene representation method.
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
- locate_matched_anchors: `['100-200K iterations, with 5000 rays as batch size.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper states this explicitly in SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup:
> 100-200K iterations, with 5000 rays as batch size.
```

### SCINERF2024_FIG3_CAPTION_COMPONENTS Figure 3 experimental setup caption
- paper: `cvpr_2024_scinerf`
- family: `figure_walkthrough`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 1 T N )) , (4) / Figure 3', 'SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 1 T N )) , (4) / Figure 3']`
- locate_matched_anchors: `['This SCI imaging system contains a CCD camera to record snapshot measurement, primary and rely lens, and a DMD to modulate input frames.', 'This SCI imaging system contains a CCD camera to record snapshot measurement, primary and rely lens, and a DMD to modulate input frames.']`
- gate[jump]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The exact caption sentence for Figure 3 is:

**“Figure 3. Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.”**

The key components listed are:
- CCD camera (to record snapshot measurement),
- Primary and relay lens,
- DMD (Digital Micromirror Device, to modulate input frames).

This sentence appears verbatim in the supporting excerpts.

Caption anchor: **Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and rely lens, and a DMD to modulate input frames.

### Library Figure![CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf Fig. 3](/api/references/asset?path=F%3A%5Cresearch-papers%5C2026%5CJan%5Celse%5Ckb_chat%5Cdb%5CCVPR-2024-SCIN
```
