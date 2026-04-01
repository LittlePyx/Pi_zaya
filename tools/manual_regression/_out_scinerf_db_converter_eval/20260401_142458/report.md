# Paper Guide SCINeRF DB Converter Eval

- Time: `2026-04-01T14:24:58`
- Base URL: `http://127.0.0.1:8000`
- Suite: `paper_guide_scinerf_db_converter_eval`

## Summary
- total_cases: `4`
- pass_cases: `4`
- failed_cases: `0`
- overall: **PASS**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| SCINERF2024_DB_CITATION_ADMM | cvpr_2024_scinerf_db | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| SCINERF2024_DB_METHOD_IMPL_ITER_BATCH | cvpr_2024_scinerf_db | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| SCINERF2024_DB_FIG3_CAPTION_COMPONENTS | cvpr_2024_scinerf_db | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | n/a | PASS |
| SCINERF2024_DB_FIG4_SCENES | cvpr_2024_scinerf_db | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | n/a | PASS |

## Findings
### SCINERF2024_DB_CITATION_ADMM ADMM citation in SCI decoding
- paper: `cvpr_2024_scinerf_db`
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
- locate_matched_anchors: `['When solving the optimization problems, instead of using conventional gradient descent algorithm, most of the existing methods employ alternating direction method of multipliers (ADMM) [4], which leads to good results and easier to adapt to']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper cites [4] for this point.
This is stated in SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 2. Related Work:
> When solving the optimization problems, instead of using conventional gradient descent algorithm, most of the existing methods employ alternating direction method of multipliers (ADMM) [4], which leads to good results and easier to adapt to different systems.
```

### SCINERF2024_DB_METHOD_IMPL_ITER_BATCH Implementation details: iterations and batch size (rays)
- paper: `cvpr_2024_scinerf_db`
- family: `method`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup / Implementation details']`
- locate_matched_anchors: `['We train our model for 100–200K iterations, with 5000 rays as batch size.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper states this explicitly in SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup / Implementation details:
> We train our model for 100–200K iterations, with 5000 rays as batch size.
```

### SCINERF2024_DB_FIG3_CAPTION_COMPONENTS Figure 3 experimental setup caption
- paper: `cvpr_2024_scinerf_db`
- family: `figure_walkthrough`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup / Figure 3', 'SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup / Figure 3', 'SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup / Figure 3']`
- locate_matched_anchors: `['**Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.', 'This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.', 'This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.']`
- gate[jump]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The exact caption sentence for Figure 3 is:

**“Figure 3. Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.”**

The key components listed are:
- CCD camera (to record snapshot measurement)
- Primary lens
- Relay lens
- DMD (to modulate input frames)

This sentence is directly supported by.

Caption anchor: **Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.

### Library Figure![CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf Fig. 3](/api/references/asset?path=F%3A%5Cresearch-papers%5C2026%5CJan%5Celse%5Ckb_chat%5Cdb%5CCVPR-2024-SCINeRF-%20Neural%20Radiance%20Fields%20from%20a%2
```

### SCINERF2024_DB_FIG4_SCENES Figure 4 synthetic scenes caption
- paper: `cvpr_2024_scinerf_db`
- family: `figure_walkthrough`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.2. Additional Study / Figure 4', 'SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.2. Additional Study / Figure 4']`
- locate_matched_anchors: `['**Figure 4.** Qualitative evaluations of our method against SOTA SCI image restoration methods on the synthetic dataset. Top to bottom shows the results for different scenes, including Cozy2room , Tanabata , Factory and Vender . The experim', '**Figure 4.** Qualitative evaluations of our method against SOTA SCI image restoration methods on the synthetic dataset. Top to bottom shows the results for different scenes, including Cozy2room , Tanabata , Factory and Vender . The experim']`
- gate[jump]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The authors state in the caption of Figure 4 that “Top to bottom shows the results for different scenes, including *Cozy2room*, *Tanabata*, *Factory* and *Vender*.”
This exact sentence appears in the supporting excerpts, and the supporting excerpts.

Caption anchor: **Figure 4.** Qualitative evaluations of our method against SOTA SCI image restoration methods on the synthetic dataset. Top to bottom shows the results for different scenes, including Cozy2room, Tanabata, Factory and Vender. The experimental results demonstrate that our method achieves superior performance on im...

### Library Figure![CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf Fig. 4](/api/references/asset?path=F%3A%5Cresearch-papers%5C2026%5CJan%5Celse%5Ckb_chat%5Cdb%5CCVPR-2024-SCINeRF-%20Neural%20Radiance%20Fields%20from%20a%20Snapshot%20Compressive%20Image%5Cassets%5Cpage_7_fig_1.png
```
