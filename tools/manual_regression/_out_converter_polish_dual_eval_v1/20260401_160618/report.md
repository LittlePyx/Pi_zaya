# Paper Guide Converter Polish Dual Eval v1

- Time: `2026-04-01T16:06:18`
- Base URL: `http://127.0.0.1:8000`
- Suite: `paper_guide_converter_polish_dual_eval_v1`

## Summary
- total_cases: `8`
- pass_cases: `6`
- failed_cases: `2`
- overall: **FAIL**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| SCINERF2024_V7_CITATION_ADMM | cvpr_2024_scinerf_polish_v10 | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| SCINERF2024_V7_METHOD_IMPL_ITER_BATCH | cvpr_2024_scinerf_polish_v10 | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| SCINERF2024_V7_FIG3_CAPTION_COMPONENTS | cvpr_2024_scinerf_polish_v10 | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | n/a | PASS |
| SCINERF2024_V7_FIG4_SCENES | cvpr_2024_scinerf_polish_v10 | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | n/a | PASS |
| NC2023_POLISH_CITATION_TRANSFORMER_FRAMEWORK | natcommun_2023_spad_sr_polish_v5 | citation_lookup | FAIL | PASS | FAIL | PASS | n/a | n/a | n/a |
| NC2023_POLISH_CITATION_PASCAL_VOC | natcommun_2023_spad_sr_polish_v5 | citation_lookup | FAIL | FAIL | FAIL | FAIL | n/a | n/a | n/a |
| NC2023_POLISH_METHOD_NETWORK_TRAINING | natcommun_2023_spad_sr_polish_v5 | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| NC2023_POLISH_FIGURE6_PANEL_B_CLAUSE | natcommun_2023_spad_sr_polish_v5 | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | PASS | PASS |

## Findings
### SCINERF2024_V7_CITATION_ADMM ADMM citation in SCI decoding
- paper: `cvpr_2024_scinerf_polish_v10`
- family: `citation_lookup`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
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

### SCINERF2024_V7_METHOD_IMPL_ITER_BATCH Implementation details: iterations and batch size (rays)
- paper: `cvpr_2024_scinerf_polish_v10`
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

### SCINERF2024_V7_FIG3_CAPTION_COMPONENTS Figure 3 experimental setup caption
- paper: `cvpr_2024_scinerf_polish_v10`
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
- CCD camera (to record snapshot measurement),
- Primary and relay lens,
- DMD (to modulate input frames).

This is directly supported by the caption in the supporting excerpts.

Caption anchor: **Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.

### Library Figure![output.pdf Fig. 3](/api/references/asset?path=F%3A%5Cresearch-papers%5C2026%5CJan%5Celse%5Ckb_chat%5Ctmp%5Creconvert_scinerf_text_polish_v10%5Cassets%5Cpage_5_fig_1.png)

*Source: output.pdf, Fig. 3. Figur
```

### SCINERF2024_V7_FIG4_SCENES Figure 4 synthetic scenes caption
- paper: `cvpr_2024_scinerf_polish_v10`
- family: `figure_walkthrough`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.2. Additional Study / Figure 4', 'SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.2. Additional Study / Figure 4', 'SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.2. Additional Study / Figure 4']`
- locate_matched_anchors: `['**Figure 4.** Qualitative evaluations of our method against SOTA SCI image restoration methods on the synthetic dataset. Top to bottom shows the results for different scenes, including Cozy2room, Tanabata, Factory and Vender. The experiment', 'Top to bottom shows the results for different scenes, including Cozy2room, Tanabata, Factory and Vender.', '**Figure 4.** Qualitative evaluations of our method against SOTA SCI image restoration methods on the synthetic dataset. Top to bottom shows the results for different scenes, including Cozy2room, Tanabata, Factory and Vender. The experiment']`
- gate[jump]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The authors state in the caption of Figure 4 that, from top to bottom, the scenes shown are: **Cozy2room, Tanabata, Factory, and Vender**.

Exact supporting sentence:
**“Top to bottom shows the results for different scenes, including Cozy2room, Tanabata, Factory and Vender.”**

Caption anchor: **Figure 4.** Qualitative evaluations of our method against SOTA SCI image restoration methods on the synthetic dataset. Top to bottom shows the results for different scenes, including Cozy2room, Tanabata, Factory and Vender. The experimental results demonstrate that our method achieves superior performance on image...

### Library Figure![output.pdf Fig. 4](/api/references/asset?path=F%3A%5Cresearch-papers%5C2026%5CJan%5Celse%5Ckb_chat%5Ctmp%5Creconvert_scinerf_text_polish_v10%5Cassets%5Cpage_7_fig_1.png)

*Source: output.pdf, Fig. 4. Figure 4*
```

### NC2023_POLISH_CITATION_TRANSFORMER_FRAMEWORK Transformer framework citations
- paper: `natcommun_2023_spad_sr_polish_v5`
- family: `citation_lookup`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `FAIL`
- gate_reason[citation]: `missing_citation_ref_num`
- gate_reason[citation]: `expected_ref_nums_missing=[33, 34]`
- gate[locate]: `PASS`
- locate_matched_headings: `['High-resolution single-photon imaging with physics-informed deep learning / Abstract']`
- locate_matched_anchors: `['The transformer framework [33,34] has recently attracted increasing attention and produced an impressive performance on multiple vision tasks [34-36] .']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `citation:missing_citation_ref_num`
- reason: `citation:expected_ref_nums_missing=[33, 34]`
- answer_preview:

```text
The paper cites [33], [34] for this point.
This is stated in High-resolution single-photon imaging with physics-informed deep learning / Abstract:
> The transformer framework [33,34] has recently attracted increasing attention and produced an impressive performance on multiple vision tasks [34-36].
```

### NC2023_POLISH_CITATION_PASCAL_VOC PASCAL VOC dataset citations
- paper: `natcommun_2023_spad_sr_polish_v5`
- family: `citation_lookup`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `FAIL`
- gate_reason[answer]: `answer_missing_contains_any`
- gate[citation]: `FAIL`
- gate_reason[citation]: `expected_ref_nums_missing=[31, 32]`
- gate_reason[citation]: `unexpected_ref_nums=[24]`
- citation_matched_ref_nums: `[24]`
- gate[locate]: `FAIL`
- gate_reason[locate]: `locate_missing_matching_segment`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `answer:answer_missing_contains_any`
- reason: `citation:expected_ref_nums_missing=[31, 32]`
- reason: `citation:unexpected_ref_nums=[24]`
- reason: `locate:locate_missing_matching_segment`
- answer_preview:

```text
The paper cites [24] for this point.
This is stated in High-resolution single-photon imaging with physics-informed deep learning / Abstract:
> One way to acquire high-resolution images is placing another CMOS or CCD camera to take images of the same target, which however introduces additional labour-some registration workload [24].
```

### NC2023_POLISH_METHOD_NETWORK_TRAINING Network training exact details
- paper: `natcommun_2023_spad_sr_polish_v5`
- family: `method`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['High-resolution single-photon imaging with physics-informed deep learning / Methods / Network training']`
- locate_matched_anchors: `['We implemented the network on Ubuntu20 operating system using the Pytorch framework, and trained 1000 epochs until convergence using Adam optimization on NVIDIA RTX3090 with the batch size set to 24. We set the initial learning rate as 0.00']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper states this explicitly in High-resolution single-photon imaging with physics-informed deep learning / Methods / Network training:
> We implemented the network on Ubuntu20 operating system using the Pytorch framework, and trained 1000 epochs until convergence using Adam optimization on NVIDIA RTX3090 with the batch size set to 24. We set the initial learning rate as 0.0003, which was decreased by 10% for every 100 epochs.
```

### NC2023_POLISH_FIGURE6_PANEL_B_CLAUSE Figure 6 panel (b) caption clause jump
- paper: `natcommun_2023_spad_sr_polish_v5`
- family: `figure_walkthrough`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['High-resolution single-photon imaging with physics-informed deep learning / Methods / Figure 6']`
- locate_matched_anchors: `['b The enhancement comparison between CNN-based U-net network and the reported transformer-based network. Benefiting from the transformer structure, our network realizes spatial-variable convolution that helps pay more attention to the regio']`
- gate[figure_panel]: `PASS`
- figure_panel_matched_headings: `['High-resolution single-photon imaging with physics-informed deep learning / Methods / Figure 6']`
- figure_panel_matched_anchors: `['b The enhancement comparison between CNN-based U-net network and the reported transformer-based network. Benefiting from the transformer structure, our network realizes spatial-variable convolution that helps pay more attention to the regio']`
- gate[jump]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
Figure 6 caption for panel (b) states:
Section: High-resolution single-photon imaging with physics-informed deep learning / Methods
> b The enhancement comparison between CNN-based U-net network and the reported transformer-based network. Benefiting from the transformer structure, our network realizes spatial-variable convolution that helps pay more attention to the regions of interests (ROI) and fine details, as the attention maps on the right side validate. following advantages: 1) The content-based interactions between image content and attention weights can be interpreted as spatially varying convolution, and the shift window mechanism can perform long-distance dependency modeling. 2) The gated fusion mechanism can deeply dig medium-frequency and high-frequency information at different levels, enabling the prevention of losing long-term memory as the network deepens and enhancement o
```
