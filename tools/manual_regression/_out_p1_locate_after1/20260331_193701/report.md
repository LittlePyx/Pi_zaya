# Paper Guide Baseline (P0 v1)

- Time: `2026-03-31T19:37:01`
- Base URL: `http://127.0.0.1:8000`
- Suite: `paper_guide_baseline_p0_v1`

## Summary
- total_cases: `5`
- pass_cases: `3`
- failed_cases: `2`
- overall: **FAIL**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| NP2019_FIGURE3_PANEL_F | natphoton_2019_spi | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | PASS | PASS |
| LSA_METHOD_REAPPLY | lsa_2026 | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| NC2023_FIGURE6_PANEL_B_CLAUSE | natcommun_2023_spad_sr | figure_walkthrough | FAIL | PASS | n/a | PASS | n/a | FAIL | PASS |
| NC2021_CITATION_DYNAMIC_SUPERSAMPLING | natcommun_2021_sph | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| NC2021_METHOD_BEAT_FREQUENCY | natcommun_2021_sph | method | FAIL | PASS | n/a | FAIL | n/a | n/a | n/a |

## Findings
### NP2019_FIGURE3_PANEL_F Figure 3 panel f
- paper: `natphoton_2019_spi`
- family: `figure_walkthrough`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `1`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['Principles and prospects for single-pixel imaging / Applications and future potential for single-pixel imaging / Figure 3']`
- locate_matched_anchors: `['(f) methane imaging using SPC$^{15}$;']`
- gate[figure_panel]: `PASS`
- figure_panel_matched_headings: `['Principles and prospects for single-pixel imaging / Applications and future potential for single-pixel imaging / Figure 3']`
- figure_panel_matched_anchors: `['(f) methane imaging using SPC$^{15}$;']`
- gate[jump]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
Figure 3 caption (panel f) states:
Section: Principles and prospects for single-pixel imaging / Applications and future potential for single-pixel imaging
> (f) methane imaging using SPC$^{15}$;
References in this clause: [15]

### Library Figure

![NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf Fig. 3](/api/references/asset?path=F%3A%5Cresearch-papers%5C2026%5CJan%5Celse%5Ckb_chat%5Cdb%5CNatPhoton-2019-Principles%20and%20prospects%20for%20single-pixel%20imaging%5Cassets%5Cpage_5_fig_2.png)

*Source: NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf, Fig. 3. Fig. 3*
```

### LSA_METHOD_REAPPLY APR exact support
- paper: `lsa_2026`
- family: `method`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['Abstract / Results / iISM with adaptive pixel-reassignment (APR)']`
- locate_matched_anchors: `['Finally, these RVT-APR shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper states this explicitly in Abstract / Results / iISM with adaptive pixel-reassignment (APR):
> Finally, these RVT-APR shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution.
```

### NC2023_FIGURE6_PANEL_B_CLAUSE Figure 6 panel (b) caption clause jump
- paper: `natcommun_2023_spad_sr`
- family: `figure_walkthrough`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['Workflow and structure of the reported deep transformer network / Figure 6']`
- locate_matched_anchors: `['b The enhancement comparison between CNN-based U-net network and the reported transformer-based network. Benefiting from the transformer structure, our network realizes spatial-variable convolution that helps pay more attention to the regio']`
- gate[figure_panel]: `FAIL`
- gate_reason[figure_panel]: `answer_missing_panel_b`
- figure_panel_matched_headings: `['Workflow and structure of the reported deep transformer network / Figure 6']`
- figure_panel_matched_anchors: `['b The enhancement comparison between CNN-based U-net network and the reported transformer-based network. Benefiting from the transformer structure, our network realizes spatial-variable convolution that helps pay more attention to the regio']`
- gate[jump]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `figure_panel:answer_missing_panel_b`
- answer_preview:

```text
Figure 6 caption (panel b) states:
Section: Workflow and structure of the reported deep transformer network
> b The enhancement comparison between CNN-based U-net network and the reported transformer-based network. Benefiting from the transformer structure, our network realizes spatial-variable convolution that helps pay more attention to the regions of interests (ROI) and fine details, as the attention maps on the right side validate.
```

### NC2021_CITATION_DYNAMIC_SUPERSAMPLING Dynamic supersampling citation lookup
- paper: `natcommun_2021_sph`
- family: `citation_lookup`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `1`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `PASS`
- citation_matched_ref_nums: `[31]`
- gate[locate]: `PASS`
- locate_matched_headings: `['ARTICLE / Imaging biological tissue with high-throughput single-pixel compressive holography']`
- locate_matched_anchors: `['Recently, adaptive and smart sensing with dynamic supersampling was reported to combine with compressive sensing in SPI. The enabling feature of this approach is to rapidly record fast-changing features by dynamically adapting to the evolut']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper cites [31] for this point.
This is stated in ARTICLE / Imaging biological tissue with high-throughput single-pixel compressive holography:
> Recently, adaptive and smart sensing with dynamic supersampling was reported to combine with compressive sensing in SPI. The enabling feature of this approach is to rapidly record fast-changing features by dynamically adapting to the evolution of the scene. Thus, it significantly shortens acquisition time without considerably sacrificing spatial information$^{31}$.
```

### NC2021_METHOD_BEAT_FREQUENCY Experimental setup: beat frequency and DAC rate
- paper: `natcommun_2021_sph`
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
The paper states this explicitly in ARTICLE / Methods / Principle of high-throughput SPH:
> Thus, the beat frequency of these two beams is 62,500 Hz, indicating a temporal period of 16 µs. Using a lens with 150-mm focal length, the combined light was collected by a photodetector (DET10A2, Thorlabs) with a bandwidth of 350 MHz, which was then digitized by a data acquisition card (DAC, USB-6251, National Instrument) with a sampling rate of 1.25 Ms/s (not shown in the figure).
```
