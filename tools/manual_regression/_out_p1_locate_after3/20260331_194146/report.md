# Paper Guide Baseline (P0 v1)

- Time: `2026-03-31T19:41:46`
- Base URL: `http://127.0.0.1:8000`
- Suite: `paper_guide_baseline_p0_v1`

## Summary
- total_cases: `1`
- pass_cases: `1`
- failed_cases: `0`
- overall: **PASS**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| NC2023_FIGURE6_PANEL_B_CLAUSE | natcommun_2023_spad_sr | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | PASS | PASS |

## Findings
### NC2023_FIGURE6_PANEL_B_CLAUSE Figure 6 panel (b) caption clause jump
- paper: `natcommun_2023_spad_sr`
- family: `figure_walkthrough`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['Workflow and structure of the reported deep transformer network / Figure 6']`
- locate_matched_anchors: `['b The enhancement comparison between CNN-based U-net network and the reported transformer-based network. Benefiting from the transformer structure, our network realizes spatial-variable convolution that helps pay more attention to the regio']`
- gate[figure_panel]: `PASS`
- figure_panel_matched_headings: `['Workflow and structure of the reported deep transformer network / Figure 6']`
- figure_panel_matched_anchors: `['b The enhancement comparison between CNN-based U-net network and the reported transformer-based network. Benefiting from the transformer structure, our network realizes spatial-variable convolution that helps pay more attention to the regio']`
- gate[jump]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
Figure 6 caption for panel (b) states:
Section: Workflow and structure of the reported deep transformer network
> b The enhancement comparison between CNN-based U-net network and the reported transformer-based network. Benefiting from the transformer structure, our network realizes spatial-variable convolution that helps pay more attention to the regions of interests (ROI) and fine details, as the attention maps on the right side validate.
```
