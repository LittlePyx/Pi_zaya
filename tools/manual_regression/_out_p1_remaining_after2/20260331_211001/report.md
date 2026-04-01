# Paper Guide Baseline (P0 v1)

- Time: `2026-03-31T21:10:01`
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
| NP2019_NOT_STATED_GPU | natphoton_2019_spi | reproduce | PASS | PASS | n/a | n/a | n/a | n/a | n/a |
| LSA_DOC_MAP | lsa_2026 | overview | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| NC2021_OVERVIEW_THROUGHPUT | natcommun_2021_sph | overview | FAIL | PASS | n/a | n/a | n/a | n/a | n/a |

## Findings
### NP2019_NOT_STATED_GPU Control question: unspecified GPU
- paper: `natphoton_2019_spi`
- family: `reproduce`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The retrieved paper does not specify the GPU model used for the single-pixel imaging experiments.
```

### LSA_DOC_MAP Doc map / outline
- paper: `lsa_2026`
- family: `overview`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['Abstract / Introduction', 'Abstract / Results / Principle of interferometric ISM (iISM) / Figure 1', 'Abstract / Results / iISM with adaptive pixel-reassignment (APR)']`
- locate_matched_anchors: `['Light microscopy is indispensable in both material and life sciences and continues to advance toward higher spatial and temporal resolution, improved sensitivity, and enhanced imaging depth.', 'In this work, we developed a new ISM microscope that includes both interferometric scattering and fluorescence detection.', 'The most widely used approach is pixel-reassignment (PR), where the shift vectors $\\vec{\\mu}$ are estimated from the detector geometry.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
Doc map (verbatim anchors by section):

1. Abstract
> Light microscopy remains indispensable in life sciences for visualizing cellular structures and dynamics in live specimens.

2. Abstract / Introduction
> Light microscopy is indispensable in both material and life sciences and continues to advance toward higher spatial and temporal resolution, improved sensitivity, and enhanced imaging depth.

3. Abstract / Results / Principle of interferometric ISM (iISM)
> In this work, we developed a new ISM microscope that includes both interferometric scattering and fluorescence detection.

4. Abstract / Results / iISM with adaptive pixel-reassignment (APR)
> The most widely used approach is pixel-reassignment (PR), where the shift vectors $\vec{\mu}$ are estimated from the detector geometry.

5. Abstract / Results / iISM imaging of intracellular organelles in live cells
> Major organelles, inclu
```

### NC2021_OVERVIEW_THROUGHPUT Overview: throughput contribution
- paper: `natcommun_2021_sph`
- family: `overview`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `1`
- raw_structured_cite_count: `1`
- gate[answer]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `FAIL`
- gate_reason[structured_markers]: `raw_structured_cite_count=1`
- reason: `structured_markers:raw_structured_cite_count=1`
- answer_preview:

```text
The authors claim a space-bandwidth-time product (SBP-T) of **41,667 pixels/s**, which is about **3× larger** than the largest previously reported value in single-pixel holography (SPH) literature [[CITE:s11b51525:26]]. This throughput is achieved using a single-pixel detector and enables reconstructed images up to $256 \times 256 = 65{,}536$ pixels—about 4× larger than prior SPH results.

They frame this high throughput as critical for biological tissue imaging because it supports large field-of-view (FOV) operation (e.g., $14.9\,\text{mm} \times 11.1\,\text{mm}$) and real-time monitoring of dynamic biological processes, while leveraging the advantages of single-pixel detectors—low dark noise, high sensitivity, broad spectral compatibility, and low cost—especially where pixel-array detectors (e.g., CCD/CMOS) are unavailable or impractical.
```
