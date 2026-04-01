# Paper Guide Baseline (P0 v1)

- Time: `2026-03-31T21:07:56`
- Base URL: `http://127.0.0.1:8000`
- Suite: `paper_guide_baseline_p0_v1`

## Summary
- total_cases: `3`
- pass_cases: `1`
- failed_cases: `2`
- overall: **FAIL**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| NP2019_NOT_STATED_GPU | natphoton_2019_spi | reproduce | PASS | PASS | n/a | n/a | n/a | n/a | n/a |
| LSA_DOC_MAP | lsa_2026 | overview | FAIL | FAIL | n/a | FAIL | n/a | n/a | n/a |
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
The retrieved context does not specify the GPU model used for the single-pixel imaging experiments. The paper discusses experimental setups involving DMDs, single-pixel detectors, and signal-processing pipelines, but no GPU hardware details are provided in the excerpts.
```

### LSA_DOC_MAP Doc map / outline
- paper: `lsa_2026`
- family: `overview`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `FAIL`
- gate_reason[answer]: `answer_missing_contains_any`
- gate[locate]: `FAIL`
- gate_reason[locate]: `locate_missing_matching_segment`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `answer:answer_missing_contains_any`
- reason: `locate:locate_missing_matching_segment`
- answer_preview:

```text
Here is a section-by-section reading map using one verbatim anchor sentence per major section, drawn directly from the retrieved context and aligned with the paper’s structure:

- **Title / Main claim**:
 *“Interferometric Image Scanning Microscopy for label-free imaging at 120 nm lateral resolution inside live cells”*.

- **Abstract / Introduction (core contribution)**:
 *“By combining the principles of ISM and iSCAT, iISM enables high resolution ($\sim$120 nm laterally), and label-free imaging inside live cells at substantially reduced illumination power.”*.

- **Principle / Figure 1 comparison**:
 *“Figure 1d, e compare the iPSFs under circular polarization for open-pinhole and closed-pinhole confocal iSCAT configurations after background normalization.”*.

- **Resolution quantification**:
 *“The closed pinhole iPSF exhibits a full width at half maximum (FWHM) of about 122 nm pm 5 nm
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
The authors claim a space-bandwidth-time product (SBP-T) of **41,667 pixels/s**, which is about **3× larger** than the largest previously reported value in single-pixel holography (SPH) literature [[CITE:s11b51525:26]]. This throughput was achieved using a single-pixel detector and enables reconstructed images up to $256 \times 256 = 65{,}536$ pixels—about 4× larger than prior SPH reports.

They frame this high throughput as critical for biological tissue imaging because it supports large field-of-view (FOV) operation (e.g., $14.9\,\text{mm} \times 11.1\,\text{mm}$) and real-time monitoring of dynamic biological processes, while leveraging the advantages of single-pixel detectors—low dark noise, high sensitivity, broad spectral compatibility, and low cost—especially where pixel-array detectors (e.g., CCD/CMOS) are unavailable or impractical.
```
