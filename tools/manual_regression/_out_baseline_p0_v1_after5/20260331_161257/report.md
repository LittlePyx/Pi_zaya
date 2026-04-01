# Paper Guide Baseline (P0 v1)

- Time: `2026-03-31T16:12:57`
- Base URL: `http://127.0.0.1:8000`
- Suite: `paper_guide_baseline_p0_v1`

## Summary
- total_cases: `5`
- pass_cases: `1`
- failed_cases: `4`
- overall: **FAIL**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| LSA_METHOD_REAPPLY | lsa_2026 | method | FAIL | PASS | n/a | FAIL | n/a | n/a | n/a |
| NC2023_METHOD_NETWORK_TRAINING | natcommun_2023_spad_sr | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| NC2021_OVERVIEW_THROUGHPUT | natcommun_2021_sph | overview | FAIL | PASS | n/a | n/a | n/a | n/a | n/a |
| NC2021_CITATION_DYNAMIC_SUPERSAMPLING | natcommun_2021_sph | citation_lookup | FAIL | PASS | PASS | FAIL | n/a | n/a | n/a |
| NC2021_METHOD_BEAT_FREQUENCY | natcommun_2021_sph | method | FAIL | PASS | n/a | FAIL | n/a | n/a | n/a |

## Findings
### LSA_METHOD_REAPPLY APR exact support
- paper: `lsa_2026`
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
The paper states this explicitly in Abstract / Results / iISM with adaptive pixel-reassignment (APR):
> Finally, these RVT-APR shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution.
```

### NC2023_METHOD_NETWORK_TRAINING Network training exact details
- paper: `natcommun_2023_spad_sr`
- family: `method`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['Workflow and structure of the reported deep transformer network / Network training']`
- locate_matched_anchors: `['We implemented the network on Ubuntu20 operating system using the Pytorch framework, and trained 1000 epochs until convergence using Adam optimization on NVIDIA RTX3090 with the batch size set to 24.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper states this explicitly in Workflow and structure of the reported deep transformer network / Network training:
> We implemented the network on Ubuntu20 operating system using the Pytorch framework, and trained 1000 epochs until convergence using Adam optimization on NVIDIA RTX3090 with the batch size set to 24. We set the initial learning rate as 0.0003, which was decreased by 10% for every 100 epochs.
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
The authors claim a space-bandwidth-time product (SBP-T) of **41,667 pixels/s**, which is about **3× higher** than the largest previously reported value in single-pixel holography (SPH) literature [[CITE:s11b51525:26]]. This throughput was achieved using a single-pixel detector and enables reconstructed images up to $256 \times 256 = 65{,}536$ pixels—about 4× larger than prior SPH reports.

They frame this improvement as critical for biological tissue imaging because high SBP-T supports large field-of-view (e.g., $14.9\,\text{mm} \times 11.1\,\text{mm}$) and high-resolution monitoring of dynamic biological processes, while leveraging the advantages of single-pixel detectors—low dark noise, high sensitivity, broad spectral compatibility, and low cost—especially where pixel-array detectors (e.g., CCD/CMOS) are unavailable or impractical.
```

### NC2021_CITATION_DYNAMIC_SUPERSAMPLING Dynamic supersampling citation lookup
- paper: `natcommun_2021_sph`
- family: `citation_lookup`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `1`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `PASS`
- citation_matched_ref_nums: `[31]`
- gate[locate]: `FAIL`
- gate_reason[locate]: `locate_missing_matching_segment`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `locate:locate_missing_matching_segment`
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
