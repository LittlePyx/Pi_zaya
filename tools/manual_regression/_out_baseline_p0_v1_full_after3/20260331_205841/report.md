# Paper Guide Baseline (P0 v1)

- Time: `2026-03-31T20:58:41`
- Base URL: `http://127.0.0.1:8000`
- Suite: `paper_guide_baseline_p0_v1`

## Summary
- total_cases: `17`
- pass_cases: `14`
- failed_cases: `3`
- overall: **FAIL**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| SCINERF2024_CITATION_ADMM | cvpr_2024_scinerf | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| SCINERF2024_METHOD_IMPL_ITER_BATCH | cvpr_2024_scinerf | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| SCINERF2024_FIG3_CAPTION_COMPONENTS | cvpr_2024_scinerf | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | n/a | PASS |
| NP2019_BOX1_ONLY | natphoton_2019_spi | box_only | PASS | PASS | n/a | PASS | PASS | n/a | n/a |
| NP2019_CITATION_DUARTE | natphoton_2019_spi | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| NP2019_FIGURE3_PANEL_F | natphoton_2019_spi | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | PASS | PASS |
| NP2019_NOT_STATED_GPU | natphoton_2019_spi | reproduce | FAIL | FAIL | n/a | n/a | n/a | n/a | n/a |
| LSA_DOC_MAP | lsa_2026 | overview | FAIL | FAIL | n/a | FAIL | n/a | n/a | n/a |
| LSA_CITATION_RVT | lsa_2026 | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| LSA_METHOD_REAPPLY | lsa_2026 | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| NC2023_CITATION_TRANSFORMER_FRAMEWORK | natcommun_2023_spad_sr | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| NC2023_METHOD_NETWORK_TRAINING | natcommun_2023_spad_sr | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| NC2023_FIGURE6_PANEL_B_CLAUSE | natcommun_2023_spad_sr | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | PASS | PASS |
| NC2021_OVERVIEW_THROUGHPUT | natcommun_2021_sph | overview | FAIL | PASS | n/a | n/a | n/a | n/a | n/a |
| NC2021_CITATION_DYNAMIC_SUPERSAMPLING | natcommun_2021_sph | citation_lookup | PASS | PASS | PASS | PASS | n/a | n/a | n/a |
| NC2021_METHOD_BEAT_FREQUENCY | natcommun_2021_sph | method | PASS | PASS | n/a | PASS | n/a | n/a | n/a |
| NC2021_FIGURE7_SETUP | natcommun_2021_sph | figure_walkthrough | PASS | PASS | n/a | PASS | n/a | n/a | PASS |

## Findings
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

### SCINERF2024_METHOD_IMPL_ITER_BATCH Implementation details: iterations and batch size (rays)
- paper: `cvpr_2024_scinerf`
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

### SCINERF2024_FIG3_CAPTION_COMPONENTS Figure 3 experimental setup caption
- paper: `cvpr_2024_scinerf`
- family: `figure_walkthrough`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup / Figure 3', 'SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup / Figure 3', 'SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image / 4. Experiments / 4.1. Experimental Setup / Figure 3']`
- locate_matched_anchors: `['This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.', 'This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.', 'This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.']`
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

This sentence is directly supported by the supporting excerpts.

Caption anchor: **Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.

### Library Figure

![CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf Fig. 3](/api/references/asset?path=F%3A%5Cresearch-papers%5C2026%5CJan%5Celse%5Ckb_chat%5Cdb%5CCVPR-2024-SCINeRF-%20Neural%20Radi
```

### NP2019_BOX1_ONLY Box 1 exact support
- paper: `natphoton_2019_spi`
- family: `box_only`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['Box 1', 'Box 1']`
- locate_matched_anchors: `['It can be shown that when the number of sampling patterns used M >= O(K log(N/K)), the image in the transform domain can be reconstructed by solving an optimization problem, most commonly \\ell_1-minimization, expressed as', 'It can be shown that when the number of sampling patterns used M >= O(K log(N/K)), the image in the transform domain can be reconstructed by solving an optimization problem, most commonly \\ell_1-minimization, expressed as']`
- gate[section_target]: `PASS`
- section_target_matched_headings: `['Box 1', 'Box 1']`
- section_target_matched_anchors: `['It can be shown that when the number of sampling patterns used M >= O(K log(N/K)), the image in the transform domain can be reconstructed by solving an optimization problem, most commonly \\ell_1-minimization, expressed as', 'It can be shown that when the number of sampling patterns used M >= O(K log(N/K)), the image in the transform domain can be reconstructed by solving an optimization problem, most commonly \\ell_1-minimization, expressed as']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
From Box 1, the authors state that the image in the transform domain can be reconstructed when the number of sampling patterns satisfies
$$
M \ge O(K \log(N/K)),
$$
by solving an optimization problem—most commonly $\ell_1$-minimization.

Exact supporting part:
> “It can be shown that when the number of sampling patterns used $M \ge O(K \log(N/K))$, the image in the transform domain can be reconstructed by solving an optimization problem, most commonly $\ell_1$-minimization, expressed as”.

Plain-text condition: M >= O(K log(N/K)).
```

### NP2019_CITATION_DUARTE Duarte citation lookup
- paper: `natphoton_2019_spi`
- family: `citation_lookup`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `1`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `PASS`
- citation_matched_ref_nums: `[4]`
- gate[locate]: `PASS`
- locate_matched_headings: `['Principles and prospects for single-pixel imaging / Acquisition and image reconstruction strategies']`
- locate_matched_anchors: `['The original concept of the single-pixel imaging approach, demonstrated by Sen et al. at Stanford University$^{3,58}$, was developed further in conjunction with compressive sensing$^{59}$ and reported soon after in a seminal paper by Duarte']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper cites [4] for this point.
This is stated in Principles and prospects for single-pixel imaging / Acquisition and image reconstruction strategies:
> The original concept of the single-pixel imaging approach, demonstrated by Sen et al. at Stanford University$^{3,58}$, was developed further in conjunction with compressive sensing$^{59}$ and reported soon after in a seminal paper by Duarte et al. at Rice University$^{4}$.
```

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
Figure 3 caption for panel (f) states:
Section: Principles and prospects for single-pixel imaging / Applications and future potential for single-pixel imaging
> (f) methane imaging using SPC$^{15}$;
References in this clause: [15]

### Library Figure

![NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf Fig. 3](/api/references/asset?path=F%3A%5Cresearch-papers%5C2026%5CJan%5Celse%5Ckb_chat%5Cdb%5CNatPhoton-2019-Principles%20and%20prospects%20for%20single-pixel%20imaging%5Cassets%5Cpage_5_fig_2.png)

*Source: NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf, Fig. 3. Fig. 3*
```

### NP2019_NOT_STATED_GPU Control question: unspecified GPU
- paper: `natphoton_2019_spi`
- family: `reproduce`
- status: **FAIL**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `FAIL`
- gate_reason[answer]: `answer_missing_contains_any`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `answer:answer_missing_contains_any`
- answer_preview:

```text
The retrieved paper does not state the GPU model used for the single-pixel imaging experiments.
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

- **Principle / Figure 1 illustration**:
 *“Figure 1d, e compare the iPSFs under circular polarization for open-pinhole and closed-pinhole confocal iSCAT configurations after background normalization.”*.

- **Resolution quantification**:
 *“The closed pinhole iPSF exhibits a full width at half maximum (FWHM) of about 122 nm pm 5 n
```

### LSA_CITATION_RVT RVT citation lookup
- paper: `lsa_2026`
- family: `citation_lookup`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `1`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `PASS`
- citation_matched_ref_nums: `[34]`
- gate[locate]: `PASS`
- locate_matched_headings: `['Abstract / Results / iISM with adaptive pixel-reassignment (APR) / Figure 2']`
- locate_matched_anchors: `['Specifically, we use the radial variance transform (RVT)[34], which converts an interferogram into an intensity-only map that reflects the local degree of symmetry.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper cites [34] for this point.
This is stated in Abstract / Results / iISM with adaptive pixel-reassignment (APR):
> Specifically, we use the radial variance transform (RVT)[34], which converts an interferogram into an intensity-only map that reflects the local degree of symmetry.
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
- locate_matched_headings: `['Abstract / Results / iISM with adaptive pixel-reassignment (APR) / Figure 2']`
- locate_matched_anchors: `['Finally, these RVT-APR shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper states this explicitly in Abstract / Results / iISM with adaptive pixel-reassignment (APR):
> Finally, these RVT-APR shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution.
```

### NC2023_CITATION_TRANSFORMER_FRAMEWORK Transformer framework citations
- paper: `natcommun_2023_spad_sr`
- family: `citation_lookup`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `2`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[citation]: `PASS`
- citation_matched_ref_nums: `[33, 34]`
- gate[locate]: `PASS`
- locate_matched_headings: `['']`
- locate_matched_anchors: `['The transformer framework$^{33,34}$ has recently attracted increasing attention and produced an impressive performance on multiple vision tasks$^{34–36}$.']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper cites [33], [34] for this point.
This is stated explicitly in the paper:
> The transformer framework$^{33,34}$ has recently attracted increasing attention and produced an impressive performance on multiple vision tasks$^{34–36}$.
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
- locate_matched_anchors: `['We implemented the network on Ubuntu20 operating system using the Pytorch framework, and trained 1000 epochs until convergence using Adam optimization on NVIDIA RTX3090 with the batch size set to 24. We set the initial learning rate as 0.00']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper states this explicitly in Workflow and structure of the reported deep transformer network / Network training:
> We implemented the network on Ubuntu20 operating system using the Pytorch framework, and trained 1000 epochs until convergence using Adam optimization on NVIDIA RTX3090 with the batch size set to 24. We set the initial learning rate as 0.0003, which was decreased by 10% for every 100 epochs.
```

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
The authors claim a space-bandwidth-time product (SBP-T) of **41,667 pixels/s**, which is about **3× larger** than the largest previously reported value in single-pixel holography (SPH) literature [[CITE:s11b51525:26]]. This throughput is achieved using a single-pixel detector and enables reconstructed images up to $256 \times 256 = 65{,}536$ pixels—about 4× larger than prior SPH reports.

They frame this improvement as critical for biological tissue imaging because high SBP-T supports large field-of-view (e.g., $14.9\,\text{mm} \times 11.1\,\text{mm}$) and high-resolution monitoring of dynamic biological processes, while leveraging the advantages of single-pixel detectors—low dark noise, high sensitivity, broad spectral compatibility, and low cost—especially where pixel-array detectors (e.g., CCD/CMOS) are unavailable or impractical.
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
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup / Figure 7']`
- locate_matched_anchors: `['Thus, the beat frequency of these two beams is 62,500 Hz, indicating a temporal period of 16 µs. Using a lens with 150-mm focal length, the combined light was collected by a photodetector (DET10A2, Thorlabs) with a bandwidth of 350 MHz, whi']`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
The paper states this explicitly in ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup:
> Thus, the beat frequency of these two beams is 62,500 Hz, indicating a temporal period of 16 µs. Using a lens with 150-mm focal length, the combined light was collected by a photodetector (DET10A2, Thorlabs) with a bandwidth of 350 MHz, which was then digitized by a data acquisition card (DAC, USB-6251, National Instrument) with a sampling rate of 1.25 Ms/s (not shown in the figure).
```

### NC2021_FIGURE7_SETUP Figure 7 setup caption
- paper: `natcommun_2021_sph`
- family: `figure_walkthrough`
- status: **PASS**
- minimum_ok: `true`
- cite_count: `0`
- raw_structured_cite_count: `0`
- gate[answer]: `PASS`
- gate[locate]: `PASS`
- locate_matched_headings: `['ARTICLE / Methods / Principle of high-throughput SPH / Figure 7', 'ARTICLE / Methods / Principle of high-throughput SPH / Figure 7']`
- locate_matched_anchors: `['a series of Hadamard-like patterns were generated and projected to the sample. HWP1-3: half-wave plate; M1-3: mirror; PBS: polarization beam splitter; AOM1-2: acousto-optic modulators that cause a frequency shift to the light passing throug', 'a series of Hadamard-like patterns were generated and projected to the sample. HWP1-3: half-wave plate; M1-3: mirror; PBS: polarization beam splitter; AOM1-2: acousto-optic modulators that cause a frequency shift to the light passing throug']`
- gate[jump]: `PASS`
- gate[quality]: `PASS`
- gate[structured_markers]: `PASS`
- reason: `none`
- answer_preview:

```text
Figure 7 includes the following components, as explicitly listed in its caption:

- A long-coherence solid-state semiconductor laser (MSL-FN-532-100mW, CNI, 532 nm) as light source
- Polarizing beam splitter (PBS) to split light into signal and reference beams
- Half-wave plates HWP1–3
- Mirrors M1–3
- Acousto-optic modulators AOM1–2 (AOM-505AF1, Intraaction), frequency-shifted by a double-channel function generator
- Lenses L1–L7 with focal lengths: $f_1 = f_3 = 7.5\,\text{mm}$, $f_2 = f_4 = 250\,\text{mm}$, $f_7 = 150\,\text{mm}$; $f_5$, $f_6$ are scalable
- Digital micromirror device (DMD) for amplitude modulation (“0”/“1”)
- Beam splitter (BS)
- Photodetector (PD)
- Upper inset: schematic of double-channel function generation producing beating frequency to drive AOMs (power amplifier omitted)

The exact supporting caption clause is:
**“Figure 7. Experiment setup of the high-throughpu
```
