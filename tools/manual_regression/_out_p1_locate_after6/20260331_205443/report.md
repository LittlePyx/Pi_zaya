# Paper Guide Baseline (P0 v1)

- Time: `2026-03-31T20:54:43`
- Base URL: `http://127.0.0.1:8000`
- Suite: `paper_guide_baseline_p0_v1`

## Summary
- total_cases: `1`
- pass_cases: `0`
- failed_cases: `1`
- overall: **FAIL**

## Cases

| Case | Paper | Family | Status | answer | cite | locate | section | panel | jump |
|---|---|---|---|---|---|---|---|---|---|
| NC2021_METHOD_BEAT_FREQUENCY | natcommun_2021_sph | method | FAIL | PASS | n/a | FAIL | n/a | n/a | n/a |

## Findings
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
The paper states this explicitly in ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup:
> Thus, the beat frequency of these two beams is 62,500 Hz, indicating a temporal period of 16 µs. Using a lens with 150-mm focal length, the combined light was collected by a photodetector (DET10A2, Thorlabs) with a bandwidth of 350 MHz, which was then digitized by a data acquisition card (DAC, USB-6251, National Instrument) with a sampling rate of 1.25 Ms/s (not shown in the figure).
```
