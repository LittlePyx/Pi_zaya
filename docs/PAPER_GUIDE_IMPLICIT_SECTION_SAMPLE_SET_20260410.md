# Paper Guide Implicit Section Sample Set (2026-04-10)

This sample set tracks real PDFs used to validate converter behavior for papers
whose first-page structure does not cleanly label `Abstract` and `Introduction`.

## Selected PDFs

1. `F:\research-papers\research-paper-pyx\NatCommun-2021-Imaging biological tissue with...pixel compressive holography.pdf`
   - Pattern: title shell + author block + implicit abstract paragraph + implicit introduction paragraphs + first explicit section `Results`
   - Expected conversion: add `## Abstract`, then add `## Introduction` before the first non-abstract body paragraph

2. `F:\research-papers\research-paper-pyx\Optics & Laser Technology-2024-Part-based image-loop network for single-pixel imaging.pdf`
   - Pattern: explicit but spaced label `A B S T R A C T`, explicit `1. Introduction`
   - Expected conversion: normalize `A B S T R A C T` to `## Abstract`

3. `F:\research-papers\research-paper-pyx\arXiv-Quantum correlation light-field microscope with extreme depth of field.pdf`
   - Pattern: implicit abstract paragraph + explicit Roman numeral heading `I. INTRODUCTION`
   - Expected conversion: add `## Abstract`, preserve explicit introduction heading, do not insert a duplicate `Introduction`

4. `F:\research-papers\research-paper-pyx\LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf`
   - Pattern: implicit abstract paragraph + explicit `1. Introduction`
   - Expected conversion: add `## Abstract`, preserve explicit `1. Introduction`

5. `F:\research-papers\research-paper-pyx\Nature-2025-Electrically driven lasing from a dual-cavity perovskite device.pdf`
   - Pattern: multi-paragraph implicit abstract followed directly by the first topical section, with no standalone `Introduction`
   - Expected conversion: add `## Abstract`, keep all opening paragraphs under abstract, do not insert a synthetic `## Introduction`

6. `F:\research-papers\research-paper-pyx\Optica-2016-Frequency-division-multiplexed single-pixel imaging with metamaterials.pdf`
   - Pattern: implicit abstract paragraph with a copyright tail and `OCIS codes` metadata before explicit `1. INTRODUCTION`
   - Expected conversion: preserve the abstract paragraph, add `## Abstract`, preserve explicit `1. INTRODUCTION`

7. `F:\research-papers\research-paper-pyx\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf`
   - Pattern: teaser figure and caption appear before explicit `Abstract`
   - Expected conversion: preserve the teaser figure block before `## Abstract`, then preserve explicit `## 1. Introduction`

8. `F:\research-papers\research-paper-pyx\Journal of Optics-2016-3D single-pixel video.pdf`
   - Pattern: reader-service block (`To cite this article`, `You may also like`, advertisement image) appears before the real title/author/abstract block
   - Expected conversion: drop the reader-service block and duplicate title, preserve the real title, `## Abstract`, and `## Introduction`

9. `F:\research-papers\research-paper-pyx\SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf`
   - Pattern: first-page copyright / journal metadata masking can overlap the true title block if frontmatter suppression is too coarse
   - Expected conversion: preserve the title, suppress author metadata lines, and recover `## Abstract` before `## INTRODUCTION`

## Covered By

- [test_post_processing_headings.py](/f:/research-papers/2026/Jan/else/kb_chat/tests/unit/test_post_processing_headings.py)
- [test_implicit_section_regression_runner.py](/f:/research-papers/2026/Jan/else/kb_chat/tests/unit/test_implicit_section_regression_runner.py)
- [post_processing.py](/f:/research-papers/2026/Jan/else/kb_chat/kb/converter/post_processing.py)
- [paper_guide_implicit_sections_v1.json](/f:/research-papers/2026/Jan/else/kb_chat/tools/manual_regression/manifests/paper_guide_implicit_sections_v1.json)
- [implicit_section_regression.py](/f:/research-papers/2026/Jan/else/kb_chat/tools/manual_regression/implicit_section_regression.py)
- [run_implicit_section_regression.py](/f:/research-papers/2026/Jan/else/kb_chat/tools/manual_regression/run_implicit_section_regression.py)
- [converter_frontmatter_pdf_v1.json](/f:/research-papers/2026/Jan/else/kb_chat/tools/manual_regression/manifests/converter_frontmatter_pdf_v1.json)
- [frontmatter_pdf_regression.py](/f:/research-papers/2026/Jan/else/kb_chat/tools/manual_regression/frontmatter_pdf_regression.py)
- [run_frontmatter_pdf_regression.py](/f:/research-papers/2026/Jan/else/kb_chat/tools/manual_regression/run_frontmatter_pdf_regression.py)

## Notes

- The current implementation is still heuristic-driven, not a full visual layout classifier.
- It is strongest on first-page patterns where abstract and introduction are separated by paragraph boundaries and/or a later explicit section such as `Results`.
- The latest real-sample regression run is [summary.json](/f:/research-papers/2026/Jan/else/kb_chat/tmp/implicit_section_regression/20260410_222429/summary.json).
- The latest end-to-end frontmatter PDF regression run is [summary.json](/f:/research-papers/2026/Jan/else/kb_chat/tmp/frontmatter_pdf_regression/20260410_225714/summary.json).
