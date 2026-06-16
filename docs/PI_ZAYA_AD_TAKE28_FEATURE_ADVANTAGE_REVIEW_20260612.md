# Pi-zaya Take28 Feature-Advantage Ad Review

Date: 2026-06-12

## Outputs

- Final video: `docs/ad_assets/pi_zaya_20260609/take28_feature_advantage_ad_2x_png_20260612/pi_zaya_ad_take28_feature_advantage_ad_bilingual_2560x1800_h264420_crf8_20260612.mp4`
- Review sheet: `docs/ad_assets/pi_zaya_20260609/take28_feature_advantage_ad_2x_png_20260612/review_frames_final/take28_feature_advantage_review_sheet.jpg`
- Builder script: `docs/ad_assets/pi_zaya_20260609/build_take28_feature_advantage_ad.py`
- Voice overrides: `docs/ad_assets/pi_zaya_20260609/take28_feature_advantage_voice_overrides.json`
- Voice timing report: `docs/ad_assets/pi_zaya_20260609/take28_feature_advantage_ad_2x_png_20260612/take13_voice_timing_report.json`

## Specs

- Duration: 105.00 s.
- Video: H.264, 2560x1800, 30 fps.
- Audio: AAC.
- Size: about 43.6 MB.
- Qwen-TTS timing audit: 0 issues.
- Narration pacing: all blocks are below the visible speed-up threshold.

## What Was Added From Take27

1. More complete positioning: Pi-zaya is introduced as a research agent for arbitrary PDF libraries, not only this single-photon imaging demo.
2. Library now explicitly covers automatic categories/tags, manual editing, reading status, conversion status, pages, refs, figures, math count, and ask-ready state.
3. Reader still explains PDF -> Markdown conversion, searchable text/formulas/sections, and evidence anchors.
4. The agent-generation section is longer, making the answer feel less like a static result.
5. The answer-source relationship is called out before opening the reference card.
6. Source card narration now explains what a good card must show: claim, original evidence, and why the paper is relevant.
7. Evidence jump remains visible and lands on highlighted original text.
8. DOI and basket were expanded: publisher verification, basket summary, source trail, original entry point, and next-turn context are all mentioned.
9. The closing still stays short enough to avoid forced TTS speed-up.

## Review Notes

- This version is a better full product-ad cut than take27. Take27 is the social short; take28 is the feature-advantage version.
- The most important missing functionality from take27 is now covered without making the ad feel like a feature checklist.
- The strongest proof sequence remains: source card -> Reader highlight -> DOI -> basket context.
- The Library segment is now more informative, but still visually dense. Full-resolution playback is acceptable; mobile-only publishing may need a vertical crop.

## Remaining Polish

- Export citation formats are still not shown.
- Quality center remains intentionally omitted except for ask-ready/conversion status signals.
- Some compositor labels are English; a final Chinese-market version should either localize or remove those labels.
- A future vertical 9:16 version should reframe Library and source-card crops for phone screens.
