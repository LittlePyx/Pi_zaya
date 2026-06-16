# Pi-zaya Take27 Short Product Ad Review

Date: 2026-06-12

## Outputs

- Final video: `docs/ad_assets/pi_zaya_20260609/take27_short_product_ad_2x_png_20260612/pi_zaya_ad_take27_short_product_ad_bilingual_2560x1800_h264420_crf8_20260612.mp4`
- Review sheet: `docs/ad_assets/pi_zaya_20260609/take27_short_product_ad_2x_png_20260612/review_frames_final/take27_short_product_ad_review_sheet.jpg`
- Builder script: `docs/ad_assets/pi_zaya_20260609/build_take27_short_product_ad.py`
- Voice overrides: `docs/ad_assets/pi_zaya_20260609/take27_short_voice_overrides.json`
- Voice timing report: `docs/ad_assets/pi_zaya_20260609/take27_short_product_ad_2x_png_20260612/take13_voice_timing_report.json`

## Specs

- Duration: 79.50 s.
- Video: H.264, 2560x1800, 30 fps.
- Audio: AAC.
- Size: about 36.9 MB.
- Qwen-TTS timing audit: 0 issues.
- Narration pacing: no hard speed-up; only the Reader segment has a tiny `tempo=1.029`, below the visible warning threshold.

## What Changed From Take26

1. Cut the full 124.83 s demo down to a 79.50 s short product-ad version.
2. Kept the rotating logo and sequential page montage at the opening.
3. Preserved the real Library manual-edit drawer from the recorded UI, including category/tag/status editing context.
4. Kept the tag-filter narrowing moment after manual edit.
5. Kept Reader as the conversion proof: Markdown text, structure, formulas, and locatable evidence anchors.
6. Combined ask and generation into one faster agent segment.
7. Kept source card, evidence jump, DOI, basket, and next-turn writing as the core research-agent workflow.
8. Rewrote the narration into shorter ad lines so Qwen-TTS can speak naturally without forced compression.

## Review Notes

- The short cut is much more publishable than the 125 s version for social platforms.
- Library now answers the user's requested point: automatic category/tag recommendation plus manual modification.
- Evidence jump remains the strongest proof moment: the Reader lands directly on highlighted original evidence.
- The DOI + basket sequence is compressed but still understandable.
- The right-side source card is readable enough at full resolution, but the next polish pass could enlarge the card even more if targeting mobile-only distribution.

## Remaining Polish

- Some compositor overlay labels are still English. For a Chinese release, either localize them or remove them and rely on subtitles.
- The manual-edit segment is visually present, but a future fresh browser recording could type one edited tag live for an even clearer "manual edit" proof.
- A separate vertical 9:16 cut should be produced for Douyin/Reels style publishing.
