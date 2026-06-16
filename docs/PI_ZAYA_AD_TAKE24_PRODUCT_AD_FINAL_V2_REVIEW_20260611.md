# Pi-zaya Take24 Product Ad Final V2 Review

Date: 2026-06-11

## Outputs

- Final video: `docs/ad_assets/pi_zaya_20260609/take24_product_ad_record_v2_2x_png_20260611/pi_zaya_ad_take24_product_ad_final_v2_bilingual_2560x1800_h264420_crf8_20260611.mp4`
- Review sheet: `docs/ad_assets/pi_zaya_20260609/take24_product_ad_record_v2_2x_png_20260611/review_frames_final_v2/take24_product_ad_final_v2_review_sheet.jpg`
- Timing report: `docs/ad_assets/pi_zaya_20260609/take24_product_ad_record_v2_2x_png_20260611/take13_voice_timing_report.json`

## Technical Result

- Video: H.264 + AAC, `2560x1800`, `124.83s`, about `29.6MB`.
- TTS: DashScope/CosyVoice.
- Timing audit: `0` issues.
- Tempo: most blocks are `1.000x`; DOI is `1.052x`, closing is `1.061x`, both below the warning threshold and not visibly rushed.

## Fixed During This Pass

1. Evidence jump no longer lands on `404 Not Found`.
   The recording now falls back to the prepared NatCommun evidence Reader when the clicked source does not have local Markdown.

2. DOI no longer lands on a Science.org security-check page.
   The DOI segment now opens a stable Nature official article page and holds on `Download PDF`.

3. Voice pacing is no longer uneven.
   Reader, question, DOI, and closing copy were shortened or adjusted, then rebuilt with `--strict-timing`.

4. Quality Center is no longer part of the final cut.
   Library now carries the metadata story: categories, tags, conversion state, page/ref counts, and ask-ready status.

5. Reader figure-caption copy remains removed.
   Reader is framed as converted Markdown with searchable structure and source-location anchors.

## Visual Review

Strong:
- Library taxonomy is much clearer than before. It shows category/tag filtering and manual metadata editing.
- Reader conversion proof is now a real converted Markdown view with formulas and structure.
- Source #4 card is more useful than the earlier template-looking card.
- Evidence jump now visibly lands on highlighted original evidence in Reader.
- DOI segment now supports the official-source claim.
- Basket/context-pack segment finally shows the next-turn context state clearly.

Still Weak:
- Opening is improved by the rotating logo, but the montage still feels like real UI cuts rather than a fully composited high-end product ad.
- Source card text is readable in close-up, but still small in the full 2560x1800 frame. A post-production macro crop would make it stronger.
- The final cut is `124.83s`, not the planned `107s`, because real UI actions add time. Pacing is now stable, but a tighter short-ad version should trim Library, DOI, and Basket.
- The product strip/spotlight helps comprehension, but a custom post-FX pass would look more premium than browser overlays.

## Next Cut Recommendation

Use this V2 as the functional proof version. For a more polished product-ad version, keep the same recording but add a post-production compositor pass:

- Opening: true page-wall parallax with source card and Reader highlight as the largest tiles.
- Evidence card: crop in tighter and freeze for readability.
- Evidence jump: snap transition from card to Reader highlight.
- Basket: animate the selected context pack with a small lift/glow.
- Closing: replace the plain title card with proven UI tiles settling behind the logo.
