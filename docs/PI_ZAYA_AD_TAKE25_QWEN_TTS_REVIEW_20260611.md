# Pi-zaya Take25 Qwen-TTS Review

Date: 2026-06-11

## Outputs

- Qwen-TTS video: `docs/ad_assets/pi_zaya_20260609/take25_qwen_voice_test_20260611/pi_zaya_ad_take25_qwen_tts_bilingual_2560x1800_h264420_crf8_20260611.mp4`
- Qwen-TTS voiceover: `docs/ad_assets/pi_zaya_20260609/take25_qwen_voice_test_20260611/take13_voiceover.wav`
- Timing report: `docs/ad_assets/pi_zaya_20260609/take25_qwen_voice_test_20260611/take13_voice_timing_report.json`
- Review sheet: `docs/ad_assets/pi_zaya_20260609/take25_qwen_voice_test_20260611/review_frames_qwen_tts/take25_qwen_tts_review_sheet.jpg`
- Opening sample: `docs/ad_assets/pi_zaya_20260609/take25_voice_tests_20260611/sample_qwen3_tts_instruct_cherry_opening.wav`

## What Changed

- Added `--tts-provider qwen-tts` to `produce_take13_final.py`.
- Uses Alibaba Cloud DashScope Qwen-TTS non-streaming API through `dashscope.MultiModalConversation.call`.
- Default model: `qwen3-tts-instruct-flash`.
- Default voice: `Cherry`.
- Uses `QWEN_API_KEY` when `DASHSCOPE_API_KEY` is not set.
- Converts `QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` to `https://dashscope.aliyuncs.com/api/v1` for the DashScope SDK.

## Voice Timing Audit

Result: pass, 0 issues.

All blocks are now `tempo=1.000`, so the final narration is not being forcibly sped up. The only overrun was the DOI line; it was shortened to:

`DOI 回到官方论文页。光标停在 Download PDF，来源能核到出版方。`

## Ad Review

1. Opening montage: partially done, not finished. The rotating logo is present, and there is a quick product/page-wall beat, but it is not yet the premium "one page zooms in, then the next page takes over" montage the ad needs.
2. Library segment: functionally correct. It shows category/tag surfaces, but the filter and tag changes are still too small. Next pass should use macro crop zooms on category, tag, manual edit, and list-state change.
3. Reader conversion: better than before because it shows converted Markdown with equations and structure. It still needs a clearer before-to-after beat: PDF upload result becomes searchable Markdown with location anchors.
4. Question/generation: now closer to an agent demo because a new writing task appears and the answer generates. It should show generation progress more cinematically, not only a finished answer view.
5. Source #4 card: much improved. It now shows the card instead of a generic template. The card is still too small for mobile viewing; next pass should punch in on claim, evidence snippet, relevance, and paper metadata.
6. Evidence jump: strong. The Reader lands on the highlighted source paragraph, which matches the product promise.
7. DOI page: strong. It lands on the official publisher page and the cursor is visibly at `Download PDF`.
8. Literature basket: correct workflow, but still visually understated. The selected papers and next-turn context strip need a larger animated emphasis.
9. Quality center: correctly de-emphasized. It should stay out of the ad unless there is a single useful signal tied to "can this paper be asked".
10. Product-ad feel: the current version is still mostly screen recording. The next real upgrade should be a post-production compositor pass with page tiles, scale moves, soft shadows, speed ramps, cursor callouts, and crop zooms.

## Next Edit Direction

- Build a true opening sequence: logo spin -> four page tiles -> sequential zooms into Library, Reader, Source Card, Evidence Highlight, Basket.
- Add a conversion beat: PDF card -> Markdown Reader -> anchor marker -> answer citation.
- Replace small UI callout cards with larger cropped product surfaces.
- Keep Qwen-TTS, but synthesize in shorter phrase groups if we need even more natural pauses.
- Make a 75-90 second short cut after the current 125 second full demo is stable.
