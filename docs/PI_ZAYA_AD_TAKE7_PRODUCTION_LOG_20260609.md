# π-zaya Take7 15-Minute Ad Production Log

Date: 2026-06-10

## Final Recommended Version

- File: `docs/ad_assets/pi_zaya_20260609/pi_zaya_ad_take7_15min_1280x800_clean_v3_pi_greek_20260610.mp4`
- Format: MP4 / H.264
- Frame size: `1280x800`
- Duration: `00:15:03.04`
- Frame rate: `25 fps`
- Size: `24,855,290 bytes`

This is the recommended release/editing master. It keeps the take7 workflow, replaces the opening and closing cards with the correct Greek brand spelling `π-zaya`, removes the initial app-shell frame and the short loading-white transition, and keeps the finished runtime at about 15 minutes.

## Public Cut Recommendations

- Public long-form cut: `docs/ad_assets/pi_zaya_20260609/pi_zaya_ad_take7_directors_cut_9min_1280x800_v4_pi_greek_20260610.mp4`
- Short cut: `docs/ad_assets/pi_zaya_20260609/pi_zaya_ad_take7_short_60s_1280x800_v4_pi_greek_20260610.mp4`

The v4 director's cut is recommended for public viewing because it keeps the full workflow but removes long waits and repetitive holds. The v4 short is recommended as the 1-minute teaser/editing base.

## Raw Recording

- File: `docs/ad_assets/pi_zaya_20260609/pi_zaya_ad_take7_15min_1280x800_20260609_235449.webm`
- Format: WebM / VP8
- Frame size: `1280x800`
- Duration: `00:15:03.04`
- Status file: `docs/ad_assets/pi_zaya_20260609/ad_take7_15min_1280x800_status.json`
- stderr log: `docs/ad_assets/pi_zaya_20260609/pi_zaya_take7_15min_20260609_235449.stderr.log`

Raw recording completed with `stderr` length `0`. The raw video is preserved as source material, but the clean v3 MP4 is preferred for review and publishing.

## Flow Covered

1. Brand opening card.
2. Literature library preparation: converted/answerable status, quality maintenance, tags/classification.
3. Real full-library question: `单光子成像为什么需要专门建模噪声？有哪些代表性做法？`
4. Structured answer with chapter navigation and subtitle overlay.
5. Reference positioning and citation card details.
6. Reader opened in expanded single-page form, using a clean paragraph region.
7. Reader features: current evidence, evidence tab, chapter context, in-paper references.
8. Literature basket causality: answer/reference card entry plus Reader/in-paper-reference entries, then seeded multi-paper basket.
9. Basket review: 6 items, detail/selection, local snapshot.
10. Export: selected/current/all scopes plus BibTeX, RIS, Markdown, GB/T, CSV.
11. Return to library and quality-maintenance close-up.
12. Stable final brand card.

## Verification

- `ffprobe` clean v3 result: `1280x800`, H.264, `25/1 fps`, `903.040000 s`.
- `blackdetect=d=0.5:pix_th=0.1` produced no `black_start` records.
- Key frame review directory: `docs/ad_assets/pi_zaya_20260609/take7_clean_v3_review_frames_20260610/`
- Raw review frames: `docs/ad_assets/pi_zaya_20260609/take7_review_frames_20260609_235449/`

Reviewed frames confirm:

- The right literature basket is not clipped at `1280x800`.
- Reader is shown as the expanded single-page Reader, not a side drawer.
- The Reader segment avoids the bad figure/OCR text region and stays on a clean paragraph.
- Export controls and formats are visible in the right panel.
- The visible cursor aligns with the target areas well enough for the page-level recording.
- Opening and ending cards are clean in the v3 master and use `π-zaya`.

## Still Images Captured During Recording

- `docs/ad_assets/pi_zaya_20260609/103_take7_ready_1280x800.png`
- `docs/ad_assets/pi_zaya_20260609/104_take7_qa_1280x800.png`
- `docs/ad_assets/pi_zaya_20260609/105_take7_reader_1280x800.png`
- `docs/ad_assets/pi_zaya_20260609/106_take7_export_1280x800.png`
- `docs/ad_assets/pi_zaya_20260609/107_take7_final_1280x800.png`

## One-Minute Short Edit Recommendation

Use the clean v3 MP4 as the source and cut a `55-65 s` short:

1. `00:00-00:05` brand card.
2. `00:14-00:20` library status and quality center.
3. `~03:20-03:45` structured full-library answer.
4. `~05:00-05:25` reference card and entry to basket.
5. `~06:00-06:35` single-page Reader evidence.
6. `~09:20-10:05` in-paper reference to literature basket.
7. `~12:00-12:40` export formats.
8. Last `5 s` final brand card.

For the short, keep only one clear sentence per scene and use faster cuts. Do not show the full generation wait.
