# π-zaya Take7 V4 Improvement Log

Date: 2026-06-10

## New Deliverables

Director's cut:

- File: `docs/ad_assets/pi_zaya_20260609/pi_zaya_ad_take7_directors_cut_9min_1280x800_v4_pi_greek_20260610.mp4`
- Duration: `00:09:10.04`
- Format: MP4 / H.264
- Frame size: `1280x800`
- Frame rate: `25 fps`

Short version:

- File: `docs/ad_assets/pi_zaya_20260609/pi_zaya_ad_take7_short_60s_1280x800_v4_pi_greek_20260610.mp4`
- Duration: `00:00:58.04`
- Format: MP4 / H.264
- Frame size: `1280x800`
- Frame rate: `25 fps`

Both versions passed `blackdetect=d=0.5:pix_th=0.1` with no `black_start` records.

## Improvements Applied

1. Shortened pacing.

   The 9-minute director's cut removes long holds and repetitive pauses from the 15-minute master while preserving the same research workflow.

2. 60-second story cut.

   The short version keeps one product claim: ask the PDF library, verify in Reader, collect references, export for writing.

3. Brand consistency.

   Both new cuts use the corrected Greek brand spelling `π-zaya` from the v3 master.

4. Future recording script improvements.

   The Playwright recording script now:

   - uses `π-zaya` as the default title-card brand,
   - moves subtitles higher on chat pages so they do not cover the composer,
   - keeps Reader subtitles lower and compact,
   - uses a cleaner natural-language Reader evidence snippet for future sessions.

## Remaining Limitation

The v4 cuts are edited from the existing v3 master, so the old Reader evidence strip is still visible in the already-recorded Reader segment. The script is fixed, but removing that artifact from the actual video requires a targeted Reader re-record or a full take8.

## Recommendation

- Use v3 as the complete 15-minute demo master.
- Use v4 director's cut as the main public long-form ad.
- Use v4 short as the social/preview cut.
- Do a targeted take8 only if the Reader evidence strip and subtitle placement must be flawless in the final published long video.

