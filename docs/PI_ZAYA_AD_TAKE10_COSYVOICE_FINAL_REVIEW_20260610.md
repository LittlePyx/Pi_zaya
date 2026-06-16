# π-zaya Take10 CosyVoice 成片验收记录

日期：2026-06-10

## 成片文件

| 类型 | 路径 |
|---|---|
| 最终带语音成片 | `docs/ad_assets/pi_zaya_20260609/pi_zaya_ad_take10_cosyvoice_effects_11min_1280x800_h264_20260610.mp4` |
| CosyVoice 时间轴音轨 | `docs/ad_assets/pi_zaya_20260609/take10_cosyvoice_voiceover_delayed_11min.wav` |
| 逐句中文字幕 | `docs/ad_assets/pi_zaya_20260609/take10_cosyvoice_aligned_zh.srt` |
| 完整旁白稿 | `docs/ad_assets/pi_zaya_20260609/take10_voiceover_script_zh.md` |
| 语音生成脚本 | `docs/ad_assets/pi_zaya_20260609/generate_take10_cosyvoice.py` |
| 视觉/字幕资产脚本 | `docs/ad_assets/pi_zaya_20260609/create_take10_voice_assets.py` |
| 最终联系表 | `docs/ad_assets/pi_zaya_20260609/reader_conversion_quality_v2_frames_20260610/take10_cosyvoice_final_contact_20s.png` |

## 严格审查结论

通过，作为当前广告长版主成片候选。

本版解决了上一轮最明显的问题：

- 旁白从约 1000 字扩展到约 3013 字，真实说话时长约 587.87 秒，信息密度更接近正式产品讲解。
- 使用 DashScope CosyVoice 生成中文旁白，替代本机系统语音；语气设置为自然、专业、温和的产品演示。
- 逐句字幕从原来的整段长字幕改成 93 条短句字幕，字幕不再长时间挂屏。
- 章节导航、功能高亮、引用卡、文献篮、Reader 单页展开、转换质量证明、限定范围问答、写作准备都进入故事线。
- 品牌名文本使用希腊字符 `π-zaya`；语音合成文本中将其读作“派 zaya”，避免 TTS 误读符号。

## 技术验收

`ffprobe`：

- 分辨率：1280x800
- 帧率：25fps
- 时长：689.92 秒
- 视频：H.264
- 音频：AAC，48 kHz，mono，689.92 秒
- 字幕：mov_text，中文轨，686.203 秒

自动检测：

- 黑屏检测：无 `black_start` 输出
- 8 秒以上静音检测：无输出
- 字符检查：`π-zaya` 存在；无 `蟺-zaya`；无替换字符 `�`

## 仍建议人工听感复核

CosyVoice 的效果已经明显优于本机系统 TTS，但最终是否达到“像真人讲的”仍建议人工听一遍，重点检查：

- `PDF`、`DOI`、`Reader`、`Markdown` 的读法是否自然。
- “派 zaya”的品牌读法是否符合预期。
- 个别长句是否需要拆短，以减少合成腔。
- 是否需要加入很低音量的品牌背景音乐。当前版本没有加音乐，避免遮盖语音和操作声。

## 成片叙事

故事线为：

1. 开场说明 π-zaya 是完整研究工作流，不是单篇 PDF 聊天。
2. Library 展示多论文入库、状态、筛选与管理。
3. 质量中心建立转换可信度。
4. Reader 用含公式、图片、表格、参考文献的论文证明转换质量。
5. 回到正式研究问题，展示全库 RAG 问答。
6. 展开引用卡，证明答案可追溯到 DOI、期刊和原文。
7. 多篇文献加入文献篮，形成研究上下文。
8. Reader 单页展开定位证据、高亮、摘录。
9. 文献篮限定范围继续问答，最后进入写作前材料整理。
