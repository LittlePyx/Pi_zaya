# Pi_zaya 广告 take23 页面墙版审查与发布文案

## 最新交付

- 成片：`docs/ad_assets/pi_zaya_20260609/take23_tech_agent_pagewall_2x_png_20260611/pi_zaya_ad_take23_tech_agent_pagewall_sapi_v2_bilingual_2560x1800_h264420_crf8_20260611.mp4`
- 审查图：`docs/ad_assets/pi_zaya_20260609/take23_tech_agent_pagewall_2x_png_20260611/review_frames/take23_pagewall_review_sheet.jpg`
- 封面图：`docs/ad_assets/pi_zaya_20260609/take23_tech_agent_pagewall_2x_png_20260611/pi_zaya_take23_pagewall_cover_1920x1080.jpg`
- 画面生成脚本：`docs/ad_assets/pi_zaya_20260609/build_take23_tech_agent.py`
- 分镜配置：`docs/ad_assets/pi_zaya_20260609/take23_tech_agent_blocks.json`

## 结构

1. Logo 开场：PDF 文献库的 research agent。
2. 页面快速扫过：知识库回答、原文证据、参考定位、文献篮、下一轮写作。
3. 页面墙：多个真实页面同屏，不使用流程图。
4. 转换成知识库：强调转换效果进入可检索、可问答、可定位材料；质量中心只作为状态背书。
5. 新提问：问题交给整个知识库，而不是单篇 PDF。
6. Agent 工作：解析、检索、阅读、组织，结论旁边带来源入口。
7. 原文证据：点击来源编号跳回论文证据。
8. 参考定位卡片：解释参考了哪些文章、为什么相关、支撑哪个判断。
9. 文献篮：收录、看概要、看来源标签、用于下一轮。
10. 收尾：Ask the library / Trace evidence / Collect papers / Continue writing。

## 本轮对齐用户反馈

- 去掉流程图式表达，改成多个真实功能页面并置的科技广告布局。
- 不再强调 Reader 图注和解释。
- 不详细展开质量中心，只作为可信状态一闪而过。
- 强化知识库回答、原文证据跳转、参考定位卡片、文献篮收录和下一轮上下文。
- 画面中加入扫描线、页面卡片、局部标注、轻微浮动与页面墙布局。

## 注意

当前 v2 成片使用 Windows 本地中文语音生成，优点是声线统一且不受 Edge 网络超时影响；缺点是自然度不如 Edge 或真人配音。若要发布正式版，建议只替换音轨，画面结构可以继续沿用这一版。

## 发布标题

1. 把 PDF 文献库变成 research agent：π-zaya 页面墙演示
2. 不只是读 PDF：回答、证据、参考定位和文献篮在同一工作台
3. 给一批 PDF，一个可问、可查、可收录的研究 agent
4. π-zaya：让文献库能回答，也能回到证据

## 发布正文

π-zaya 不是单篇 PDF 聊天，也不是普通阅读器。

它把一批 PDF 转成可检索、可问答、可定位的知识库。提问时，agent 会围绕整个文献库检索和组织回答；点击来源编号，可以跳回原文证据；参考定位卡片会解释参考了哪些文章、为什么相关；有用文献可以收录进文献篮，并作为下一轮写作上下文。

这一版用页面墙展示真实功能页面：回答、证据、参考、收录，都围绕同一批 PDF 工作。

## 短文案

一批 PDF，不只是被读完。  
π-zaya 让文献库变成 research agent：能回答、能跳回原文证据、能解释参考了哪些文章，也能把有用文献收进下一轮上下文。

## 标签

`#AI科研` `#ResearchAgent` `#文献管理` `#PDF知识库` `#RAG` `#论文写作` `#引用管理` `#知识库问答`
