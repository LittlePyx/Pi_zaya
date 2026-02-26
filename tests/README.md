# Python 测试覆盖指南 (Tests)

本目录包含了对于 `kb` 模块以及 PDF 转换核心流程的自动化测试。

## 测试结构

- **`unit/` (单元测试)**
  包含了针对特定函数和小型模块逻辑的不依赖外部接口的快速执行测试：
  - **`test_config.py`**：配置文件解析模块（校验环境变量读取、默认值的解析和兼容处理等）。
  - **`test_chunking.py`**：文本分块逻辑测试（确保大文本能正确按照规则切片）。
  - **`test_converter_pipeline.py`**：核心转换流水分发测试。
  - **`test_geometry_utils.py`**：版面底层几何工具测试（包围盒的宽高、面积、交并比运算等维度验证）。
  - **`test_block_classifier.py`**：解析器核心块分类启发式测试（校验大段文本、特殊坐标能否正确判定为表格、代码、数学公式或目录 TOC）。
  - **`test_heuristics_metadata.py`**：测试元数据提取（如从文档尝试提取出作者、日期等推断信息）的准确性。
  - **`test_layout_visual_rects.py`**：版面布局坐标/可视化包围盒测试（这通常与 PDF/图片解析定位有关）。
  - **`test_markdown_analyzer.py`**：Markdown 格式分析器（拦截、发现如未闭合公式或残缺表格的警告）。
  - **`test_pipeline_math_guardrails.py`**：数学公式护栏机制（确保生成阶段的 LaTeX 代码完好）。
  - **`test_post_processing_captions.py`**：Markdown 结果的题注和图表标题清理格式测试。
  - **`test_post_processing_citations.py`**：引用文本及序号的合并、识别与后处理。
  - **`test_post_processing_headings.py`**：将裸露的标题格式推算并升级为 Markdown 各级 Headings 的逻辑。
  - **`test_post_processing_math_unwrap.py`**：负责把行内外的数学公式块多余包裹解开的准确度测试。
  - **`test_post_processing_references.py`**：文档参考文献抽取与排版还原测试。
  - **`test_post_processing_typora_math.py`**：对于 Typora 及其他主流 Markdown 阅读器标准数学公式块的兼容替换。
  - **`test_task_runtime_bg_task.py`**：测试异步任务或后台队列系统的稳定性。
  - **`test_text_utils.py`**：针对普通字符串裁剪、去重、清洗的函数覆盖。

- **`sanity/` (健康度/集成测试)**
  用于跨模块或带真实运行框架的启停端到端粗筛：
  - **`test_app_startup.py`**：测试 App 能否正常加载依赖、不报错地启动初始化流程（包括所有环境变量及懒加载路径注册）。

## 如何运行测试

得益于根目录下的 `pytest.ini`，您现在可以随时在终端安全地执行测试，且不会误执行或采集主项目根目录中同名的开发脚手架脚本（如 `test_converter.py`）。

直接运行全量测试：
```powershell
python -m pytest tests/
或
pytest
```

## 常见问题 (FAQ)

### 为什么根目录下的运行脚本（如 `test_converter.py`）不能当做测试？
根目录下的以 `test_` 打头的 `.py` 脚本最初设计为 **独立命令行的入口工具**，它们不仅具有例如 `test_convert(pdf_path: str ...)` 这样需要传参的入口函数，而且其逻辑是用于真实转换体验而非自动化断言。
依靠这个配置，`pytest` 只会老老实实地去 `tests/` 文件夹下方搜集代码，而不会错误提取并引发找不到参数配置 (`fixture not found`) 的 ERROR。
因此建议永远遵循 `python -m pytest tests/` 或单纯 `pytest` (由于配置文件已经被约束) 来执行测试过程。
