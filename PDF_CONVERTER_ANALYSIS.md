# PDF转MD功能重构分析报告

## 📋 当前情况

### 1. 架构概览

协作者将PDF转MD功能重构为一个模块化的 `kb/converter` 包，包含以下核心组件：

```
kb/converter/
├── __init__.py          # 导出主要接口
├── config.py            # 配置类 (ConvertConfig, LlmConfig)
├── pipeline.py          # 主转换流程 (PDFConverter)
├── llm_worker.py       # LLM处理逻辑
├── models.py            # 数据模型 (TextBlock)
├── block_classifier.py  # 块分类器
├── layout_analysis.py  # 布局分析
├── tables.py           # 表格提取
├── post_processing.py  # 后处理
├── geometry_utils.py   # 几何工具
├── text_utils.py       # 文本工具
├── heuristics.py       # 启发式规则
└── runner.py           # CLI入口
```

### 2. 核心流程

**主转换流程** (`pipeline.py`):
1. 打开PDF文档
2. 预扫描噪声文本 (`build_repeated_noise_texts`)
3. 逐页处理：
   - 分析布局 (`detect_body_font_size`, `_collect_visual_rects`)
   - 提取表格 (`_extract_tables_by_layout`)
   - 提取文本块 (`_extract_text_blocks`)
   - LLM增强（可选）(`_enhance_blocks_with_llm`)
   - 渲染为Markdown (`_render_blocks_to_markdown`)
4. 后处理 (`postprocess_markdown`)
5. 保存到 `output.md`

### 3. 当前问题

#### ❌ 问题1: 未集成到主流程
- `kb/pdf_tools.py` 的 `run_pdf_to_md()` 仍在使用旧的 `test2.py`
- 新模块 `kb/converter` 尚未被主应用调用
- 存在两套并行的转换系统

#### ❌ 问题2: 代码不完整
- `pipeline.py:158` 使用了 `_rect_intersection_area` 但未导入
- `pipeline.py:259` 尝试修改 `TextBlock.heading_level`，但 `TextBlock` 是 `BaseModel`（不可变）
- `_process_batch_llm` 只是简单调用 `_process_batch_fast`，未实现真正的LLM并行处理
- `_enhance_blocks_with_llm` 中的块修改逻辑有问题（尝试修改不可变对象）

#### ❌ 问题3: 配置不匹配
- `ConvertConfig` 需要 `pdf_path`, `out_dir` 等字段，但 `runner.py` 只设置了部分字段
- `LlmConfig` 是 `frozen` dataclass，但 `runner.py` 尝试创建 `cfg.LLMConfig()`（应该是 `LlmConfig`）

#### ❌ 问题4: 功能缺失
- 缺少进度回调支持
- 缺少取消机制
- 缺少错误恢复
- 缺少调试输出选项
- 缺少页面范围选择（`start_page`, `end_page`）

#### ⚠️ 问题5: 性能问题
- 逐页顺序处理，未利用并行处理
- 图像提取使用固定DPI (200)，未使用配置中的 `dpi`
- 缺少缓存机制

## 🔧 优化建议

### 优先级1: 修复关键Bug

1. **修复导入问题**
   ```python
   # pipeline.py 需要添加
   from .geometry_utils import _rect_intersection_area
   ```

2. **修复不可变对象修改**
   ```python
   # 方案A: 使用 Pydantic 的 copy 方法
   # 方案B: 重构为可变模型
   # 方案C: 返回新对象列表
   ```

3. **修复配置创建**
   ```python
   # runner.py 中应该使用
   from .config import LlmConfig
   cfg.llm = LlmConfig(...)  # 而不是 cfg.LLMConfig()
   ```

### 优先级2: 集成到主流程

1. **在 `kb/pdf_tools.py` 中添加新转换器选项**
   ```python
   def run_pdf_to_md(...):
       # 优先使用新转换器
       if use_new_converter:
           from kb.converter import PDFConverter, ConvertConfig, LlmConfig
           # 构建配置并调用
       else:
           # 回退到旧 test2.py
   ```

2. **保持向后兼容**
   - 保留 `test2.py` 作为fallback
   - 通过环境变量或配置选择转换器

### 优先级3: 完善功能

1. **添加进度回调**
   ```python
   def convert(self, ..., progress_cb=None):
       for i, page in enumerate(doc):
           if progress_cb:
               progress_cb(i+1, total_pages, f"Processing page {i+1}")
   ```

2. **实现真正的并行处理**
   ```python
   def _process_batch_llm(self, doc, ...):
       with ThreadPoolExecutor(max_workers=self.cfg.llm_workers) as ex:
           futures = {ex.submit(self._process_page, ...): i 
                     for i, page in enumerate(doc)}
           # 收集结果
   ```

3. **添加页面范围支持**
   ```python
   start = max(0, self.cfg.start_page)
   end = min(total_pages, self.cfg.end_page if self.cfg.end_page >= 0 else total_pages)
   pages = list(doc)[start:end]
   ```

4. **使用配置的DPI**
   ```python
   pix = page.get_pixmap(clip=rect, dpi=self.cfg.dpi or 200)
   ```

### 优先级4: 代码质量

1. **统一错误处理**
   - 添加异常捕获和日志
   - 提供有意义的错误消息

2. **添加类型提示**
   - 完善所有函数的类型注解

3. **添加文档字符串**
   - 为所有公共方法添加docstring

4. **添加单元测试**
   - 为核心功能添加测试用例

## 📝 具体修复步骤

### Step 1: 修复导入和类型错误
- [ ] 修复 `_rect_intersection_area` 导入
- [ ] 修复 `TextBlock` 不可变问题
- [ ] 修复 `LlmConfig` 创建问题

### Step 2: 集成到主流程
- [ ] 在 `kb/pdf_tools.py` 中添加新转换器调用
- [ ] 添加配置选项选择转换器
- [ ] 测试集成

### Step 3: 完善功能
- [ ] 添加进度回调
- [ ] 实现并行处理
- [ ] 添加页面范围支持
- [ ] 使用配置的DPI

### Step 4: 优化和测试
- [ ] 性能测试
- [ ] 错误处理测试
- [ ] 与旧系统对比测试

## 🎯 建议的实施顺序

1. **立即修复**: Bug修复（优先级1）
2. **短期**: 集成到主流程（优先级2）
3. **中期**: 功能完善（优先级3）
4. **长期**: 代码质量和测试（优先级4）

## 📊 代码统计

- **新模块代码量**: ~2900行（14个文件）
- **旧系统代码量**: ~2000行（test2.py中的PdfToMarkdown类）
- **重构优势**: 模块化、可维护、可测试
- **当前状态**: 功能基本完整，但存在集成和Bug问题
