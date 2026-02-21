#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量测试PDF转换器 - 使用research-paper-pyx目录中的文件
"""
import sys
import os
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from kb.converter.pipeline import PDFConverter
from kb.converter.config import ConvertConfig, LlmConfig

def test_convert(pdf_path: str, output_dir: str = None, use_llm: bool = False):
    """测试PDF转换"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"错误: PDF文件不存在: {pdf_path}")
        return False
    
    if output_dir is None:
        # 使用PDF文件名作为输出目录名
        output_dir = f"test_output_{pdf_path.stem}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"正在转换: {pdf_path.name}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}")
    
    try:
        llm_config = None
        if use_llm:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                print("警告: 未设置API密钥，禁用LLM")
                use_llm = False
            else:
                base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.openai.com/v1"
                model = os.environ.get("OPENAI_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "gpt-4o-mini"
                llm_config = LlmConfig(
                    api_key=api_key,
                    base_url=base_url,
                    model=model
                )
        
        # 创建配置
        cfg = ConvertConfig(
            pdf_path=pdf_path,
            out_dir=output_dir,
            translate_zh=False,
            start_page=0,
            end_page=-1,
            skip_existing=False,
            keep_debug=True,
            llm=llm_config
        )
        
        # 创建转换器
        converter = PDFConverter(cfg)
        converter.dpi = 200
        converter.analyze_quality = True
        
        # 执行转换
        converter.convert(str(pdf_path), str(output_dir))
        
        print(f"\n[✓] 转换完成!")
        print(f"  输出文件: {output_dir / 'output.md'}")
        if (output_dir / "quality_report.md").exists():
            print(f"  质量报告: {output_dir / 'quality_report.md'}")
        
        return True
    except Exception as e:
        print(f"\n[✗] 转换失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数 - 批量测试"""
    pdf_dir = Path(r"F:\research-papers\research-paper-pyx")
    
    if not pdf_dir.exists():
        print(f"错误: 目录不存在: {pdf_dir}")
        sys.exit(1)
    
    # 获取所有PDF文件
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"错误: 在 {pdf_dir} 中未找到PDF文件")
        sys.exit(1)
    
    print(f"找到 {len(pdf_files)} 个PDF文件")
    print("\n文件列表:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf.name}")
    
    # 检查是否使用LLM
    use_llm = "--llm" in sys.argv
    
    # 检查是否只测试一个文件
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        # 测试指定的文件
        pdf_name = sys.argv[1]
        pdf_file = pdf_dir / pdf_name
        if not pdf_file.exists():
            print(f"错误: 文件不存在: {pdf_file}")
            sys.exit(1)
        test_convert(pdf_file, use_llm=use_llm)
    else:
        # 批量测试所有文件
        print(f"\n开始批量转换 (LLM: {'启用' if use_llm else '禁用'})...")
        
        success_count = 0
        fail_count = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}]")
            if test_convert(pdf_file, use_llm=use_llm):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n{'='*80}")
        print(f"批量转换完成!")
        print(f"  成功: {success_count}/{len(pdf_files)}")
        print(f"  失败: {fail_count}/{len(pdf_files)}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
