# P0 Release Checklist

更新时间：2026-03-06

## A. 环境与构建

1. `.\tools\stability\doctor.ps1 -Strict` 通过
2. 前端构建通过：`cd web && npm run build`
3. 单测通过：`.venv\Scripts\python.exe -m pytest tests/unit -q`
4. CI 检查通过（GitHub Actions 全绿）

## B. 文献篮一致性

1. Case-01 通过（切会话不丢失）
2. Case-02 通过（切 Chat/Library 往返不丢失）
3. Case-03 通过（刷新后不降级）
4. 无“跨会话串写”现象

## C. 切换稳定性

1. Case-04 通过（30-50 次切换无假死）
2. 页面交互无明显阻塞
3. 无异常错误弹窗/控制台关键报错

## D. 引用与映射正确性

1. Case-05 通过（图/公式定位正确）
2. NatPhoton 类已知样本回归通过
3. 引用详情字段可追溯（source/section/snippet 正常）

## E. 跨平台与路径

1. Case-06 通过（Windows 样式路径解析正常）
2. 与路径相关单测通过
3. 不出现 CI 专有失败

## F. 发布前记录

1. 本次 commit hash
2. 执行人
3. 执行日期
4. 失败项（若有）与处理结论

## G. 准入规则

1. A/B/C/D/E 任一失败，不可合并/发布。
2. P0 阶段至少连续 3 轮 checklist 全通过，才视为稳定封口。
