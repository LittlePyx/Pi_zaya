# P0 Baseline

更新时间：2026-03-06

## 1. 执行环境

1. 分支：`main`
2. commit：`38e3886`
3. Python：`3.10.11`
4. Node：`24.13.0`
5. npm：`11.6.2`

## 2. 基线检查结果（2026-03-06 12:00 CST）

### 2.1 环境体检

命令：

```powershell
.\tools\stability\doctor.ps1 -Strict
```

结果：`PASS`

### 2.2 前端构建

命令：

```powershell
cd web
npm run build
```

结果：`PASS`  
备注：存在 chunk size 警告（非阻断）。

### 2.3 后端单测

命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -q
```

结果：`PASS`（`118 passed, 6 warnings`）

## 3. 当前结论

1. P0/WP0 基线校验通过，可进入 `WP1`（转换链路映射正确性）。
2. 后续每次阶段性提交前，至少重复执行本页 3 个命令并记录结果。
