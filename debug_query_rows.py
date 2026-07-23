#!/usr/bin/env python3
"""debug _query_rows — 逐步展示 mx-smi 输出被解析的每一步"""

import re
import subprocess
import sys
import pdb


# ── 运行 mx-smi ──
proc = subprocess.run(["mx-smi"], capture_output=True, text=True, timeout=5, check=False)
output = proc.stdout
# pdb.set_trace()
# ── 第 1 步：提取 table_lines ──
all_lines = output.splitlines()
table_lines = [line for line in all_lines if line.strip().startswith("|")]

print("=" * 80)
print(f"第 1 步：提取以 | 开头的行")
print(f"  总行数: {len(all_lines)}, table_lines: {len(table_lines)}")
print("=" * 80)
print()

# ── 第 2 步：逐两行解析 GPU ──
GPU_RE = re.compile(r"\|\s*(\d+)\s+")
PERCENT_RE = re.compile(r"(\d+)%\s+(?:Disabled|Enabled)")
MEM_RE = re.compile(r"(\d+)/(\d+)\s*MiB")
SKIP_KW = re.compile(r"GPU|Pwr|Process|Memory-Usage|Usage\(MiB\)|PID|Board|Kernel|MACA|MX-SMI")

idx = 0
gpu_count = 0
skipped = 0
process_rows = 0

while idx + 1 < len(table_lines):
    gpu_line = table_lines[idx]
    mem_line = table_lines[idx + 1]

    # 检查是否应该跳过
    gpu_match = GPU_RE.match(gpu_line)
    if not gpu_match:
        print(f"[跳过] idx={idx:2d}  行首无 GPU index")
        print(f"  gpu_line: {gpu_line[:100]}")
        print()
        idx += 1
        skipped += 1
        continue

    # 检查是否是 Process 行（有 GPU index 但无 utilization%）
    util_match = PERCENT_RE.search(gpu_line)
    mem_match = MEM_RE.search(mem_line)

    if not util_match and not mem_match:
        print(f"[跳过] idx={idx:2d}  疑似 Process 行（有 index 但无 % 和 MiB）")
        print(f"  gpu_line: {gpu_line[:100]}")
        print(f"  mem_line: {mem_line[:100]}")
        print(f"  GPU_RE → {gpu_match.group(0)!r}")
        print()
        idx += 2
        process_rows += 1
        continue

    if not mem_match:
        print(f"[跳过] idx={idx:2d}  mem_line 无 MiB 匹配（可能跨阶段行）")
        print(f"  gpu_line: {gpu_line[:100]}")
        print(f"  mem_line: {mem_line[:100]}")
        print()
        idx += 2
        skipped += 1
        continue

    # ✅ 匹配成功
    gpu_index = gpu_match.group(1)
    used_mb = mem_match.group(1)
    total_mb = mem_match.group(2)
    util_pct = util_match.group(1) if util_match else "0"

    print(f"[✓ 匹配] idx={idx:2d}  →  GPU {gpu_index}")
    print(f"  gpu_line  : {gpu_line.strip()}")
    print(f"  mem_line  : {mem_line.strip()}")
    print(f"  GPU_RE    : {gpu_match.group(0)!r}  →  index={gpu_index}")
    print(f"  PERCENT_RE: {util_match.group(0) if util_match else 'N/A'}  →  util={util_pct}%")
    print(f"  MEM_RE    : {mem_match.group(0)!r}  →  used={used_mb} MiB, total={total_mb} MiB")
    print(f"  memory_pct: {100.0 * int(used_mb) / int(total_mb):.1f}%")
    print()

    gpu_count += 1
    idx += 2

print("=" * 80)
print(f"汇总: 匹配 GPU={gpu_count}, 跳过={skipped}, Process行={process_rows}")
print("=" * 80)

# ── 第 3 步：展示所有 table_lines 的分类 ──
print()
print("─" * 80)
print("所有 table_lines 分类一览")
print("─" * 80)

for i, line in enumerate(table_lines):
    gpu_match = GPU_RE.match(line)
    util_match = PERCENT_RE.search(line)
    mem_match = MEM_RE.search(line)
    has_pct = PERCENT_RE.search(line)

    tags = []
    if not gpu_match:
        tags.append("无GPU index")
    else:
        tags.append(f"GPU {gpu_match.group(1)}")
    if has_pct:
        tags.append(f"util={has_pct.group(1)}%")
    if mem_match:
        tags.append(f"mem={mem_match.group(1)}/{mem_match.group(2)} MiB")
    if re.search(r"Process|PID|GPU Memory", line):
        tags.append("PROCESS区")

    tag_str = " | ".join(tags) if tags else "无匹配"
    print(f"  [{i:2d}] {tag_str}")
    print(f"       {line[:120]}")
