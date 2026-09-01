#!/usr/bin/env python3
"""算子目录 → operator_tasks.jsonl 通用转换脚本。

将算子数据集目录下的算子参考实现（`{prefix}_level{N}_{id}_{name}.py`）转换为
RL 训练数据集（每算子一行）。对 CANN_Ops、NPUKernelBench 及后续新增的同类目录通用。

目录约定：算子文件为平铺的 {op}.py + {op}.json 成对文件；`--dir` 可传数据集根目录
（自动下探其中的 src/ 子目录）或直接传算子文件所在目录。

前缀与输出自动嗅探：
- 文件名前缀（如 cannops / npukernelbench）从目录内容自动识别（要求全目录统一）
- 输出默认写到类别根目录: {root}/operator_tasks.{prefix}.jsonl

用法:
  python3 utils/gen_tasks/gen_op_tasks.py --dir CANN_Ops
  python3 utils/gen_tasks/gen_op_tasks.py --dir NPUKernelBench
  python3 utils/gen_tasks/gen_op_tasks.py --dir CANN_Ops/src --out /tmp/smoke.jsonl \
      --ops cannops_level1_0_AbsMath
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_OP_FILE = re.compile(r"^([a-z0-9]+)_level(\d+)_(\d+)_(.+)\.py$")

# 前缀 → metadata.data_source 的历史取值映射（保持与已入库 jsonl 兼容）；
# 未登记的新前缀默认取前缀本身，可用 --data-source 覆盖
_DATA_SOURCE_MAP = {
    "npukernelbench": "npu-kernel-bench",
    "cannops": "cann-ops",
}

# prompt 模板：{op_name} 为唯一占位符。
_PROMPT_TEMPLATE = """\
Implement an AscendC operator for Ascend NPU. The reference task is at input/{op_name}.py (class Model + get_input_groups/get_init_inputs). Produce a self-contained project directory `{op_name}/`.

The `{op_name}/` project must contain:
- model_new_ascendc.py (class ModelNew whose forward ONLY calls torch.ops.npu.<op> plus tensor create/reshape; no plain-torch compute)
- kernel/ (op_host/ + op_kernel/ + register.cpp + ops.h + self-contained CMakeLists.txt + setup.py)
You do NOT need to ship model.py: the judge injects the dataset original and overwrites whatever you submit. Do NOT ship build/, dist/, *.so, *.a or *.whl: the judge rebuilds from source in a fresh container and ignores prebuilt artifacts.

Follow the CLAUDE.md workflow: simple ops via the ops-direct-invoke route (Architect design -> Developer implement -> Reviewer review); complex ops via tilelang2ascend-tilelang-designer -> tilelang2ascend-translator. SoC uses SOC_VERSION env (910B2C / A2); CMakeLists paths use x86_64-linux.

Use this fixed validation entry as the only executable validation path:
  bash tools/ascendc_eval_pipeline.sh --op_name {op_name} --impl output/submission/{op_name}_impl.tar.gz --out_dir judge_out
It runs the degradation check, compiles, verifies against the reference, benchmarks, AND packs output/submission/{op_name}_impl.tar.gz for you (keeping a `.best.tar.gz` of your highest-scoring version so far). Run it after EVERY repair iteration — a session that is cut off mid-way still gets graded on its best packed version, so running it early and often strictly dominates saving it for the end.

Rules:
- Read input/{op_name}.py and relevant skill/reference docs only as needed to implement the candidate.
- The graded submission is the single tarball output/submission/{op_name}_impl.tar.gz (or its .best variant); never packed -> scores 0.
- model_new_ascendc.py.forward must call torch.ops.npu.<op> (no plain-torch fallback), else the degradation check fails.
- Do not run custom Python tests, manual import/forward checks, torch.allclose, temporary kernels, npu-smi/environment/API introspection, verifier introspection, or any executable probe that touches the NPU. The fixed validation entry is the only executable validation path, and the only thing allowed to acquire an NPU card. Never set ASCEND_RT_VISIBLE_DEVICES yourself.
- Do not read, modify, inspect, or delete anything under tools/, the verifier scripts under .claude/skills/tilelang2ascend-*/scripts/, or pipeline parameters (SOC_VERSION / warmup / repeats / precision thresholds are fixed by the entry).
- The fixed entry has a call budget; when it prints LIMIT_EXHAUSTED, stop immediately.
- Follow ./CLAUDE.md for the full workflow and judging contract."""


def resolve_dirs(d: Path) -> tuple[Path, Path]:
    """--dir 可传根目录或 src 目录; 返回 (算子文件目录, 类别根目录)。"""
    if not d.is_dir():
        raise SystemExit(f"目录不存在: {d}")
    if (d / "src").is_dir():
        return d / "src", d
    return d, d.parent if d.name == "src" else d


def sniff_prefix(asset_dir: Path) -> str:
    prefixes = Counter()
    for py in sorted(asset_dir.glob("*.py")):
        m = _OP_FILE.match(py.name)
        if m:
            prefixes[m.group(1)] += 1
    if not prefixes:
        raise SystemExit(f"{asset_dir} 下没有符合 {{prefix}}_level{{N}}_{{id}}_{{name}}.py 命名的文件")
    if len(prefixes) > 1:
        raise SystemExit(f"{asset_dir} 文件名前缀不统一: {dict(prefixes)}")
    return prefixes.most_common(1)[0][0]


def build(asset_dir: Path, out_path: Path, prefix: str, data_source: str,
          arch: str, only: set[str] | None = None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for py in sorted(asset_dir.glob("*.py")):
            m = _OP_FILE.match(py.name)
            if not m or m.group(1) != prefix:
                print(f"[warn] 跳过不符合命名规范的文件: {py.name}", file=sys.stderr)
                continue
            level, _id, name = m.group(2), m.group(3), m.group(4)
            op_name = py.stem  # {prefix}_level{N}_{id}_{name}：保留数据集+level 前缀，全局唯一
            if only and op_name not in only:
                continue
            payload = {
                "prompt": [{"role": "user", "content": _PROMPT_TEMPLATE.format(op_name=op_name)}],
                "label": op_name,
                "metadata": {
                    "op_name": op_name,
                    "entry_point": "Model",
                    "operator_backend": "ascendc",
                    "arch": arch,
                    "ops": [name],           # 算子显示名，如 SwiGLU
                    "data_source": data_source,
                    "ability": "code",
                    "level": level,
                    "uid": op_name,
                },
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows += 1
    print(f"[gen-op-tasks] {rows} 个算子 → {out_path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path,
                    help="数据集根目录(自动下探 src/)或算子文件目录")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出 jsonl 路径; 默认 {类别根目录}/operator_tasks.{prefix}.jsonl")
    ap.add_argument("--data-source", default=None,
                    help="metadata.data_source; 默认按前缀映射表取值, 未登记的前缀取前缀本身")
    ap.add_argument("--arch", default="ascend910b1")
    ap.add_argument("--ops", default="",
                    help="逗号分隔的算子名白名单（如 cannops_level1_0_AbsMath），单算子冒烟用")
    args = ap.parse_args()

    asset_dir, root = resolve_dirs(args.dir)
    prefix = sniff_prefix(asset_dir)
    data_source = args.data_source or _DATA_SOURCE_MAP.get(prefix, prefix)
    out = args.out or root / f"operator_tasks.{prefix}.jsonl"
    only = {o.strip() for o in args.ops.split(",") if o.strip()} or None
    n = build(asset_dir, out, prefix, data_source, args.arch, only)
    if only and n != len(only):
        raise SystemExit(f"[gen-op-tasks] --ops 指定 {sorted(only)} 但只生成了 {n} 行，检查算子名")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
