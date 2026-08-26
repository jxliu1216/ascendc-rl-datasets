#!/usr/bin/env python3
"""NPUKernelBench → operator_tasks.jsonl 转换脚本。

将 NPUKernelBench/src 目录下的算子参考实现（level{N}_{id}_{name}.py）转换为
RL 训练数据集（每算子一行）。

用法:
  python3 gen_npukernelbench_tasks.py \
      --benchmark-dir ../../NPUKernelBench \
      --out ../../operator_tasks.npukernelbench.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_OP_FILE = re.compile(r"^level(\d+)_(\d+)_(.+)\.py$")

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


def build(benchmark_dir: Path, out_path: Path,
          arch: str = "ascend910b1", only: set[str] | None = None) -> int:
    if not benchmark_dir.is_dir():
        raise SystemExit(f"benchmark 目录不存在: {benchmark_dir}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for py in sorted(benchmark_dir.glob("*.py")):
            m = _OP_FILE.match(py.name)
            if not m:
                print(f"[warn] 跳过不符合命名规范的文件: {py.name}", file=sys.stderr)
                continue
            level, _id, name = m.group(1), m.group(2), m.group(3)
            op_name = py.stem  # level1_2_SwiGLU：保留 level 前缀，全局唯一
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
                    "data_source": "npu-kernel-bench",
                    "ability": "code",
                    "level": level,
                    "uid": op_name,
                },
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows += 1
    print(f"[gen-npukernelbench] {rows} 个算子 → {out_path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-dir", required=True, type=Path,
                    help="NPUKernelBench 目录（含 level{N}_{id}_{name}.py）")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arch", default="ascend910b1")
    ap.add_argument("--ops", default="",
                    help="逗号分隔的算子名白名单（如 level1_2_SwiGLU），单算子冒烟用")
    args = ap.parse_args()
    only = {o.strip() for o in args.ops.split(",") if o.strip()} or None
    n = build(args.benchmark_dir, args.out, args.arch, only)
    if only and n != len(only):
        raise SystemExit(f"[gen-npukernelbench] --ops 指定 {sorted(only)} 但只生成了 {n} 行，检查算子名")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
