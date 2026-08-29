#!/usr/bin/env python3
"""Run all CANN_Ops operators' reference Models on NPU, all cases.
3 workers pinned to different cards via ASCEND_RT_VISIBLE_DEVICES.
Writes CANN_Ops/_npu_report.md.

Usage: npu_run.py [new_base ...]   # default: all 296
"""

import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "CANN_Ops", "src")
REPORT_DIR = os.path.join(HERE, "..", "..", "CANN_Ops", "report")
PY = "/root/miniconda3/envs/coding_env/bin/python"
SET_ENV = "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
N_WORKERS = 16


def run_one(base, worker):
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(worker)
    for attempt in (0, 1):
        try:
            p = subprocess.run(
                ["bash", "-c", "%s && %s %s %s" % (SET_ENV, PY, os.path.join(HERE, "npu_one.py"), base)],
                capture_output=True, text=True, timeout=1800, env=env)
            lines = p.stdout.strip().splitlines()
            if not lines:
                r = {"op": base, "total": 0, "passed": 0,
                     "fails": ["subprocess died: " + p.stderr.strip()[-200:]]}
            else:
                r = json.loads(lines[-1])
            if r["total"] > 0 or attempt == 1:
                return r  # retry once on transient harness crash (0/0)
        except Exception as e:
            r = {"op": base, "total": 0, "passed": 0, "fails": ["runner: " + repr(e)]}
    return r


def main():
    if len(sys.argv) > 1:
        bases = sys.argv[1:]
    else:
        bases = sorted(os.path.basename(f)[:-3]
                       for f in glob.glob(os.path.join(OUT_DIR, "cannops_*.py")))
    print("total ops: %d" % len(bases), flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {}
        for i, base in enumerate(bases):
            futs[ex.submit(run_one, base, i % N_WORKERS)] = base
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print("[%d/%d] %s %s" % (r["passed"], r["total"], r["op"],
                                     "" if r["passed"] == r["total"] else "FAIL"), flush=True)

    results.sort(key=lambda r: r["op"])
    bad = [r for r in results if r["passed"] != r["total"]]
    total_cases = sum(r["total"] for r in results)
    passed_cases = sum(r["passed"] for r in results)
    lines = ["# CANN_Ops NPU 全用例运行报告", "",
             "| 项目 | 数量 |", "|---|---|",
             "| 算子总数 | %d |" % len(results),
             "| 全部用例通过的算子 | %d |" % (len(results) - len(bad)),
             "| 存在失败用例的算子 | %d |" % len(bad),
             "| 用例通过率 | %d/%d |" % (passed_cases, total_cases), ""]
    if bad:
        lines += ["## 存在失败的算子", "", "| 算子 | 通过/总数 | 首个错误 |", "|---|---|---|"]
        for r in bad:
            first = (r["fails"][0] if r["fails"] else "").replace("|", "\\|")[:150]
            lines.append("| %s | %d/%d | %s |" % (r["op"], r["passed"], r["total"], first))
        lines.append("")
    with open(os.path.join(REPORT_DIR, "_npu_report.md"), "w") as f:
        f.write("\n".join(lines))
    with open("/tmp/npu_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("NPU FINAL: ops %d/%d fully passed, cases %d/%d"
          % (len(results) - len(bad), len(results), passed_cases, total_cases))


if __name__ == "__main__":
    main()
