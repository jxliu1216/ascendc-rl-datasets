#!/usr/bin/env python3
"""NPU verification orchestrator: split all CANN_Ops ops into 16 chunks, one
long-lived npu_batch.py subprocess per card, then retry any op with
total <= 0 (crashes) once more. Merges into CANN_Ops/_npu_report.md.

Usage: npu_run16.py [new_base ...]   # default: all 296
"""

import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "CANN_Ops", "src")
REPORT_DIR = os.path.join(HERE, "..", "..", "CANN_Ops", "report")
PY = "/root/miniconda3/envs/coding_env/bin/python"
SET_ENV = "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
N_WORKERS = 16


def run_chunk(worker, bases):
    out_path = "/tmp/npu_chunk_%d.jsonl" % worker
    if os.path.exists(out_path):
        os.remove(out_path)
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(worker)
    cmd = "%s && %s %s --out %s %s" % (
        SET_ENV, PY, os.path.join(HERE, "npu_batch.py"), out_path, " ".join(bases))
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                       timeout=7200, env=env)
    results = []
    if os.path.exists(out_path):
        for line in open(out_path):
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
    seen = {r["op"] for r in results}
    for b in bases:
        if b not in seen:
            results.append({"op": b, "total": -1, "passed": 0,
                            "fails": ["chunk died: " + p.stderr.strip()[-200:]]})
    return results


def sweep(bases, tag):
    chunks = [bases[i::N_WORKERS] for i in range(N_WORKERS)]
    chunks = [c for c in chunks if c]
    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_chunk, w, c): w for w, c in enumerate(chunks)}
        for fut in as_completed(futs):
            results.extend(fut.result())
            print("[%s] chunk done (%d results)" % (tag, len(results)), flush=True)
    return results


def main():
    if len(sys.argv) > 1:
        bases = sys.argv[1:]
    else:
        bases = sorted(os.path.basename(f)[:-3]
                       for f in glob.glob(os.path.join(OUT_DIR, "cannops_*.py")))
    print("total ops: %d" % len(bases), flush=True)
    t0 = time.time()
    results = sweep(bases, "pass1")
    # retry crashed ops once
    crashed = [r["op"] for r in results if r["total"] <= 0]
    if crashed:
        print("retrying %d crashed ops" % len(crashed), flush=True)
        retry = sweep(crashed, "pass2")
        ok = {r["op"]: r for r in retry if r["total"] > 0}
        results = [ok.get(r["op"], r) for r in results]

    results.sort(key=lambda r: r["op"])
    bad = [r for r in results if r["passed"] != r["total"]]
    total_cases = sum(max(r["total"], 0) for r in results)
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
    print("NPU FINAL: ops %d/%d fully passed, cases %d/%d, %.0fs"
          % (len(results) - len(bad), len(results), passed_cases, total_cases,
             time.time() - t0))


if __name__ == "__main__":
    main()
