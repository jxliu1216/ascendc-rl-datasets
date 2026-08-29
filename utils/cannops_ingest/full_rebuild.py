#!/usr/bin/env python3
"""Full rebuild of CANN_Ops/ from cann_ops_tmp/ using the current converter:
every op converted + verified in its own subprocess (3 workers), then
_manifest.json and _ingest_report.md are rewritten from scratch.

Usage: full_rebuild.py
"""

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "..", "..", "cann_ops_tmp")
OUT_DIR = os.path.join(HERE, "..", "..", "CANN_Ops")
PY = "/root/miniconda3/envs/coding_env/bin/python"

sys.path.insert(0, HERE)
import verify_incr  # for write_report


def collect_ops():
    ops = []
    for level in ("level1", "level2", "level3"):
        lvl_dir = os.path.join(SRC_DIR, level)
        lvl = []
        for fn in os.listdir(lvl_dir):
            m = re.match(r"^(\d+)_(.+)\.py$", fn)
            if m:
                lvl.append((int(m.group(1)), m.group(2)))
        lvl.sort()
        for new_id, (old_id, op) in enumerate(lvl):
            ops.append((level, old_id, op, new_id))
    return ops


def run_one(t):
    level, old_id, op, new_id = t
    try:
        p = subprocess.run(
            [PY, os.path.join(HERE, "fix_one.py"), level, str(old_id), op, str(new_id)],
            capture_output=True, text=True, timeout=1800)
        lines = p.stdout.strip().splitlines()
        if not lines:
            return t, None, {"op": "%s/%d_%s" % (level, old_id, op), "status": "FAIL",
                             "struct": [], "review": [],
                             "error": ["subprocess died (OOM?): " + p.stderr.strip()[-300:]]}
        payload = json.loads(lines[-1])
        return t, payload["convert_ok"], payload["result"]
    except Exception as e:
        return t, None, {"op": "%s/%d_%s" % (level, old_id, op), "status": "FAIL",
                         "struct": [], "review": [], "error": ["runner: " + repr(e)]}


def main():
    # clean previous outputs
    for fn in os.listdir(OUT_DIR):
        if fn.startswith("cannops_") or fn in ("_manifest.json", "_ingest_report.md"):
            os.remove(os.path.join(OUT_DIR, fn))
    ops = collect_ops()
    print("total ops: %d" % len(ops), flush=True)

    manifest, results = [], []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run_one, t): t for t in ops}
        for fut in as_completed(futs):
            (level, old_id, op, new_id), convert_ok, r = fut.result()
            base = "cannops_%s_%d_%s" % (level, new_id, op)
            manifest.append({"level": level, "old_id": old_id, "op": op,
                             "new": base, "status": "ok" if convert_ok else "fail"})
            results.append(r)
            print("[%s] %s" % (r["status"], r["op"]), flush=True)

    manifest.sort(key=lambda m: (m["level"], m["old_id"]))
    with open(os.path.join(OUT_DIR, "_manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open("/tmp/verify_partial.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_pass, n_rev, n_fail = verify_incr.write_report(results)
    print("FINAL: PASS %d / REVIEW %d / FAIL %d" % (n_pass, n_rev, n_fail))


if __name__ == "__main__":
    main()
