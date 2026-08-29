#!/usr/bin/env python3
"""Full CPU verification of every op in CANN_Ops/report/_manifest.json,
without reconversion. 3 subprocess workers (OOM isolation). Rewrites
CANN_Ops/report/_ingest_report.md. Exit 0 iff zero FAIL.

Usage: verify_all.py
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/root/miniconda3/envs/coding_env/bin/python"

sys.path.insert(0, HERE)
import verify_incr


def main():
    report_dir = os.path.join(HERE, "..", "..", "CANN_Ops", "report")
    manifest = json.load(open(os.path.join(report_dir, "_manifest.json")))
    targets = [(m["level"], m["old_id"], m["op"], int(m["new"].split("_")[2]), m["new"])
               for m in manifest if m["status"] == "ok"]
    print("verify targets: %d" % len(targets), flush=True)

    def run(t):
        level, old_id, op, new_id, base = t
        p = subprocess.run([PY, os.path.join(HERE, "fix_one.py"),
                            level, str(old_id), op, str(new_id), "--verify-only"],
                           capture_output=True, text=True, timeout=1800)
        lines = [l for l in p.stdout.strip().splitlines() if l.startswith("{")]
        if not lines:
            return {"op": base, "status": "FAIL", "struct": [], "review": [],
                    "error": ["subprocess died: " + p.stderr.strip()[-200:]]}
        return json.loads(lines[-1])["result"]

    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run, t): t for t in targets}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print("[%s] %s" % (r["status"], r["op"]), flush=True)

    results.sort(key=lambda r: r["op"])
    with open("/tmp/verify_partial.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_pass, n_rev, n_fail = verify_incr.write_report(results)
    print("CPU FINAL: PASS %d / REVIEW %d / FAIL %d" % (n_pass, n_rev, n_fail))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
