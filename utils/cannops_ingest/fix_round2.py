#!/usr/bin/env python3
"""Orchestrate per-op re-conversion+verification in separate processes.

Targets: ops that are FAIL in the latest /tmp/verify_partial.jsonl state or
failed at convert time in CANN_Ops/_manifest.json. Each target runs in a fresh
subprocess (fix_one.py) so a crash/OOM only kills one op. Results are appended
to /tmp/verify_partial.jsonl incrementally; the final report dedupes by op
keeping the LAST entry.

Usage: fix_round2.py [op ...]   # default: all current FAILs
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARTIAL = "/tmp/verify_partial.jsonl"
PY = "/root/miniconda3/envs/coding_env/bin/python"

sys.path.insert(0, HERE)
import verify_incr  # for write_report


def load_state():
    results = {}
    if os.path.exists(PARTIAL):
        for line in open(PARTIAL):
            line = line.strip()
            if line:
                r = json.loads(line)
                results[r["op"]] = r
    return results


def main():
    out_dir = verify_incr.verify.OUT_DIR
    manifest = json.load(open(os.path.join(out_dir, "_manifest.json")))
    results = load_state()

    # new_id per (level, old_id, op): sorted order within level
    by_level = {}
    for m in manifest:
        by_level.setdefault(m["level"], []).append(m)
    new_ids = {}
    for level, ops in by_level.items():
        ops.sort(key=lambda x: x["old_id"])
        for i, m in enumerate(ops):
            new_ids[(m["level"], m["old_id"], m["op"])] = i

    only = set(sys.argv[1:])
    targets = []
    for m in manifest:
        key = (m["level"], m["old_id"], m["op"])
        base = m.get("new")
        is_fail = m["status"] != "ok" or (base in results and results[base]["status"] == "FAIL") \
                  or ("%s/%d_%s" % (m["level"], m["old_id"], m["op"]) in results and
                      results["%s/%d_%s" % (m["level"], m["old_id"], m["op"])]["status"] == "FAIL")
        if only:
            is_fail = (base in only) or ("%s/%d_%s" % key in only) or (m["op"] in only)
        if is_fail:
            targets.append((m["level"], m["old_id"], m["op"], new_ids[key]))

    print("targets: %d" % len(targets), flush=True)
    out = open(PARTIAL, "a")

    def run_one(t):
        level, old_id, op, new_id = t
        try:
            p = subprocess.run(
                [PY, os.path.join(HERE, "fix_one.py"), level, str(old_id), op, str(new_id)],
                capture_output=True, text=True, timeout=1800)
            lines = p.stdout.strip().splitlines()
            if not lines:
                return {"op": "%s/%d_%s" % (level, old_id, op), "status": "FAIL",
                        "struct": [], "review": [],
                        "error": ["subprocess died (OOM?): " + p.stderr.strip()[-300:]]}
            try:
                return json.loads(lines[-1])["result"]
            except Exception:
                return {"op": "%s/%d_%s" % (level, old_id, op), "status": "FAIL",
                        "struct": [], "review": [],
                        "error": ["bad fix_one output: " + (p.stdout + p.stderr)[-300:]]}
        except Exception as e:
            return {"op": "%s/%d_%s" % (level, old_id, op), "status": "FAIL",
                    "struct": [], "review": [], "error": ["runner: " + repr(e)]}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run_one, t): t for t in targets}
        for fut in as_completed(futs):
            r = fut.result()
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            print("[%s] %s" % (r["status"], r["op"]), flush=True)
    out.close()

    results = load_state()
    # drop convert-fail alias keys whose op now has a passing new_base entry
    aliases = {}
    for m in manifest:
        if m["status"] == "ok" and m.get("new"):
            aliases["%s/%d_%s" % (m["level"], m["old_id"], m["op"])] = m["new"]
    for alias, base in aliases.items():
        if alias in results and base in results:
            del results[alias]
    n_pass, n_rev, n_fail = verify_incr.write_report(list(results.values()))
    print("FINAL: PASS %d / REVIEW %d / FAIL %d" % (n_pass, n_rev, n_fail))


if __name__ == "__main__":
    main()
