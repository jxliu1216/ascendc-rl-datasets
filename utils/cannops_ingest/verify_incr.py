#!/usr/bin/env python3
"""Incremental verifier: watches /tmp/convert_full.log for newly converted ops
and verifies them while the full conversion is still running. Writes
/tmp/verify_partial.jsonl incrementally; when conversion is done, emits the
final CANN_Ops/_ingest_report.md from all collected results."""

import json
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify

LOG = "/tmp/convert_full.log"
PARTIAL = "/tmp/verify_partial.jsonl"
REPORT = os.path.join(verify.OUT_DIR, "..", "report", "_ingest_report.md")

OK_RE = re.compile(r"\[ok\] (level\d)/(\d+)_(\S+) -> (\S+)")
FAIL_RE = re.compile(r"\[FAIL\] (level\d)/(\d+)_(\S+): (.*)")


def scan_log():
    ops, fails = {}, {}
    if not os.path.exists(LOG):
        return ops, fails, False
    done = False
    for line in open(LOG):
        line = line.rstrip("\n")
        m = OK_RE.match(line)
        if m:
            ops[m.group(4)] = (m.group(1), int(m.group(2)), m.group(3))
            continue
        m = FAIL_RE.match(line)
        if m:
            fails["%s/%s_%s" % (m.group(1), m.group(2), m.group(3))] = m.group(4)
        if line.startswith("converted "):
            done = True
    return ops, fails, done


def write_report(results):
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_rev = sum(1 for r in results if r["status"] == "REVIEW")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    lines = ["# CANN_Ops 入库验证报告", "",
             "| 状态 | 数量 |", "|---|---|",
             "| PASS | %d |" % n_pass, "| REVIEW | %d |" % n_rev,
             "| FAIL | %d |" % n_fail, ""]
    for st in ("FAIL", "REVIEW"):
        rows = [r for r in results if r["status"] == st]
        if not rows:
            continue
        lines += ["## %s (%d)" % (st, len(rows)), "", "| 算子 | 详情 |", "|---|---|"]
        for r in rows:
            detail = "<br>".join(str(x) for x in (r.get("error") or r.get("review") or [])[:6])
            lines.append("| %s | %s |" % (r["op"], detail.replace("|", "\\|")))
        lines.append("")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    return n_pass, n_rev, n_fail


def main():
    verified = {}
    if os.path.exists(PARTIAL):
        for line in open(PARTIAL):
            line = line.strip()
            if line:
                r = json.loads(line)
                verified[r["op"]] = r
    out = open(PARTIAL, "a")
    while True:
        ops, fails, conv_done = scan_log()
        for new_base, (level, old_id, op) in sorted(ops.items()):
            if new_base in verified:
                continue
            try:
                r = verify.verify_op(level, old_id, op, new_base)
            except Exception as e:
                r = {"op": new_base, "status": "FAIL", "struct": [], "review": [],
                     "error": ["verify crash: " + repr(e), traceback.format_exc(limit=3)]}
            verified[new_base] = r
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            print("[%s] %s" % (r["status"], new_base), flush=True)
        if conv_done:
            break
        time.sleep(20)
    out.close()
    results = list(verified.values())
    for op, err in fails.items():
        results.append({"op": op, "status": "FAIL", "struct": [], "review": [],
                        "error": ["convert: " + err]})
    n_pass, n_rev, n_fail = write_report(results)
    print("FINAL: PASS %d / REVIEW %d / FAIL %d -> %s" % (n_pass, n_rev, n_fail, REPORT))


if __name__ == "__main__":
    main()
