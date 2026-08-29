#!/usr/bin/env python3
"""Re-convert and re-verify failed ops after converter fixes, then rewrite the
final report. Run after verify_incr.py has finished its first pass.

Usage: fix_round.py            # re-convert all FAIL ops, re-verify, rewrite report
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert
import verify
import verify_incr

PARTIAL = "/tmp/verify_partial.jsonl"


def main():
    manifest = json.load(open(os.path.join(verify.OUT_DIR, "_manifest.json")))
    results = {}
    for line in open(PARTIAL):
        line = line.strip()
        if line:
            r = json.loads(line)
            results[r["op"]] = r

    # targets: convert-time failures + verify FAILs
    targets = []
    for m in manifest:
        if m["status"] != "ok":
            targets.append((m, "convert-fail"))
        elif m["new"] in results and results[m["new"]]["status"] == "FAIL":
            targets.append((m, "verify-fail"))

    print("re-converting %d ops" % len(targets))
    for m, why in targets:
        new_id = int(m["new"].split("_")[2]) if m["status"] == "ok" else None
        if new_id is None:
            # find intended new id: position in sorted order within level
            lvl = [(x["old_id"], x["op"]) for x in manifest if x["level"] == m["level"]]
            lvl.sort()
            new_id = [i for i, (oid, op) in enumerate(lvl)
                      if oid == m["old_id"] and op == m["op"]][0]
        src_py = os.path.join(convert.SRC_DIR, m["level"], "%d_%s.py" % (m["old_id"], m["op"]))
        try:
            new_base, _info = convert.convert_op(m["level"], m["old_id"], m["op"], src_py, new_id)
            m["status"], m["new"] = "ok", new_base
            print("[ok] %s (%s)" % (new_base, why), flush=True)
        except Exception as e:
            m["status"] = "ok"  # files may not exist; mark verified fail below
            results["%s/%d_%s" % (m["level"], m["old_id"], m["op"])] = {
                "op": "%s/%d_%s" % (m["level"], m["old_id"], m["op"]),
                "status": "FAIL", "struct": [], "review": [],
                "error": ["convert: " + repr(e)]}
            print("[FAIL-convert] %s/%d_%s: %r" % (m["level"], m["old_id"], m["op"], e), flush=True)
            continue
        try:
            r = verify.verify_op(m["level"], m["old_id"], m["op"], new_base)
        except Exception as e:
            r = {"op": new_base, "status": "FAIL", "struct": [], "review": [],
                 "error": ["verify crash: " + repr(e), traceback.format_exc(limit=3)]}
        results[new_base] = r
        print("[%s] %s" % (r["status"], new_base), flush=True)

    # drop stale entries for old convert-fail keys now converted
    stale = [k for k in results
             if "/" in k and any(m["status"] == "ok" and
                                 "%s/%d_%s" % (m["level"], m["old_id"], m["op"]) == k and
                                 m.get("new") in results
                                 for m in manifest)]
    for k in stale:
        del results[k]

    with open(PARTIAL, "w") as f:
        for r in results.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_pass, n_rev, n_fail = verify_incr.write_report(list(results.values()))
    print("FINAL: PASS %d / REVIEW %d / FAIL %d" % (n_pass, n_rev, n_fail))


if __name__ == "__main__":
    main()
