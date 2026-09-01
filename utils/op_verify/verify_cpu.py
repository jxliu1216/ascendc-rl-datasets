#!/usr/bin/env python3
"""CPU full verification of an op asset directory.

For every {op}.py with a sibling {op}.json under --dir: load the module,
run get_input_groups()/get_init_inputs(), and execute Model.forward for every
case on CPU (no_grad first, grad-path fallback for *Grad ops; inputs cloned).
Output finiteness is scanned and reported as WARN (use --strict-finite to make
it a FAIL).

Each op runs in its own subprocess (OOM isolation); failures are retried once.

Examples:
    python utils/op_verify/verify_cpu.py --dir CANN_Ops/src
    python utils/op_verify/verify_cpu.py --dir NPUKernelBench/src --only npukernelbench_level1_3_Add
    python utils/op_verify/verify_cpu.py --dir CANN_Ops/src --jobs 4 --strict-finite

Exit code is non-zero iff any op FAILs.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "report")


def find_ops(d, only=None):
    pys = sorted(glob.glob(os.path.join(d, "*.py")))
    ops = [os.path.basename(p)[:-3] for p in pys
           if os.path.exists(p[:-3] + ".json") and not os.path.basename(p).startswith("_")]
    if only:
        ops = [o for o in ops if o in set(only)]
    return ops


def run_one(d, base):
    """Run one op in a fresh subprocess (re-invoke this script in worker mode)."""
    p = subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--worker", "--dir", d, "--only", base],
                       capture_output=True, text=True, timeout=1800)
    lines = [l for l in p.stdout.strip().splitlines() if l.startswith("{")]
    if not lines:
        return {"op": base, "total": 0, "passed": 0, "warns": [],
                "fails": ["subprocess died: " + p.stderr.strip()[-200:]]}
    return json.loads(lines[-1])


def main():
    ap = argparse.ArgumentParser(description="CPU full verification of op assets")
    ap.add_argument("--dir", required=True, help="op asset dir ({op}.py + {op}.json)")
    ap.add_argument("--only", nargs="*", default=None, help="only these ops")
    ap.add_argument("--jobs", type=int, default=3, help="parallel subprocesses (default 3)")
    ap.add_argument("--strict-finite", action="store_true",
                    help="treat non-finite outputs as FAIL instead of WARN")
    ap.add_argument("--report", default=None, help="report path (default: utils/op_verify/report/...)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        sys.path.insert(0, HERE)
        import _common
        r = _common.run_op(os.path.join(args.dir, args.only[0] + ".py"), device=None)
        print(json.dumps(r, ensure_ascii=False))
        return

    if not os.path.isdir(args.dir):
        print("ERROR: --dir %s is not a directory" % args.dir)
        sys.exit(2)
    sys.path.insert(0, HERE)
    import _common

    ops = find_ops(args.dir, args.only)
    print("total ops: %d (dir=%s)" % (len(ops), args.dir), flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, args.dir, o): o for o in ops}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            ok = r["passed"] == r["total"] and not r["fails"]
            print("[%d/%d] %s %s" % (r["passed"], r["total"], r["op"],
                                     "" if ok else "FAIL"), flush=True)

    # one retry for transient failures
    bad = [r["op"] for r in results if r["passed"] != r["total"] or r["fails"]]
    if bad:
        print("retrying %d failed ops" % len(bad), flush=True)
        retry = {r["op"]: r for r in (run_one(args.dir, o) for o in bad)}
        results = [retry.get(r["op"], r) if r["op"] in retry and
                   retry[r["op"]]["passed"] == retry[r["op"]]["total"] else r
                   for r in results]

    ts = time.strftime("%Y%m%d_%H%M%S")
    report = args.report or os.path.join(
        REPORT_DIR, "verify_cpu_%s_%s.md" % (os.path.basename(args.dir.rstrip("/")), ts))
    n_ok, n_warn, n_fail = _common.write_report(
        results, report, "CPU 全量验证报告 (%s)" % args.dir, args.strict_finite)
    print("CPU FINAL: ops %d/%d passed, WARN %d, FAIL %d (%.0fs) -> %s"
          % (n_ok, len(results), n_warn, n_fail, time.time() - t0, report))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
