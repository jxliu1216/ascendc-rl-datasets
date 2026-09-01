#!/usr/bin/env python3
"""NPU full verification of an op asset directory.

For every {op}.py with a sibling {op}.json under --dir: load the module,
generate inputs on CPU, move them to NPU, and execute Model.forward for every
case (no_grad first, grad-path fallback; inputs cloned). Output finiteness is
scanned and reported as WARN (use --strict-finite to make it a FAIL).

Ops are split into N chunks, one long-lived worker process per card
(ASCEND_RT_VISIBLE_DEVICES), with staggered init and a 3-pass retry:
  pass1: all ops, chunked across cards
  pass2: retry crashed / all-cases-init-failed ops (concurrent)
  pass3: remaining init failures, strictly sequential (contention-proof)

Requires: torch_npu + sourced Ascend toolkit env (the script sources
/usr/local/Ascend/ascend-toolkit/set_env.sh for workers itself).

Examples:
    python utils/op_verify/verify_npu.py --dir CANN_Ops/src
    python utils/op_verify/verify_npu.py --dir CANN_Ops/src --only cannops_level1_0_AbsMath
    python utils/op_verify/verify_npu.py --dir CANN_Ops/src --workers 8

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
SET_ENV = "source /usr/local/Ascend/ascend-toolkit/set_env.sh"


def find_ops(d, only=None):
    pys = sorted(glob.glob(os.path.join(d, "*.py")))
    ops = [os.path.basename(p)[:-3] for p in pys
           if os.path.exists(p[:-3] + ".json") and not os.path.basename(p).startswith("_")]
    if only:
        ops = [o for o in ops if o in set(only)]
    return ops


def run_chunk(d, worker, bases, stagger=1.5):
    """One long-lived worker process on one card, processing its chunk."""
    out_path = "/tmp/opverify_npu_chunk_%d.jsonl" % worker
    if os.path.exists(out_path):
        os.remove(out_path)
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(worker)
    cmd = "sleep %.1f && %s && %s %s --worker --dir %s --out %s --only %s" % (
        worker * stagger, SET_ENV, sys.executable, os.path.abspath(__file__),
        d, out_path, " ".join(bases))
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                       timeout=7200, env=env)
    results = []
    if os.path.exists(out_path):
        for line in open(out_path):
            line = line.strip()
            if line.startswith("{"):
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
    seen = {r["op"] for r in results}
    for b in bases:
        if b not in seen:
            results.append({"op": b, "total": -1, "passed": 0, "warns": [],
                            "fails": ["chunk died: " + p.stderr.strip()[-200:]]})
    return results


def sweep(d, bases, tag, workers, stagger=1.5):
    chunks = [bases[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_chunk, d, w, c, stagger): w for w, c in enumerate(chunks)}
        for fut in as_completed(futs):
            results.extend(fut.result())
            print("[%s] chunk done (%d results)" % (tag, len(results)), flush=True)
    return results


def needs_retry(r):
    if r["total"] <= 0:
        return True
    return r["passed"] == 0 and r["fails"] and "Initialize" in r["fails"][0]


def main():
    ap = argparse.ArgumentParser(description="NPU full verification of op assets")
    ap.add_argument("--dir", required=True, help="op asset dir ({op}.py + {op}.json)")
    ap.add_argument("--only", nargs="*", default=None, help="only these ops")
    ap.add_argument("--workers", type=int, default=16, help="NPU cards to use (default 16)")
    ap.add_argument("--strict-finite", action="store_true",
                    help="treat non-finite outputs as FAIL instead of WARN")
    ap.add_argument("--report", default=None, help="report path (default: utils/op_verify/report/...)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        import torch  # noqa
        import torch_npu  # noqa
        sys.path.insert(0, HERE)
        import _common
        with open(args.out, "a") as out:
            for base in args.only:
                r = _common.run_op(os.path.join(args.dir, base + ".py"), device="npu")
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                try:
                    import gc
                    gc.collect()
                    torch.npu.empty_cache()
                except Exception:
                    pass
                print("[%d/%d] %s" % (r["passed"], r["total"], base), flush=True)
        return

    if not os.path.isdir(args.dir):
        print("ERROR: --dir %s is not a directory" % args.dir)
        sys.exit(2)
    sys.path.insert(0, HERE)
    import _common

    ops = find_ops(args.dir, args.only)
    print("total ops: %d (dir=%s, workers=%d)" % (len(ops), args.dir, args.workers),
          flush=True)
    t0 = time.time()
    results = sweep(args.dir, ops, "pass1", args.workers)
    bad = [r["op"] for r in results if needs_retry(r)]
    if bad:
        print("pass2: retrying %d crashed/init-failed ops" % len(bad), flush=True)
        retry = sweep(args.dir, bad, "pass2", args.workers)
        ok = {r["op"]: r for r in retry if not needs_retry(r)}
        results = [ok.get(r["op"], r) for r in results]
        still = [r["op"] for r in results if needs_retry(r)]
        if still:
            print("pass3: sequential retry for %d ops" % len(still), flush=True)
            retry3 = sweep(args.dir, still, "pass3", workers=1, stagger=0.0)
            ok3 = {r["op"]: r for r in retry3 if not needs_retry(r)}
            results = [ok3.get(r["op"], r) for r in results]

    ts = time.strftime("%Y%m%d_%H%M%S")
    report = args.report or os.path.join(
        REPORT_DIR, "verify_npu_%s_%s.md" % (os.path.basename(args.dir.rstrip("/")), ts))
    n_ok, n_warn, n_fail = _common.write_report(
        results, report, "NPU 全量验证报告 (%s)" % args.dir, args.strict_finite)
    total_cases = sum(max(r["total"], 0) for r in results)
    passed_cases = sum(r["passed"] for r in results)
    print("NPU FINAL: ops %d/%d passed, cases %d/%d, WARN %d (%.0fs) -> %s"
          % (n_ok, len(results), passed_cases, total_cases, n_warn,
             time.time() - t0, report))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
