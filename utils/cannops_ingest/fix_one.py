#!/usr/bin/env python3
"""Convert + verify ONE operator. Prints a single JSON line to stdout:
{"op": ..., "convert_ok": bool, "result": <verify result dict>}

Usage: fix_one.py <level> <old_id> <op_name> <new_id>
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    level, old_id, op, new_id = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    import convert
    import verify

    src_py = os.path.join(convert.SRC_DIR, level, "%d_%s.py" % (old_id, op))
    try:
        new_base, info = convert.convert_op(level, old_id, op, src_py, new_id)
    except Exception as e:
        print(json.dumps({"op": "%s/%d_%s" % (level, old_id, op), "convert_ok": False,
                          "result": {"op": "%s/%d_%s" % (level, old_id, op), "status": "FAIL",
                                     "struct": [], "review": [],
                                     "error": ["convert: " + repr(e)]}},
                         ensure_ascii=False))
        return
    try:
        r = verify.verify_op(level, old_id, op, new_base)
    except Exception as e:
        r = {"op": new_base, "status": "FAIL", "struct": [], "review": [],
             "error": ["verify crash: " + repr(e), traceback.format_exc(limit=3)]}
    print(json.dumps({"op": new_base, "convert_ok": True, "result": r}, ensure_ascii=False))


if __name__ == "__main__":
    main()
