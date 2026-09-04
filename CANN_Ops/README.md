# CANN_Ops Dataset

CANN operator dataset for AscendC operator-generation RL training: **280 operators**, each with a
torch reference implementation and test cases, plus the training task list.

## Directory contents

```
CANN_Ops/
├── src/                              # op assets, flat: {op}.py + {op}.json pairs (280 x 2)
└── operator_tasks.cannops.jsonl      # task list, one {prompt, label, metadata} row per op
```

## Naming and format

- Files are named `cannops_level{N}_{id}_{OpName}.py` / `.json` (N in 1/2/3, `id` renumbered
  contiguously from 0 within each level, `OpName` in CamelCase)
- **`.py`** — torch reference implementation: `class Model` + `get_input_groups()` /
  `get_init_inputs()`. `get_input_groups()` hardcodes the sibling `.json` filename (the judge
  renames the .py to `model.py` when injecting it, so `__file__`-derived paths are forbidden).
  Reference implementations use only public aten ops, so they run on both CPU and NPU
- **`.json`** — test cases, JSON Lines, one `{"inputs": [...]}` case per line. Entry schema follows
  the NPUKernelBench convention: tensors `{name, type: "tensor", required, dtype, shape}`,
  attributes `{name, type: "attr", required, dtype, value}`, tensor lists
  `{name, type: "tensor_list", dtype, shapes}`



## Verification status

Both full sweeps pass at 100% (see `utils/op_verify/`):

| Side | Result |
|---|---|
| CPU (`verify_cpu.py --dir CANN_Ops/src`) | 280/280 ops |
| NPU (`verify_npu.py --dir CANN_Ops/src`) | 280/280 ops, 3382/3382 cases |

16 additional operators are retained on the `cann-ops-dev` branch pending analysis (they fail on
torch NPU for reasons unrelated to ingestion: complex64 coverage of `aclnnAbs`/`aclnnIndex`,
Models with hardcoded CPU devices or direct `.numpy()` calls, kernels missing in the current CANN
version); they merge into main once fixed and confirmed.

## Regenerating the task list

```bash
python utils/gen_tasks/gen_op_tasks.py --dir CANN_Ops
# prefix auto-detected -> CANN_Ops/operator_tasks.cannops.jsonl
```

Newly added op files must pass both verification scripts before committing (see the repository
README for the full workflow).
