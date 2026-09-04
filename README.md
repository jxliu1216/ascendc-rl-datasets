# ascendc-rl-datasets

Datasets for AscendC operator-generation RL training: every operator ships with a **torch reference
implementation + test cases.

## Repository layout

```
ascendc-rl-datasets/
├── CANN_Ops/                       # CANN operator dataset (280 ops)
│   ├── src/                        # op assets: cannops_level{N}_{id}_{Name}.py + .json pairs
│   └── operator_tasks.cannops.jsonl    # task list (280 rows)
├── NPUKernelBench/                 # NPUKernelBench dataset (43 ops, see its README)
│   ├── src/                        # full-case assets: npukernelbench_level{N}_{id}_{Name}.py + .json (full cases)
│   ├── src_simple/                 # simpilified-case assets: each .py with simplified .json (10 cases)
│   ├── operator_tasks.npukernelbench.jsonl
│   └── README.md
└── utils/
    ├── gen_tasks/gen_op_tasks.py   # task-list generation (works for any same-format directory)
    ├── op_verify/                  # CPU/NPU full verification (self-contained, has its own README)
    └── test_case_simplify/         # test-case trimming tool (has its own README)
```

## Dataset format conventions

Each operator is a pair of same-named files stored flat in one directory:

| File | Content |
|---|---|
| `{prefix}_level{N}_{id}_{Name}.py` | torch reference implementation: `class Model` + `get_input_groups()`/`get_init_inputs()`. `get_input_groups()` hardcodes the sibling `{op}.json` filename (the judge renames the .py to `model.py` when injecting it, so `__file__`-derived paths are forbidden) |
| `{prefix}_level{N}_{id}_{Name}.json` | test cases, JSON Lines, one `{"inputs": [...]}` case per line (tensors carry `dtype`/`shape`, attributes carry `value`) |

- `prefix` identifies the source dataset: `cannops_` (CANN_Ops), `npukernelbench_` (NPUKernelBench)
- `id` is numbered contiguously from 0 within each level
- the task list `operator_tasks.{prefix}.jsonl` lives at the **category root**, one
  `{prompt, label, metadata}` row per op; training resolves op files via `metadata.op_name`

## Verification (the hard gate for ingestion and regression)

```bash
# CPU full sweep (any plain torch environment)
python utils/op_verify/verify_cpu.py --dir CANN_Ops/src

# NPU full sweep (requires torch_npu + Ascend drivers; 16 cards in parallel with
# staggered init and self-healing retries)
python utils/op_verify/verify_npu.py --dir CANN_Ops/src
```

- Both scripts require an explicit `--dir` and work for any same-format op directory
- **Ingestion bar on main: 100% of cases pass on both CPU and NPU** (currently CANN_Ops:
  280/280 ops, 3382/3382 cases)
- Note: `verify_cpu.py` does NOT apply to NPUKernelBench — its reference implementations call
  `torch_npu.npu_*` ops directly, which only have NPU-backend kernels; use `verify_npu.py`
  there instead. See `utils/op_verify/README.md` for details

## Adding new operators

1. Drop `{op}.py` + `{op}.json` into the dataset's `src/`, following the format conventions above
2. Run both verifications: `verify_cpu.py --dir <dataset>/src` and `verify_npu.py --dir <dataset>/src` — all cases must pass
3. Regenerate the task list: `python utils/gen_tasks/gen_op_tasks.py --dir <dataset>`
   (prefix auto-detected, output lands at the category root)
4. Commit

## Branches

| Branch | Content |
|---|---|
| `main` | stable subset: CANN_Ops 280 ops, 100% pass on both CPU and NPU |
| `cann-ops-dev` | full archive: 296 ops, including 16 that do not pass torch NPU yet (Ccopy on complex64, Models with hardcoded CPU devices or direct `.numpy()` calls, kernels missing in the current CANN version); merged into main once fixed and confirmed |
