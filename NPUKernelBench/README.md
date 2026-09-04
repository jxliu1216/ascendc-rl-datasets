# NPUKernelBench Dataset

Task data for AscendC operator-generation RL training (operators are continuously being added).

## Directory contents

- `operator_tasks.npukernelbench.jsonl` — task list, one `{prompt, label, metadata}` row per op;
  consumed by the training side (slime `--prompt-data`); `metadata.op_name` is the key that links
  each row to its op files
- `src/` — op assets with the **full** case set
- `src_simple/` — op assets with **manually trimmed** case sets

## Difference between src/ and src_simple/

The two directories have **identical structure**: one file pair per operator:

| File | Content |
|---|---|
| `{op}.py` | torch reference implementation (`class Model` + `get_input_groups()`/`get_init_inputs()`). `get_input_groups()` hardcodes the sibling `{op}.json` filename, so the judge can still find the case file after renaming the reference implementation to `model.py` when injecting it |
| `{op}.json` | test cases, JSON Lines, one `{"inputs": [...]}` case per line |

The only difference is the **number of cases** in `{op}.json`:

| Directory | Case set | Purpose |
|---|---|---|
| `src/` | full cases | offline re-runs, final acceptance and other cost-insensitive scenarios |
| `src_simple/` | manually trimmed (representative dtype/shape/attr combinations kept) | **online RL training** (evaluation runs cases serially while holding the NPU card lock, so case count directly drives session length and timeout rate; reward only looks at binary correctness + speedup, case count does not enter the reward formula) |


## Notes

- Op files are uniformly named `npukernelbench_level{N}_{id}_{Name}.py` / `.json`; the
  `npukernelbench_` prefix distinguishes this dataset from others added later
- When adding operators, the jsonl and both directories' `{op}.py` / `{op}.json` must all be
  updated together; the json filename inside `get_input_groups()` **must be hardcoded** as
  `{op}.json` — never derive it from the `__file__` basename (the judge renames the file to
  `model.py` when injecting it)
- After editing a `.py` in `src/`, sync it to `src_simple/` (the .py files in both directories are identical)
- Keep **10 cases per op** in the trimmed set
