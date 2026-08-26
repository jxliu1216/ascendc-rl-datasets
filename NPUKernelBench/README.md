# NPUKernelBench 数据集目录说明

本目录提供 AscendC 算子生成 RL 训练的任务数据(算子持续补充中)。

## 文件构成

- `operator_tasks.npukernelbench.jsonl` — 任务清单,每行 `{prompt, label, metadata}`;训练侧(slime `--prompt-data`)读取,`metadata.op_name` 是关联算子文件的键
- `src/` — **全量用例版**算子资产目录
- `src_simple/` — **精简用例版**算子资产目录(人工精简)

## src/ 与 src_simple/ 的差异

两个目录**结构完全相同**,每个算子一对文件:

| 文件 | 内容 |
|---|---|
| `{op}.py` | torch 参考实现(`class Model` + `get_input_groups()`/`get_init_inputs()`)。`get_input_groups()` 读取**同目录、同名**的 `{op}.json`(文件名为硬编码,保证 judge 将参考实现重命名为 `model.py` 注入工程目录后仍能命中同目录的用例文件) |
| `{op}.json` | 测试用例,JSON Lines,每行一个 `{"inputs": [...]}` case |

唯一差异是 **`{op}.json` 的用例数量**:

| 目录 | 用例规模 | 用途 |
|---|---|---|
| `src/` | 全量用例 | 离线复测、最终验收等不计成本的场景 |
| `src_simple/` | 人工精简用例(保留 dtype/shape/attr 代表性组合) | **RL 在线训练**(评测逐 case 串行且独占 NPU 卡锁,用例数直接决定 session 时长与超时率;reward 只看 correctness 二值 + speedup,case 数不进入 reward 公式) |

`src/` 中另保留 `{op}_simple.json`,为精简用例的原始存档(与 `src_simple/{op}.json` 内容一致)。

## 使用方式

训练时按需将配置指向对应目录(两个目录都可直接被 pipeline 消费):

- `operator_runtime.task_assets_dir`(profile.t2a.yaml)→ 用例 JSON 来源
- `operator_tasks_dir`(polar_config.yaml)→ 参考实现 .py 来源(task_source 内联通道)

## 注意事项

- 新增算子时:jsonl、两个目录的 `{op}.py` / `{op}.json` 需同步补齐;`{op}.py` 中 `get_input_groups()` 的 json 文件名**必须硬编码**为 `{op}.json`,不得使用 `__file__` basename 推导(judge 注入时文件会被重命名为 `model.py`)
- 修改 `src/` 的 `.py` 后需同步到 `src_simple/`(两目录 .py 内容一致)
