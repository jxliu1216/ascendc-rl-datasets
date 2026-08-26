# 训练数据集生成脚本（gen_npukernelbench_tasks.py）

## 一、功能说明

`gen_npukernelbench_tasks.py` 将 **NPUKernelBench 数据集**目录下的算子参考实现（`level{N}_{id}_{name}.py`）批量转换为 **`.jsonl` 格式的 RL 训练数据集**，每个算子一行。


## 二、使用方法

```bash
# 在 ascendc-rl-datasets/ 根目录下执行
python3 utils/gen_tasks/gen_npukernelbench_tasks.py \
    --benchmark-dir NPUKernelBench/src \
    --out NPUKernelBench/operator_tasks.npukernelbench.jsonl
```

脚本扫描 `--benchmark-dir` 下所有符合 `level{N}_{id}_{name}.py` 命名规范的文件，逐个生成一行 jsonl；**命名不符的文件会打印 `[warn]` 跳过**，不会中断。新增算子放入目录后重跑即可。

单算子冒烟：

```bash
python3 utils/gen_tasks/gen_npukernelbench_tasks.py \
    --benchmark-dir NPUKernelBench/src \
    --out /tmp/smoke.jsonl \
    --ops level1_3_Add
```

## 三、参数说明

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--benchmark-dir` | 是 | — | NPUKernelBench 目录路径（含 `level{N}_{id}_{name}.py` 文件） |
| `--out` | 是 | — | 输出 jsonl 文件路径，父目录不存在时自动创建 |
| `--arch` | 否 | `ascend910b1` | 写入 `metadata.arch` 的目标架构 |
| `--ops` | 否 | — | 逗号分隔的算子名白名单（如 `level1_3_Add,level2_9_TopKTopP`），只生成这些行，用于单算子冒烟；若白名单中的算子未全部命中，脚本以非零码退出并报错 |

## 四、输出格式

每行一个 JSON 对象：

```json
{
  "prompt": [{"role": "user", "content": "Implement an AscendC operator for Ascend NPU. ..."}],
  "label": "level1_2_SwiGLU",
  "metadata": {
    "op_name": "level1_2_SwiGLU",
    "entry_point": "Model",
    "operator_backend": "ascendc",
    "arch": "ascend910b1",
    "ops": ["SwiGLU"],
    "data_source": "npu-kernel-bench",
    "ability": "code",
    "level": "1",
    "uid": "level1_2_SwiGLU"
  }
}
```

字段取值规则：

- **`op_name` / `label` / `uid`**：py 文件名 stem（保留 `level` 前缀，全局唯一）
- **`ops`**：算子显示名（文件名中 `level{N}_{id}_` 之后的部分）
- **`level`**：从文件名 `level{N}_` 前缀解析（`"1"` / `"2"`）
- **`entry_point` / `operator_backend` / `data_source` / `ability`**：固定为 `Model` / `ascendc` / `npu-kernel-bench` / `code`
