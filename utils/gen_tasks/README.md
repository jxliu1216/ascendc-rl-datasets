# 训练数据集生成脚本（gen_op_tasks.py）

## 一、功能说明

`gen_op_tasks.py` 将算子数据集目录下的算子参考实现（`{prefix}_level{N}_{id}_{name}.py`）批量转换为 **`.jsonl` 格式的 RL 训练数据集**，每个算子一行。

对 CANN_Ops、NPUKernelBench 及后续新增的同类目录通用，目录约定：

- 算子文件为平铺的 `{op}.py`（参考实现）+ `{op}.json`（测试用例）成对文件
- 文件名遵循 `{prefix}_level{N}_{id}_{name}` 规范，同一目录内前缀统一
- 数据集根目录下用 `src/` 子目录存放算子文件（`--dir` 传根目录时自动下探）

前缀从目录内容**自动嗅探**；生成的 jsonl 默认写到**类别根目录**：`{root}/operator_tasks.{prefix}.jsonl`。

## 二、使用方法

```bash
# 在 ascendc-rl-datasets/ 根目录下执行
python3 utils/gen_tasks/gen_op_tasks.py --dir CANN_Ops
#   → CANN_Ops/operator_tasks.cannops.jsonl (280 行)

python3 utils/gen_tasks/gen_op_tasks.py --dir NPUKernelBench
#   → NPUKernelBench/operator_tasks.npukernelbench.jsonl (43 行, 与历史版本逐字节一致)
```

命名不符的文件打印 `[warn]` 跳过，不会中断。新增算子放入目录后重跑即可。

单算子冒烟：

```bash
python3 utils/gen_tasks/gen_op_tasks.py \
    --dir CANN_Ops/src --out /tmp/smoke.jsonl \
    --ops cannops_level1_0_AbsMath
```

## 三、参数说明

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--dir` | 是 | — | 数据集根目录（自动下探 `src/`）或算子文件所在目录 |
| `--out` | 否 | `{类别根目录}/operator_tasks.{prefix}.jsonl` | 输出 jsonl 路径 |
| `--data-source` | 否 | 前缀映射表 | `metadata.data_source`；内置映射 `npukernelbench→npu-kernel-bench`、`cannops→cann-ops`，未登记的新前缀取前缀本身 |
| `--arch` | 否 | `ascend910b1` | 写入 `metadata.arch` 的目标架构 |
| `--ops` | 否 | — | 逗号分隔的算子名白名单，单算子冒烟用；白名单未全部命中时以非零码退出 |

## 四、输出格式

每行一个 JSON 对象：

```json
{
  "prompt": [{"role": "user", "content": "Implement an AscendC operator for Ascend NPU. ..."}],
  "label": "cannops_level1_0_AbsMath",
  "metadata": {
    "op_name": "cannops_level1_0_AbsMath",
    "entry_point": "Model",
    "operator_backend": "ascendc",
    "arch": "ascend910b1",
    "ops": ["AbsMath"],
    "data_source": "cann-ops",
    "ability": "code",
    "level": "1",
    "uid": "cannops_level1_0_AbsMath"
  }
}
```

字段取值规则：

- **`op_name` / `label` / `uid`**：py 文件名 stem（保留 `{prefix}_level` 前缀，全局唯一），训练侧以 `metadata.op_name` 关联算子文件
- **`ops`**：算子显示名（文件名中 `{prefix}_level{N}_{id}_` 之后的部分）
- **`level`**：从文件名 `level{N}` 解析
- **`entry_point` / `operator_backend` / `ability`**：固定为 `Model` / `ascendc` / `code`

## 五、变更记录

- 旧版 `gen_npukernelbench_tasks.py`（仅支持 NPUKernelBench、硬编码前缀）已由本通用版替代并移除，可从 git 历史追溯。
