# op_verify — 算子资产通用验证工具

对任意算子资产目录（`{op}.py` + 同名 `{op}.json` 成对存放，如 `CANN_Ops/src`、`NPUKernelBench/src`）
做全量健康检查：**模块可加载 → 输入可生成 → 每条用例前向跑通（CPU 或 NPU）**。
新增算子入库前后都可用它做全量回归。

## 用法

```bash
# CPU 全量（任何有 torch 的机器）
python utils/op_verify/verify_cpu.py --dir CANN_Ops/src

# NPU 全量（需要 torch_npu + 昇腾驱动；脚本自动 source set_env.sh）
python utils/op_verify/verify_npu.py --dir CANN_Ops/src --workers 16

# 只验证指定算子（增量）
python utils/op_verify/verify_cpu.py --dir CANN_Ops/src --only cannops_level1_0_AbsMath
```

## 参数（两脚本一致）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dir` | **必填** | 算子资产目录 |
| `--only` | 全部 | 只测列出的算子名（不含扩展名） |
| `--jobs`（CPU）/ `--workers`（NPU） | 3 / 16 | CPU 并发子进程数 / NPU 卡数 |
| `--strict-finite` | 关 | 输出含 inf/nan 时从 WARN 升级为 FAIL |
| `--report` | 自动生成 | 报告输出路径（默认 `utils/op_verify/report/verify_{cpu,npu}_{目录名}_{时间戳}.md`，该目录不入库） |

## 检查项

1. 模块加载（语法/依赖）
2. `get_input_groups()`/`get_init_inputs()` 可执行，且 case 数与 json 行数一致
3. 每条 case 前向通过：先 `no_grad`，失败自动走 `requires_grad_` 梯度路径（覆盖 *Grad 算子）；输入先 clone（防参考实现就地改写污染）
4. 输出有限性扫描：含 inf/nan 记 WARN（Segsum fp16 溢出、IsFinite 故意注入等属合法行为，故默认不判 FAIL）

## 工程特性

- **CPU**：每算子独立子进程（OOM 隔离），失败自动重试一次
- **NPU**：每卡一个长驻 worker 进程（摊销 torch_npu/ACL 初始化），错峰启动；
  三段重试自愈共享环境下的 `SetDevice` 初始化竞争（pass1 全量 → pass2 并发重试 → pass3 单卡串行重试）
- exit code 非 0 当且仅当存在 FAIL；报告含失败首错与 WARN 清单

## 环境依赖

- CPU 侧：`torch`（本仓库使用 `/root/miniconda3/envs/coding_env`）
- NPU 侧：另需 `torch_npu` 与昇腾驱动（worker 自动 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`）

## 与 utils/cannops_ingest/ 的区别

`cannops_ingest/` 是**入库转换期**工具（与 cann_ops_tmp 原版逐 case 对照），依赖本地未入库的
`cann_ops_tmp/` 目录；`op_verify/` 是**入库后长期维护**的自包含验证，只依赖入库文件本身。
