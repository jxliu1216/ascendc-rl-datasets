import torch
import torch.nn as nn
import json
import os

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "complex64": torch.complex64,
}

from typing import List
import torch
import torch.nn as nn

class Model(nn.Module):
    """使用 PyTorch 原生算子的参考实现（golden model）。

    ThreeInterpolateBackward: 反向传播算子。
    给定 grad_x (B, C, N), idx (B, N, 3), weight (B, N, 3)，
    计算 grad_y (B, C, M)，其中 M 由属性 m 指定。

    计算公式:
        grad_y[b, c, idx[b, n, k]] += weight[b, n, k] * grad_x[b, c, n]
        对所有 n in [0, N) 和 k in [0, 3)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_x, idx, weight, m):
        B, C, N = grad_x.shape
        grad_y = torch.zeros(B, C, m, dtype=grad_x.dtype, device=grad_x.device)
        for k in range(3):
            idx_k = idx[:, :, k].long().unsqueeze(1).expand(-1, C, -1)
            weight_k = weight[:, :, k].unsqueeze(1)
            contrib = grad_x * weight_k
            grad_y.scatter_add_(2, idx_k, contrib)
        return grad_y

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_148_ThreeInterpolateBackward.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_x_info = inputs[0]
        idx_info = inputs[1]
        weight_info = inputs[2]
        m_info = inputs[3]

        if "data" in grad_x_info:
            grad_x = torch.tensor(grad_x_info["data"], dtype=DTYPE_MAP[grad_x_info["dtype"]]).reshape(grad_x_info["shape"])
        else:
            grad_x = torch.randn(grad_x_info["shape"], dtype=DTYPE_MAP[grad_x_info["dtype"]])
        if "data" in idx_info:
            idx = torch.tensor(idx_info["data"], dtype=DTYPE_MAP[idx_info["dtype"]]).reshape(idx_info["shape"])
        else:
            idx = torch.randint(idx_info["range"][0], idx_info["range"][1] + 1, tuple(idx_info["shape"]), dtype=DTYPE_MAP[idx_info["dtype"]])
        if "data" in weight_info:
            weight = torch.tensor(weight_info["data"], dtype=DTYPE_MAP[weight_info["dtype"]]).reshape(weight_info["shape"])
        else:
            weight = torch.randn(weight_info["shape"], dtype=DTYPE_MAP[weight_info["dtype"]])
        m = m_info["value"]

        input_groups.append([grad_x, idx, weight, m])
    return input_groups


def get_init_inputs():
    return []
