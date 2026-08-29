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

def exp_segsum_reference(x: torch.Tensor) -> torch.Tensor:
    """
    ExpSegsum 参考实现：
    输入 x (..., N)，输出 (..., N, N)。
    下三角（不含对角）为 cumsum 后 exp，对角为 1，上三角为 0（-inf 再 exp）。
    """
    shape = list(x.shape)
    n = shape[-1]
    prefix = shape[:-1]
    x_expand = x.unsqueeze(-1)
    x_broadcast = x_expand.expand(prefix + [n, n])
    mask_keep = torch.tril(torch.ones(n, n, device=x.device, dtype=torch.bool), -1)
    mask_keep = mask_keep.reshape([1] * len(prefix) + [n, n]).expand(prefix + [n, n])
    mat = torch.where(mask_keep, x_broadcast, torch.zeros_like(x_broadcast, device=x.device))
    cum = mat.cumsum(dim=-2)
    mask_upper = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), 1)
    mask_upper = mask_upper.reshape([1] * len(prefix) + [n, n]).expand(prefix + [n, n])
    cum = torch.where(mask_upper, torch.full_like(cum, float('-inf')), cum)
    out = torch.exp(cum)
    return out.to(x.dtype)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return exp_segsum_reference(x)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_61_Segsum.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])

        input_groups.append([x])
    return input_groups


def get_init_inputs():
    return []
