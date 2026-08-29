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

from typing import List, Tuple
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, cost: torch.Tensor, tol: float=0.0001):
        super(Model, self).__init__()
        self.cost = cost
        self.tol = tol

    def forward(self, cost: torch.Tensor, tol: float=0.0001) -> torch.Tensor:
        """
        手动实现 Sinkhorn 算法（CPU）
        Args:
            cost: 输入成本矩阵 [S*B, num_experts]，dtype: float16/bf16/f32
            tol: 收敛容差，float32

        Returns:
            p: 输出运输方案，shape=[S*B, num_experts]
        """
        golden_p = self.sinkhorn(cost, tol)
        return golden_p

    def sinkhorn(self, cost: torch.Tensor, tol: float=0.0001):
        """Sinkhorn based MoE routing function"""
        cost = torch.exp(cost)
        d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
        d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)
        eps = 1e-08
        error = 1000000000.0
        d1_old = d1
        while error > tol:
            d0 = 1 / d0.size(0) * 1 / (torch.sum(d1 * cost, 1) + eps)
            d1 = 1 / d1.size(0) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
            error = torch.mean(torch.abs(d1_old - d1))
            d1_old = d1
        return d1 * cost * d0.unsqueeze(1)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_63_Sinkhorn.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        cost_info = inputs[0]
        tol_info = inputs[1]

        if "data" in cost_info:
            cost = torch.tensor(cost_info["data"], dtype=DTYPE_MAP[cost_info["dtype"]]).reshape(cost_info["shape"])
        else:
            cost = torch.rand(cost_info["shape"], dtype=DTYPE_MAP[cost_info["dtype"]])
        tol = tol_info["value"]

        input_groups.append([cost, tol])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_63_Sinkhorn.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        cost_info = entries[0]
        tol_info = entries[1]
        if "data" in cost_info:
            cost = torch.tensor(cost_info["data"], dtype=DTYPE_MAP[cost_info["dtype"]]).reshape(cost_info["shape"])
        else:
            cost = torch.rand(cost_info["shape"], dtype=DTYPE_MAP[cost_info["dtype"]])
        tol = tol_info["value"]
        init_groups.append([cost, tol])
    return init_groups
