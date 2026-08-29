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
    """
    实现InplaceScatterAdd算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """
        实现InplaceScatterAdd算子功能。

        Args:
            var: 被更新的张量 [M, N]
            indices: 索引张量 [K]
            updates: 更新值张量 [K, N]

        Returns:
            更新后的var张量
        """
        result = var.clone()
        result.scatter_add_(0, indices.unsqueeze(1).expand_as(updates).long(), updates)
        return result

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_32_InplaceScatterAdd.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        var_info = inputs[0]
        indices_info = inputs[1]
        updates_info = inputs[2]

        if "data" in var_info:
            var = torch.tensor(var_info["data"], dtype=DTYPE_MAP[var_info["dtype"]]).reshape(var_info["shape"])
        else:
            _dt = DTYPE_MAP[var_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                var = torch.randint(var_info["range"][0], var_info["range"][1] + 1, tuple(var_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                var = torch.rand(var_info["shape"]) > 0.5
            else:
                var = torch.randn(var_info["shape"], dtype=_dt)
        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])
        if "data" in updates_info:
            updates = torch.tensor(updates_info["data"], dtype=DTYPE_MAP[updates_info["dtype"]]).reshape(updates_info["shape"])
        else:
            _dt = DTYPE_MAP[updates_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                updates = torch.randint(updates_info["range"][0], updates_info["range"][1] + 1, tuple(updates_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                updates = torch.rand(updates_info["shape"]) > 0.5
            else:
                updates = torch.randn(updates_info["shape"], dtype=_dt)

        input_groups.append([var, indices, updates])
    return input_groups


def get_init_inputs():
    return []
