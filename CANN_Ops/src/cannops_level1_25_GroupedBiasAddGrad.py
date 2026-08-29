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

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from typing import List

class Model(nn.Module):

    def __init__(self, grad_y: Tensor, group_idx: Tensor):
        super(Model, self).__init__()
        self.grad_y = grad_y
        self.group_idx = group_idx

    def forward(self, grad_y: Tensor, group_idx: Tensor) -> Tensor:
        """
        CPU 实现 grouped_bias_add_grad，与 golden 一致
        Args:
            grad_y: shape [C, H] 或 [G, C, H]，dtype: float32/float16/bfloat16
            group_idx: shape [G]，int32/int64，可选

        Returns:
            out: shape [G, H]，dtype 与 grad_y 一致
        """
        if group_idx is None:
            grad_y = grad_y.float()
            out = torch.sum(grad_y, dim=1)
            return out.to(grad_y.dtype)
        else:
            group_idx = group_idx.int()
            grad_y = grad_y.float()
            out = torch.tensor([]).to(grad_y.device)
            for i in range(len(group_idx)):
                start = group_idx[i - 1] if i > 0 else 0
                end = group_idx[i]
                chunk = grad_y[start:end, :]
                tmp = torch.sum(chunk, dim=0, keepdim=True)
                out = torch.cat((out, tmp), dim=0)
            return out.to(grad_y.dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_25_GroupedBiasAddGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_y_info = inputs[0]
        group_idx_info = inputs[1]

        if "data" in grad_y_info:
            grad_y = torch.tensor(grad_y_info["data"], dtype=DTYPE_MAP[grad_y_info["dtype"]]).reshape(grad_y_info["shape"])
        else:
            grad_y = torch.rand(grad_y_info["shape"], dtype=DTYPE_MAP[grad_y_info["dtype"]])
        if group_idx_info["type"] == "attr":
            if group_idx_info.get("dtype") == "none":
                group_idx = None
            else:
                group_idx = group_idx_info["value"]
        else:
            if "data" in group_idx_info:
                group_idx = torch.tensor(group_idx_info["data"], dtype=DTYPE_MAP[group_idx_info["dtype"]]).reshape(group_idx_info["shape"])
            else:
                group_idx = torch.randint(group_idx_info["range"][0], group_idx_info["range"][1] + 1, tuple(group_idx_info["shape"]), dtype=DTYPE_MAP[group_idx_info["dtype"]])

        input_groups.append([grad_y, group_idx])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_25_GroupedBiasAddGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        grad_y_info = entries[0]
        group_idx_info = entries[1]
        if "data" in grad_y_info:
            grad_y = torch.tensor(grad_y_info["data"], dtype=DTYPE_MAP[grad_y_info["dtype"]]).reshape(grad_y_info["shape"])
        else:
            grad_y = torch.rand(grad_y_info["shape"], dtype=DTYPE_MAP[grad_y_info["dtype"]])
        if group_idx_info["type"] == "attr":
            if group_idx_info.get("dtype") == "none":
                group_idx = None
            else:
                group_idx = group_idx_info["value"]
        else:
            if "data" in group_idx_info:
                group_idx = torch.tensor(group_idx_info["data"], dtype=DTYPE_MAP[group_idx_info["dtype"]]).reshape(group_idx_info["shape"])
            else:
                group_idx = torch.randint(group_idx_info["range"][0], group_idx_info["range"][1] + 1, tuple(group_idx_info["shape"]), dtype=DTYPE_MAP[group_idx_info["dtype"]])
        init_groups.append([grad_y, group_idx])
    return init_groups
