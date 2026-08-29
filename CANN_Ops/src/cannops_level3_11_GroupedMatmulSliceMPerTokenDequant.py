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

from typing import List, Optional
import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    实现add算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, per_token_scale: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        results = []
        start_idx = 0
        for ind, end_idx in enumerate(group_list):
            m = end_idx - start_idx
            if m > 0:
                group_a = a[start_idx:end_idx]
                group_b = b[ind]
                result = torch.matmul(group_a.to(torch.int32), group_b.to(torch.int32))
                result = result.to(torch.float32) * scale[ind].unsqueeze(0).to(torch.float32) * per_token_scale[start_idx:end_idx].unsqueeze(1).to(torch.float32)
                results.append(result.flatten())
                start_idx = end_idx
        return torch.cat(results)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_11_GroupedMatmulSliceMPerTokenDequant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        a_info = inputs[0]
        b_info = inputs[1]
        scale_info = inputs[2]
        per_token_scale_info = inputs[3]
        group_list_info = inputs[4]

        if "data" in a_info:
            a = torch.tensor(a_info["data"], dtype=DTYPE_MAP[a_info["dtype"]]).reshape(a_info["shape"])
        else:
            a = torch.randint(a_info["range"][0], a_info["range"][1] + 1, tuple(a_info["shape"]), dtype=DTYPE_MAP[a_info["dtype"]])
        if "data" in b_info:
            b = torch.tensor(b_info["data"], dtype=DTYPE_MAP[b_info["dtype"]]).reshape(b_info["shape"])
        else:
            b = torch.randint(b_info["range"][0], b_info["range"][1] + 1, tuple(b_info["shape"]), dtype=DTYPE_MAP[b_info["dtype"]])
        if "data" in scale_info:
            scale = torch.tensor(scale_info["data"], dtype=DTYPE_MAP[scale_info["dtype"]]).reshape(scale_info["shape"])
        else:
            scale = torch.rand(scale_info["shape"], dtype=DTYPE_MAP[scale_info["dtype"]])
        if "data" in per_token_scale_info:
            per_token_scale = torch.tensor(per_token_scale_info["data"], dtype=DTYPE_MAP[per_token_scale_info["dtype"]]).reshape(per_token_scale_info["shape"])
        else:
            per_token_scale = torch.rand(per_token_scale_info["shape"], dtype=DTYPE_MAP[per_token_scale_info["dtype"]])
        if "data" in group_list_info:
            group_list = torch.tensor(group_list_info["data"], dtype=DTYPE_MAP[group_list_info["dtype"]]).reshape(group_list_info["shape"])
        else:
            group_list = torch.randint(group_list_info["range"][0], group_list_info["range"][1] + 1, tuple(group_list_info["shape"]), dtype=DTYPE_MAP[group_list_info["dtype"]])

        input_groups.append([a, b, scale, per_token_scale, group_list])
    return input_groups


def get_init_inputs():
    return []
