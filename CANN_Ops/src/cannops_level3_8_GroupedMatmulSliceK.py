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

    def forward(self, a: torch.Tensor, b: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        results = []
        m = a.shape[0]
        n = b.shape[1]
        offset_a = 0
        offset_b = 0
        a = a.flatten()
        b = b.flatten()
        start_idx = 0
        for end_idx in group_list:
            k = end_idx - start_idx
            if k > 0:
                size_a = m * k
                size_b = k * n
                group_a_flat = a[offset_a:offset_a + size_a]
                group_b_flat = b[offset_b:offset_b + size_b]
                group_a = group_a_flat.view(k, m).transpose(0, 1)
                group_b = group_b_flat.view(k, n)
                result = torch.matmul(group_a, group_b).flatten()
                results.append(result)
                offset_a += size_a
                offset_b += size_b
                start_idx = end_idx
            else:
                results.append(torch.zeros([m, n], device=a.device, dtype=a.dtype).flatten())
        return torch.cat(results)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_8_GroupedMatmulSliceK.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        a_info = inputs[0]
        b_info = inputs[1]
        group_list_info = inputs[2]

        if "data" in a_info:
            a = torch.tensor(a_info["data"], dtype=DTYPE_MAP[a_info["dtype"]]).reshape(a_info["shape"])
        else:
            a = torch.rand(a_info["shape"], dtype=DTYPE_MAP[a_info["dtype"]]) * (a_info["range"][1] - a_info["range"][0]) + a_info["range"][0]
        if "data" in b_info:
            b = torch.tensor(b_info["data"], dtype=DTYPE_MAP[b_info["dtype"]]).reshape(b_info["shape"])
        else:
            b = torch.rand(b_info["shape"], dtype=DTYPE_MAP[b_info["dtype"]]) * (b_info["range"][1] - b_info["range"][0]) + b_info["range"][0]
        if "data" in group_list_info:
            group_list = torch.tensor(group_list_info["data"], dtype=DTYPE_MAP[group_list_info["dtype"]]).reshape(group_list_info["shape"])
        else:
            group_list = torch.randint(group_list_info["range"][0], group_list_info["range"][1] + 1, tuple(group_list_info["shape"]), dtype=DTYPE_MAP[group_list_info["dtype"]])

        input_groups.append([a, b, group_list])
    return input_groups


def get_init_inputs():
    return []
