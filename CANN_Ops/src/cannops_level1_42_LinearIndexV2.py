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
    实现LinearIndexV2算子功能的模型 - 将多维索引转换为线性索引。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, indices_list: List[torch.Tensor], stride: torch.Tensor, value_size: torch.Tensor) -> torch.Tensor:
        """
        实现LinearIndexV2算子功能。
        
        计算公式: output += (indices[i] % value_size[i]) * stride[i]

        Args:
            indices_list: 索引张量列表
            stride: 步长张量 [dim_num]
            value_size: 维度大小张量 [dim_num]

        Returns:
            线性索引张量 (int32)
        """
        output = torch.zeros_like(indices_list[0], dtype=torch.int32)
        for i, indices in enumerate(indices_list):
            indices_int64 = indices.to(torch.int64)
            value_size_val = value_size[i].item()
            stride_val = stride[i].item()
            quotient = torch.div(indices_int64, value_size_val, rounding_mode='floor')
            remainder = indices_int64 - quotient * value_size_val
            contribution = (remainder * stride_val).to(torch.int32)
            output = output + contribution
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_42_LinearIndexV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        indices_list_info = inputs[0]
        stride_info = inputs[1]
        value_size_info = inputs[2]

        indices_list = []
        for _shape in indices_list_info["shapes"]:
            _t = torch.randint({"dtype": indices_list_info["dtype"], "shape": _shape, "range": indices_list_info.get("range", [0, 1]), "mean": indices_list_info.get("mean", 0.0), "std": indices_list_info.get("std", 1.0), "value": indices_list_info.get("value")}["range"][0], {"dtype": indices_list_info["dtype"], "shape": _shape, "range": indices_list_info.get("range", [0, 1]), "mean": indices_list_info.get("mean", 0.0), "std": indices_list_info.get("std", 1.0), "value": indices_list_info.get("value")}["range"][1] + 1, tuple({"dtype": indices_list_info["dtype"], "shape": _shape, "range": indices_list_info.get("range", [0, 1]), "mean": indices_list_info.get("mean", 0.0), "std": indices_list_info.get("std", 1.0), "value": indices_list_info.get("value")}["shape"]), dtype=DTYPE_MAP[{"dtype": indices_list_info["dtype"], "shape": _shape, "range": indices_list_info.get("range", [0, 1]), "mean": indices_list_info.get("mean", 0.0), "std": indices_list_info.get("std", 1.0), "value": indices_list_info.get("value")}["dtype"]])
            indices_list.append(_t)
        if "data" in stride_info:
            stride = torch.tensor(stride_info["data"], dtype=DTYPE_MAP[stride_info["dtype"]]).reshape(stride_info["shape"])
        else:
            stride = torch.randint(stride_info["range"][0], stride_info["range"][1] + 1, tuple(stride_info["shape"]), dtype=DTYPE_MAP[stride_info["dtype"]])
        if "data" in value_size_info:
            value_size = torch.tensor(value_size_info["data"], dtype=DTYPE_MAP[value_size_info["dtype"]]).reshape(value_size_info["shape"])
        else:
            value_size = torch.randint(value_size_info["range"][0], value_size_info["range"][1] + 1, tuple(value_size_info["shape"]), dtype=DTYPE_MAP[value_size_info["dtype"]])

        input_groups.append([indices_list, stride, value_size])
    return input_groups


def get_init_inputs():
    return []
