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
    实现TopKV3算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, self_tensor: torch.Tensor, k: int, dim: int, largest: bool, sorted: bool):
        """
        实现TopKV3算子功能。

        Args:
            self_tensor: 输入张量
            k: 计算维度上输出的极值个数
            dim: 计算维度
            largest: 布尔型，True表示计算维度上的结果应由大到小输出，False表示计算维度上的结果由小到大输出
            sorted: 布尔型，True表示输出结果排序，False表示输出结果不排序

        Returns:
            输入张量在指定维度上的k个极值及索引
        """
        largest = bool(largest)
        sorted = bool(sorted)
        values, indices = torch.topk(self_tensor, k=k, dim=dim, largest=largest, sorted=sorted)
        return [values, indices]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_50_TopKV3.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        self_tensor_info = inputs[0]
        k_info = inputs[1]
        dim_info = inputs[2]
        largest_info = inputs[3]
        sorted_info = inputs[4]

        if "data" in self_tensor_info:
            self_tensor = torch.tensor(self_tensor_info["data"], dtype=DTYPE_MAP[self_tensor_info["dtype"]]).reshape(self_tensor_info["shape"])
        else:
            self_tensor = torch.randn(self_tensor_info["shape"], dtype=DTYPE_MAP[self_tensor_info["dtype"]])
        k = k_info["value"]
        dim = dim_info["value"]
        largest = largest_info["value"]
        sorted = sorted_info["value"]

        input_groups.append([self_tensor, k, dim, largest, sorted])
    return input_groups


def get_init_inputs():
    return []
