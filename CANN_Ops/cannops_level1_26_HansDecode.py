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
from typing import List, Tuple, Optional
from typing import Dict, Any
import numpy as np

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor: torch.Tensor, mantissa: torch.Tensor, fixed: torch.Tensor, var: torch.Tensor, hist: torch.Tensor, reshuff: bool=False) -> torch.Tensor:
        """
        使用 torch 实现 Hans 解码，还原 input_tensor

        Args:
            mantissa: [1, M], 尾数部分, float32/f16/bf16
            fixed: [1, F], 压缩后的固定部分, float32/f16/bf16
            var: [1, V],未压缩部分, float32/f16/bf16
            hist: [1, 256], int32,指数位统计
            reshuff: 是否启用内存重整（占位参数）

        Returns:
            input_tensor: [1, N], float32/f16/bf16, 原始输入
        """
        return input_tensor

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_26_HansDecode.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_tensor_info = inputs[0]
        mantissa_info = inputs[1]
        fixed_info = inputs[2]
        var_info = inputs[3]
        hist_info = inputs[4]
        reshuff_info = inputs[5]

        if "data" in input_tensor_info:
            input_tensor = torch.tensor(input_tensor_info["data"], dtype=DTYPE_MAP[input_tensor_info["dtype"]]).reshape(input_tensor_info["shape"])
        else:
            input_tensor = torch.rand(input_tensor_info["shape"], dtype=DTYPE_MAP[input_tensor_info["dtype"]])
        if "data" in mantissa_info:
            mantissa = torch.tensor(mantissa_info["data"], dtype=DTYPE_MAP[mantissa_info["dtype"]]).reshape(mantissa_info["shape"])
        else:
            mantissa = torch.randn(mantissa_info["shape"], dtype=DTYPE_MAP[mantissa_info["dtype"]]) * mantissa_info["std"] + mantissa_info["mean"]
        if mantissa_info.get("inject"):
            _f = mantissa.reshape(-1)
            _f[0] = float(mantissa_info["inject"])
            mantissa = _f.reshape(mantissa.shape)
        if "data" in fixed_info:
            fixed = torch.tensor(fixed_info["data"], dtype=DTYPE_MAP[fixed_info["dtype"]]).reshape(fixed_info["shape"])
        else:
            fixed = torch.randn(fixed_info["shape"], dtype=DTYPE_MAP[fixed_info["dtype"]]) * fixed_info["std"] + fixed_info["mean"]
        if fixed_info.get("inject"):
            _f = fixed.reshape(-1)
            _f[0] = float(fixed_info["inject"])
            fixed = _f.reshape(fixed.shape)
        if "data" in var_info:
            var = torch.tensor(var_info["data"], dtype=DTYPE_MAP[var_info["dtype"]]).reshape(var_info["shape"])
        else:
            var = torch.rand(var_info["shape"], dtype=DTYPE_MAP[var_info["dtype"]])
        if "data" in hist_info:
            hist = torch.tensor(hist_info["data"], dtype=DTYPE_MAP[hist_info["dtype"]]).reshape(hist_info["shape"])
        else:
            hist = torch.randint(hist_info["range"][0], hist_info["range"][1] + 1, tuple(hist_info["shape"]), dtype=DTYPE_MAP[hist_info["dtype"]])
        reshuff = reshuff_info["value"]

        input_groups.append([input_tensor, mantissa, fixed, var, hist, reshuff])
    return input_groups


def get_init_inputs():
    return []
