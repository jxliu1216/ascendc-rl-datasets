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

    def __init__(self, gamma: torch.Tensor, epsilon: float):
        super(Model, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            y1: FP32 归一化结果
            y2: FP16/BF16 归一化结果
            rstd: 倒数标准差 (FP32)
            x: 残差加法后的中间结果 (FP16/BF16)
        """
        dtype = x1.dtype
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        x = x1 + x2
        variance = torch.mean(x.pow(2), dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + self.epsilon)
        y1 = x * rstd * self.gamma
        y2 = y1.to(x1.dtype)
        return [y1, y2, rstd, x]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_8_AddRmsNormCast.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]

        if "data" in x1_info:
            x1 = torch.tensor(x1_info["data"], dtype=DTYPE_MAP[x1_info["dtype"]]).reshape(x1_info["shape"])
        else:
            x1 = torch.rand(x1_info["shape"], dtype=DTYPE_MAP[x1_info["dtype"]])
        if "data" in x2_info:
            x2 = torch.tensor(x2_info["data"], dtype=DTYPE_MAP[x2_info["dtype"]]).reshape(x2_info["shape"])
        else:
            x2 = torch.rand(x2_info["shape"], dtype=DTYPE_MAP[x2_info["dtype"]])

        input_groups.append([x1, x2])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_8_AddRmsNormCast.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        gamma_info = entries[0]
        epsilon_info = entries[1]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        epsilon = epsilon_info["value"]
        init_groups.append([gamma, epsilon])
    return init_groups
