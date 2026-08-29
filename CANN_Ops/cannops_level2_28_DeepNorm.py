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

    def __init__(self, beta: torch.Tensor, gamma: torch.Tensor, alpha: float, epsilon: float):
        super(Model, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, gx: torch.Tensor) -> List[torch.Tensor]:
        x_fp32 = x.to(torch.float32)
        gx_fp32 = gx.to(torch.float32)
        beta_fp32 = self.beta.to(torch.float32)
        gamma_fp32 = self.gamma.to(torch.float32)
        x_add = x_fp32 * self.alpha + gx_fp32
        mean = x_add.mean(-1, keepdim=True)
        diff = x_add - mean
        variance = diff.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + self.epsilon)
        y_out = gamma_fp32 * diff * rstd + beta_fp32
        mean = mean.to(x.dtype)
        rstd = rstd.to(x.dtype)
        y_out = y_out.to(x.dtype)
        return [mean, rstd, y_out]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_28_DeepNorm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        gx_info = inputs[1]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in gx_info:
            gx = torch.tensor(gx_info["data"], dtype=DTYPE_MAP[gx_info["dtype"]]).reshape(gx_info["shape"])
        else:
            gx = torch.rand(gx_info["shape"], dtype=DTYPE_MAP[gx_info["dtype"]])

        input_groups.append([x, gx])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_28_DeepNorm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        beta_info = entries[0]
        gamma_info = entries[1]
        alpha_info = entries[2]
        epsilon_info = entries[3]
        if "data" in beta_info:
            beta = torch.tensor(beta_info["data"], dtype=DTYPE_MAP[beta_info["dtype"]]).reshape(beta_info["shape"])
        else:
            beta = torch.rand(beta_info["shape"], dtype=DTYPE_MAP[beta_info["dtype"]])
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        alpha = alpha_info["value"]
        epsilon = epsilon_info["value"]
        init_groups.append([beta, gamma, alpha, epsilon])
    return init_groups
