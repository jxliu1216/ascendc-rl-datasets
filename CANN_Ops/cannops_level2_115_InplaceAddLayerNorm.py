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

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float, additional_out: bool):
        super(Model, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.epsilon = float(epsilon)
        self.additional_out = additional_out

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, bias: torch.Tensor) -> List[torch.Tensor]:
        dtype = x1.dtype
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        if bias is not None:
            bias = bias.to(torch.float32)
        x1.add_(x2)
        if bias is not None:
            x1.add_(bias)
        if self.additional_out:
            x2.copy_(x1)
        mean = x1.mean(dim=-1, keepdim=True)
        var = x1.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + self.epsilon)
        x1.sub_(mean)
        x1.mul_(rstd)
        x1.mul_(self.gamma).add_(self.beta)
        x1 = x1.to(dtype)
        x2 = x2.to(dtype)
        if self.additional_out:
            return [x1, mean, rstd, x2]
        else:
            return [x1]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_115_InplaceAddLayerNorm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]
        bias_info = inputs[2]

        if "data" in x1_info:
            x1 = torch.tensor(x1_info["data"], dtype=DTYPE_MAP[x1_info["dtype"]]).reshape(x1_info["shape"])
        else:
            x1 = torch.rand(x1_info["shape"], dtype=DTYPE_MAP[x1_info["dtype"]])
        if "data" in x2_info:
            x2 = torch.tensor(x2_info["data"], dtype=DTYPE_MAP[x2_info["dtype"]]).reshape(x2_info["shape"])
        else:
            x2 = torch.rand(x2_info["shape"], dtype=DTYPE_MAP[x2_info["dtype"]])
        if bias_info["type"] == "attr":
            if bias_info.get("dtype") == "none":
                bias = None
            else:
                bias = bias_info["value"]
        else:
            if "data" in bias_info:
                bias = torch.tensor(bias_info["data"], dtype=DTYPE_MAP[bias_info["dtype"]]).reshape(bias_info["shape"])
            else:
                bias = torch.rand(bias_info["shape"], dtype=DTYPE_MAP[bias_info["dtype"]])

        input_groups.append([x1, x2, bias])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_115_InplaceAddLayerNorm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        gamma_info = entries[0]
        beta_info = entries[1]
        epsilon_info = entries[2]
        additional_out_info = entries[3]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        if "data" in beta_info:
            beta = torch.tensor(beta_info["data"], dtype=DTYPE_MAP[beta_info["dtype"]]).reshape(beta_info["shape"])
        else:
            beta = torch.rand(beta_info["shape"], dtype=DTYPE_MAP[beta_info["dtype"]])
        epsilon = epsilon_info["value"]
        additional_out = additional_out_info["value"]
        init_groups.append([gamma, beta, epsilon, additional_out])
    return init_groups
