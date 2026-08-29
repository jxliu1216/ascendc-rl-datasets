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

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor, gamma: torch.Tensor) -> List[torch.Tensor]:
        normalized_dim_size = x.shape[-1]
        dgamma_reduction_dims = tuple(range(dy.dim() - gamma.dim()))
        dgamma = (dy * x * rstd).sum(dim=dgamma_reduction_dims, keepdim=False)
        dx = dy * gamma * rstd - (dy * gamma * x * rstd.pow(3)).sum(dim=-1, keepdim=True) * x / normalized_dim_size
        return [dx, dgamma]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_139_RmsNormGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dy_info = inputs[0]
        x_info = inputs[1]
        rstd_info = inputs[2]
        gamma_info = inputs[3]

        if "data" in dy_info:
            dy = torch.tensor(dy_info["data"], dtype=DTYPE_MAP[dy_info["dtype"]]).reshape(dy_info["shape"])
        else:
            dy = torch.rand(dy_info["shape"], dtype=DTYPE_MAP[dy_info["dtype"]])
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in rstd_info:
            rstd = torch.tensor(rstd_info["data"], dtype=DTYPE_MAP[rstd_info["dtype"]]).reshape(rstd_info["shape"])
        else:
            rstd = torch.rand(rstd_info["shape"], dtype=DTYPE_MAP[rstd_info["dtype"]]) * (rstd_info["range"][1] - rstd_info["range"][0]) + rstd_info["range"][0]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])

        input_groups.append([dy, x, rstd, gamma])
    return input_groups


def get_init_inputs():
    return []
