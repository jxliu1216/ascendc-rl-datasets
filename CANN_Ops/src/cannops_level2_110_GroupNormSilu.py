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

def _group_norm_silu_ref(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, eps: float, activate_silu: bool) -> List[torch.Tensor]:
    """PyTorch reference: GroupNorm + optional SiLU (x * sigmoid(x)). Returns [y] for compare."""
    input_dtype = x.dtype
    N, C = (x.shape[0], x.shape[1])
    remaining = x.shape[2:]
    HxW = 1
    for s in remaining:
        HxW *= s
    x_fp32 = x.to(torch.float32)
    gamma_fp32 = gamma.to(torch.float32)
    beta_fp32 = beta.to(torch.float32)
    gn_out, mean_out, rstd_out = torch.ops.aten.native_group_norm(x_fp32, gamma_fp32, beta_fp32, N, C, HxW, num_groups, eps)
    if activate_silu:
        final_out = gn_out * torch.sigmoid(gn_out)
    else:
        final_out = gn_out
    return [final_out.to(input_dtype)]

class Model(nn.Module):

    def __init__(self, num_channels: int, num_groups: int, eps: float, activate_silu: bool):
        super(Model, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.activate_silu = activate_silu

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> List[torch.Tensor]:
        return _group_norm_silu_ref(x, gamma, beta, self.num_groups, self.eps, self.activate_silu)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_110_GroupNormSilu.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        gamma_info = inputs[1]
        beta_info = inputs[2]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        if "data" in beta_info:
            beta = torch.tensor(beta_info["data"], dtype=DTYPE_MAP[beta_info["dtype"]]).reshape(beta_info["shape"])
        else:
            beta = torch.rand(beta_info["shape"], dtype=DTYPE_MAP[beta_info["dtype"]]) * (beta_info["range"][1] - beta_info["range"][0]) + beta_info["range"][0]

        input_groups.append([x, gamma, beta])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_110_GroupNormSilu.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        num_channels_info = entries[0]
        num_groups_info = entries[1]
        eps_info = entries[2]
        activate_silu_info = entries[3]
        num_channels = num_channels_info["value"]
        num_groups = num_groups_info["value"]
        eps = eps_info["value"]
        activate_silu = activate_silu_info["value"]
        init_groups.append([num_channels, num_groups, eps, activate_silu])
    return init_groups
