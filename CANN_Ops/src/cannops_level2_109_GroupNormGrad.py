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

def _group_norm_grad_ref(dy, x, mean, rstd, gamma, num_groups):
    """PyTorch reference: GroupNorm backward. Returns dx, dgamma, dbeta as list."""
    dtype_orig = x.dtype
    dy_hp = dy.to(torch.float32)
    mean_hp = mean.to(torch.float32)
    rstd_hp = rstd.to(torch.float32)
    x_hp = x.to(torch.float32)
    gamma_hp = gamma.to(torch.float32)
    batch_num, num_channels = (x_hp.size(0), x_hp.size(1))
    remaining = x_hp.size()[2:]
    hw = 1
    for s in remaining:
        hw *= s
    num_per_group_channel = num_channels // num_groups
    num_per_group_total = float(num_per_group_channel * hw)
    x_reshaped = x_hp.reshape((batch_num, num_channels, hw))
    dy_reshaped = dy_hp.reshape((batch_num, num_channels, hw))
    dgamma_sum = torch.zeros_like(gamma_hp)
    dbeta_sum = torch.zeros_like(gamma_hp)
    dx_out = torch.zeros_like(x_reshaped)
    for n_i in range(batch_num):
        for g_i in range(num_groups):
            ch_start = g_i * num_per_group_channel
            ch_end = (g_i + 1) * num_per_group_channel
            x_g = x_reshaped[n_i, ch_start:ch_end, :]
            dy_g = dy_reshaped[n_i, ch_start:ch_end, :]
            mean_x = mean_hp[n_i, g_i]
            rstd_x = rstd_hp[n_i, g_i]
            x_norm = (x_g - mean_x) * rstd_x
            gamma_g = gamma_hp[ch_start:ch_end].view(num_per_group_channel, 1)
            temp_1 = torch.sum(dy_g, dim=1)
            temp_2 = torch.sum(dy_g * x_norm, dim=1)
            dbeta_sum[ch_start:ch_end] += temp_1
            dgamma_sum[ch_start:ch_end] += temp_2
            c1 = torch.sum(temp_1 * gamma_g.squeeze(1)) / num_per_group_total
            c2 = torch.sum(temp_2 * gamma_g.squeeze(1)) / num_per_group_total
            dx_g = rstd_x * (dy_g * gamma_g - x_norm * c2 - c1)
            dx_out[n_i, ch_start:ch_end, :] = dx_g
    dx_out = dx_out.reshape(x_hp.shape)
    return [dx_out.to(dtype_orig), dgamma_sum.to(dtype_orig), dbeta_sum.to(dtype_orig)]

class Model(nn.Module):

    def __init__(self, num_groups: int, data_format: str):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.data_format = data_format

    def forward(self, dy: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor, dx_is_require: bool, dgamma_is_require: bool, dbeta_is_require: bool) -> List[torch.Tensor]:
        dx, dgamma, dbeta = _group_norm_grad_ref(dy, x, mean, rstd, gamma, self.num_groups)
        out = []
        if dx_is_require:
            out.append(dx)
        if dgamma_is_require:
            out.append(dgamma)
        if dbeta_is_require:
            out.append(dbeta)
        return out

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_109_GroupNormGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dy_info = inputs[0]
        mean_info = inputs[1]
        rstd_info = inputs[2]
        x_info = inputs[3]
        gamma_info = inputs[4]
        dx_is_require_info = inputs[5]
        dgamma_is_require_info = inputs[6]
        dbeta_is_require_info = inputs[7]

        if "data" in dy_info:
            dy = torch.tensor(dy_info["data"], dtype=DTYPE_MAP[dy_info["dtype"]]).reshape(dy_info["shape"])
        else:
            dy = torch.rand(dy_info["shape"], dtype=DTYPE_MAP[dy_info["dtype"]]) * (dy_info["range"][1] - dy_info["range"][0]) + dy_info["range"][0]
        if "data" in mean_info:
            mean = torch.tensor(mean_info["data"], dtype=DTYPE_MAP[mean_info["dtype"]]).reshape(mean_info["shape"])
        else:
            mean = torch.rand(mean_info["shape"], dtype=DTYPE_MAP[mean_info["dtype"]]) * (mean_info["range"][1] - mean_info["range"][0]) + mean_info["range"][0]
        if "data" in rstd_info:
            rstd = torch.tensor(rstd_info["data"], dtype=DTYPE_MAP[rstd_info["dtype"]]).reshape(rstd_info["shape"])
        else:
            rstd = torch.rand(rstd_info["shape"], dtype=DTYPE_MAP[rstd_info["dtype"]]) * (rstd_info["range"][1] - rstd_info["range"][0]) + rstd_info["range"][0]
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]]) * (gamma_info["range"][1] - gamma_info["range"][0]) + gamma_info["range"][0]
        dx_is_require = dx_is_require_info["value"]
        dgamma_is_require = dgamma_is_require_info["value"]
        dbeta_is_require = dbeta_is_require_info["value"]

        input_groups.append([dy, mean, rstd, x, gamma, dx_is_require, dgamma_is_require, dbeta_is_require])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_109_GroupNormGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        num_groups_info = entries[0]
        data_format_info = entries[1]
        num_groups = num_groups_info["value"]
        data_format = data_format_info["value"]
        init_groups.append([num_groups, data_format])
    return init_groups
