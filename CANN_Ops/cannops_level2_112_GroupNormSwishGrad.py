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

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, num_groups: int, swish_scale: float):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.swish_scale = swish_scale

    def forward(self, dy: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, dgamma_is_require: bool, dbeta_is_require: bool) -> torch.Tensor:
        dtype_orig = x.dtype
        dy_hp = dy.to(torch.float32)
        mean_hp = mean.to(torch.float32)
        rstd_hp = rstd.to(torch.float32)
        x_hp = x.to(torch.float32)
        gamma_hp = gamma.to(torch.float32)
        beta_hp = beta.to(torch.float32)
        batch_num = x_hp.size(0)
        num_channels = x_hp.size(1)
        remaining_dims = x_hp.size()[2:]
        hw = 1
        for size in remaining_dims:
            hw *= size
        num_per_group_channel = num_channels // self.num_groups
        num_per_group_total = float(num_per_group_channel * hw)
        x_reshaped = x_hp.reshape((batch_num, num_channels, hw))
        dy_reshaped = dy_hp.reshape((batch_num, num_channels, hw))
        dL_dgamma_sum = torch.zeros_like(gamma_hp)
        dL_dbeta_sum = torch.zeros_like(beta_hp)
        dL_dx_out = torch.zeros_like(x_reshaped)
        for n_i in range(batch_num):
            for g_i in range(self.num_groups):
                ch_start = g_i * num_per_group_channel
                ch_end = (g_i + 1) * num_per_group_channel
                x_group_slice = x_reshaped[n_i, ch_start:ch_end, :]
                dy_group_slice = dy_reshaped[n_i, ch_start:ch_end, :]
                mean_x = mean_hp[n_i, g_i]
                rstd_x = rstd_hp[n_i, g_i]
                x_norm_i = (x_group_slice - mean_x) * rstd_x
                gamma_group_slice = gamma_hp[ch_start:ch_end].view(num_per_group_channel, 1)
                beta_group_slice = beta_hp[ch_start:ch_end].view(num_per_group_channel, 1)
                gn_output_group = x_norm_i * gamma_group_slice + beta_group_slice
                dswish_dz_intermediate = gn_output_group * -self.swish_scale
                dswish_dz_intermediate = torch.exp(dswish_dz_intermediate)
                dswish_dz_intermediate = dswish_dz_intermediate + 1.0
                tmp_res_val = gn_output_group / dswish_dz_intermediate
                tmp_res_val = gn_output_group - tmp_res_val
                tmp_res_val = tmp_res_val + 1.0
                dswish_dz = tmp_res_val / dswish_dz_intermediate
                d_gn_output = dswish_dz * dy_group_slice
                temp_1 = torch.sum(d_gn_output, dim=1)
                temp_2 = torch.sum(d_gn_output * x_norm_i, dim=1)
                dL_dbeta_sum[ch_start:ch_end] += temp_1
                dL_dgamma_sum[ch_start:ch_end] += temp_2
                c1 = torch.sum(temp_1 * gamma_group_slice.squeeze(1)) / num_per_group_total
                c2 = torch.sum(temp_2 * gamma_group_slice.squeeze(1)) / num_per_group_total
                dL_dx_G_C = torch.zeros_like(x_group_slice)
                for i in range(num_per_group_channel):
                    dL_dx_G_C[i] = rstd_x * (d_gn_output[i] * gamma_group_slice[i] - x_norm_i[i] * c2 - c1)
                dL_dx_out[n_i, ch_start:ch_end, :] = dL_dx_G_C
        dL_dx_out = dL_dx_out.reshape(x_hp.shape)
        dx_result = dL_dx_out.to(dtype_orig)
        return dx_result

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_112_GroupNormSwishGrad.json')
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
        beta_info = inputs[5]
        dgamma_is_require_info = inputs[6]
        dbeta_is_require_info = inputs[7]

        if "data" in dy_info:
            dy = torch.tensor(dy_info["data"], dtype=DTYPE_MAP[dy_info["dtype"]]).reshape(dy_info["shape"])
        else:
            dy = torch.rand(dy_info["shape"], dtype=DTYPE_MAP[dy_info["dtype"]]) * (dy_info["range"][1] - dy_info["range"][0]) + dy_info["range"][0]
        if "data" in mean_info:
            mean = torch.tensor(mean_info["data"], dtype=DTYPE_MAP[mean_info["dtype"]]).reshape(mean_info["shape"])
        else:
            mean = torch.rand(mean_info["shape"], dtype=DTYPE_MAP[mean_info["dtype"]])
        if "data" in rstd_info:
            rstd = torch.tensor(rstd_info["data"], dtype=DTYPE_MAP[rstd_info["dtype"]]).reshape(rstd_info["shape"])
        else:
            rstd = torch.rand(rstd_info["shape"], dtype=DTYPE_MAP[rstd_info["dtype"]])
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
            beta = torch.rand(beta_info["shape"], dtype=DTYPE_MAP[beta_info["dtype"]])
        dgamma_is_require = dgamma_is_require_info["value"]
        dbeta_is_require = dbeta_is_require_info["value"]

        input_groups.append([dy, mean, rstd, x, gamma, beta, dgamma_is_require, dbeta_is_require])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_112_GroupNormSwishGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        num_groups_info = entries[0]
        swish_scale_info = entries[1]
        num_groups = num_groups_info["value"]
        swish_scale = swish_scale_info["value"]
        init_groups.append([num_groups, swish_scale])
    return init_groups
