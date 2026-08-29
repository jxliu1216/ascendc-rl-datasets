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

    def forward(self, dy: torch.Tensor, x: torch.Tensor, gx: torch.Tensor, gamma: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, alpha: float) -> List[torch.Tensor]:
        dy_fp32 = dy.to(torch.float32)
        x_fp32 = x.to(torch.float32)
        gx_fp32 = gx.to(torch.float32)
        gamma_fp32 = gamma.to(torch.float32)
        mean_fp32 = mean.to(torch.float32)
        rstd_fp32 = rstd.to(torch.float32)
        D = float(torch.prod(torch.tensor(gamma_fp32.shape)))
        tmpone = dy_fp32 * gamma_fp32
        tmptwo = alpha * x_fp32 + gx_fp32 - mean_fp32
        reduction_dims = tuple(range(x_fp32.dim() - gamma_fp32.dim(), x_fp32.dim()))
        d_var = torch.sum(-0.5 * tmpone * tmptwo * rstd_fp32.pow(3), dim=reduction_dims, keepdim=True)
        d_mean = torch.sum(-1.0 * tmpone * rstd_fp32, dim=reduction_dims, keepdim=True)
        dgx = tmpone * rstd_fp32 + 2.0 / D * d_var * tmptwo + 1.0 / D * d_mean
        dx = alpha * dgx
        d_reduction_dims_for_gamma_beta = tuple(range(dy_fp32.dim() - gamma_fp32.dim()))
        dbeta = torch.sum(dy_fp32, dim=d_reduction_dims_for_gamma_beta, keepdim=False)
        dgamma = torch.sum(dy_fp32 * rstd_fp32 * tmptwo, dim=d_reduction_dims_for_gamma_beta, keepdim=False)
        dx = dx.to(x.dtype)
        dgx = dgx.to(gx.dtype)
        return [dx, dgx, dbeta, dgamma]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_29_DeepNormGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dy_info = inputs[0]
        x_info = inputs[1]
        gx_info = inputs[2]
        gamma_info = inputs[3]
        mean_info = inputs[4]
        rstd_info = inputs[5]
        alpha_info = inputs[6]

        if "data" in dy_info:
            dy = torch.tensor(dy_info["data"], dtype=DTYPE_MAP[dy_info["dtype"]]).reshape(dy_info["shape"])
        else:
            dy = torch.rand(dy_info["shape"], dtype=DTYPE_MAP[dy_info["dtype"]])
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in gx_info:
            gx = torch.tensor(gx_info["data"], dtype=DTYPE_MAP[gx_info["dtype"]]).reshape(gx_info["shape"])
        else:
            gx = torch.rand(gx_info["shape"], dtype=DTYPE_MAP[gx_info["dtype"]])
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        if "data" in mean_info:
            mean = torch.tensor(mean_info["data"], dtype=DTYPE_MAP[mean_info["dtype"]]).reshape(mean_info["shape"])
        else:
            mean = torch.rand(mean_info["shape"], dtype=DTYPE_MAP[mean_info["dtype"]]) * (mean_info["range"][1] - mean_info["range"][0]) + mean_info["range"][0]
        if "data" in rstd_info:
            rstd = torch.tensor(rstd_info["data"], dtype=DTYPE_MAP[rstd_info["dtype"]]).reshape(rstd_info["shape"])
        else:
            rstd = torch.rand(rstd_info["shape"], dtype=DTYPE_MAP[rstd_info["dtype"]]) * (rstd_info["range"][1] - rstd_info["range"][0]) + rstd_info["range"][0]
        alpha = alpha_info["value"]

        input_groups.append([dy, x, gx, gamma, mean, rstd, alpha])
    return input_groups


def get_init_inputs():
    return []
