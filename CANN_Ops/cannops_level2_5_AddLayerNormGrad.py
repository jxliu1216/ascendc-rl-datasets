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

    def __init__(self, normalized_shape: List[int]):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(self, dy: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, rstd: torch.Tensor, mean: torch.Tensor, gamma: torch.Tensor, dsum: Optional[torch.Tensor]) -> List[torch.Tensor]:
        dy = dy.float()
        gamma = gamma.float()
        x = x1.float() + x2.float()
        d = float(torch.prod(torch.tensor(self.normalized_shape)))
        batch_axis = tuple(range(x.dim() - len(self.normalized_shape)))
        feature_axis = tuple(range(x.dim() - len(self.normalized_shape), x.dim()))
        pd_xl = dy * gamma
        x_hat = x - mean
        pd_var_first_part = -0.5 * pd_xl * x_hat * torch.pow(rstd, 3)
        pd_var = torch.sum(pd_var_first_part, dim=feature_axis, keepdim=True)
        pd_mean_first_part = torch.sum(-1.0 * pd_xl * rstd, dim=feature_axis, keepdim=True)
        pd_mean_second_part = torch.sum(x_hat, dim=feature_axis, keepdim=True)
        pd_mean = pd_mean_first_part + pd_var * (-2.0 / d) * pd_mean_second_part
        pd_x_first_part = pd_xl * rstd
        pd_x_second_part = pd_var * (2.0 / d) * x_hat + pd_mean * (1.0 / d)
        golden_x = pd_x_first_part + pd_x_second_part
        if dsum is not None:
            golden_x += dsum.float()
        golden_gamma = torch.sum(dy * x_hat * rstd, dim=batch_axis)
        golden_beta = torch.sum(dy, dim=batch_axis)
        return [golden_x.to(dy.dtype), golden_gamma, golden_beta]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_5_AddLayerNormGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dy_info = inputs[0]
        x1_info = inputs[1]
        x2_info = inputs[2]
        rstd_info = inputs[3]
        mean_info = inputs[4]
        gamma_info = inputs[5]
        dsum_info = inputs[6]

        if "data" in dy_info:
            dy = torch.tensor(dy_info["data"], dtype=DTYPE_MAP[dy_info["dtype"]]).reshape(dy_info["shape"])
        else:
            dy = torch.rand(dy_info["shape"], dtype=DTYPE_MAP[dy_info["dtype"]])
        if "data" in x1_info:
            x1 = torch.tensor(x1_info["data"], dtype=DTYPE_MAP[x1_info["dtype"]]).reshape(x1_info["shape"])
        else:
            x1 = torch.rand(x1_info["shape"], dtype=DTYPE_MAP[x1_info["dtype"]])
        if "data" in x2_info:
            x2 = torch.tensor(x2_info["data"], dtype=DTYPE_MAP[x2_info["dtype"]]).reshape(x2_info["shape"])
        else:
            x2 = torch.rand(x2_info["shape"], dtype=DTYPE_MAP[x2_info["dtype"]])
        if "data" in rstd_info:
            rstd = torch.tensor(rstd_info["data"], dtype=DTYPE_MAP[rstd_info["dtype"]]).reshape(rstd_info["shape"])
        else:
            rstd = torch.rand(rstd_info["shape"], dtype=DTYPE_MAP[rstd_info["dtype"]]) * (rstd_info["range"][1] - rstd_info["range"][0]) + rstd_info["range"][0]
        if "data" in mean_info:
            mean = torch.tensor(mean_info["data"], dtype=DTYPE_MAP[mean_info["dtype"]]).reshape(mean_info["shape"])
        else:
            mean = torch.rand(mean_info["shape"], dtype=DTYPE_MAP[mean_info["dtype"]]) * (mean_info["range"][1] - mean_info["range"][0]) + mean_info["range"][0]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        if dsum_info["type"] == "attr":
            if dsum_info.get("dtype") == "none":
                dsum = None
            else:
                dsum = dsum_info["value"]
        else:
            if "data" in dsum_info:
                dsum = torch.tensor(dsum_info["data"], dtype=DTYPE_MAP[dsum_info["dtype"]]).reshape(dsum_info["shape"])
            else:
                dsum = torch.rand(dsum_info["shape"], dtype=DTYPE_MAP[dsum_info["dtype"]])

        input_groups.append([dy, x1, x2, rstd, mean, gamma, dsum])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_5_AddLayerNormGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        normalized_shape_info = entries[0]
        normalized_shape = normalized_shape_info["value"]
        init_groups.append([normalized_shape])
    return init_groups
