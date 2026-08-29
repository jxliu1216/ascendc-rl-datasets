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

from typing import List, Optional
import torch
import torch.nn as nn

def _calc_output(output_y: torch.Tensor) -> tuple:
    """Golden formula: out_scales = max(|y|, dim=-1, keepdim)/127, y_int8 = round(y/out_scales)."""
    max_y_0 = torch.max(torch.abs(output_y), dim=-1, keepdim=True)[0]
    out_scales = max_y_0 / 127.0
    x_quant = output_y / (out_scales + 1e-12)
    y_int8 = torch.round(x_quant).clamp(-128, 127).to(torch.int8)
    return (y_int8, out_scales.squeeze(-1))

class Model(nn.Module):
    """Reference aligned with TBE golden: add -> layernorm -> quant(y*scale) with out_scale = max(|z|)/127."""

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, bias: Optional[torch.Tensor]=None, scales1: Optional[torch.Tensor]=None, scales2: Optional[torch.Tensor]=None, zero_points1: Optional[torch.Tensor]=None, zero_points2: Optional[torch.Tensor]=None, epsilon: float=1e-05, additional_output: bool=True, div_mode: bool=True):
        super(Model, self).__init__()
        self.gamma = gamma.to(torch.float32).to('cpu')
        self.beta = beta.to(torch.float32).to('cpu')
        self.bias = bias.to(torch.float32).to('cpu') if bias is not None else None
        self.scales1 = scales1
        self.scales2 = scales2
        self.zero_points1 = zero_points1
        self.zero_points2 = zero_points2
        self.epsilon = epsilon
        self.additional_output = additional_output
        self.div_mode = div_mode

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        dtype_hp = torch.float32
        x1 = x1.to(dtype_hp)
        x2 = x2.to(dtype_hp)
        gamma = self.gamma.to(x1.device).to(dtype_hp)
        beta = self.beta.to(x1.device).to(dtype_hp)
        x = x1 + x2
        if self.bias is not None:
            x = x + self.bias.to(x.device).to(dtype_hp)
        mean = x.mean(dim=-1, keepdim=True)
        var = torch.mean(torch.pow(x - mean, 2), dim=-1, keepdim=True)
        rstd = 1.0 / torch.sqrt(var + self.epsilon)
        y = (x - mean) * rstd * gamma + beta
        y = y.to(torch.float32)
        if self.scales1 is not None and self.scales2 is not None:
            s1 = self.scales1.to(y.device).to(torch.float32)
            s2 = self.scales2.to(y.device).to(torch.float32)
            output_y1 = y * s1
            output_y2 = y * s2
            y1_int, out_scale1 = _calc_output(output_y1)
            y2_int, out_scale2 = _calc_output(output_y2)
        else:
            y1_int, out_scale1 = _calc_output(y)
            y2_int = torch.zeros_like(y, dtype=torch.int8)
            out_scale2 = out_scale1
        x_out = x.to(x1.dtype) if x1.dtype != torch.float32 else x
        return [y1_int, y2_int, x_out, out_scale1, out_scale2]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_6_AddLayerNormQuant.json')
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
            x1 = torch.rand(x1_info["shape"], dtype=DTYPE_MAP[x1_info["dtype"]]) * (x1_info["range"][1] - x1_info["range"][0]) + x1_info["range"][0]
        if "data" in x2_info:
            x2 = torch.tensor(x2_info["data"], dtype=DTYPE_MAP[x2_info["dtype"]]).reshape(x2_info["shape"])
        else:
            x2 = torch.rand(x2_info["shape"], dtype=DTYPE_MAP[x2_info["dtype"]]) * (x2_info["range"][1] - x2_info["range"][0]) + x2_info["range"][0]

        input_groups.append([x1, x2])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_6_AddLayerNormQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        gamma_info = entries[0]
        beta_info = entries[1]
        bias_info = entries[2]
        scales1_info = entries[3]
        scales2_info = entries[4]
        zero_points1_info = entries[5]
        zero_points2_info = entries[6]
        epsilon_info = entries[7]
        additional_output_info = entries[8]
        div_mode_info = entries[9]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        if "data" in beta_info:
            beta = torch.tensor(beta_info["data"], dtype=DTYPE_MAP[beta_info["dtype"]]).reshape(beta_info["shape"])
        else:
            beta = torch.rand(beta_info["shape"], dtype=DTYPE_MAP[beta_info["dtype"]])
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
        if scales1_info["type"] == "attr":
            if scales1_info.get("dtype") == "none":
                scales1 = None
            else:
                scales1 = scales1_info["value"]
        else:
            if "data" in scales1_info:
                scales1 = torch.tensor(scales1_info["data"], dtype=DTYPE_MAP[scales1_info["dtype"]]).reshape(scales1_info["shape"])
            else:
                scales1 = torch.rand(scales1_info["shape"], dtype=DTYPE_MAP[scales1_info["dtype"]])
        if scales2_info["type"] == "attr":
            if scales2_info.get("dtype") == "none":
                scales2 = None
            else:
                scales2 = scales2_info["value"]
        else:
            if "data" in scales2_info:
                scales2 = torch.tensor(scales2_info["data"], dtype=DTYPE_MAP[scales2_info["dtype"]]).reshape(scales2_info["shape"])
            else:
                scales2 = torch.rand(scales2_info["shape"], dtype=DTYPE_MAP[scales2_info["dtype"]])
        if zero_points1_info["type"] == "attr":
            if zero_points1_info.get("dtype") == "none":
                zero_points1 = None
            else:
                zero_points1 = zero_points1_info["value"]
        else:
            if "data" in zero_points1_info:
                zero_points1 = torch.tensor(zero_points1_info["data"], dtype=DTYPE_MAP[zero_points1_info["dtype"]]).reshape(zero_points1_info["shape"])
            else:
                zero_points1 = torch.full(zero_points1_info["shape"], zero_points1_info["fill"], dtype=DTYPE_MAP[zero_points1_info["dtype"]])
        if zero_points2_info["type"] == "attr":
            if zero_points2_info.get("dtype") == "none":
                zero_points2 = None
            else:
                zero_points2 = zero_points2_info["value"]
        else:
            if "data" in zero_points2_info:
                zero_points2 = torch.tensor(zero_points2_info["data"], dtype=DTYPE_MAP[zero_points2_info["dtype"]]).reshape(zero_points2_info["shape"])
            else:
                zero_points2 = torch.full(zero_points2_info["shape"], zero_points2_info["fill"], dtype=DTYPE_MAP[zero_points2_info["dtype"]])
        epsilon = epsilon_info["value"]
        additional_output = additional_output_info["value"]
        div_mode = div_mode_info["value"]
        init_groups.append([gamma, beta, bias, scales1, scales2, zero_points1, zero_points2, epsilon, additional_output, div_mode])
    return init_groups
