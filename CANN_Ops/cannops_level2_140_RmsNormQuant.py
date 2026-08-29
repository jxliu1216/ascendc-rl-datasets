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

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    """与 ATK npu_add_rms_norm_quant_golden 一致: factor = 1/sqrt(mean(x^2)+eps), return x * factor * weight + bias."""
    square_sum = torch.sum(torch.square(x), dim=-1, keepdim=True)
    factor = 1.0 / torch.sqrt(square_sum / x.shape[-1] + eps)
    return x * factor * weight + bias

class Model(nn.Module):
    """
    标杆与 ATK function_rms_norm_quant 完全一致:
    quant_in = rms_norm(x, gamma, beta, eps=epsilon);
    y = quantize_per_tensor(quant_in, scale, offset, qint8).int_repr()，scale/offset 取标量。
    误差来源说明：当某行 mean(x^2) 很小时，rms≈sqrt(epsilon)。若 kernel 侧 epsilon 与标杆不一致
    （如未注入 attr 时用默认 1e-12），kernel 的 factor 会远大于标杆，导致该行 quant_in 被放大，
    出现 72/127 等异常值，从而产生 max_abs_error=130 量级的误差。
    """

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, epsilon: float):
        super(Model, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.scale = scale
        self.offset = offset
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        input_x = x.float()
        input_gamma = self.gamma.float()
        input_beta = self.beta.float()
        input_scale = self.scale.float().flatten()[0].item()
        input_offset = self.offset.flatten()[0].item()
        quant_in = _rms_norm(input_x, weight=input_gamma, bias=input_beta, eps=self.epsilon)
        output_q = torch.quantize_per_tensor(quant_in, input_scale, input_offset, torch.qint8)
        y_np = output_q.int_repr().detach().clone().cpu()
        out_int8 = y_np.to(torch.int8).reshape(x.shape)
        return [out_int8]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_140_RmsNormQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]

        input_groups.append([x])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_140_RmsNormQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        gamma_info = entries[0]
        beta_info = entries[1]
        scale_info = entries[2]
        offset_info = entries[3]
        epsilon_info = entries[4]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]]) * (gamma_info["range"][1] - gamma_info["range"][0]) + gamma_info["range"][0]
        if "data" in beta_info:
            beta = torch.tensor(beta_info["data"], dtype=DTYPE_MAP[beta_info["dtype"]]).reshape(beta_info["shape"])
        else:
            beta = torch.randn(beta_info["shape"], dtype=DTYPE_MAP[beta_info["dtype"]]) * beta_info["std"] + beta_info["mean"]
        if "data" in scale_info:
            scale = torch.tensor(scale_info["data"], dtype=DTYPE_MAP[scale_info["dtype"]]).reshape(scale_info["shape"])
        else:
            scale = torch.full(scale_info["shape"], scale_info["fill"], dtype=DTYPE_MAP[scale_info["dtype"]])
        if "data" in offset_info:
            offset = torch.tensor(offset_info["data"], dtype=DTYPE_MAP[offset_info["dtype"]]).reshape(offset_info["shape"])
        else:
            offset = torch.full(offset_info["shape"], offset_info["fill"], dtype=DTYPE_MAP[offset_info["dtype"]])
        epsilon = epsilon_info["value"]
        init_groups.append([gamma, beta, scale, offset, epsilon])
    return init_groups
