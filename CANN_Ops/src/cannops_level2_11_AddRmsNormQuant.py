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

class Model(nn.Module):

    def __init__(self, gamma: torch.Tensor, scales1: torch.Tensor, scales2: Optional[torch.Tensor]=None, zero_points1: Optional[torch.Tensor]=None, zero_points2: Optional[torch.Tensor]=None, axis: int=-1, epsilon: float=1e-06, div_mode: bool=True):
        super(Model, self).__init__()
        self.gamma = gamma.to(torch.float32).to('cpu')
        self.scales1 = scales1.to(torch.float32).to('cpu')
        self.scales2 = scales2
        if zero_points1 is not None:
            self.zero_points1 = zero_points1.to(torch.float32).to('cpu')
        else:
            self.zero_points1 = torch.zeros(self.gamma.shape, dtype=torch.float32, device='cpu')
        self.zero_points2 = zero_points2
        self.axis = axis
        self.epsilon = epsilon
        self.div_mode = div_mode

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        x = x1 + x2
        rms = torch.sqrt(x.pow(2).mean(dim=self.axis, keepdim=True) + self.epsilon)
        if self.div_mode:
            x_norm = x / rms
        else:
            x_norm = x * torch.rsqrt(rms + self.epsilon)
        y = x_norm * self.gamma
        if not self.div_mode:
            self.scales1 = 1.0 / self.scales1
        y1 = torch.quantize_per_channel(y, self.scales1, self.zero_points1, len(x1.shape) - len(self.gamma.shape), torch.qint8)
        return [y1.int_repr()]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_11_AddRmsNormQuant.json')
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
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_11_AddRmsNormQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        gamma_info = entries[0]
        scales1_info = entries[1]
        scales2_info = entries[2]
        zero_points1_info = entries[3]
        zero_points2_info = entries[4]
        axis_info = entries[5]
        epsilon_info = entries[6]
        div_mode_info = entries[7]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        if "data" in scales1_info:
            scales1 = torch.tensor(scales1_info["data"], dtype=DTYPE_MAP[scales1_info["dtype"]]).reshape(scales1_info["shape"])
        else:
            scales1 = torch.rand(scales1_info["shape"], dtype=DTYPE_MAP[scales1_info["dtype"]])
        scales2 = None
        if "data" in zero_points1_info:
            zero_points1 = torch.tensor(zero_points1_info["data"], dtype=DTYPE_MAP[zero_points1_info["dtype"]]).reshape(zero_points1_info["shape"])
        else:
            _dt = DTYPE_MAP[zero_points1_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                zero_points1 = torch.randint(zero_points1_info["range"][0], zero_points1_info["range"][1] + 1, tuple(zero_points1_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                zero_points1 = torch.rand(zero_points1_info["shape"]) > 0.5
            else:
                zero_points1 = torch.rand(zero_points1_info["shape"], dtype=_dt)
        zero_points2 = None
        axis = axis_info["value"]
        epsilon = epsilon_info["value"]
        div_mode = div_mode_info["value"]
        init_groups.append([gamma, scales1, scales2, zero_points1, zero_points2, axis, epsilon, div_mode])
    return init_groups
