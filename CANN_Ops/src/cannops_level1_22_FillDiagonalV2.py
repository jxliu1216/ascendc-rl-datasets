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

import torch
import torch.nn as nn

class Model(nn.Module):
    """CPU 参考：与 torch.fill_diagonal_ 语义一致（就地填主对角）。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, fill_tensor: torch.Tensor, wrap: bool) -> torch.Tensor:
        out = x.clone()
        f = fill_tensor.to(device=out.device, dtype=out.dtype).contiguous().reshape(())
        out.fill_diagonal_(f, wrap=wrap)
        return out

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_22_FillDiagonalV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        fill_tensor_info = inputs[1]
        wrap_info = inputs[2]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            _dt = DTYPE_MAP[x_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                x = torch.randint(x_info["range"][0], x_info["range"][1] + 1, tuple(x_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                x = torch.rand(x_info["shape"]) > 0.5
            else:
                x = torch.randn(x_info["shape"], dtype=_dt)
        if "data" in fill_tensor_info:
            fill_tensor = torch.tensor(fill_tensor_info["data"], dtype=DTYPE_MAP[fill_tensor_info["dtype"]]).reshape(fill_tensor_info["shape"])
        else:
            _dt = DTYPE_MAP[fill_tensor_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                fill_tensor = torch.randint(fill_tensor_info["range"][0], fill_tensor_info["range"][1] + 1, tuple(fill_tensor_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                fill_tensor = torch.full(fill_tensor_info["shape"], fill_tensor_info["value"], dtype=torch.bool)
            else:
                fill_tensor = torch.rand(fill_tensor_info["shape"], dtype=_dt) * (fill_tensor_info["range"][1] - fill_tensor_info["range"][0]) + fill_tensor_info["range"][0]
        wrap = wrap_info["value"]

        input_groups.append([x, fill_tensor, wrap])
    return input_groups


def get_init_inputs():
    return []
