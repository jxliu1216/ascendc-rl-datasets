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

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, yGrad: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, scale: float=1.0) -> torch.Tensor:
        orig_dtype = yGrad.dtype
        yGrad_f = yGrad.to(torch.float32)
        y_f = y.to(torch.float32)
        dy_mul_y = yGrad_f * y_f
        sum_dy_y = dy_mul_y.sum(dim=-1, keepdim=True)
        xGrad_f = y_f * (yGrad_f - sum_dy_y)
        if scale != 1.0:
            xGrad_f = xGrad_f * scale
        mask_f = mask.to(torch.float32)
        xGrad_f = xGrad_f * (1.0 - mask_f)
        return xGrad_f.to(orig_dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_46_ScaledMaskedSoftmaxGradV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        yGrad_info = inputs[0]
        y_info = inputs[1]
        mask_info = inputs[2]
        scale_info = inputs[3]

        if "data" in yGrad_info:
            yGrad = torch.tensor(yGrad_info["data"], dtype=DTYPE_MAP[yGrad_info["dtype"]]).reshape(yGrad_info["shape"])
        else:
            yGrad = torch.randn(yGrad_info["shape"], dtype=DTYPE_MAP[yGrad_info["dtype"]])
        if "data" in y_info:
            y = torch.tensor(y_info["data"], dtype=DTYPE_MAP[y_info["dtype"]]).reshape(y_info["shape"])
        else:
            y = torch.rand(y_info["shape"], dtype=DTYPE_MAP[y_info["dtype"]]) * (y_info["range"][1] - y_info["range"][0]) + y_info["range"][0]
        if "data" in mask_info:
            mask = torch.tensor(mask_info["data"], dtype=DTYPE_MAP[mask_info["dtype"]]).reshape(mask_info["shape"])
        else:
            mask = torch.rand(mask_info["shape"]) < mask_info.get("true_frac", 0.5)
        scale = scale_info["value"]

        input_groups.append([yGrad, y, mask, scale])
    return input_groups


def get_init_inputs():
    return []
