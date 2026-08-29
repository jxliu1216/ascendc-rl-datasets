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

    def forward(self, x: torch.Tensor, mask: torch.Tensor, scale: float=1.0) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.to(torch.float32)
        x_f = x_f * scale
        mask_f = mask.to(torch.float32)
        x_f = x_f * (1.0 - mask_f) + -10000.0 * mask_f
        y_f = torch.softmax(x_f, dim=-1)
        return y_f.to(orig_dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_47_ScaledMaskedSoftmaxV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        mask_info = inputs[1]
        scale_info = inputs[2]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in mask_info:
            mask = torch.tensor(mask_info["data"], dtype=DTYPE_MAP[mask_info["dtype"]]).reshape(mask_info["shape"])
        else:
            mask = torch.rand(mask_info["shape"]) < mask_info.get("true_frac", 0.5)
        scale = scale_info["value"]

        input_groups.append([x, mask, scale])
    return input_groups


def get_init_inputs():
    return []
