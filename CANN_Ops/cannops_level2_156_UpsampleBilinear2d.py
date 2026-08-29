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
import torch.nn.functional as F

class Model(nn.Module):
    """使用 PyTorch 原生算子的参考实现（golden model）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, size, align_corners=False):
        out_h, out_w = (size[0].item(), size[1].item())
        return F.interpolate(input, size=[out_h, out_w], mode='bilinear', align_corners=align_corners)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_156_UpsampleBilinear2d.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_info = inputs[0]
        size_info = inputs[1]
        align_corners_info = inputs[2]

        if "data" in input_info:
            input = torch.tensor(input_info["data"], dtype=DTYPE_MAP[input_info["dtype"]]).reshape(input_info["shape"])
        else:
            input = torch.randn(input_info["shape"], dtype=DTYPE_MAP[input_info["dtype"]])
        if "data" in size_info:
            size = torch.tensor(size_info["data"], dtype=DTYPE_MAP[size_info["dtype"]]).reshape(size_info["shape"])
        else:
            size = torch.randint(size_info["range"][0], size_info["range"][1] + 1, tuple(size_info["shape"]), dtype=DTYPE_MAP[size_info["dtype"]])
        align_corners = align_corners_info["value"]

        input_groups.append([input, size, align_corners])
    return input_groups


def get_init_inputs():
    return []
