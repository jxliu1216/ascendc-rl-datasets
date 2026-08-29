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
    """PyTorch native torch.gather_elements reference implementation."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.gather(x, dim, index.long())

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_23_GatherElementsV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        index_info = inputs[1]
        dim_info = inputs[2]

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
        if "data" in index_info:
            index = torch.tensor(index_info["data"], dtype=DTYPE_MAP[index_info["dtype"]]).reshape(index_info["shape"])
        else:
            index = torch.randint(index_info["range"][0], index_info["range"][1] + 1, tuple(index_info["shape"]), dtype=DTYPE_MAP[index_info["dtype"]])
        dim = dim_info["value"]

        input_groups.append([x, index, dim])
    return input_groups


def get_init_inputs():
    return []
