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

    def __init__(self, perm_cpu: torch.Tensor):
        super().__init__()
        self._perm = tuple((int(x) for x in perm_cpu.detach().cpu().tolist()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self._perm).contiguous()

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_73_TransposeV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])

        input_groups.append([x])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_73_TransposeV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        perm_cpu_info = entries[0]
        if "data" in perm_cpu_info:
            perm_cpu = torch.tensor(perm_cpu_info["data"], dtype=DTYPE_MAP[perm_cpu_info["dtype"]]).reshape(perm_cpu_info["shape"])
        else:
            perm_cpu = torch.randperm(perm_cpu_info["shape"][0], dtype=DTYPE_MAP[perm_cpu_info["dtype"]]) + perm_cpu_info["range"][0]
        init_groups.append([perm_cpu])
    return init_groups
