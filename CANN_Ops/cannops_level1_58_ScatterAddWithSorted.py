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
    """
    Model implementing the ScatterAddWithSorted operator logic (Golden).
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, value: torch.Tensor, sorted_index: torch.Tensor, pos: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            var: Tensor to be updated [N, D] or [..., N, D]
            value: Updates tensor.
            sorted_index: Indices into var [K]
            pos: Indices into value [K], optional.
        """
        out = var.clone()
        inner_dim = var.shape[-1]
        out_flat = out.view(-1, inner_dim)
        value_flat = value.view(-1, inner_dim)
        sorted_index_long = sorted_index.long()
        if pos is not None:
            pos_long = pos.long()
            updates = value_flat[pos_long]
        else:
            updates = value_flat[:sorted_index.numel()]
        out_flat.index_add_(0, sorted_index_long, updates)
        return out_flat.view(var.shape)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_58_ScatterAddWithSorted.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        var_info = inputs[0]
        value_info = inputs[1]
        sorted_index_info = inputs[2]
        pos_info = inputs[3]

        if "data" in var_info:
            var = torch.tensor(var_info["data"], dtype=DTYPE_MAP[var_info["dtype"]]).reshape(var_info["shape"])
        else:
            _dt = DTYPE_MAP[var_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                var = torch.randint(var_info["range"][0], var_info["range"][1] + 1, tuple(var_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                var = torch.rand(var_info["shape"]) < var_info.get("true_frac", 0.5)
            else:
                var = torch.randn(var_info["shape"], dtype=_dt)
        if "data" in value_info:
            value = torch.tensor(value_info["data"], dtype=DTYPE_MAP[value_info["dtype"]]).reshape(value_info["shape"])
        else:
            _dt = DTYPE_MAP[value_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                value = torch.randint(value_info["range"][0], value_info["range"][1] + 1, tuple(value_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                value = torch.rand(value_info["shape"]) < value_info.get("true_frac", 0.5)
            else:
                value = torch.randn(value_info["shape"], dtype=_dt)
        if "data" in sorted_index_info:
            sorted_index = torch.tensor(sorted_index_info["data"], dtype=DTYPE_MAP[sorted_index_info["dtype"]]).reshape(sorted_index_info["shape"])
        else:
            sorted_index = torch.randint(sorted_index_info["range"][0], sorted_index_info["range"][1] + 1, tuple(sorted_index_info["shape"]), dtype=DTYPE_MAP[sorted_index_info["dtype"]])
        if pos_info["type"] == "attr":
            if pos_info.get("dtype") == "none":
                pos = None
            else:
                pos = pos_info["value"]
        else:
            if "data" in pos_info:
                pos = torch.tensor(pos_info["data"], dtype=DTYPE_MAP[pos_info["dtype"]]).reshape(pos_info["shape"])
            else:
                pos = torch.randint(pos_info["range"][0], pos_info["range"][1] + 1, tuple(pos_info["shape"]), dtype=DTYPE_MAP[pos_info["dtype"]])

        input_groups.append([var, value, sorted_index, pos])
    return input_groups


def get_init_inputs():
    return []
