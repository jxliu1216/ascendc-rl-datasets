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

from typing import List, Tuple
import torch
import torch.nn as nn

def _flatten(lst):
    flat = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            flat.extend(_flatten(item))
        elif isinstance(item, torch.Tensor):
            flat.append(item)
        else:
            raise TypeError('Unexpected element type')
    return flat

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: List[torch.Tensor], round_mode: torch.Tensor) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        round_mode = round_mode.item()
        for x_tensor in x:
            if round_mode == 1:
                result_tensor = torch.round(x_tensor)
            elif round_mode == 2:
                result_tensor = torch.floor(x_tensor)
            elif round_mode == 3:
                result_tensor = torch.ceil(x_tensor)
            elif round_mode == 4:
                result_tensor = torch.where(x_tensor >= 0, (x_tensor + 0.5).floor(), (x_tensor - 0.5).ceil())
            elif round_mode == 5:
                result_tensor = torch.trunc(x_tensor)
            elif round_mode == 6:
                int_part = x_tensor.trunc()
                frac_part_abs = (x_tensor - int_part).abs()
                result_tensor = torch.where(frac_part_abs == 0.5, torch.where(int_part % 2 == 0, int_part + torch.sign(x_tensor), int_part), torch.round(x_tensor))
            elif round_mode == 7:
                result_tensor = torch.frac(x_tensor)
            else:
                result_tensor = x_tensor
            results.append(result_tensor.to(x_tensor.dtype))
        return results

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_88_ForeachRoundOffNumber.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        round_mode_info = inputs[1]

        x = []
        for _shape in x_info["shapes"]:
            _t = torch.rand({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["dtype"]]) * ({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][1] - {"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][0]) + {"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][0]
            x.append(_t)
        if "data" in round_mode_info:
            round_mode = torch.tensor(round_mode_info["data"], dtype=DTYPE_MAP[round_mode_info["dtype"]]).reshape(round_mode_info["shape"])
        else:
            round_mode = torch.rand(round_mode_info["shape"], dtype=DTYPE_MAP[round_mode_info["dtype"]]) * (round_mode_info["range"][1] - round_mode_info["range"][0]) + round_mode_info["range"][0]

        input_groups.append([x, round_mode])
    return input_groups


def get_init_inputs():
    return []
