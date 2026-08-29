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

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, grads: List[torch.Tensor], exponent: torch.Tensor) -> List[torch.Tensor]:
        return [x ** exponent.item() for x in grads]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_84_ForeachPowScalar.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grads_info = inputs[0]
        exponent_info = inputs[1]

        grads = []
        for _shape in grads_info["shapes"]:
            if "data" in {"dtype": grads_info["dtype"], "shape": _shape, "range": grads_info.get("range", [0, 1]), "mean": grads_info.get("mean", 0.0), "std": grads_info.get("std", 1.0), "value": grads_info.get("value")}:
                _t = torch.tensor({"dtype": grads_info["dtype"], "shape": _shape, "range": grads_info.get("range", [0, 1]), "mean": grads_info.get("mean", 0.0), "std": grads_info.get("std", 1.0), "value": grads_info.get("value")}["data"], dtype=DTYPE_MAP[{"dtype": grads_info["dtype"], "shape": _shape, "range": grads_info.get("range", [0, 1]), "mean": grads_info.get("mean", 0.0), "std": grads_info.get("std", 1.0), "value": grads_info.get("value")}["dtype"]]).reshape({"dtype": grads_info["dtype"], "shape": _shape, "range": grads_info.get("range", [0, 1]), "mean": grads_info.get("mean", 0.0), "std": grads_info.get("std", 1.0), "value": grads_info.get("value")}["shape"])
            else:
                _dt = DTYPE_MAP[{"dtype": grads_info["dtype"], "shape": _shape, "range": grads_info.get("range", [0, 1]), "mean": grads_info.get("mean", 0.0), "std": grads_info.get("std", 1.0), "value": grads_info.get("value")}["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    _t = torch.randint({"dtype": grads_info["dtype"], "shape": _shape, "range": grads_info.get("range", [0, 1]), "mean": grads_info.get("mean", 0.0), "std": grads_info.get("std", 1.0), "value": grads_info.get("value")}["range"][0], {"dtype": grads_info["dtype"], "shape": _shape, "range": grads_info.get("range", [0, 1]), "mean": grads_info.get("mean", 0.0), "std": grads_info.get("std", 1.0), "value": grads_info.get("value")}["range"][1] + 1, tuple({"dtype": grads_info["dtype"], "shape": _shape, "range": grads_info.get("range", [0, 1]), "mean": grads_info.get("mean", 0.0), "std": grads_info.get("std", 1.0), "value": grads_info.get("value")}["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    _t = torch.rand({"dtype": grads_info["dtype"], "shape": _shape, "range": grads_info.get("range", [0, 1]), "mean": grads_info.get("mean", 0.0), "std": grads_info.get("std", 1.0), "value": grads_info.get("value")}["shape"]) < {"dtype": grads_info["dtype"], "shape": _shape, "range": grads_info.get("range", [0, 1]), "mean": grads_info.get("mean", 0.0), "std": grads_info.get("std", 1.0), "value": grads_info.get("value")}.get("true_frac", 0.5)
                else:
                    _t = torch.randn({"dtype": grads_info["dtype"], "shape": _shape, "range": grads_info.get("range", [0, 1]), "mean": grads_info.get("mean", 0.0), "std": grads_info.get("std", 1.0), "value": grads_info.get("value")}["shape"], dtype=_dt)
            grads.append(_t)
        if "data" in exponent_info:
            exponent = torch.tensor(exponent_info["data"], dtype=DTYPE_MAP[exponent_info["dtype"]]).reshape(exponent_info["shape"])
        else:
            _dt = DTYPE_MAP[exponent_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                exponent = torch.full(exponent_info["shape"], exponent_info["fill"], dtype=_dt)
            elif _dt == torch.bool:
                exponent = torch.rand(exponent_info["shape"]) < exponent_info.get("true_frac", 0.5)
            else:
                exponent = torch.full(exponent_info["shape"], exponent_info["fill"], dtype=_dt)

        input_groups.append([grads, exponent])
    return input_groups


def get_init_inputs():
    return []
