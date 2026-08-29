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
    """
    Simple model that performs abs operation.
    """

    def __init__(self):
        """
        Initialize the model for abs operation.
        No parameters needed for basic abs operation.
        """
        super(Model, self).__init__()

    def forward(self, x: List[torch.Tensor], scalars: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes the abs of input elements.

        Args:
            inputs: Input tensor list of any shape.

        Returns:
            Output tensor of same shape as input with abs applied elementwise.
        """
        res = torch._foreach_pow(x, scalars)
        res_list = []
        for i in range(len(res)):
            t = res[i]
            t = torch.where(torch.isinf(t) & (t > 0), torch.tensor(1, dtype=t.dtype, device=t.device), t).to(x[0].dtype)
            t = torch.where(torch.isinf(t) & (t < 0), torch.tensor(-1, dtype=t.dtype, device=t.device), t).to(x[0].dtype)
            t = torch.where(torch.isnan(t), torch.tensor(0, dtype=t.dtype, device=t.device), t).to(x[0].dtype)
            res_list.append(t)
        return res_list

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_86_ForeachPowScalarList.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        scalars_info = inputs[1]

        x = []
        for _shape in x_info["shapes"]:
            if "data" in {"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}:
                _t = torch.tensor({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["data"], dtype=DTYPE_MAP[{"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["dtype"]]).reshape({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["shape"])
            else:
                _dt = DTYPE_MAP[{"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    _t = torch.randint({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][0], {"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][1] + 1, tuple({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    _t = torch.rand({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["shape"]) > 0.5
                else:
                    _t = torch.rand({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["shape"], dtype=_dt) * ({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][1] - {"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][0]) + {"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][0]
            x.append(_t)
        scalars = []
        for _shape in scalars_info["shapes"]:
            if "data" in {"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}:
                _t = torch.tensor({"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["data"], dtype=DTYPE_MAP[{"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["dtype"]]).reshape({"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["shape"])
            else:
                _dt = DTYPE_MAP[{"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    _t = torch.randint({"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["range"][0], {"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["range"][1] + 1, tuple({"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    _t = torch.rand({"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["shape"]) > 0.5
                else:
                    _t = torch.rand({"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["shape"], dtype=_dt) * ({"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["range"][1] - {"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["range"][0]) + {"dtype": scalars_info["dtype"], "shape": _shape, "range": scalars_info.get("range", [0, 1]), "mean": scalars_info.get("mean", 0.0), "std": scalars_info.get("std", 1.0), "value": scalars_info.get("value")}["range"][0]
            scalars.append(_t)

        input_groups.append([x, scalars])
    return input_groups


def get_init_inputs():
    return []
