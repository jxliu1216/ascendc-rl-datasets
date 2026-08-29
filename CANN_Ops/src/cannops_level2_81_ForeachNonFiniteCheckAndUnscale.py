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

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, scaled_grads: List[torch.Tensor], found_inf_tensor: torch.Tensor, in_scale_tensor: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        scale_value = in_scale_tensor.item()
        local_found_inf = torch.tensor(0.0, dtype=torch.float, device=found_inf_tensor.device)
        for grad in scaled_grads:
            non_finite = torch.isnan(grad) | torch.isinf(grad)
            if non_finite.any():
                local_found_inf.fill_(1.0)
            if scale_value == 0.0:
                grad.fill_(0.0)
            else:
                grad.mul_(scale_value)
        found_inf_tensor.copy_(local_found_inf)
        return [torch.concat([x.flatten() for x in scaled_grads]), found_inf_tensor]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_81_ForeachNonFiniteCheckAndUnscale.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        scaled_grads_info = inputs[0]
        found_inf_tensor_info = inputs[1]
        in_scale_tensor_info = inputs[2]

        scaled_grads = []
        for _shape in scaled_grads_info["shapes"]:
            _t = torch.rand({"dtype": scaled_grads_info["dtype"], "shape": _shape, "range": scaled_grads_info.get("range", [0, 1]), "mean": scaled_grads_info.get("mean", 0.0), "std": scaled_grads_info.get("std", 1.0), "value": scaled_grads_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": scaled_grads_info["dtype"], "shape": _shape, "range": scaled_grads_info.get("range", [0, 1]), "mean": scaled_grads_info.get("mean", 0.0), "std": scaled_grads_info.get("std", 1.0), "value": scaled_grads_info.get("value")}["dtype"]])
            scaled_grads.append(_t)
        if scaled_grads_info.get("inject"):
            _f = scaled_grads[0].reshape(-1)
            _f[0] = float(scaled_grads_info["inject"])
            scaled_grads[0] = _f.reshape(scaled_grads[0].shape)
        if "data" in found_inf_tensor_info:
            found_inf_tensor = torch.tensor(found_inf_tensor_info["data"], dtype=DTYPE_MAP[found_inf_tensor_info["dtype"]]).reshape(found_inf_tensor_info["shape"])
        else:
            found_inf_tensor = torch.full(found_inf_tensor_info["shape"], found_inf_tensor_info["fill"], dtype=DTYPE_MAP[found_inf_tensor_info["dtype"]])
        if "data" in in_scale_tensor_info:
            in_scale_tensor = torch.tensor(in_scale_tensor_info["data"], dtype=DTYPE_MAP[in_scale_tensor_info["dtype"]]).reshape(in_scale_tensor_info["shape"])
        else:
            in_scale_tensor = torch.rand(in_scale_tensor_info["shape"], dtype=DTYPE_MAP[in_scale_tensor_info["dtype"]]) * (in_scale_tensor_info["range"][1] - in_scale_tensor_info["range"][0]) + in_scale_tensor_info["range"][0]

        input_groups.append([scaled_grads, found_inf_tensor, in_scale_tensor])
    return input_groups


def get_init_inputs():
    return []
