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
import torch.nn.functional as F

def _ref_masked_softmax_with_rel_pos_bias(x: torch.Tensor, atten_mask: Optional[torch.Tensor], relative_pos_bias: torch.Tensor, scale_value: float=1.0) -> torch.Tensor:
    """Reference: out = softmax(scale_value * x + atten_mask + relative_pos_bias)."""
    scaled = x * scale_value
    out = scaled + relative_pos_bias
    if atten_mask is not None:
        out = out + atten_mask
    return F.softmax(out, dim=-1)

class Model(torch.nn.Module):

    def forward(self, x: torch.Tensor, atten_mask: Optional[torch.Tensor], relative_pos_bias: torch.Tensor, scale_value: float=1.0) -> List[torch.Tensor]:
        y = _ref_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value)
        return [y]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_125_MaskedSoftmaxWithRelPosBias.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        atten_mask_info = inputs[1]
        relative_pos_bias_info = inputs[2]
        scale_value_info = inputs[3]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if atten_mask_info["type"] == "attr":
            if atten_mask_info.get("dtype") == "none":
                atten_mask = None
            else:
                atten_mask = atten_mask_info["value"]
        else:
            if "data" in atten_mask_info:
                atten_mask = torch.tensor(atten_mask_info["data"], dtype=DTYPE_MAP[atten_mask_info["dtype"]]).reshape(atten_mask_info["shape"])
            else:
                atten_mask = torch.rand(atten_mask_info["shape"], dtype=DTYPE_MAP[atten_mask_info["dtype"]])
        if "data" in relative_pos_bias_info:
            relative_pos_bias = torch.tensor(relative_pos_bias_info["data"], dtype=DTYPE_MAP[relative_pos_bias_info["dtype"]]).reshape(relative_pos_bias_info["shape"])
        else:
            relative_pos_bias = torch.rand(relative_pos_bias_info["shape"], dtype=DTYPE_MAP[relative_pos_bias_info["dtype"]])
        scale_value = scale_value_info["value"]

        input_groups.append([x, atten_mask, relative_pos_bias, scale_value])
    return input_groups


def get_init_inputs():
    return []
