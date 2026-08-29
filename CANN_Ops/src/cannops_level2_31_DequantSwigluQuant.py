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
import torch.nn.functional as F
from typing import List
from typing import Optional, Tuple

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x_tensor: torch.Tensor, weight_scale: Optional[torch.Tensor]=None, activate_scale: Optional[torch.Tensor]=None, bias: Optional[torch.Tensor]=None, quant_scale: Optional[torch.Tensor]=None, quant_offset: Optional[torch.Tensor]=None, group_index: Optional[torch.Tensor]=None, activate_left: bool=False, quant_mode: str='static') -> List[torch.Tensor]:
        if group_index is None:
            group_index = torch.tensor([x_tensor.shape[0]])
        x_shape = list(x_tensor.shape)
        x_shape[-1] //= 2
        res_y = torch.zeros(x_shape, dtype=torch.float32, device=x_tensor.device)
        input_dtype = x_tensor.dtype
        offset = 0
        for g_idx in range(group_index.shape[0]):
            groupIdx = group_index[g_idx]
            x = x_tensor[offset:offset + groupIdx].float()
            if input_dtype == torch.int32:
                if bias is not None:
                    x = x + bias
                x = x * weight_scale[g_idx]
                if activate_scale is not None:
                    x = x * activate_scale[offset:offset + groupIdx]
            gate, up = torch.chunk(x, 2, dim=-1)
            if activate_left:
                output = torch.nn.functional.silu(gate) * up
            else:
                output = torch.nn.functional.silu(up) * gate
            if quant_mode == 'static':
                output = output / quant_scale[g_idx] + quant_offset[g_idx]
            elif quant_mode == 'dynamic':
                output = output * quant_scale[g_idx]
                abs = torch.abs(output)
                max_values = torch.amax(abs, dim=-1)
                scale_out = max_values / 127
                max_values = 127 / max_values
                output = output * max_values.unsqueeze(1)
            output = torch.clamp(output, -128, 127)
            output = torch.round(output)
            res_y[offset:offset + groupIdx] = output
            offset = offset + groupIdx
        return res_y.to(torch.int8)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_31_DequantSwigluQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_tensor_info = inputs[0]
        weight_scale_info = inputs[1]
        activate_scale_info = inputs[2]
        bias_info = inputs[3]
        quant_scale_info = inputs[4]
        quant_offset_info = inputs[5]
        group_index_info = inputs[6]
        activate_left_info = inputs[7]
        quant_mode_info = inputs[8]

        if "data" in x_tensor_info:
            x_tensor = torch.tensor(x_tensor_info["data"], dtype=DTYPE_MAP[x_tensor_info["dtype"]]).reshape(x_tensor_info["shape"])
        else:
            _dt = DTYPE_MAP[x_tensor_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                x_tensor = torch.randint(x_tensor_info["range"][0], x_tensor_info["range"][1] + 1, tuple(x_tensor_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                x_tensor = torch.rand(x_tensor_info["shape"]) > 0.5
            else:
                x_tensor = torch.rand(x_tensor_info["shape"], dtype=_dt)
        if weight_scale_info["type"] == "attr":
            if weight_scale_info.get("dtype") == "none":
                weight_scale = None
            else:
                weight_scale = weight_scale_info["value"]
        else:
            if "data" in weight_scale_info:
                weight_scale = torch.tensor(weight_scale_info["data"], dtype=DTYPE_MAP[weight_scale_info["dtype"]]).reshape(weight_scale_info["shape"])
            else:
                weight_scale = torch.rand(weight_scale_info["shape"], dtype=DTYPE_MAP[weight_scale_info["dtype"]])
        if activate_scale_info["type"] == "attr":
            if activate_scale_info.get("dtype") == "none":
                activate_scale = None
            else:
                activate_scale = activate_scale_info["value"]
        else:
            if "data" in activate_scale_info:
                activate_scale = torch.tensor(activate_scale_info["data"], dtype=DTYPE_MAP[activate_scale_info["dtype"]]).reshape(activate_scale_info["shape"])
            else:
                activate_scale = torch.rand(activate_scale_info["shape"], dtype=DTYPE_MAP[activate_scale_info["dtype"]])
        if bias_info["type"] == "attr":
            if bias_info.get("dtype") == "none":
                bias = None
            else:
                bias = bias_info["value"]
        else:
            if "data" in bias_info:
                bias = torch.tensor(bias_info["data"], dtype=DTYPE_MAP[bias_info["dtype"]]).reshape(bias_info["shape"])
            else:
                bias = torch.randint(bias_info["range"][0], bias_info["range"][1] + 1, tuple(bias_info["shape"]), dtype=DTYPE_MAP[bias_info["dtype"]])
        if "data" in quant_scale_info:
            quant_scale = torch.tensor(quant_scale_info["data"], dtype=DTYPE_MAP[quant_scale_info["dtype"]]).reshape(quant_scale_info["shape"])
        else:
            quant_scale = torch.rand(quant_scale_info["shape"], dtype=DTYPE_MAP[quant_scale_info["dtype"]])
        if quant_offset_info["type"] == "attr":
            if quant_offset_info.get("dtype") == "none":
                quant_offset = None
            else:
                quant_offset = quant_offset_info["value"]
        else:
            if "data" in quant_offset_info:
                quant_offset = torch.tensor(quant_offset_info["data"], dtype=DTYPE_MAP[quant_offset_info["dtype"]]).reshape(quant_offset_info["shape"])
            else:
                quant_offset = torch.rand(quant_offset_info["shape"], dtype=DTYPE_MAP[quant_offset_info["dtype"]])
        if group_index_info["type"] == "attr":
            if group_index_info.get("dtype") == "none":
                group_index = None
            else:
                group_index = group_index_info["value"]
        else:
            if "data" in group_index_info:
                group_index = torch.tensor(group_index_info["data"], dtype=DTYPE_MAP[group_index_info["dtype"]]).reshape(group_index_info["shape"])
            else:
                group_index = torch.randint(group_index_info["range"][0], group_index_info["range"][1] + 1, tuple(group_index_info["shape"]), dtype=DTYPE_MAP[group_index_info["dtype"]])
        activate_left = activate_left_info["value"]
        quant_mode = quant_mode_info["value"]

        input_groups.append([x_tensor, weight_scale, activate_scale, bias, quant_scale, quant_offset, group_index, activate_left, quant_mode])
    return input_groups


def get_init_inputs():
    return []
