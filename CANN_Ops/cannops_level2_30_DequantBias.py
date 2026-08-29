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

"""
CPU golden 与 ops-nn-dev ST 对齐：
`ops-nn-dev/quant/dequant_bias/tests/st/aclnnDequantBias/executor_aclnnDequantBias.py`
"""
from typing import List, Optional
import torch
import torch.nn as nn
GE_DT_FP16 = 1
GE_DT_BF16 = 27

def _golden_dequant_bias(x: torch.Tensor, weight_scale: torch.Tensor, activate_scale: Optional[torch.Tensor], bias: Optional[torch.Tensor], output_dtype: int) -> torch.Tensor:
    """复刻 executor_aclnnDequantBias.__call__ 分支逻辑。"""
    x_data = x
    weight_scale_data = weight_scale
    activate_scale_data = activate_scale
    bias_data = bias
    if bias_data is not None:
        if bias_data.dtype == torch.int32:
            if activate_scale_data is None:
                y = (x_data.to(torch.float32) + bias_data.to(torch.float32)) * weight_scale_data.to(torch.float32)
            else:
                a = activate_scale_data.to(torch.float32)[:, None]
                y = (x_data.to(torch.float32) + bias_data.to(torch.float32)) * weight_scale_data.to(torch.float32) * a
        elif bias_data.dtype in (torch.float16, torch.bfloat16, torch.float32):
            if activate_scale_data is None:
                y = x_data.to(torch.float32) * weight_scale_data.to(torch.float32) + bias_data.to(torch.float32)
            else:
                a = activate_scale_data.to(torch.float32)[:, None]
                y = x_data.to(torch.float32) * weight_scale_data.to(torch.float32) * a + bias_data.to(torch.float32)
        else:
            raise ValueError(f'Unsupported bias dtype: {bias_data.dtype}')
    elif activate_scale_data is None:
        y = x_data.to(torch.float32) * weight_scale_data.to(torch.float32)
    else:
        a = activate_scale_data.to(torch.float32)[:, None]
        y = x_data.to(torch.float32) * weight_scale_data.to(torch.float32) * a
    if output_dtype == GE_DT_FP16:
        return y.to(torch.float16)
    if output_dtype == GE_DT_BF16:
        return y.to(torch.bfloat16)
    raise ValueError(f'golden 仅支持 output_dtype 1(fp16) 或 27(bf16)，收到 {output_dtype}')

class Model(nn.Module):

    def __init__(self, output_dtype: int):
        super().__init__()
        self.output_dtype = int(output_dtype)

    def forward(self, x: torch.Tensor, weight_scale: torch.Tensor, activate_scale: Optional[torch.Tensor]=None, bias: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        y = _golden_dequant_bias(x, weight_scale, activate_scale, bias, self.output_dtype)
        return [y]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_30_DequantBias.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        weight_scale_info = inputs[1]
        activate_scale_info = inputs[2]
        bias_info = inputs[3]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randint(x_info["range"][0], x_info["range"][1] + 1, tuple(x_info["shape"]), dtype=DTYPE_MAP[x_info["dtype"]])
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
                activate_scale = torch.rand(activate_scale_info["shape"], dtype=DTYPE_MAP[activate_scale_info["dtype"]]) * (activate_scale_info["range"][1] - activate_scale_info["range"][0]) + activate_scale_info["range"][0]
        if bias_info["type"] == "attr":
            if bias_info.get("dtype") == "none":
                bias = None
            else:
                bias = bias_info["value"]
        else:
            if "data" in bias_info:
                bias = torch.tensor(bias_info["data"], dtype=DTYPE_MAP[bias_info["dtype"]]).reshape(bias_info["shape"])
            else:
                _dt = DTYPE_MAP[bias_info["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    bias = torch.randint(bias_info["range"][0], bias_info["range"][1] + 1, tuple(bias_info["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    bias = torch.rand(bias_info["shape"]) < bias_info.get("true_frac", 0.5)
                else:
                    bias = torch.rand(bias_info["shape"], dtype=_dt) * (bias_info["range"][1] - bias_info["range"][0]) + bias_info["range"][0]

        input_groups.append([x, weight_scale, activate_scale, bias])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_30_DequantBias.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        output_dtype_info = entries[0]
        output_dtype = output_dtype_info["value"]
        init_groups.append([output_dtype])
    return init_groups
