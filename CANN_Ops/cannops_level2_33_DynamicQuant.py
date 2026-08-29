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
CPU golden 与 AICore 对称动态量化路径一致（见 answer/0/op_kernel/dynamic_quant.h::Compute）：
Cast(x)->fp32；可选乘 smooth；Abs 后沿最后一维求 max；scale_fp = 127/max_abs；
输出 scale 张量为 max_abs/127（与内核 scaleLocal.SetValue(i, 1 / scale) 一致）；
量化：tempFp32 * (127/max_abs) 后 CAST_RINT 再落到 int8。
Golden 用 float32 + numpy.rint 逼近 CAST_RINT。
"""
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

def golden_dynamic_quant_int8(x: torch.Tensor, smooth: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    x_f = x.detach().float()
    if smooth is not None:
        x_f = x_f * smooth.detach().float()
    last = x_f.shape[-1]
    flat = x_f.reshape(-1, last)
    max_abs = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    inv = 127.0 / max_abs
    qf = flat * inv
    q_np = np.rint(qf.cpu().numpy())
    q = torch.from_numpy(q_np).to(device=x.device, dtype=torch.float32).clamp(-128, 127).to(torch.int8)
    y = q.reshape_as(x)
    scale = (max_abs / 127.0).squeeze(-1).reshape(x.shape[:-1]).to(torch.float32)
    return (y, scale)

class Model(nn.Module):

    def forward(self, x: torch.Tensor, smooth: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        y, scale = golden_dynamic_quant_int8(x, smooth)
        return [y, scale]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_33_DynamicQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        smooth_info = inputs[1]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
        if smooth_info["type"] == "attr":
            if smooth_info.get("dtype") == "none":
                smooth = None
            else:
                smooth = smooth_info["value"]
        else:
            if "data" in smooth_info:
                smooth = torch.tensor(smooth_info["data"], dtype=DTYPE_MAP[smooth_info["dtype"]]).reshape(smooth_info["shape"])
            else:
                smooth = torch.rand(smooth_info["shape"], dtype=DTYPE_MAP[smooth_info["dtype"]])

        input_groups.append([x, smooth])
    return input_groups


def get_init_inputs():
    return []
