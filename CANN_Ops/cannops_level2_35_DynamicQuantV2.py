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
CPU golden 对齐 answer/0/op_kernel 同目录下 dynamic_quant.h::ComputAsymmetric（int8）：
NPU 上 offset 输出缓冲区非空 → InitParams 将 isAsymmetrical 置 true，走非对称路径。
scale = max((max-min)/255, eps)；offset = 127 - max/scale；逐行 x/scale+offset 后 CAST_RINT → int8。
输出 scale/offset 形状与 infershape 默认 pertoken 一致（x 去掉最后一维）。
"""
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

def golden_dynamic_quant_v2_int8_pertoken(x: torch.Tensor, smooth: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_f = x.detach().float()
    if smooth is not None:
        x_f = x_f * smooth.detach().float()
    last = x_f.shape[-1]
    flat = x_f.reshape(-1, last)
    max_v = flat.amax(dim=1, keepdim=True)
    min_v = flat.amin(dim=1, keepdim=True)
    eps = 1e-12
    scale = torch.maximum((max_v - min_v) / 255.0, torch.full_like(max_v, eps))
    offset = 127.0 - max_v / scale
    t = flat / scale + offset
    q_np = np.rint(t.cpu().numpy())
    q = torch.from_numpy(q_np).to(device=x.device).clamp(-128, 127).to(torch.int8)
    y = q.reshape_as(x)
    scale_out = scale.squeeze(-1).reshape(x.shape[:-1]).to(torch.float32)
    offset_out = offset.squeeze(-1).reshape(x.shape[:-1]).to(torch.float32)
    return (y, scale_out, offset_out)

class Model(nn.Module):

    def __init__(self, dst_type: int):
        super().__init__()
        self.dst_type = int(dst_type)

    def forward(self, x: torch.Tensor, smooth: Optional[torch.Tensor]=None, group_index: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        _ = self.dst_type
        _ = group_index
        y, s, o = golden_dynamic_quant_v2_int8_pertoken(x, smooth)
        return [y, s, o]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_35_DynamicQuantV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        smooth_info = inputs[1]
        group_index_info = inputs[2]

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
        group_index = None

        input_groups.append([x, smooth, group_index])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_35_DynamicQuantV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        dst_type_info = entries[0]
        dst_type = dst_type_info["value"]
        init_groups.append([dst_type])
    return init_groups
