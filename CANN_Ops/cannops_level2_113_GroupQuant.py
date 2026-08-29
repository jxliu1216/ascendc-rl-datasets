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
CPU 金标准对齐 kernel group_quant_base.h::VecCompute：
x*scale(+offset) 在 fp32 上先 CAST_RINT→int32，再经 fp16 回读，最后 CAST_RINT→int8；
与 ops-nn-dev executor_aclnnGroupQuant 的分组边界语义一致。
"""
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
DTYPE_INT8 = 2
DTYPE_INT4 = 29

def _ascend_int8_from_float(y_fp32: np.ndarray) -> np.ndarray:
    i32 = np.rint(y_fp32).astype(np.int32)
    h = i32.astype(np.float16).astype(np.float32)
    out = np.rint(h)
    return np.clip(out, -128, 127).astype(np.int8)

def _ascend_int4_from_float(y_fp32: np.ndarray) -> np.ndarray:
    i32 = np.rint(y_fp32).astype(np.int32)
    h = i32.astype(np.float16).astype(np.float32)
    out = np.rint(h)
    return np.clip(out, -8, 7).astype(np.int8)

def golden_group_quant(x: torch.Tensor, scale: torch.Tensor, group_index: torch.Tensor, offset: Optional[torch.Tensor], dst_type: int) -> torch.Tensor:
    x_np = x.detach().float().cpu().numpy()
    scale_np = scale.detach().float().cpu().numpy()
    gi = group_index.detach().cpu().numpy().astype(np.int64)
    dim_s, dim_h = x_np.shape
    dim_e, dim_h2 = scale_np.shape
    if dim_h != dim_h2 or gi.shape[0] != dim_e:
        raise ValueError('shape mismatch x/scale/group_index')
    if int(gi[-1]) != dim_s:
        raise ValueError('group_index[-1] must equal S')
    off = 0.0
    if offset is not None:
        off = float(offset.detach().float().cpu().reshape(-1)[0])
    parts = []
    for row_scale in range(dim_e):
        r0 = 0 if row_scale == 0 else int(gi[row_scale - 1])
        r1 = int(gi[row_scale])
        if r0 < r1:
            blk = x_np[r0:r1] * scale_np[row_scale:row_scale + 1]
            blk = blk + off
            parts.append(blk)
    y_fp32 = np.concatenate(parts, axis=0)
    if int(dst_type) == DTYPE_INT8:
        y_np = _ascend_int8_from_float(y_fp32)
    elif int(dst_type) == DTYPE_INT4:
        y_np = _ascend_int4_from_float(y_fp32)
    else:
        raise ValueError('dst_type must be 2 (int8) or 29 (int4)')
    return torch.from_numpy(y_np).to(device=x.device)

class Model(nn.Module):

    def __init__(self, dst_type: int):
        super().__init__()
        self.dst_type = int(dst_type)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, group_index: torch.Tensor, offset: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        y = golden_group_quant(x, scale, group_index, offset, self.dst_type)
        return [y]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_113_GroupQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        scale_info = inputs[1]
        group_index_info = inputs[2]
        offset_info = inputs[3]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
        if "data" in scale_info:
            scale = torch.tensor(scale_info["data"], dtype=DTYPE_MAP[scale_info["dtype"]]).reshape(scale_info["shape"])
        else:
            scale = torch.rand(scale_info["shape"], dtype=DTYPE_MAP[scale_info["dtype"]])
        if "data" in group_index_info:
            group_index = torch.tensor(group_index_info["data"], dtype=DTYPE_MAP[group_index_info["dtype"]]).reshape(group_index_info["shape"])
        else:
            group_index = torch.randint(group_index_info["range"][0], group_index_info["range"][1] + 1, tuple(group_index_info["shape"]), dtype=DTYPE_MAP[group_index_info["dtype"]])
        if offset_info["type"] == "attr":
            if offset_info.get("dtype") == "none":
                offset = None
            else:
                offset = offset_info["value"]
        else:
            if "data" in offset_info:
                offset = torch.tensor(offset_info["data"], dtype=DTYPE_MAP[offset_info["dtype"]]).reshape(offset_info["shape"])
            else:
                offset = torch.rand(offset_info["shape"], dtype=DTYPE_MAP[offset_info["dtype"]])

        input_groups.append([x, scale, group_index, offset])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_113_GroupQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        dst_type_info = entries[0]
        dst_type = dst_type_info["value"]
        init_groups.append([dst_type])
    return init_groups
