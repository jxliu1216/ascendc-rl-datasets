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
CPU golden：与 trans_quant_param_v2.h 中 round_mode==0 路径一致：
pack_scale = (float32_bits(scale) & 0xFFFFE000) | (1<<46)；
offset 按 CAST_RINT 到 int32 后 clamp 到 [-256,255]，取低 9 bit 左移 37 后与 pack_scale 按位或。
"""
import struct
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
DEQ_SCALE_MUL = 4294959104
QUANT_SCALE = 1 << 46
QUANT_MASK_0 = 511
OFFSET_DEVIATION = 37

def _pack_scale_rm0(f: float) -> np.uint64:
    u32 = struct.unpack('I', struct.pack('f', np.float32(f)))[0]
    return np.uint64(u32 & DEQ_SCALE_MUL | QUANT_SCALE)

def _offset_bits(f: float) -> np.uint64:
    v = int(np.rint(np.float32(f)))
    v = max(-256, min(255, v))
    return (np.uint64(v) & np.uint64(QUANT_MASK_0)) << np.uint64(OFFSET_DEVIATION)

def golden(scale: torch.Tensor, offset: Optional[torch.Tensor], round_mode: int) -> torch.Tensor:
    if int(round_mode) != 0:
        raise ValueError('validation 仅对齐 aclnnTransQuantParamV2（内部 roundMode 固定为 0）')
    s = scale.detach().cpu().numpy().astype(np.float32).reshape(-1)
    n = s.size
    out = np.zeros(n, dtype=np.uint64)
    if offset is None:
        for i in range(n):
            out[i] = _pack_scale_rm0(float(s[i]))
        return torch.from_numpy(out).to(device=scale.device)
    o = offset.detach().cpu().numpy().astype(np.float32).reshape(-1)
    m = o.size
    if m == 1 and n > 1:
        ob = _offset_bits(float(o[0]))
        for i in range(n):
            out[i] = _pack_scale_rm0(float(s[i])) | ob
    elif m == n:
        for i in range(n):
            out[i] = _pack_scale_rm0(float(s[i])) | _offset_bits(float(o[i]))
    else:
        raise ValueError(f'unsupported scale len {n} vs offset len {m}')
    return torch.from_numpy(out).to(device=scale.device)

class Model(nn.Module):

    def __init__(self, round_mode: int):
        super().__init__()
        self.round_mode = int(round_mode)

    def forward(self, scale: torch.Tensor, offset=None):
        return golden(scale, offset, self.round_mode)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_149_TransQuantParamV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        scale_info = inputs[0]
        offset_info = inputs[1]

        if "data" in scale_info:
            scale = torch.tensor(scale_info["data"], dtype=DTYPE_MAP[scale_info["dtype"]]).reshape(scale_info["shape"])
        else:
            scale = torch.rand(scale_info["shape"], dtype=DTYPE_MAP[scale_info["dtype"]])
        if offset_info["type"] == "attr":
            if offset_info.get("dtype") == "none":
                offset = None
            else:
                offset = offset_info["value"]
        else:
            if "data" in offset_info:
                offset = torch.tensor(offset_info["data"], dtype=DTYPE_MAP[offset_info["dtype"]]).reshape(offset_info["shape"])
            else:
                offset = torch.rand(offset_info["shape"], dtype=DTYPE_MAP[offset_info["dtype"]]) * (offset_info["range"][1] - offset_info["range"][0]) + offset_info["range"][0]

        input_groups.append([scale, offset])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_149_TransQuantParamV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        round_mode_info = entries[0]
        round_mode = round_mode_info["value"]
        init_groups.append([round_mode])
    return init_groups
