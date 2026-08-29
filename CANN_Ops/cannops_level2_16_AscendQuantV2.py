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
`ops-nn-dev/quant/ascend_quant/tests/st/aclnnAscendQuant/executor_aclnnAscendQuant.py`
（.codex/TODO.md：AscendQuantV2 → quant → ops-nn-dev；L2 接口名为 aclnnAscendQuant，axis 固定 -1）。
"""
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
_GOLDEN_AXIS = -1

def _golden_aclnn_ascend_quant_int8(x: torch.Tensor, scale: torch.Tensor, offset: Optional[torch.Tensor], sqrt_mode: bool, round_mode: str, dst_type: int) -> torch.Tensor:
    """
    复刻 executor 中 __call__ 的计算顺序与舍入（float32 → numpy → np.round 等），仅实现 dst_type==2 (int8)。
    """
    x_t = x.to(torch.float32)
    scale_t = scale.to(torch.float32)
    offset_t = offset.to(torch.float32) if offset is not None else None
    if len(scale_t.shape) == 1:
        scale_new_shape = [1] * len(x_t.shape)
        scale_new_shape[_GOLDEN_AXIS] = scale_t.shape[0]
        scale_t = torch.reshape(scale_t, scale_new_shape)
        if offset_t is not None:
            offset_t = torch.reshape(offset_t, scale_new_shape)
    x_np = x_t.detach().cpu().numpy()
    scale_np = scale_t.detach().cpu().numpy()
    offset_np = offset_t.detach().cpu().numpy() if offset_t is not None else None
    if sqrt_mode:
        scale_sqrt = x_np * scale_np
        scale_rst = scale_sqrt * scale_np
    else:
        scale_rst = x_np * scale_np
    if offset_np is not None:
        add_offset = scale_rst + offset_np
    else:
        add_offset = scale_rst
    rm = str(round_mode).strip()
    if rm == 'round':
        round_data = np.round(add_offset, 0)
    elif rm == 'floor':
        round_data = np.floor(add_offset)
    elif rm == 'ceil':
        round_data = np.ceil(add_offset)
    elif rm == 'trunc':
        round_data = np.trunc(add_offset)
    else:
        raise ValueError(f'unsupported round_mode: {round_mode}')
    if dst_type == 2:
        round_data = np.clip(round_data, -128, 127)
        return torch.from_numpy(round_data.astype(np.int8)).to(x.device)
    raise ValueError(f'golden 仅支持 dst_type==2 (int8)，收到 {dst_type}')

class Model(nn.Module):

    def __init__(self, sqrt_mode: bool, round_mode: str, dst_type: int):
        super().__init__()
        self.sqrt_mode = bool(sqrt_mode)
        self.round_mode = str(round_mode)
        self.dst_type = int(dst_type)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, offset: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        assert self.dst_type == 2, 'validation 仅校验 int8 输出 (ge::DT_INT8 == 2)'
        y = _golden_aclnn_ascend_quant_int8(x, scale, offset, self.sqrt_mode, self.round_mode, self.dst_type)
        return [y]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_16_AscendQuantV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        scale_info = inputs[1]
        offset_info = inputs[2]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
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

        input_groups.append([x, scale, offset])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_16_AscendQuantV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        sqrt_mode_info = entries[0]
        round_mode_info = entries[1]
        dst_type_info = entries[2]
        sqrt_mode = sqrt_mode_info["value"]
        round_mode = round_mode_info["value"]
        dst_type = dst_type_info["value"]
        init_groups.append([sqrt_mode, round_mode, dst_type])
    return init_groups
