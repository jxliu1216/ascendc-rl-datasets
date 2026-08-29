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
CPU 金标准与 `posembedding/rotary_position_embedding_grad/README.md` / aclnn 文档中 half、interleave 的反向公式一致。
当前 Bench 目标核（如 910B）与 RotaryPositionEmbedding 一致，仅覆盖 mode 0、1；dcos/dsin 对 broadcast 轴求和。
"""
from typing import List, Tuple
import torch
import torch.nn as nn

def _sum_grad_to_broadcast_input(grad_full: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """将 [B,N,S,D] 上的梯度规约到 ref（如 cos/sin）的 shape。"""
    g = grad_full
    for d in range(grad_full.dim()):
        if ref.shape[d] == 1 and g.shape[d] != 1:
            g = g.sum(dim=d, keepdim=True)
    return g

def golden_rotary_position_embedding_grad(dy: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, x: torch.Tensor, mode: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dy_f = dy.float()
    cos_b = torch.broadcast_to(cos.float(), dy.shape)
    sin_b = torch.broadcast_to(sin.float(), dy.shape)
    xf = torch.broadcast_to(x.float(), dy.shape)
    d = dy.shape[-1]
    if mode == 0:
        half = d // 2
        dy1, dy2 = (dy_f[..., :half], dy_f[..., half:])
        cos1, cos2 = (cos_b[..., :half], cos_b[..., half:])
        sin1, sin2 = (sin_b[..., :half], sin_b[..., half:])
        x1, x2 = (xf[..., :half], xf[..., half:])
        dx = torch.cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1)
        g_dcos = dy_f * xf
        g_dsin = dy_f * torch.cat((-x2, x1), dim=-1)
    elif mode == 1:
        dy1, dy2 = (dy_f[..., ::2], dy_f[..., 1::2])
        cos1, cos2 = (cos_b[..., ::2], cos_b[..., 1::2])
        sin1, sin2 = (sin_b[..., ::2], sin_b[..., 1::2])
        x1, x2 = (xf[..., ::2], xf[..., 1::2])
        dx = torch.stack((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1).reshape(dy.shape)
        g_dcos = dy_f * xf
        rot_x = torch.stack((-x2, x1), dim=-1).reshape(xf.shape)
        g_dsin = dy_f * rot_x
    else:
        raise ValueError(f'unsupported mode {mode} for this validation')
    dcos = _sum_grad_to_broadcast_input(g_dcos, cos.float())
    dsin = _sum_grad_to_broadcast_input(g_dsin, sin.float())
    return (dx.to(dy.dtype), dcos.to(cos.dtype), dsin.to(sin.dtype))

class Model(nn.Module):

    def __init__(self, mode: int=0):
        super().__init__()
        self.mode = int(mode)

    def forward(self, dy: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, x: torch.Tensor, _dx: torch.Tensor, _dcos: torch.Tensor, _dsin: torch.Tensor) -> List[torch.Tensor]:
        del _dx, _dcos, _dsin
        dx, dcos, dsin = golden_rotary_position_embedding_grad(dy, cos, sin, x, self.mode)
        return [dx, dcos, dsin]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_45_RotaryPositionEmbeddingGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dy_info = inputs[0]
        cos_info = inputs[1]
        sin_info = inputs[2]
        x_info = inputs[3]
        _dx_info = inputs[4]
        _dcos_info = inputs[5]
        _dsin_info = inputs[6]

        if "data" in dy_info:
            dy = torch.tensor(dy_info["data"], dtype=DTYPE_MAP[dy_info["dtype"]]).reshape(dy_info["shape"])
        else:
            dy = torch.rand(dy_info["shape"], dtype=DTYPE_MAP[dy_info["dtype"]]) * (dy_info["range"][1] - dy_info["range"][0]) + dy_info["range"][0]
        if "data" in cos_info:
            cos = torch.tensor(cos_info["data"], dtype=DTYPE_MAP[cos_info["dtype"]]).reshape(cos_info["shape"])
        else:
            cos = torch.rand(cos_info["shape"], dtype=DTYPE_MAP[cos_info["dtype"]]) * (cos_info["range"][1] - cos_info["range"][0]) + cos_info["range"][0]
        if "data" in sin_info:
            sin = torch.tensor(sin_info["data"], dtype=DTYPE_MAP[sin_info["dtype"]]).reshape(sin_info["shape"])
        else:
            sin = torch.rand(sin_info["shape"], dtype=DTYPE_MAP[sin_info["dtype"]]) * (sin_info["range"][1] - sin_info["range"][0]) + sin_info["range"][0]
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
        _dx = torch.empty(_dx_info["shape"], dtype=DTYPE_MAP[_dx_info["dtype"]])
        _dcos = torch.empty(_dcos_info["shape"], dtype=DTYPE_MAP[_dcos_info["dtype"]])
        _dsin = torch.empty(_dsin_info["shape"], dtype=DTYPE_MAP[_dsin_info["dtype"]])

        input_groups.append([dy, cos, sin, x, _dx, _dcos, _dsin])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_45_RotaryPositionEmbeddingGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        mode_info = entries[0]
        mode = mode_info["value"]
        init_groups.append([mode])
    return init_groups
