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
CPU golden 与 AICore 中 dynamic_quant_update_scatter_comm.h::ComputeQuant（int8）一致：
对 axis=-2 上每个长度为 D3 的切片单独做动态量化（与 tiling 中 quantReptNum * updateAxisShape 分段一致；
可选 smooth；127/abs 后 ReduceMin；乘回后 int32→fp16→int8 与内核 CAST 链对齐）。
scatter 偏移与 GetDetOffsetNeg2（1D / 2D indices）一致；axis 须为 -2（与 tiling 支持一致）。
"""
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

def _quantize_segment_fp32(u_fp32: np.ndarray, smooth_fp32: Optional[np.ndarray]) -> Tuple[np.ndarray, np.float32]:
    """u_fp32: 1D float32, length L. 返回 (q_int8[L], scale_fp)。与内核 CAST_RINT→int32→fp16→int8 对齐。"""
    x = u_fp32.astype(np.float64, copy=True)
    if smooth_fp32 is not None:
        x = x * smooth_fp32.astype(np.float64)
    ax = np.abs(x)
    ax = np.maximum(ax, 1e-38)
    inv_scale_per = 127.0 / ax
    inv_scale = float(np.min(inv_scale_per))
    scale_out = np.float32(1.0 / inv_scale)
    xs = x * inv_scale
    t = torch.from_numpy(xs.astype(np.float32))
    i32 = torch.round(t).to(torch.int32)
    h = i32.to(torch.float16)
    y = h.to(torch.int8).numpy()
    return (y, scale_out)

def golden_scatter(var: torch.Tensor, var_scale: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, smooth: Optional[torch.Tensor], axis: int, indices_rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if axis != -2:
        raise ValueError('golden only implements axis=-2 to match op_host tiling')
    d0, d1, d2, d3 = (int(var.shape[i]) for i in range(4))
    if tuple(var.shape) != tuple(updates.shape):
        raise ValueError('var and updates must share shape')
    exp_scale = (d0, d1, d2, 1)
    if tuple(var_scale.shape) != exp_scale:
        raise ValueError('var_scale must be (*var.shape[:-1], 1), same rank as var')
    dst_bs_stride = d2 * d3
    num_head = d1
    size_per_head = d3
    total = d0 * d1
    src_bs_stride = dst_bs_stride
    var_o = var.detach().clone()
    sc_o = var_scale.detach().clone()
    flat_v = var_o.view(-1)
    flat_s = sc_o.view(-1)
    upd = updates.detach().float()
    sm_np: Optional[np.ndarray] = None
    if smooth is not None:
        sm_np = smooth.detach().float().cpu().numpy().reshape(-1)
        if sm_np.size != d3:
            raise ValueError('smooth must match last dim')
    ind = indices.detach().cpu()
    if indices_rank == 1:
        ind1 = ind.numpy().astype(np.int64, copy=False).reshape(-1)
        if ind1.size < d0:
            raise ValueError('1D indices length must be >= d0')
        for g in range(total):
            u0, u1 = (g // d1, g % d1)
            index_idx = g // num_head
            valid_idx = int(ind1[index_idx])
            dst_offset = g * dst_bs_stride + valid_idx * size_per_head
            blk = np.empty((d2, d3), dtype=np.int8)
            scales_row = []
            for j in range(d2):
                uvec = upd[u0, u1, j, :].reshape(-1).cpu().numpy()
                q, sc = _quantize_segment_fp32(uvec, sm_np)
                blk[j, :] = q
                scales_row.append(sc)
            flat_v[dst_offset:dst_offset + src_bs_stride] = torch.from_numpy(blk.reshape(-1)).to(var_o.dtype)
            base_s = dst_offset // d3
            for j in range(d2):
                flat_s[base_s + j] = torch.tensor(scales_row[j], dtype=sc_o.dtype)
    elif indices_rank == 2:
        ind2 = ind.numpy().astype(np.int64, copy=False)
        if ind2.ndim != 2 or ind2.shape[1] != 2:
            raise ValueError('2D indices must be [K,2]')
        for g in range(total):
            u0, u1 = (g // d1, g % d1)
            index_idx = g // num_head
            bs_idx = int(ind2[index_idx, 0])
            valid_idx = int(ind2[index_idx, 1])
            actual_batch = bs_idx * num_head + g % num_head
            dst_offset = actual_batch * dst_bs_stride + valid_idx * size_per_head
            blk = np.empty((d2, d3), dtype=np.int8)
            scales_row = []
            for j in range(d2):
                uvec = upd[u0, u1, j, :].reshape(-1).cpu().numpy()
                q, sc = _quantize_segment_fp32(uvec, sm_np)
                blk[j, :] = q
                scales_row.append(sc)
            flat_v[dst_offset:dst_offset + src_bs_stride] = torch.from_numpy(blk.reshape(-1)).to(var_o.dtype)
            base_s = dst_offset // d3
            for j in range(d2):
                flat_s[base_s + j] = torch.tensor(scales_row[j], dtype=sc_o.dtype)
    else:
        raise ValueError('indices_rank must be 1 or 2')
    return (var_o, sc_o)

class Model(nn.Module):

    def __init__(self, reduce_str: str, axis: int, indices_rank: int):
        super().__init__()
        self.reduce_str = reduce_str
        self.axis = int(axis)
        self.indices_rank = int(indices_rank)

    def forward(self, var: torch.Tensor, var_scale: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, smooth: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        y, s = golden_scatter(var, var_scale, indices, updates, smooth, self.axis, self.indices_rank)
        return [y, s]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_34_DynamicQuantUpdateScatter.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        var_info = inputs[0]
        var_scale_info = inputs[1]
        indices_info = inputs[2]
        updates_info = inputs[3]
        smooth_info = inputs[4]

        if "data" in var_info:
            var = torch.tensor(var_info["data"], dtype=DTYPE_MAP[var_info["dtype"]]).reshape(var_info["shape"])
        else:
            var = torch.randint(var_info["range"][0], var_info["range"][1] + 1, tuple(var_info["shape"]), dtype=DTYPE_MAP[var_info["dtype"]])
        if "data" in var_scale_info:
            var_scale = torch.tensor(var_scale_info["data"], dtype=DTYPE_MAP[var_scale_info["dtype"]]).reshape(var_scale_info["shape"])
        else:
            var_scale = torch.rand(var_scale_info["shape"], dtype=DTYPE_MAP[var_scale_info["dtype"]])
        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])
        if "data" in updates_info:
            updates = torch.tensor(updates_info["data"], dtype=DTYPE_MAP[updates_info["dtype"]]).reshape(updates_info["shape"])
        else:
            updates = torch.rand(updates_info["shape"], dtype=DTYPE_MAP[updates_info["dtype"]]) * (updates_info["range"][1] - updates_info["range"][0]) + updates_info["range"][0]
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

        input_groups.append([var, var_scale, indices, updates, smooth])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_34_DynamicQuantUpdateScatter.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        reduce_str_info = entries[0]
        axis_info = entries[1]
        indices_rank_info = entries[2]
        reduce_str = reduce_str_info["value"]
        axis = axis_info["value"]
        indices_rank = indices_rank_info["value"]
        init_groups.append([reduce_str, axis, indices_rank])
    return init_groups
