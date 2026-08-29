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
NsaCompress：CPU 参考为按 batch 段内滑窗（步长 compressStride）的加权行求和，
与文档中滑窗压缩语义一致；累加使用 float32 再回 cast 到输入 dtype。
"""
import ast
import torch
import torch.nn as nn

def _compress_out_rows(act_cumsum, compress_block_size: int, compress_stride: int) -> int:
    pre = 0
    total = 0
    for end in act_cumsum:
        cur = end - pre
        if cur >= compress_block_size:
            total += (cur - compress_block_size + compress_stride) // compress_stride
        pre = end
    return total

def _ceil_power2_u32(n: int) -> int:
    """与内核 getCeilPower2 一致（n>=1）。"""
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1

def reduce_block_like_kernel(mul_rows: torch.Tensor) -> torch.Tensor:
    """
    与 AscendC ReduceBlock 一致：mul_rows 为 [L, dim] float32。
    先将 mul_rows[0:add_n] += mul_rows[align:align+add_n]，再对折归约至首行。
    """
    l_len, dim = mul_rows.shape
    if l_len == 1:
        return mul_rows[0]
    align = _ceil_power2_u32(l_len) // 2
    add_n = l_len - align
    x = mul_rows.clone()
    x[0:add_n] = x[0:add_n] + x[align:align + add_n]
    while align > 1:
        align = align >> 1
        x[0:align] = x[0:align] + x[align:2 * align]
    return x[0]

def nsa_compress_reference(input_tensor: torch.Tensor, weight: torch.Tensor, act_seq_len_cumsum: torch.Tensor, compress_block_size: int, compress_stride: int) -> torch.Tensor:
    """CPU 金标准：TND，actSeqLen 为 cumsum。"""
    t, n, d = input_tensor.shape
    l_blk, n_w = weight.shape
    assert l_blk == compress_block_size and n_w == n
    cum = act_seq_len_cumsum.detach().cpu().tolist()
    m = _compress_out_rows(cum, compress_block_size, compress_stride)
    out = torch.empty((m, n, d), dtype=input_tensor.dtype, device=input_tensor.device)
    inp_f = input_tensor.float()
    w_f = weight.float()
    pre = 0
    out_idx = 0
    flat_dim = n * d
    for end in cum:
        seg = inp_f[pre:end]
        t_seg = seg.shape[0]
        k = 0
        while k * compress_stride + compress_block_size <= t_seg:
            s = k * compress_stride
            block = seg[s:s + compress_block_size]
            mul = block * w_f.unsqueeze(-1)
            mul_flat = mul.reshape(compress_block_size, flat_dim)
            red = reduce_block_like_kernel(mul_flat)
            out[out_idx].copy_(red.view(n, d).to(input_tensor.dtype))
            out_idx += 1
            k += 1
        pre = end
    assert out_idx == m
    return out

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: torch.Tensor, weight: torch.Tensor, act_seq_len_cumsum: torch.Tensor, compress_block_size: int, compress_stride: int, act_seq_len_type: int):
        del act_seq_len_type
        return nsa_compress_reference(input_tensor, weight, act_seq_len_cumsum, compress_block_size, compress_stride)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_47_NsaCompress.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_tensor_info = inputs[0]
        weight_info = inputs[1]
        act_seq_len_cumsum_info = inputs[2]
        compress_block_size_info = inputs[3]
        compress_stride_info = inputs[4]
        act_seq_len_type_info = inputs[5]

        if "data" in input_tensor_info:
            input_tensor = torch.tensor(input_tensor_info["data"], dtype=DTYPE_MAP[input_tensor_info["dtype"]]).reshape(input_tensor_info["shape"])
        else:
            input_tensor = torch.randn(input_tensor_info["shape"], dtype=DTYPE_MAP[input_tensor_info["dtype"]])
        if "data" in weight_info:
            weight = torch.tensor(weight_info["data"], dtype=DTYPE_MAP[weight_info["dtype"]]).reshape(weight_info["shape"])
        else:
            weight = torch.randn(weight_info["shape"], dtype=DTYPE_MAP[weight_info["dtype"]])
        if "data" in act_seq_len_cumsum_info:
            act_seq_len_cumsum = torch.tensor(act_seq_len_cumsum_info["data"], dtype=DTYPE_MAP[act_seq_len_cumsum_info["dtype"]]).reshape(act_seq_len_cumsum_info["shape"])
        else:
            act_seq_len_cumsum = torch.randint(act_seq_len_cumsum_info["range"][0], act_seq_len_cumsum_info["range"][1] + 1, tuple(act_seq_len_cumsum_info["shape"]), dtype=DTYPE_MAP[act_seq_len_cumsum_info["dtype"]])
        compress_block_size = compress_block_size_info["value"]
        compress_stride = compress_stride_info["value"]
        act_seq_len_type = act_seq_len_type_info["value"]

        input_groups.append([input_tensor, weight, act_seq_len_cumsum, compress_block_size, compress_stride, act_seq_len_type])
    return input_groups


def get_init_inputs():
    return []
