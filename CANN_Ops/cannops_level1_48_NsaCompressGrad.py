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
NsaCompressGrad：CPU 金标准通过对可微的 forward（与 NsaCompress 参考同构的树形规约）做 autograd 得到 inputGrad / weightGrad。
"""
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
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1

def reduce_block_like_torch(mul_rows: torch.Tensor) -> torch.Tensor:
    """与 NsaCompress `reduce_block_like_kernel` 同构、可反传。"""
    x = mul_rows
    l_len, dim = x.shape
    if l_len == 1:
        return x[0]
    align = _ceil_power2_u32(l_len) // 2
    add_n = l_len - align
    new = x.clone()
    new[0:add_n] = x[0:add_n] + x[align:align + add_n]
    x = new
    while align > 1:
        align = align >> 1
        new = x.clone()
        new[0:align] = x[0:align] + x[align:2 * align]
        x = new
    return x[0]

def nsa_compress_forward_torch(input_tensor: torch.Tensor, weight: torch.Tensor, act_seq_len_cumsum: torch.Tensor, compress_block_size: int, compress_stride: int) -> torch.Tensor:
    """可微 forward（float32），与 `NsaCompress` 参考语义一致。"""
    t, n, d = input_tensor.shape
    l_blk, n_w = weight.shape
    assert l_blk == compress_block_size and n_w == n
    cum = act_seq_len_cumsum.detach().cpu().tolist()
    m = _compress_out_rows(cum, compress_block_size, compress_stride)
    out_rows = []
    inp_f = input_tensor
    w_f = weight
    pre = 0
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
            red = reduce_block_like_torch(mul_flat)
            out_rows.append(red.view(n, d))
            k += 1
        pre = end
    assert len(out_rows) == m
    return torch.stack(out_rows, dim=0)

def nsa_compress_grad_reference(output_grad: torch.Tensor, input_tensor: torch.Tensor, weight: torch.Tensor, act_seq_len_cumsum: torch.Tensor, compress_block_size: int, compress_stride: int, act_seq_len_type: int):
    del act_seq_len_type
    inp = input_tensor.float().detach().clone().requires_grad_(True)
    w = weight.float().detach().clone().requires_grad_(True)
    out = nsa_compress_forward_torch(inp, w, act_seq_len_cumsum, compress_block_size, compress_stride)
    loss = (out * output_grad.float()).sum()
    gi, gw = torch.autograd.grad(loss, (inp, w), retain_graph=False, create_graph=False)
    return (gi.to(input_tensor.dtype), gw.to(weight.dtype))

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output_grad: torch.Tensor, input_tensor: torch.Tensor, weight: torch.Tensor, act_seq_len_cumsum: torch.Tensor, compress_block_size: int, compress_stride: int, act_seq_len_type: int):
        ig, wg = nsa_compress_grad_reference(output_grad, input_tensor, weight, act_seq_len_cumsum, compress_block_size, compress_stride, act_seq_len_type)
        return [ig, wg]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_48_NsaCompressGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        output_grad_info = inputs[0]
        input_tensor_info = inputs[1]
        weight_info = inputs[2]
        act_seq_len_cumsum_info = inputs[3]
        compress_block_size_info = inputs[4]
        compress_stride_info = inputs[5]
        act_seq_len_type_info = inputs[6]

        if "data" in output_grad_info:
            output_grad = torch.tensor(output_grad_info["data"], dtype=DTYPE_MAP[output_grad_info["dtype"]]).reshape(output_grad_info["shape"])
        else:
            output_grad = torch.randn(output_grad_info["shape"], dtype=DTYPE_MAP[output_grad_info["dtype"]])
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

        input_groups.append([output_grad, input_tensor, weight, act_seq_len_cumsum, compress_block_size, compress_stride, act_seq_len_type])
    return input_groups


def get_init_inputs():
    return []
