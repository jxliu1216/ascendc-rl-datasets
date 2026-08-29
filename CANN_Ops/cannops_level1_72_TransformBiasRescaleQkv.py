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

from typing import List
import torch
import torch.nn as nn

def _ref_transform_bias_rescale_qkv(qkv: torch.Tensor, qkv_bias: torch.Tensor, num_heads: int) -> List[torch.Tensor]:
    """Reference implementation: split qkv to q,k,v, add bias, rescale q by 1/sqrt(dim_per_head)."""
    batch, token, triple_dim = qkv.shape
    dim = triple_dim // 3
    dim_per_head = dim // num_heads
    scale = dim_per_head ** (-0.5)
    qkv_flat = (qkv + qkv_bias).float()
    qkv_flat = qkv_flat.view(batch, token, 3, num_heads, dim_per_head)
    q = qkv_flat[:, :, 0, :, :].permute(0, 2, 1, 3)
    k = qkv_flat[:, :, 1, :, :].permute(0, 2, 1, 3)
    v = qkv_flat[:, :, 2, :, :].permute(0, 2, 1, 3)
    q = (q * scale).to(qkv.dtype)
    return [q, k, v]

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, qkv: torch.Tensor, qkv_bias: torch.Tensor, num_heads: int) -> List[torch.Tensor]:
        return _ref_transform_bias_rescale_qkv(qkv, qkv_bias, num_heads)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_72_TransformBiasRescaleQkv.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        qkv_info = inputs[0]
        qkv_bias_info = inputs[1]
        num_heads_info = inputs[2]

        if "data" in qkv_info:
            qkv = torch.tensor(qkv_info["data"], dtype=DTYPE_MAP[qkv_info["dtype"]]).reshape(qkv_info["shape"])
        else:
            qkv = torch.randn(qkv_info["shape"], dtype=DTYPE_MAP[qkv_info["dtype"]])
        if "data" in qkv_bias_info:
            qkv_bias = torch.tensor(qkv_bias_info["data"], dtype=DTYPE_MAP[qkv_bias_info["dtype"]]).reshape(qkv_bias_info["shape"])
        else:
            qkv_bias = torch.randn(qkv_bias_info["shape"], dtype=DTYPE_MAP[qkv_bias_info["dtype"]])
        num_heads = num_heads_info["value"]

        input_groups.append([qkv, qkv_bias, num_heads])
    return input_groups


def get_init_inputs():
    return []
