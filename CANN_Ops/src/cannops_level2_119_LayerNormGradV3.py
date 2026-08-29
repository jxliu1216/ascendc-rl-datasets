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

from typing import List, Optional
import torch
import torch.nn as nn

def _layer_norm_grad_ref_pytorch(dy: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], normalized_shape: List[int]) -> List[torch.Tensor]:
    """标杆：直接用 PyTorch 官方 native_layer_norm_backward，公式与舍入与 PyTorch 完全一致."""
    output_mask = (True, True, True)
    dx, dgamma, dbeta = torch.ops.aten.native_layer_norm_backward(dy, x, normalized_shape, mean, rstd, weight, bias, output_mask)
    dtype_out = weight.dtype if weight is not None else x.dtype
    dev = x.device
    return [dx, dgamma if dgamma is not None else torch.zeros(normalized_shape, dtype=dtype_out, device=dev), dbeta if dbeta is not None else torch.zeros(normalized_shape, dtype=dtype_out, device=dev)]

def _layer_norm_grad_ref_fallback(dy: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, weight: Optional[torch.Tensor], normalized_shape: List[int]) -> List[torch.Tensor]:
    """无 aten 时的回退：手写公式，与 aclnn 对齐——dgamma/dbeta 在 float32 下 sum 再 cast 到输出 dtype."""
    input_dim = x.dim()
    normalized_dim = len(normalized_shape)
    reduction_dims = tuple(range(input_dim - normalized_dim, input_dim))
    N = 1
    for i in reduction_dims:
        N *= x.shape[i]
    dtype_orig = x.dtype
    compute_dtype = dtype_orig if dtype_orig in (torch.float16, torch.bfloat16) else torch.float32
    dy_c = dy.to(compute_dtype)
    x_c = x.to(compute_dtype)
    mean_c = mean.to(compute_dtype)
    rstd_c = rstd.to(compute_dtype)
    weight_c = weight.to(compute_dtype) if weight is not None else None
    x_norm = (x_c - mean_c) * rstd_c
    dy_weighted = dy_c * weight_c if weight_c is not None else dy_c
    N_t = torch.tensor(N, dtype=compute_dtype, device=x.device)
    sum1 = dy_weighted.sum(dim=reduction_dims, keepdim=True).to(compute_dtype)
    sum2 = (dy_weighted * x_norm).sum(dim=reduction_dims, keepdim=True).to(compute_dtype)
    c1 = sum1 / N_t
    c2 = sum2 / N_t
    dx = (dy_weighted - c1 - x_norm * c2) * rstd_c
    dx = dx.to(dtype_orig)
    batch_dims = tuple(range(0, input_dim - normalized_dim))
    dgamma = (dy_weighted.float() * x_norm.float()).sum(dim=batch_dims)
    dbeta = dy_c.float().sum(dim=batch_dims)
    dtype_out = weight.dtype if weight is not None else dtype_orig
    dgamma = dgamma.to(dtype_out)
    dbeta = dbeta.to(dtype_out)
    return [dx, dgamma, dbeta]

def _layer_norm_grad_ref(dy: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], normalized_shape: List[int]) -> List[torch.Tensor]:
    """优先用 PyTorch 官方 backward，否则回退到手写公式."""
    try:
        return _layer_norm_grad_ref_pytorch(dy, x, mean, rstd, weight, bias, normalized_shape)
    except Exception:
        return _layer_norm_grad_ref_fallback(dy, x, mean, rstd, weight, normalized_shape)

class Model(nn.Module):

    def __init__(self, normalized_shape: List[int]):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(self, dy: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]) -> List[torch.Tensor]:
        return _layer_norm_grad_ref(dy, x, mean, rstd, weight, bias, self.normalized_shape)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_119_LayerNormGradV3.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dy_info = inputs[0]
        x_info = inputs[1]
        mean_info = inputs[2]
        rstd_info = inputs[3]
        weight_info = inputs[4]
        bias_info = inputs[5]

        if "data" in dy_info:
            dy = torch.tensor(dy_info["data"], dtype=DTYPE_MAP[dy_info["dtype"]]).reshape(dy_info["shape"])
        else:
            dy = torch.rand(dy_info["shape"], dtype=DTYPE_MAP[dy_info["dtype"]])
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in mean_info:
            mean = torch.tensor(mean_info["data"], dtype=DTYPE_MAP[mean_info["dtype"]]).reshape(mean_info["shape"])
        else:
            mean_ms = mean_info["shape"]
            mean_red = tuple(_i for _i in range(len(x.shape)) if mean_ms[_i] == 1 and x.shape[_i] != 1)
            mean = x.mean(dim=mean_red, keepdim=True)
        if "data" in rstd_info:
            rstd = torch.tensor(rstd_info["data"], dtype=DTYPE_MAP[rstd_info["dtype"]]).reshape(rstd_info["shape"])
        else:
            rstd_red = tuple(_i for _i in range(len(x.shape)) if rstd_info["shape"][_i] == 1 and x.shape[_i] != 1)
            rstd = torch.rsqrt((x - mean).pow(2).mean(dim=rstd_red, keepdim=True) + 1e-05)
        if "data" in weight_info:
            weight = torch.tensor(weight_info["data"], dtype=DTYPE_MAP[weight_info["dtype"]]).reshape(weight_info["shape"])
        else:
            weight = torch.rand(weight_info["shape"], dtype=DTYPE_MAP[weight_info["dtype"]])
        if "data" in bias_info:
            bias = torch.tensor(bias_info["data"], dtype=DTYPE_MAP[bias_info["dtype"]]).reshape(bias_info["shape"])
        else:
            bias = torch.rand(bias_info["shape"], dtype=DTYPE_MAP[bias_info["dtype"]])

        input_groups.append([dy, x, mean, rstd, weight, bias])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_119_LayerNormGradV3.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        normalized_shape_info = entries[0]
        normalized_shape = normalized_shape_info["value"]
        init_groups.append([normalized_shape])
    return init_groups
