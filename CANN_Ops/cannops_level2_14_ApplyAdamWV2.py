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


class Model(nn.Module):
    """Self-contained AdamW reference (does not depend on torch.optim internals)."""

    def __init__(self):
        super().__init__()

    def forward(self, var_ref: torch.Tensor, m_ref: torch.Tensor, v_ref: torch.Tensor,
                grad: torch.Tensor, step: torch.Tensor, max_grad_norm_ref: torch.Tensor,
                lr: float, beta1: float, beta2: float, weight_decay: float, eps: float,
                amsgrad: bool, maximize: bool) -> List[torch.Tensor]:
        dtype1 = var_ref.dtype
        if grad.dtype != dtype1:
            grad = grad.to(dtype1)
        if max_grad_norm_ref.dtype != dtype1:
            max_grad_norm_ref = max_grad_norm_ref.to(dtype1)

        compute_dtype = torch.float32
        p = var_ref.to(compute_dtype).clone()
        exp_avg = m_ref.to(compute_dtype).clone()
        exp_avg_sq = v_ref.to(compute_dtype).clone()
        g = grad.to(compute_dtype).clone()
        if maximize:
            g = -g
        # Source CSV stores (T-1); kernel adds 1 internally. Use T for bias.
        t = (float(step.item()) if isinstance(step, torch.Tensor) else float(step)) + 1.0

        p.mul_(1.0 - lr * weight_decay)
        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

        bias1 = 1.0 - beta1 ** t
        bias2 = 1.0 - beta2 ** t
        step_size = lr / bias1
        bias_correction2_sqrt = bias2 ** 0.5

        if amsgrad:
            max_exp = max_grad_norm_ref.to(compute_dtype).clone()
            torch.maximum(max_exp, exp_avg_sq, out=max_exp)
            denom = (max_exp.sqrt() / bias_correction2_sqrt).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        p.addcdiv_(exp_avg, denom, value=-step_size)

        var_ref.copy_(p.to(dtype1))
        m_ref.copy_(exp_avg.to(dtype1))
        v_ref.copy_(exp_avg_sq.to(dtype1))
        if amsgrad:
            max_grad_norm_ref.copy_(max_exp.to(dtype1))
            return [var_ref, m_ref, v_ref, max_grad_norm_ref]
        return [var_ref, m_ref, v_ref]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_14_ApplyAdamWV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        var_ref_info = inputs[0]
        m_ref_info = inputs[1]
        v_ref_info = inputs[2]
        grad_info = inputs[3]
        step_info = inputs[4]
        max_grad_norm_ref_info = inputs[5]
        lr_info = inputs[6]
        beta1_info = inputs[7]
        beta2_info = inputs[8]
        weight_decay_info = inputs[9]
        eps_info = inputs[10]
        amsgrad_info = inputs[11]
        maximize_info = inputs[12]

        if "data" in var_ref_info:
            var_ref = torch.tensor(var_ref_info["data"], dtype=DTYPE_MAP[var_ref_info["dtype"]]).reshape(var_ref_info["shape"])
        else:
            var_ref = torch.rand(var_ref_info["shape"], dtype=DTYPE_MAP[var_ref_info["dtype"]])
        if "data" in m_ref_info:
            m_ref = torch.tensor(m_ref_info["data"], dtype=DTYPE_MAP[m_ref_info["dtype"]]).reshape(m_ref_info["shape"])
        else:
            m_ref = torch.rand(m_ref_info["shape"], dtype=DTYPE_MAP[m_ref_info["dtype"]])
        if "data" in v_ref_info:
            v_ref = torch.tensor(v_ref_info["data"], dtype=DTYPE_MAP[v_ref_info["dtype"]]).reshape(v_ref_info["shape"])
        else:
            v_ref = torch.rand(v_ref_info["shape"], dtype=DTYPE_MAP[v_ref_info["dtype"]])
        if "data" in grad_info:
            grad = torch.tensor(grad_info["data"], dtype=DTYPE_MAP[grad_info["dtype"]]).reshape(grad_info["shape"])
        else:
            grad = torch.rand(grad_info["shape"], dtype=DTYPE_MAP[grad_info["dtype"]])
        if "data" in step_info:
            step = torch.tensor(step_info["data"], dtype=DTYPE_MAP[step_info["dtype"]]).reshape(step_info["shape"])
        else:
            step = torch.arange(step_info["range"][0], step_info["range"][0] + step_info["shape"][0], dtype=DTYPE_MAP[step_info["dtype"]]).reshape(step_info["shape"])
        if "data" in max_grad_norm_ref_info:
            max_grad_norm_ref = torch.tensor(max_grad_norm_ref_info["data"], dtype=DTYPE_MAP[max_grad_norm_ref_info["dtype"]]).reshape(max_grad_norm_ref_info["shape"])
        else:
            max_grad_norm_ref = torch.rand(max_grad_norm_ref_info["shape"], dtype=DTYPE_MAP[max_grad_norm_ref_info["dtype"]])
        lr = lr_info["value"]
        beta1 = beta1_info["value"]
        beta2 = beta2_info["value"]
        weight_decay = weight_decay_info["value"]
        eps = eps_info["value"]
        amsgrad = amsgrad_info["value"]
        maximize = maximize_info["value"]

        input_groups.append([var_ref, m_ref, v_ref, grad, step, max_grad_norm_ref, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize])
    return input_groups


def get_init_inputs():
    return []
