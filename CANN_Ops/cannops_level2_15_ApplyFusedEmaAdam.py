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
import math

class Model(nn.Module):
    """
    实现FusedEmaAdam融合优化器功能的模型。
    """

    def __init__(self, lr=0.001, emaDecay=0.999, beta1=0.9, beta2=0.999, eps=1e-08, mode=0, biasCorrection=True, weightDecay=0.0):
        """
        初始化模型。
        """
        super(Model, self).__init__()
        self.lr = lr
        self.emaDecay = emaDecay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mode = mode
        self.biasCorrection = biasCorrection
        self.weightDecay = weightDecay

    def forward(self, grad: torch.Tensor, varRef: torch.Tensor, mRef: torch.Tensor, vRef: torch.Tensor, sRef: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """
        实现Adam+EMA优化计算逻辑

        参数:
            grad: 梯度
            varRef: 变量引用
            mRef: 一阶动量引用
            vRef: 二阶动量引用
            sRef: EMA平均值引用
            step: 当前步数

        返回:
            更新后的[变量, 一阶动量, 二阶动量, EMA平均值]
        """
        original_dtype = grad.dtype
        need_cast = original_dtype == torch.float16 or original_dtype == torch.bfloat16
        if need_cast:
            grad = grad.to(torch.float32)
            varRef = varRef.to(torch.float32)
            mRef = mRef.to(torch.float32)
            vRef = vRef.to(torch.float32)
            sRef = sRef.to(torch.float32)
        if self.biasCorrection:
            beta1_correction = 1.0 - self.beta1 ** step
            beta2_correction = 1.0 - self.beta2 ** step
        else:
            beta1_correction = 1.0
            beta2_correction = 1.0
        if self.mode == 0:
            grad_ = grad + self.weightDecay * varRef
        elif self.mode == 1:
            grad_ = grad
        m_ = self.beta1 * mRef + (1 - self.beta1) * grad_
        v_ = self.beta2 * vRef + (1 - self.beta2) * grad_ * grad_
        next_m = m_ / beta1_correction
        next_v = v_ / beta2_correction
        denom = torch.sqrt(next_v) + self.eps
        if self.mode == 0:
            update = next_m / denom
        elif self.mode == 1:
            update = next_m / denom + self.weightDecay * varRef
        var_ = varRef - self.lr * update
        s_ = self.emaDecay * sRef + (1 - self.emaDecay) * var_
        if need_cast:
            var_ = var_.to(original_dtype)
            m_ = m_.to(original_dtype)
            v_ = v_.to(original_dtype)
            s_ = s_.to(original_dtype)
        return [var_, m_, v_, s_]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_15_ApplyFusedEmaAdam.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_info = inputs[0]
        varRef_info = inputs[1]
        mRef_info = inputs[2]
        vRef_info = inputs[3]
        sRef_info = inputs[4]
        step_info = inputs[5]

        if "data" in grad_info:
            grad = torch.tensor(grad_info["data"], dtype=DTYPE_MAP[grad_info["dtype"]]).reshape(grad_info["shape"])
        else:
            grad = torch.randn(grad_info["shape"], dtype=DTYPE_MAP[grad_info["dtype"]])
        if "data" in varRef_info:
            varRef = torch.tensor(varRef_info["data"], dtype=DTYPE_MAP[varRef_info["dtype"]]).reshape(varRef_info["shape"])
        else:
            varRef = torch.randn(varRef_info["shape"], dtype=DTYPE_MAP[varRef_info["dtype"]])
        if "data" in mRef_info:
            mRef = torch.tensor(mRef_info["data"], dtype=DTYPE_MAP[mRef_info["dtype"]]).reshape(mRef_info["shape"])
        else:
            mRef = torch.full(mRef_info["shape"], mRef_info["fill"], dtype=DTYPE_MAP[mRef_info["dtype"]])
        if "data" in vRef_info:
            vRef = torch.tensor(vRef_info["data"], dtype=DTYPE_MAP[vRef_info["dtype"]]).reshape(vRef_info["shape"])
        else:
            vRef = torch.full(vRef_info["shape"], vRef_info["fill"], dtype=DTYPE_MAP[vRef_info["dtype"]])
        if "data" in sRef_info:
            sRef = torch.tensor(sRef_info["data"], dtype=DTYPE_MAP[sRef_info["dtype"]]).reshape(sRef_info["shape"])
        else:
            sRef = torch.full(sRef_info["shape"], sRef_info["fill"], dtype=DTYPE_MAP[sRef_info["dtype"]])
        if "data" in step_info:
            step = torch.tensor(step_info["data"], dtype=DTYPE_MAP[step_info["dtype"]]).reshape(step_info["shape"])
        else:
            step = torch.arange(step_info["range"][0], step_info["range"][0] + step_info["shape"][0], dtype=DTYPE_MAP[step_info["dtype"]]).reshape(step_info["shape"])

        input_groups.append([grad, varRef, mRef, vRef, sRef, step])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_15_ApplyFusedEmaAdam.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        lr_info = entries[0]
        emaDecay_info = entries[1]
        beta1_info = entries[2]
        beta2_info = entries[3]
        eps_info = entries[4]
        mode_info = entries[5]
        biasCorrection_info = entries[6]
        weightDecay_info = entries[7]
        lr = lr_info["value"]
        emaDecay = emaDecay_info["value"]
        beta1 = beta1_info["value"]
        beta2 = beta2_info["value"]
        eps = eps_info["value"]
        mode = mode_info["value"]
        biasCorrection = biasCorrection_info["value"]
        weightDecay = weightDecay_info["value"]
        init_groups.append([lr, emaDecay, beta1, beta2, eps, mode, biasCorrection, weightDecay])
    return init_groups
