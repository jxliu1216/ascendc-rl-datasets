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

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    实现RepeatInterleaveGrad算子功能的标杆模型。
    对应 torch.repeat_interleave 的反向逻辑。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, y_grad: torch.Tensor, repeats: torch.Tensor, axis: int) -> torch.Tensor:
        """
        Args:
            y_grad: 输入梯度张量，形状为 repeat_interleave 后的形状
            repeats: 重复次数张量 (1D Tensor) 或 标量张量
            axis: 维度

        Returns:
            x_grad: 输出梯度张量，形状为 repeat_interleave 前的形状
        """
        if axis < 0:
            axis += y_grad.ndim
        repeats_cpu = repeats.cpu()
        if repeats_cpu.numel() == 1:
            r = int(repeats_cpu.item())
            if r == 0:
                return torch.zeros_like(y_grad)
            return y_grad.unfold(axis, r, r).sum(dim=-1)
        else:
            split_sizes = repeats_cpu.tolist()
            chunks = torch.split(y_grad, split_sizes, dim=axis)
            sums = [chunk.sum(dim=axis, keepdim=True) for chunk in chunks]
            return torch.cat(sums, dim=axis)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_54_RepeatInterleaveGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        y_grad_info = inputs[0]
        repeats_info = inputs[1]
        axis_info = inputs[2]

        if "data" in y_grad_info:
            y_grad = torch.tensor(y_grad_info["data"], dtype=DTYPE_MAP[y_grad_info["dtype"]]).reshape(y_grad_info["shape"])
        else:
            y_grad = torch.randn(y_grad_info["shape"], dtype=DTYPE_MAP[y_grad_info["dtype"]])
        if "data" in repeats_info:
            repeats = torch.tensor(repeats_info["data"], dtype=DTYPE_MAP[repeats_info["dtype"]]).reshape(repeats_info["shape"])
        else:
            repeats = torch.randint(repeats_info["range"][0], repeats_info["range"][1] + 1, tuple(repeats_info["shape"]), dtype=DTYPE_MAP[repeats_info["dtype"]])
        axis = axis_info["value"]

        input_groups.append([y_grad, repeats, axis])
    return input_groups


def get_init_inputs():
    return []
