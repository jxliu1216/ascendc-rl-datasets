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

class Model(nn.Module):
    """
    实现InplaceIndexAddWithSorted算子功能的模型(torch标杆)。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, value: torch.Tensor, sorted_indices: torch.Tensor, pos: torch.Tensor, axis: int, alpha: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        实现InplaceIndexAddWithSorted算子功能。

        Args:
            var: 待更新的张量
            value: 更新值张量
            sorted_indices: 已排序的索引张量
            pos: 位置索引张量
            axis: 操作的维度
            alpha: 可选的缩放因子

        Returns:
            更新后的var张量
        """
        result = var.clone()
        result_dtype = result.dtype
        if result_dtype == torch.bfloat16:
            result = result.float()
        alpha_value = alpha.item() if alpha is not None else 1.0
        for i in range(len(sorted_indices)):
            idx = sorted_indices[i].item()
            p = pos[i].item()
            if axis == 0:
                result[idx] = result[idx] + alpha_value * value[p]
            else:
                raise NotImplementedError('Only axis=0 is supported')
        if result_dtype == torch.bfloat16:
            result = result.to(torch.bfloat16)
        return result

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_31_InplaceIndexAddWithSorted.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        var_info = inputs[0]
        value_info = inputs[1]
        sorted_indices_info = inputs[2]
        pos_info = inputs[3]
        axis_info = inputs[4]
        alpha_info = inputs[5]

        if "data" in var_info:
            var = torch.tensor(var_info["data"], dtype=DTYPE_MAP[var_info["dtype"]]).reshape(var_info["shape"])
        else:
            _dt = DTYPE_MAP[var_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                var = torch.randint(var_info["range"][0], var_info["range"][1] + 1, tuple(var_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                var = torch.rand(var_info["shape"]) > 0.5
            else:
                var = torch.randn(var_info["shape"], dtype=_dt)
        if "data" in value_info:
            value = torch.tensor(value_info["data"], dtype=DTYPE_MAP[value_info["dtype"]]).reshape(value_info["shape"])
        else:
            _dt = DTYPE_MAP[value_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                value = torch.randint(value_info["range"][0], value_info["range"][1] + 1, tuple(value_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                value = torch.rand(value_info["shape"]) > 0.5
            else:
                value = torch.randn(value_info["shape"], dtype=_dt)
        if "data" in sorted_indices_info:
            sorted_indices = torch.tensor(sorted_indices_info["data"], dtype=DTYPE_MAP[sorted_indices_info["dtype"]]).reshape(sorted_indices_info["shape"])
        else:
            sorted_indices = torch.randint(sorted_indices_info["range"][0], sorted_indices_info["range"][1] + 1, tuple(sorted_indices_info["shape"]), dtype=DTYPE_MAP[sorted_indices_info["dtype"]])
        if "data" in pos_info:
            pos = torch.tensor(pos_info["data"], dtype=DTYPE_MAP[pos_info["dtype"]]).reshape(pos_info["shape"])
        else:
            pos = torch.randint(pos_info["range"][0], pos_info["range"][1] + 1, tuple(pos_info["shape"]), dtype=DTYPE_MAP[pos_info["dtype"]])
        axis = axis_info["value"]
        if alpha_info["type"] == "attr":
            if alpha_info.get("dtype") == "none":
                alpha = None
            else:
                alpha = alpha_info["value"]
        else:
            if "data" in alpha_info:
                alpha = torch.tensor(alpha_info["data"], dtype=DTYPE_MAP[alpha_info["dtype"]]).reshape(alpha_info["shape"])
            else:
                alpha = torch.full(alpha_info["shape"], alpha_info["fill"], dtype=DTYPE_MAP[alpha_info["dtype"]])

        input_groups.append([var, value, sorted_indices, pos, axis, alpha])
    return input_groups


def get_init_inputs():
    return []
