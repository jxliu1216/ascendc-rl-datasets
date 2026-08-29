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
    实现 LinearIndex 算子功能的标杆模型。
    """

    def __init__(self, axis=-1, combine=False):
        super(Model, self).__init__()
        self.axis = axis
        self.combine = combine

    def forward(self, indices: torch.Tensor, shape_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: 输入索引张量
            shape_tensor: 表示维度的 1D Int32 张量
        Returns:
            计算后的 int32 索引张量
        """
        shape_list = shape_tensor.tolist()
        rank = len(shape_list)
        axis = self.axis
        if axis < 0:
            axis += rank
        target_dim = shape_list[axis]
        out = torch.where(indices < 0, indices + target_dim, indices)
        if self.combine and rank == 3:
            stride = shape_list[1]
            if indices.dim() >= 2:
                if axis == 0:
                    cols = indices.shape[1]
                    col_indices = torch.arange(cols, device=indices.device, dtype=indices.dtype).unsqueeze(0)
                    col_indices = col_indices.expand_as(indices)
                    out = out * stride + col_indices
                elif axis == 1:
                    rows = indices.shape[0]
                    row_indices = torch.arange(rows, device=indices.device, dtype=indices.dtype).unsqueeze(1)
                    row_indices = row_indices.expand_as(indices)
                    out = out + stride * row_indices
        return out.to(torch.int32)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_41_LinearIndex.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        indices_info = inputs[0]
        shape_tensor_info = inputs[1]

        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])
        if "data" in shape_tensor_info:
            shape_tensor = torch.tensor(shape_tensor_info["data"], dtype=DTYPE_MAP[shape_tensor_info["dtype"]]).reshape(shape_tensor_info["shape"])
        else:
            shape_tensor = torch.randint(shape_tensor_info["range"][0], shape_tensor_info["range"][1] + 1, tuple(shape_tensor_info["shape"]), dtype=DTYPE_MAP[shape_tensor_info["dtype"]])

        input_groups.append([indices, shape_tensor])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_41_LinearIndex.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        axis_info = entries[0]
        combine_info = entries[1]
        axis = axis_info["value"]
        combine = combine_info["value"]
        init_groups.append([axis, combine])
    return init_groups
