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
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var_ref: torch.Tensor, input_value: torch.Tensor, begin: torch.Tensor, end: torch.Tensor, strides: torch.Tensor, axes_optional: torch.Tensor) -> torch.Tensor:
        """
        Performs a strided slice assignment on var_ref, with all slice parameters
        (begin, end, strides, axes_optional) passed as explicit torch.Tensor inputs.

        Args:
            var_ref (torch.Tensor): The reference tensor to be modified.
            input_value (torch.Tensor): The tensor whose values will be assigned.
            begin (torch.Tensor): Tensor containing start indices for slicing (int64, 1D).
            end (torch.Tensor): Tensor containing end indices for slicing (int64, 1D).
            strides (torch.Tensor): Tensor containing step sizes for slicing (int64, 1D).
            axes_optional (torch.Tensor): Optional tensor specifying the axes along which
                                        to slice (int64, 1D). If empty, slicing occurs
                                        along sequential dimensions.

        Returns:
            torch.Tensor: The modified var_ref tensor.
        """
        output_var_ref = var_ref.clone()
        begin_list = begin.tolist()
        end_list = end.tolist()
        strides_list = strides.tolist()
        axes_list = axes_optional.tolist()
        num_dims = output_var_ref.dim()
        slices: List[slice] = [slice(None)] * num_dims
        if not axes_list:
            for i in range(len(begin_list)):
                if i < num_dims:
                    s_begin = begin_list[i] if i < len(begin_list) else 0
                    s_end = end_list[i] if i < len(end_list) else output_var_ref.shape[i]
                    s_stride = strides_list[i] if i < len(strides_list) else 1
                    slices[i] = slice(s_begin, s_end, s_stride)
        else:
            for i, axis_idx in enumerate(axes_list):
                if axis_idx >= 0 and axis_idx < num_dims:
                    s_begin = begin_list[i] if i < len(begin_list) else 0
                    s_end = end_list[i] if i < len(end_list) else output_var_ref.shape[axis_idx]
                    s_stride = strides_list[i] if i < len(strides_list) else 1
                    slices[axis_idx] = slice(s_begin, s_end, s_stride)
                else:
                    raise IndexError(f'Axis index {axis_idx} out of bounds for tensor with {num_dims} dimensions.')
        output_var_ref[tuple(slices)] = input_value
        return output_var_ref

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_143_StridedSliceAssignV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        var_ref_info = inputs[0]
        input_value_info = inputs[1]
        begin_info = inputs[2]
        end_info = inputs[3]
        strides_info = inputs[4]
        axes_optional_info = inputs[5]

        if "data" in var_ref_info:
            var_ref = torch.tensor(var_ref_info["data"], dtype=DTYPE_MAP[var_ref_info["dtype"]]).reshape(var_ref_info["shape"])
        else:
            var_ref = torch.rand(var_ref_info["shape"], dtype=DTYPE_MAP[var_ref_info["dtype"]])
        if "data" in input_value_info:
            input_value = torch.tensor(input_value_info["data"], dtype=DTYPE_MAP[input_value_info["dtype"]]).reshape(input_value_info["shape"])
        else:
            input_value = torch.rand(input_value_info["shape"], dtype=DTYPE_MAP[input_value_info["dtype"]])
        if "data" in begin_info:
            begin = torch.tensor(begin_info["data"], dtype=DTYPE_MAP[begin_info["dtype"]]).reshape(begin_info["shape"])
        else:
            begin = torch.randint(begin_info["range"][0], begin_info["range"][1] + 1, tuple(begin_info["shape"]), dtype=DTYPE_MAP[begin_info["dtype"]])
        if "data" in end_info:
            end = torch.tensor(end_info["data"], dtype=DTYPE_MAP[end_info["dtype"]]).reshape(end_info["shape"])
        else:
            end = torch.randint(end_info["range"][0], end_info["range"][1] + 1, tuple(end_info["shape"]), dtype=DTYPE_MAP[end_info["dtype"]])
        if "data" in strides_info:
            strides = torch.tensor(strides_info["data"], dtype=DTYPE_MAP[strides_info["dtype"]]).reshape(strides_info["shape"])
        else:
            strides = torch.randint(strides_info["range"][0], strides_info["range"][1] + 1, tuple(strides_info["shape"]), dtype=DTYPE_MAP[strides_info["dtype"]])
        if "data" in axes_optional_info:
            axes_optional = torch.tensor(axes_optional_info["data"], dtype=DTYPE_MAP[axes_optional_info["dtype"]]).reshape(axes_optional_info["shape"])
        else:
            axes_optional = torch.arange(axes_optional_info["range"][0], axes_optional_info["range"][0] + axes_optional_info["shape"][0], dtype=DTYPE_MAP[axes_optional_info["dtype"]]).reshape(axes_optional_info["shape"])

        input_groups.append([var_ref, input_value, begin, end, strides, axes_optional])
    return input_groups


def get_init_inputs():
    return []
