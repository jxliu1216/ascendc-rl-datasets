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
    实现ScatterList算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, varRef: list[torch.Tensor], indice: torch.Tensor, updates: torch.Tensor, mask: Optional[torch.Tensor], reduce: str, axis: int) -> torch.Tensor:
        """
        实现ScatterList算子功能。

        Args:
            varRef: 第一个输入张量
            indice: 索引张量
            updates: 更新张量
            mask: 可选的掩码张量
            reduce: 规约操作类型
            axis: 指定的轴

        Returns:
            经过ScatterList操作后的结果张量
        """
        for i in range(len(varRef)):
            if mask[i] == False:
                continue
            dest_block_slice = slice(indice[i][0], indice[i][0] + indice[i][1])
            source_block_slice = slice(0, indice[i][1])
            num_dims = varRef[i].ndim
            dest_slicer = [slice(None)] * num_dims
            src_slicer = [slice(None)] * num_dims
            dest_slicer[axis] = dest_block_slice
            src_slicer[axis] = source_block_slice
            dest_slicer = tuple(dest_slicer)
            src_slicer = tuple(src_slicer)
            source_block = updates[i][src_slicer]
            varRef[i][dest_slicer] = source_block
        return varRef

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_60_ScatterList.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        varRef_info = inputs[0]
        indice_info = inputs[1]
        updates_info = inputs[2]
        mask_info = inputs[3]
        reduce_info = inputs[4]
        axis_info = inputs[5]

        varRef = []
        for _shape in varRef_info["shapes"]:
            _t = torch.randn({"dtype": varRef_info["dtype"], "shape": _shape, "range": varRef_info.get("range", [0, 1]), "mean": varRef_info.get("mean", 0.0), "std": varRef_info.get("std", 1.0), "value": varRef_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": varRef_info["dtype"], "shape": _shape, "range": varRef_info.get("range", [0, 1]), "mean": varRef_info.get("mean", 0.0), "std": varRef_info.get("std", 1.0), "value": varRef_info.get("value")}["dtype"]])
            varRef.append(_t)
        if "data" in indice_info:
            indice = torch.tensor(indice_info["data"], dtype=DTYPE_MAP[indice_info["dtype"]]).reshape(indice_info["shape"])
        else:
            indice = torch.randint(indice_info["range"][0], indice_info["range"][1] + 1, tuple(indice_info["shape"]), dtype=DTYPE_MAP[indice_info["dtype"]])
        if "data" in updates_info:
            updates = torch.tensor(updates_info["data"], dtype=DTYPE_MAP[updates_info["dtype"]]).reshape(updates_info["shape"])
        else:
            updates = torch.randn(updates_info["shape"], dtype=DTYPE_MAP[updates_info["dtype"]])
        if "data" in mask_info:
            mask = torch.tensor(mask_info["data"], dtype=DTYPE_MAP[mask_info["dtype"]]).reshape(mask_info["shape"])
        else:
            mask = torch.rand(mask_info["shape"]) > 0.5
        reduce = reduce_info["value"]
        axis = axis_info["value"]

        input_groups.append([varRef, indice, updates, mask, reduce, axis])
    return input_groups


def get_init_inputs():
    return []
