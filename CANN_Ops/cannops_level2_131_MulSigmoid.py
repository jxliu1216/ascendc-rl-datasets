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
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    实现MulSigmoid算子功能的动态形状处理模型
    计算公式：
    1. tmp = sigmoid(x1 * t1)
    2. sel = where(tmp < t2, tmp, 2*tmp)
    3. res = sel * x2 * t3
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, t1: float, t2: float, t3: float) -> torch.Tensor:
        tmp = torch.sigmoid(x1 * t1)
        sel = torch.where(tmp < t2, tmp, 2 * tmp)
        x2_flat_dim = torch.prod(torch.tensor(x2.shape[1:])).item()
        x2_reshaped = x2.reshape(1, x2_flat_dim)
        sel_reshaped = sel.reshape(-1, x2_flat_dim)
        res = sel_reshaped * x2_reshaped * t3
        output_shape = (x1.shape[0],) + x2.shape[1:]
        res = res.reshape(output_shape)
        return res

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_131_MulSigmoid.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]
        t1_info = inputs[2]
        t2_info = inputs[3]
        t3_info = inputs[4]

        if "data" in x1_info:
            x1 = torch.tensor(x1_info["data"], dtype=DTYPE_MAP[x1_info["dtype"]]).reshape(x1_info["shape"])
        else:
            x1 = torch.rand(x1_info["shape"], dtype=DTYPE_MAP[x1_info["dtype"]])
        if "data" in x2_info:
            x2 = torch.tensor(x2_info["data"], dtype=DTYPE_MAP[x2_info["dtype"]]).reshape(x2_info["shape"])
        else:
            x2 = torch.rand(x2_info["shape"], dtype=DTYPE_MAP[x2_info["dtype"]])
        t1 = t1_info["value"]
        t2 = t2_info["value"]
        t3 = t3_info["value"]

        input_groups.append([x1, x2, t1, t2, t3])
    return input_groups


def get_init_inputs():
    return []
