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
    """
    实现ComplexMatDot算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, matx: torch.Tensor, maty: torch.Tensor, m: int, n: int) -> torch.Tensor:
        """
        实现ComplexMatDot算子功能。

        Args:
            matx: 第一个输入复数矩阵
            maty: 第二个输入复数矩阵
            m: 矩阵行数
            n: 矩阵列数

        Returns:
            两个输入复数矩阵逐点乘后的结果矩阵
        """
        return matx * maty

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_13_ComplexMatDot.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        matx_info = inputs[0]
        maty_info = inputs[1]
        m_info = inputs[2]
        n_info = inputs[3]

        if "data" in matx_info:
            matx = torch.tensor(matx_info["data"], dtype=DTYPE_MAP[matx_info["dtype"]]).reshape(matx_info["shape"])
        else:
            matx_re = torch.rand(matx_info["shape"], dtype=torch.float32)
            matx_im = torch.rand(matx_info["shape"], dtype=torch.float32)
            matx = torch.complex(matx_re, matx_im).to(DTYPE_MAP[matx_info["dtype"]])
        if "data" in maty_info:
            maty = torch.tensor(maty_info["data"], dtype=DTYPE_MAP[maty_info["dtype"]]).reshape(maty_info["shape"])
        else:
            maty_re = torch.rand(maty_info["shape"], dtype=torch.float32)
            maty_im = torch.rand(maty_info["shape"], dtype=torch.float32)
            maty = torch.complex(maty_re, maty_im).to(DTYPE_MAP[maty_info["dtype"]])
        m = m_info["value"]
        n = n_info["value"]

        input_groups.append([matx, maty, m, n])
    return input_groups


def get_init_inputs():
    return []
