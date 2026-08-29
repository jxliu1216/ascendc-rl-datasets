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
from torch import Tensor
from typing import List, Tuple, Optional
from typing import Dict, Any
import numpy as np

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor: Tensor, pdf_ref: Tensor, statistic: bool=True, reshuff: bool=False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        使用torch实现压缩做对比标杆。
        
        参数:
            input_tensor (Tensor): shape支持多维，要求数据数量是32768的倍数，dtype=float16/bfloat16/f32
            pdf_ref (Tensor): shape [1, 256]，dtype=int32
            statistic (bool): 是否进行 PDF 统计，默认为True
            reshuff (bool): 是否对各核编码后的结果进行内存重整
            
        返回:
            pdf_out（Tensor）: [1, 256], int32 指数位统计结果
            mantissa_out（Tensor）: 表示输出的尾数部分，dtype与input一致
            fixed_out（Tensor）: 表示压缩的第一段输出，dtype与input一致
            var_out（Tensor）: 表示压缩超过fixedOut后的输出，dtype与input一致
        """
        input_cpu = input_tensor.cpu().contiguous()
        dtype = input_cpu.dtype
        if dtype == torch.float32:
            exp_bytes = input_cpu.view(torch.uint8).view(-1, 4)[:, 3]
        elif dtype in [torch.float16, torch.bfloat16]:
            exp_bytes = input_cpu.view(torch.uint8).view(-1, 2)[:, 1]
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')
        pdf_out = torch.bincount(exp_bytes, minlength=256).to(torch.int32)
        pdf_out_tensor = pdf_out.to(input_tensor.device)
        return pdf_out_tensor

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_27_HansEncode.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_tensor_info = inputs[0]
        pdf_ref_info = inputs[1]
        statistic_info = inputs[2]
        reshuff_info = inputs[3]

        if "data" in input_tensor_info:
            input_tensor = torch.tensor(input_tensor_info["data"], dtype=DTYPE_MAP[input_tensor_info["dtype"]]).reshape(input_tensor_info["shape"])
        else:
            input_tensor = torch.randn(input_tensor_info["shape"], dtype=DTYPE_MAP[input_tensor_info["dtype"]])
        if "data" in pdf_ref_info:
            pdf_ref = torch.tensor(pdf_ref_info["data"], dtype=DTYPE_MAP[pdf_ref_info["dtype"]]).reshape(pdf_ref_info["shape"])
        else:
            pdf_ref = torch.full(pdf_ref_info["shape"], pdf_ref_info["fill"], dtype=DTYPE_MAP[pdf_ref_info["dtype"]])
        statistic = statistic_info["value"]
        reshuff = reshuff_info["value"]

        input_groups.append([input_tensor, pdf_ref, statistic, reshuff])
    return input_groups


def get_init_inputs():
    return []
