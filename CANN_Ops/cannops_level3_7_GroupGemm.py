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
    实现add算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, m_list: list[int], k_list: list[int], n_list: list[int]) -> torch.Tensor:
        """
        实现add算子功能。

        Args:
            a: 第一个输入张量
            b: 第二个输入张量

        Returns:
            两个输入张量的和
        """
        results = []
        offset_a = 0
        offset_b = 0
        offset_c = 0
        for ind, (m, k, n) in enumerate(zip(m_list, k_list, n_list)):
            size_a = m * k
            size_b = k * n
            size_c = m * n
            group_a_flat = a[offset_a:offset_a + size_a]
            group_b_flat = b[offset_b:offset_b + size_b]
            group_c_flat = c[offset_c:offset_c + size_c]
            group_a = group_a_flat.view(m, k)
            group_b = group_b_flat.view(k, n)
            group_c = group_c_flat.view(m, n)
            result = (torch.matmul(alpha[ind] * group_a.to(torch.float32), group_b.to(torch.float32)) + beta[ind] * group_c.to(torch.float32)).flatten()
            results.append(result)
            offset_a += size_a
            offset_b += size_b
            offset_c += size_c
        return torch.cat(results)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_7_GroupGemm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        a_info = inputs[0]
        b_info = inputs[1]
        c_info = inputs[2]
        alpha_info = inputs[3]
        beta_info = inputs[4]
        m_list_info = inputs[5]
        k_list_info = inputs[6]
        n_list_info = inputs[7]

        if "data" in a_info:
            a = torch.tensor(a_info["data"], dtype=DTYPE_MAP[a_info["dtype"]]).reshape(a_info["shape"])
        else:
            a = torch.rand(a_info["shape"], dtype=DTYPE_MAP[a_info["dtype"]]) * (a_info["range"][1] - a_info["range"][0]) + a_info["range"][0]
        if "data" in b_info:
            b = torch.tensor(b_info["data"], dtype=DTYPE_MAP[b_info["dtype"]]).reshape(b_info["shape"])
        else:
            b = torch.rand(b_info["shape"], dtype=DTYPE_MAP[b_info["dtype"]]) * (b_info["range"][1] - b_info["range"][0]) + b_info["range"][0]
        if "data" in c_info:
            c = torch.tensor(c_info["data"], dtype=DTYPE_MAP[c_info["dtype"]]).reshape(c_info["shape"])
        else:
            c = torch.rand(c_info["shape"], dtype=DTYPE_MAP[c_info["dtype"]]) * (c_info["range"][1] - c_info["range"][0]) + c_info["range"][0]
        if "data" in alpha_info:
            alpha = torch.tensor(alpha_info["data"], dtype=DTYPE_MAP[alpha_info["dtype"]]).reshape(alpha_info["shape"])
        else:
            alpha = torch.rand(alpha_info["shape"], dtype=DTYPE_MAP[alpha_info["dtype"]]) * (alpha_info["range"][1] - alpha_info["range"][0]) + alpha_info["range"][0]
        if "data" in beta_info:
            beta = torch.tensor(beta_info["data"], dtype=DTYPE_MAP[beta_info["dtype"]]).reshape(beta_info["shape"])
        else:
            beta = torch.rand(beta_info["shape"], dtype=DTYPE_MAP[beta_info["dtype"]]) * (beta_info["range"][1] - beta_info["range"][0]) + beta_info["range"][0]
        m_list = m_list_info["value"]
        k_list = k_list_info["value"]
        n_list = n_list_info["value"]

        input_groups.append([a, b, c, alpha, beta, m_list, k_list, n_list])
    return input_groups


def get_init_inputs():
    return []
