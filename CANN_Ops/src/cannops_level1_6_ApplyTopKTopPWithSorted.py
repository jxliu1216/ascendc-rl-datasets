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
import torch.nn.functional as F

class Model(nn.Module):
    """
    实现ApplyTopKTopPWithSorted算子功能的模型（PyTorch标杆）。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, sorted_value: torch.Tensor, sorted_indices: torch.Tensor, p: Optional[torch.Tensor]=None, k: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        实现ApplyTopKTopPWithSorted算子功能。

        Args:
            sorted_value: 已排序的概率值张量 [batch, vocab]
            sorted_indices: 对应的索引张量 [batch, vocab]
            p: Top-P阈值 [batch] (可选)
            k: Top-K数量 [batch] (可选)

        Returns:
            输出张量，满足top-k/top-p条件的保留原值，其余位置为-inf
        """
        if not k.dim() == 0:
            kth_idx = sorted_value.size(1) - k.long()
            kth_value = sorted_value.gather(1, kth_idx.unsqueeze(dim=1))
            top_k_mask = sorted_value < kth_value
            sorted_value.masked_fill_(top_k_mask, -float('inf'))
        if not p.dim() == 0:
            softmax_res = sorted_value.to(torch.float32).softmax(dim=-1)
            cumsum_res = softmax_res.cumsum(dim=-1)
            top_p_mask = cumsum_res <= 1 - p.unsqueeze(dim=1)
            top_p_mask[:, -1] = False
            sorted_value.masked_fill_(top_p_mask, -float('inf'))
        out = torch.empty_like(sorted_value).scatter_(dim=-1, index=sorted_indices.long(), src=sorted_value)
        return out

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_6_ApplyTopKTopPWithSorted.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        sorted_value_info = inputs[0]
        sorted_indices_info = inputs[1]
        p_info = inputs[2]
        k_info = inputs[3]

        if "data" in sorted_value_info:
            sorted_value = torch.tensor(sorted_value_info["data"], dtype=DTYPE_MAP[sorted_value_info["dtype"]]).reshape(sorted_value_info["shape"])
        else:
            sorted_value = torch.randn(sorted_value_info["shape"], dtype=DTYPE_MAP[sorted_value_info["dtype"]])
        if "data" in sorted_indices_info:
            sorted_indices = torch.tensor(sorted_indices_info["data"], dtype=DTYPE_MAP[sorted_indices_info["dtype"]]).reshape(sorted_indices_info["shape"])
        else:
            sorted_indices = torch.randint(sorted_indices_info["range"][0], sorted_indices_info["range"][1] + 1, tuple(sorted_indices_info["shape"]), dtype=DTYPE_MAP[sorted_indices_info["dtype"]])
        if "data" in p_info:
            p = torch.tensor(p_info["data"], dtype=DTYPE_MAP[p_info["dtype"]]).reshape(p_info["shape"])
        else:
            p = torch.rand(p_info["shape"], dtype=DTYPE_MAP[p_info["dtype"]])
        if "data" in k_info:
            k = torch.tensor(k_info["data"], dtype=DTYPE_MAP[k_info["dtype"]]).reshape(k_info["shape"])
        else:
            _dt = DTYPE_MAP[k_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                k = torch.randint(k_info["range"][0], k_info["range"][1] + 1, tuple(k_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                k = torch.rand(k_info["shape"]) < k_info.get("true_frac", 0.5)
            else:
                k = torch.rand(k_info["shape"], dtype=_dt) * (k_info["range"][1] - k_info["range"][0]) + k_info["range"][0]

        input_groups.append([sorted_value, sorted_indices, p, k])
    return input_groups


def get_init_inputs():
    return []
