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

from typing import List, Tuple
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    实现EmbeddingDenseGradV2算子功能的模型（PyTorch标杆）。
    """

    def __init__(self, num_weights: int, padding_idx: int=-1, scale_grad_by_freq: bool=False):
        """
        初始化模型。
        
        Args:
            num_weights: 权重数量
            padding_idx: 填充索引
            scale_grad_by_freq: 是否按频率缩放梯度
        """
        super(Model, self).__init__()
        self.num_weights = num_weights
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq

    def forward(self, grad: torch.Tensor, sort_indices: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
        """
        实现EmbeddingDenseGrad功能。

        Args:
            grad: 输入梯度张量
            sort_indices: 排序后的索引
            pos_idx: 位置索引

        Returns:
            输出梯度张量
        """
        batch_shape = list(grad.shape[:-1])
        embedding_dim = grad.shape[-1]
        output = torch.zeros(self.num_weights, embedding_dim, dtype=grad.dtype, device=grad.device)
        grad_flat = grad.reshape(-1, embedding_dim)
        for i in range(len(sort_indices)):
            idx = sort_indices[i].item()
            if idx == self.padding_idx:
                continue
            pos = pos_idx[i].item()
            output[idx] += grad_flat[pos]
        if self.scale_grad_by_freq:
            unique_indices, counts = torch.unique(sort_indices, return_counts=True)
            for idx, count in zip(unique_indices, counts):
                if idx != self.padding_idx:
                    output[idx] = output[idx] / count.float()
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_18_EmbeddingDenseGradV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_info = inputs[0]
        sort_indices_info = inputs[1]
        pos_idx_info = inputs[2]

        if "data" in grad_info:
            grad = torch.tensor(grad_info["data"], dtype=DTYPE_MAP[grad_info["dtype"]]).reshape(grad_info["shape"])
        else:
            grad = torch.randn(grad_info["shape"], dtype=DTYPE_MAP[grad_info["dtype"]])
        if "data" in sort_indices_info:
            sort_indices = torch.tensor(sort_indices_info["data"], dtype=DTYPE_MAP[sort_indices_info["dtype"]]).reshape(sort_indices_info["shape"])
        else:
            sort_indices = torch.randint(sort_indices_info["range"][0], sort_indices_info["range"][1] + 1, tuple(sort_indices_info["shape"]), dtype=DTYPE_MAP[sort_indices_info["dtype"]])
        if "data" in pos_idx_info:
            pos_idx = torch.tensor(pos_idx_info["data"], dtype=DTYPE_MAP[pos_idx_info["dtype"]]).reshape(pos_idx_info["shape"])
        else:
            pos_idx = torch.randperm(pos_idx_info["shape"][0], dtype=DTYPE_MAP[pos_idx_info["dtype"]]) + pos_idx_info["range"][0]

        input_groups.append([grad, sort_indices, pos_idx])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_18_EmbeddingDenseGradV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        num_weights_info = entries[0]
        padding_idx_info = entries[1]
        scale_grad_by_freq_info = entries[2]
        num_weights = num_weights_info["value"]
        padding_idx = padding_idx_info["value"]
        scale_grad_by_freq = scale_grad_by_freq_info["value"]
        init_groups.append([num_weights, padding_idx, scale_grad_by_freq])
    return init_groups
