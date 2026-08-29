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

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    实现EmbeddingBag算子功能的模型（PyTorch标杆）。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, weight: torch.Tensor, indices: torch.Tensor, offsets: torch.Tensor, per_sample_weights: Optional[torch.Tensor], mode: str, scale_grad_by_freq: bool, sparse: bool, include_last_offset: bool, padding_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        实现EmbeddingBag算子功能。

        Args:
            weight: 嵌入权重矩阵 (num_embeddings, embedding_dim)
            indices: 索引张量
            offsets: 偏移量张量
            per_sample_weights: 可选的每样本权重
            mode: 聚合模式 ('sum', 'mean', 'max')
            scale_grad_by_freq: 是否按频率缩放梯度
            sparse: 是否使用稀疏梯度
            include_last_offset: offsets是否包含最后一个偏移
            padding_idx: 填充索引

        Returns:
            (y, offset2bag, bag_size, max_indices)
        """
        num_bags = offsets.size(0)
        if include_last_offset:
            num_bags -= 1
        y, offset2bag, bag_size, max_indices = torch.ops.aten._embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, 0 if mode == 'sum' else 1 if mode == 'mean' else 2, sparse, per_sample_weights, include_last_offset, padding_idx)
        return (y, offset2bag, bag_size, max_indices)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_17_EmbeddingBag.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        weight_info = inputs[0]
        indices_info = inputs[1]
        offsets_info = inputs[2]
        per_sample_weights_info = inputs[3]
        mode_info = inputs[4]
        scale_grad_by_freq_info = inputs[5]
        sparse_info = inputs[6]
        include_last_offset_info = inputs[7]
        padding_idx_info = inputs[8]

        if "data" in weight_info:
            weight = torch.tensor(weight_info["data"], dtype=DTYPE_MAP[weight_info["dtype"]]).reshape(weight_info["shape"])
        else:
            weight = torch.randn(weight_info["shape"], dtype=DTYPE_MAP[weight_info["dtype"]])
        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])
        if "data" in offsets_info:
            offsets = torch.tensor(offsets_info["data"], dtype=DTYPE_MAP[offsets_info["dtype"]]).reshape(offsets_info["shape"])
        else:
            offsets = torch.randint(offsets_info["range"][0], offsets_info["range"][1] + 1, tuple(offsets_info["shape"]), dtype=DTYPE_MAP[offsets_info["dtype"]])
        per_sample_weights = None
        mode = mode_info["value"]
        scale_grad_by_freq = scale_grad_by_freq_info["value"]
        sparse = sparse_info["value"]
        include_last_offset = include_last_offset_info["value"]
        padding_idx = padding_idx_info["value"]

        input_groups.append([weight, indices, offsets, per_sample_weights, mode, scale_grad_by_freq, sparse, include_last_offset, padding_idx])
    return input_groups


def get_init_inputs():
    return []
