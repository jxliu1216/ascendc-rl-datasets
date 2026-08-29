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
import numpy as np
import torch.nn as nn

class Model(nn.Module):
    """
    CPU golden model for MoeComputeExpertTokens operator.
    
    Function: Calculate total token count before each expert based on sorted expert indices.
    """

    def __init__(self):
        """
        Initialize model.
        """
        super(Model, self).__init__()

    def forward(self, sorted_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
        """
        CPU golden implementation: Calculate total token count before each expert.
        
        Args:
            sorted_experts: Sorted expert indices, shape [num_tokens], dtype int32.
            num_experts: Number of experts.
        
        Returns:
            total_rows_before_expert: Total token count before each expert, 
                                      shape [num_experts], dtype int32.
        """
        sorted_experts_np = sorted_experts.cpu().numpy()
        num_experts = int(num_experts)
        arr_length = sorted_experts_np.shape[-1]
        res = np.arange(num_experts)
        for i in range(num_experts):
            target = i
            low = 0
            high = arr_length - 1
            target_location = -1
            while low <= high:
                mid = (low + high) // 2
                if sorted_experts_np[mid] > target:
                    high = mid - 1
                else:
                    low = mid + 1
                    target_location = mid
            res[i] = target_location + 1
        res = res.astype(np.int32)
        return torch.from_numpy(res)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_17_MoeComputeExpertTokens.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        sorted_experts_info = inputs[0]
        num_experts_info = inputs[1]

        if "data" in sorted_experts_info:
            sorted_experts = torch.tensor(sorted_experts_info["data"], dtype=DTYPE_MAP[sorted_experts_info["dtype"]]).reshape(sorted_experts_info["shape"])
        else:
            sorted_experts = torch.randint(sorted_experts_info["range"][0], sorted_experts_info["range"][1] + 1, tuple(sorted_experts_info["shape"]), dtype=DTYPE_MAP[sorted_experts_info["dtype"]])
        num_experts = num_experts_info["value"]

        input_groups.append([sorted_experts, num_experts])
    return input_groups


def get_init_inputs():
    return []
