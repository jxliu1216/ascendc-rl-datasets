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

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, xyz1, xyz2, idx1, idx2, gradDist1, gradDist2):
        """
        Args:
            xyz1: [B, N, 2] - 点集1坐标
            xyz2: [B, N, 2] - 点集2坐标
            idx1: [B, N] - 每个xyz1点对应xyz2中的最近点索引（int32）
            idx2: [B, N] - 每个xyz2点对应xyz1中的最近点索引（int32）
            gradDist1: [B, N] - 从xyz1到最近xyz2点的距离梯度
            gradDist2: [B, N] - 从xyz2到最近xyz1点的距离梯度
        Returns:
            gradXyz1: [B, N, 2] - xyz1的梯度
            gradXyz2: [B, N, 2] - xyz2的梯度
        """
        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape
        gradXyz1 = torch.zeros_like(xyz1)
        gradXyz2 = torch.zeros_like(xyz2)
        for b in range(B):
            for n in range(N):
                x1, y1 = (xyz1[b, n, 0].item(), xyz1[b, n, 1].item())
                idx = idx1[b, n].item()
                x2, y2 = (xyz2[b, idx, 0].item(), xyz2[b, idx, 1].item())
                g = gradDist1[b, n].item() * 2.0
                gradXyz1[b, n, 0] += (x1 - x2) * g
                gradXyz1[b, n, 1] += (y1 - y2) * g
                gradXyz2[b, idx, 0] -= (x1 - x2) * g
                gradXyz2[b, idx, 1] -= (y1 - y2) * g
            for m in range(M):
                x2, y2 = (xyz2[b, m, 0].item(), xyz2[b, m, 1].item())
                idx = idx2[b, m].item()
                x1, y1 = (xyz1[b, idx, 0].item(), xyz1[b, idx, 1].item())
                g = gradDist2[b, m].item() * 2.0
                gradXyz2[b, m, 0] += (x2 - x1) * g
                gradXyz2[b, m, 1] += (y2 - y1) * g
                gradXyz1[b, idx, 0] -= (x2 - x1) * g
                gradXyz1[b, idx, 1] -= (y2 - y1) * g
        return [gradXyz1, gradXyz2]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_22_ChamferDistanceGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        xyz1_info = inputs[0]
        xyz2_info = inputs[1]
        idx1_info = inputs[2]
        idx2_info = inputs[3]
        gradDist1_info = inputs[4]
        gradDist2_info = inputs[5]

        if "data" in xyz1_info:
            xyz1 = torch.tensor(xyz1_info["data"], dtype=DTYPE_MAP[xyz1_info["dtype"]]).reshape(xyz1_info["shape"])
        else:
            xyz1 = torch.rand(xyz1_info["shape"], dtype=DTYPE_MAP[xyz1_info["dtype"]])
        if "data" in xyz2_info:
            xyz2 = torch.tensor(xyz2_info["data"], dtype=DTYPE_MAP[xyz2_info["dtype"]]).reshape(xyz2_info["shape"])
        else:
            xyz2 = torch.rand(xyz2_info["shape"], dtype=DTYPE_MAP[xyz2_info["dtype"]])
        if "data" in idx1_info:
            idx1 = torch.tensor(idx1_info["data"], dtype=DTYPE_MAP[idx1_info["dtype"]]).reshape(idx1_info["shape"])
        else:
            idx1 = torch.randint(idx1_info["range"][0], idx1_info["range"][1] + 1, tuple(idx1_info["shape"]), dtype=DTYPE_MAP[idx1_info["dtype"]])
        if "data" in idx2_info:
            idx2 = torch.tensor(idx2_info["data"], dtype=DTYPE_MAP[idx2_info["dtype"]]).reshape(idx2_info["shape"])
        else:
            idx2 = torch.randint(idx2_info["range"][0], idx2_info["range"][1] + 1, tuple(idx2_info["shape"]), dtype=DTYPE_MAP[idx2_info["dtype"]])
        if "data" in gradDist1_info:
            gradDist1 = torch.tensor(gradDist1_info["data"], dtype=DTYPE_MAP[gradDist1_info["dtype"]]).reshape(gradDist1_info["shape"])
        else:
            gradDist1 = torch.rand(gradDist1_info["shape"], dtype=DTYPE_MAP[gradDist1_info["dtype"]])
        if "data" in gradDist2_info:
            gradDist2 = torch.tensor(gradDist2_info["data"], dtype=DTYPE_MAP[gradDist2_info["dtype"]]).reshape(gradDist2_info["shape"])
        else:
            gradDist2 = torch.rand(gradDist2_info["shape"], dtype=DTYPE_MAP[gradDist2_info["dtype"]])

        input_groups.append([xyz1, xyz2, idx1, idx2, gradDist1, gradDist2])
    return input_groups


def get_init_inputs():
    return []
