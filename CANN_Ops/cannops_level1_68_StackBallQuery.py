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

def stack_ball_query_cpu(xyz: torch.Tensor, center_xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor, center_xyz_batch_cnt: torch.Tensor, max_radius: float, sample_num: int) -> torch.Tensor:
    """
    xyz: [3, N] planar；center_xyz: [M, 3]。
    与两条 kernel 分支（FP32 / FP16）在「顺序扫描 + 平方距离与 max_radius^2 比较」语义上一致；
    距离在 float32 中计算（FP16 输入先提升），避免半精度边界与参考不一致。
    无命中：首元素 -1，其余填 0；有命中：输出 batch 内局部下标，不足槽位用第一个命中下标填充。
    """
    m = center_xyz.shape[0]
    max_r2 = max_radius * max_radius
    xb = xyz_batch_cnt.detach().cpu().tolist()
    cb = center_xyz_batch_cnt.detach().cpu().tolist()
    batch_size = len(xb)

    def center_batch(global_idx: int):
        cum = 0
        for b in range(batch_size):
            if global_idx < cum + cb[b]:
                return (b, global_idx - cum)
            cum += cb[b]
        return (batch_size - 1, 0)

    def xyz_off(b: int):
        return sum(xb[:b])
    out = torch.empty(m * sample_num, device=xyz.device, dtype=torch.int32)
    cx = center_xyz[:, 0].to(torch.float32)
    cy = center_xyz[:, 1].to(torch.float32)
    cz = center_xyz[:, 2].to(torch.float32)
    px = xyz[0].to(torch.float32)
    py = xyz[1].to(torch.float32)
    pz = xyz[2].to(torch.float32)
    for mi in range(m):
        b, _ = center_batch(mi)
        cxv, cyv, czv = (cx[mi].item(), cy[mi].item(), cz[mi].item())
        off = xyz_off(b)
        cnt = xb[b]
        collected = []
        for i in range(cnt):
            dx = px[off + i].item() - cxv
            dy = py[off + i].item() - cyv
            dz = pz[off + i].item() - czv
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < max_r2:
                collected.append(i)
                if len(collected) >= sample_num:
                    break
        if len(collected) == 0:
            out[mi * sample_num] = -1
            for s in range(1, sample_num):
                out[mi * sample_num + s] = 0
        else:
            fr = collected[0]
            for s in range(sample_num):
                if s < len(collected):
                    out[mi * sample_num + s] = collected[s]
                else:
                    out[mi * sample_num + s] = fr
    return out

class Model(nn.Module):

    def __init__(self, max_radius: float, sample_num: int):
        super().__init__()
        self.max_radius = max_radius
        self.sample_num = sample_num

    def forward(self, xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt):
        return stack_ball_query_cpu(xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, self.max_radius, self.sample_num)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_68_StackBallQuery.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        xyz_info = inputs[0]
        center_xyz_info = inputs[1]
        xyz_batch_cnt_info = inputs[2]
        center_xyz_batch_cnt_info = inputs[3]

        if "data" in xyz_info:
            xyz = torch.tensor(xyz_info["data"], dtype=DTYPE_MAP[xyz_info["dtype"]]).reshape(xyz_info["shape"])
        else:
            xyz = torch.rand(xyz_info["shape"], dtype=DTYPE_MAP[xyz_info["dtype"]]) * (xyz_info["range"][1] - xyz_info["range"][0]) + xyz_info["range"][0]
        if "data" in center_xyz_info:
            center_xyz = torch.tensor(center_xyz_info["data"], dtype=DTYPE_MAP[center_xyz_info["dtype"]]).reshape(center_xyz_info["shape"])
        else:
            center_xyz = torch.rand(center_xyz_info["shape"], dtype=DTYPE_MAP[center_xyz_info["dtype"]]) * (center_xyz_info["range"][1] - center_xyz_info["range"][0]) + center_xyz_info["range"][0]
        if "data" in xyz_batch_cnt_info:
            xyz_batch_cnt = torch.tensor(xyz_batch_cnt_info["data"], dtype=DTYPE_MAP[xyz_batch_cnt_info["dtype"]]).reshape(xyz_batch_cnt_info["shape"])
        else:
            xyz_batch_cnt = torch.randint(xyz_batch_cnt_info["range"][0], xyz_batch_cnt_info["range"][1] + 1, tuple(xyz_batch_cnt_info["shape"]), dtype=DTYPE_MAP[xyz_batch_cnt_info["dtype"]])
        if "data" in center_xyz_batch_cnt_info:
            center_xyz_batch_cnt = torch.tensor(center_xyz_batch_cnt_info["data"], dtype=DTYPE_MAP[center_xyz_batch_cnt_info["dtype"]]).reshape(center_xyz_batch_cnt_info["shape"])
        else:
            center_xyz_batch_cnt = torch.randint(center_xyz_batch_cnt_info["range"][0], center_xyz_batch_cnt_info["range"][1] + 1, tuple(center_xyz_batch_cnt_info["shape"]), dtype=DTYPE_MAP[center_xyz_batch_cnt_info["dtype"]])

        input_groups.append([xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_68_StackBallQuery.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        max_radius_info = entries[0]
        sample_num_info = entries[1]
        max_radius = max_radius_info["value"]
        sample_num = sample_num_info["value"]
        init_groups.append([max_radius, sample_num])
    return init_groups
