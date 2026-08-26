import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, kv: torch.Tensor, gamma: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                index: torch.Tensor, k_cache: torch.Tensor, ckv_cache: torch.Tensor,
                k_rope_scale: torch.Tensor = None, c_kv_scale: torch.Tensor = None,
                k_rope_offset: torch.Tensor = None, c_kv_offset: torch.Tensor = None,
                epsilon: float = 1e-5, cache_mode: str = 'Norm', is_output_kv: bool = False) -> tuple:
        """
        KV RMSNorm RoPE Cache - NPU reference implementation.
        Inputs: kv[B,N,S,d], gamma[rms_size], cos[B,N,S,rope_size], sin[B,N,S,rope_size],
                index [...], k_cache [...], ckv_cache [...], epsilon, cache_mode, is_output_kv
        Returns: (k_cache_out, ckv_cache_out, k_embed_out, y_out)
        """
        return torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                                    k_rope_scale=k_rope_scale, c_kv_scale=c_kv_scale,
                                                    k_rope_offset=k_rope_offset, c_kv_offset=c_kv_offset,
                                                    epsilon=epsilon, cache_mode=cache_mode,
                                                    is_output_kv=is_output_kv)


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), 'level2_12_KvRmsnormRopeCache.json')
    input_groups = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case['inputs']
            tensors = {}
            attrs = {}
            for inp in inputs:
                if inp['type'] == 'tensor':
                    name = inp['name']
                    dtype_str = inp.get('dtype', 'float32')
                    shape = inp.get('shape')
                    if shape is None:
                        tensors[name] = None
                    elif dtype_str == 'bool':
                        tensors[name] = (torch.rand(shape) > 0.5).to(torch.bool)
                    elif dtype_str in ('int32', 'int64', 'int8'):
                        max_val = {'int32': 1000, 'int64': 10000, 'int8': 127}.get(dtype_str, 100)
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                        tensors[name] = torch.randint(0, max_val, shape, dtype=dtype)
                    else:
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}.get(dtype_str, torch.float32)
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']

            index = tensors.get('index')
            cache_mode = attrs.get('cache_mode', 'Norm')
            k_cache = tensors.get('k_cache')
            if index is not None and k_cache is not None:
                device = index.device
                if cache_mode == 'Norm':
                    max_seq = k_cache.shape[2]
                    if index.dim() == 2:
                        B, S = index.shape
                        total = B * S
                        index = (torch.arange(total, dtype=torch.int32, device=device) % max_seq).reshape(B, S)
                    else:
                        total = index.numel()
                        index = (torch.arange(total, dtype=torch.int32, device=device) % max_seq).reshape(index.shape)
                elif cache_mode in ('PA', 'PA_BNSD', 'PA_NZ'):
                    index = torch.arange(index.numel(), dtype=torch.int32, device=device)
                elif cache_mode in ('PA_BLK_BNSD', 'PA_BLK_NZ'):
                    block_size = k_cache.shape[1]
                    length = index.numel()
                    index = torch.arange(length, dtype=torch.int32, device=device) * block_size
                tensors['index'] = index

            group = [
                tensors['kv'], tensors['gamma'], tensors['cos'], tensors['sin'],
                tensors['index'], tensors['k_cache'], tensors['ckv_cache'],
                tensors.get('k_rope_scale'), tensors.get('c_kv_scale'),
                tensors.get('k_rope_offset'), tensors.get('c_kv_offset'),
                attrs.get('epsilon', 1e-5), attrs.get('cache_mode', 'Norm'),
                attrs.get('is_output_kv', False)
            ]
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
