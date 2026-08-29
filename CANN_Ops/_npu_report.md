# CANN_Ops NPU 全用例运行报告

| 项目 | 数量 |
|---|---|
| 算子总数 | 296 |
| 全部用例通过的算子 | 279 |
| 存在失败用例的算子 | 17 |
| 用例通过率 | 3382/3619 |

## 存在失败的算子

| 算子 | 通过/总数 | 首个错误 |
|---|---|---|
| cannops_level1_0_AbsMath | 15/18 | case 15: RuntimeError('abs:third_party/op-plugin/op_plugin/ops/opapi/StructKernelNpuOpApi.cpp:225 NPU function error: call aclnnAbs failed, error code |
| cannops_level1_43_MulAddn | 0/6 | case 0: RuntimeError('Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.\n[E |
| cannops_level1_9_Ccopy | 0/5 | case 0: RuntimeError('index_high_dims_op_api:third_party/op-plugin/op_plugin/ops/opapi/IndexKernelNpuOpApi.cpp:73 NPU function error: call aclnnIndex  |
| cannops_level2_11_AddRmsNormQuant | 0/5 | case 0: RuntimeError('Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.\n[E |
| cannops_level2_126_MaxPool3DGradWithArgmax | 0/6 | case 0: RuntimeError('max_pool3d_with_indices_backward:third_party/op-plugin/op_plugin/ops/opapi/MaxPool3dWithIndicesBackwardKernelNpuOpApi.cpp:86 NPU |
| cannops_level2_2_AdaptiveMaxPool3DGrad | 0/6 | case 0: RuntimeError('adaptive_max_pool3d_backward:third_party/op-plugin/op_plugin/ops/opapi/AdaptiveMaxPool3dBackwardKernelNpuOpApi.cpp:67 NPU functi |
| cannops_level3_11_GroupedMatmulSliceMPerTokenDequant | 0/8 | case 0: RuntimeError('matmul_implement_npu:third_party/op-plugin/op_plugin/ops/opapi/MatmulKernelNpuOpApi.cpp:92 NPU function error: call aclnnMatmul  |
| cannops_level3_23_MoeInitRouting | 0/25 | case 0: TypeError("can't convert npu:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.") |
| cannops_level3_24_MoeInitRoutingQuant | 0/14 | case 0: TypeError("can't convert npu:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.") |
| cannops_level3_25_MoeInitRoutingQuantV2 | 0/25 | case 0: TypeError("can't convert npu:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.") |
| cannops_level3_26_MoeInitRoutingV2 | 0/25 | case 0: TypeError("can't convert npu:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.") |
| cannops_level3_27_MoeInitRoutingV2Grad | 0/20 | case 0: RuntimeError('Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.\n[E |
| cannops_level3_32_MoeTokenUnpermuteGrad | 0/30 | case 0: RuntimeError('Expected all tensors to be on the same device, but found at least two devices, cpu and npu:0! (when checking argument for argume |
| cannops_level3_33_MoeTokenUnpermuteWithEp | 0/21 | case 0: RuntimeError('Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.\n[E |
| cannops_level3_35_MoeTokenUnpermuteWithRoutingMap | 0/20 | case 0: RuntimeError('Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.\n[E |
| cannops_level3_39_QuantMatmul | 0/10 | case 0: RuntimeError('matmul_implement_npu:third_party/op-plugin/op_plugin/ops/opapi/MatmulKernelNpuOpApi.cpp:92 NPU function error: call aclnnMatmul  |
| cannops_level3_9_GroupedMatmulSliceKPerTokenDequant | 0/8 | case 0: RuntimeError('matmul_implement_npu:third_party/op-plugin/op_plugin/ops/opapi/MatmulKernelNpuOpApi.cpp:92 NPU function error: call aclnnMatmul  |
