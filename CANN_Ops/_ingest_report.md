# CANN_Ops 入库验证报告 (main 稳定子集 280 口径)

验证内容: 与 cann_ops_tmp 原版结构等价(case数/参数/dtype/shape/attr) + CPU 前向全用例(含梯度路径) + 生成分布比对。
AbsMath 已按 drop_cases 删除 3 条 complex64 用例后验证。REVIEW 为分布近似提示(结构与正确性均通过)。

| 状态 | 数量 |
|---|---|
| PASS | 257 |
| REVIEW | 23 |
| FAIL | 0 |

## REVIEW (23)

| 算子 | 详情 |
|---|---|
| cannops_level1_26_HansDecode | arg 1 (mantissa): nonfinite-frac 1.000 vs 0.015<br>arg 2 (fixed): nonfinite-frac 1.000 vs 0.000 |
| cannops_level1_29_Icamax | arg 0 (x): std 1.004 vs orig 0.5792; float range [-3.954,3.93] vs orig [-0.9999,0.9996] |
| cannops_level1_30_Icamin | arg 0 (x): std 0.9978 vs orig 0.5771; float range [-4.006,4.366] vs orig [-0.9999,1] |
| cannops_level1_33_IsFinite | arg 0 (x): nonfinite-frac 1.000 vs 0.000 |
| cannops_level1_34_IsInf | arg 0 (x): nonfinite-frac 1.000 vs 0.000 |
| cannops_level1_36_Isamin | arg 0 (x): mean 0.1331 vs orig -0.9416 |
| cannops_level1_52_Pows | arg 0 (x1): mean 2.45 vs orig 0.7979; std 1.434 vs orig 0.6027 |
| cannops_level1_64_Snrm2 | arg 0 (x): mean 0.5067 vs orig -0.1366 |
| cannops_level1_66_Sqrt | arg 0 (input): mean 18.79 vs orig 7.956; std 11.16 vs orig 6.001 |
| cannops_level1_6_ApplyTopKTopPWithSorted | arg 0 (sorted_value): nonfinite-frac 0.125 vs 0.266 |
| cannops_level2_109_GroupNormGrad | arg 1 (mean): mean 0.3839 vs orig 0.05061; std 0.3225 vs orig 0.02991<br>arg 2 (rstd): mean 0.3797 vs orig 0.05144; std 0.3177 vs orig 0.02901<br>arg 4 (gamma): std 0.1986 vs orig 0.02913 |
| cannops_level2_110_GroupNormSilu | arg 2 (beta): std 0.2641 vs orig 0.02842 |
| cannops_level2_112_GroupNormSwishGrad | arg 1 (mean): mean 0.4932 vs orig 0.05029; std 0.2924 vs orig 0.02871<br>arg 2 (rstd): mean 0.494 vs orig 0.0512; std 0.2904 vs orig 0.02885<br>arg 4 (gamma): std 0.2453 vs orig 0.02848<br>arg 5 (beta): mean 0.3606 vs orig 0.04946; std 0.3143 vs orig 0.0292 |
| cannops_level2_119_LayerNormGradV3 | arg 2 (mean): std 0.2864 vs orig 0.01496 |
| cannops_level2_16_AscendQuantV2 | arg 1 (scale): mean 0.4131 vs orig 0.1306; std 0.2951 vs orig 0.04015 |
| cannops_level2_20_CTCLossV3 | arg 0 (log_probs): mean -1.972 vs orig -1.046 |
| cannops_level2_30_DequantBias | arg 1 (weight_scale): mean 0.2628 vs orig 0.03522; std 0.305 vs orig 0.01428 |
| cannops_level3_12_InterleaveRope | arg 3 (_out): nonfinite-frac 1.000 vs 0.030 |
| cannops_level3_3_DequantRopeQuantKvcache | arg 6 (scale_k): mean 0.1978 vs orig 0.03494; std 0.278 vs orig 0.01461<br>arg 7 (scale_v): mean 0.2109 vs orig 0.03531; std 0.2912 vs orig 0.01431<br>arg 10 (weight_scale): mean 0.4665 vs orig 0.05988; std 0.2876 vs orig 0.02753<br>arg 13 (_q_buf): nonfinite-frac 0.939 vs 0.051<br>arg 14 (_k_buf): nonfinite-frac 0.869 vs 0.044<br>arg 15 (_v_buf): nonfinite-frac 0.866 vs 0.062 |
| cannops_level3_42_RopeQuantKvcache | arg 3 (quant_scale): std 0.2447 vs orig 0.01415<br>arg 8 (_q_buf): nonfinite-frac 0.935 vs 0.072<br>arg 9 (_k_buf): nonfinite-frac 0.873 vs 0.121<br>arg 10 (_v_buf): nonfinite-frac 0.869 vs 0.133 |
| cannops_level3_44_RotaryPositionEmbedding | arg 3 (_out): nonfinite-frac 1.000 vs 0.013 |
| cannops_level3_45_RotaryPositionEmbeddingGrad | arg 4 (_dx): nonfinite-frac 1.000 vs 0.010<br>arg 5 (_dcos): nonfinite-frac 0.973 vs 0.007<br>arg 6 (_dsin): nonfinite-frac 0.973 vs 0.032 |
| cannops_level3_46_ScaledMaskedSoftmaxGradV2 | arg 1 (y): mean 0.5 vs orig 0.004794; std 0.2886 vs orig 0.007533 |
