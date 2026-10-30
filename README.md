# Ling-3.0-flash on one DGX Spark

A sparkrun recipe for `inclusionAI/Ling-3.0-flash-fp4` on a single GB10,
built one small step at a time.

## Goal

Reproduce what NVIDIA reported for three MoE speedups on one DGX Spark,
at batch size 1. Decode tok/s:

| Step | Change                                       | Synthetic¹ | GSM8K² |
|------|----------------------------------------------|-----------:|-------:|
| 0    | Baseline: flashinfer_mxfp4 MoE, FP32 LM head | 36.1       | 37.8   |
| 1    | Humming MoE backend                          | 39.5       | —      |
| 2    | + online FP8 LM head                         | 43.9       | —      |
| 3    | + MTP speculative decoding (NEXTN, 3 steps)  | 102.9      | 53.9   |

¹ 32K-token prompt, 2K-token answer, greedy. Repetitive output, so MTP's
  guesses are accepted unusually often (3.90 per step).
² All 1,319 GSM8K questions, temperature 0.6. MTP accepts 2.68 per step.
  Accuracy 96.89% → 96.82%.

The article reports steps 1 and 2 on the synthetic test only.

## Sources

- NVIDIA: <https://developer.nvidia.com/blog/3-ways-to-boost-moe-inference-on-nvidia-dgx-spark/>
- inclusionAI cookbook notebook (Humming + FP8 LM head + MTP):
  <https://github.com/inclusionAI/ling-cookbook/blob/main/guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-ling-3-flash-mxfp4-humming.ipynb>
