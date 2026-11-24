> **已合并到 [P100 LLM Toolkit](https://github.com/klklklibg/p100-llm-toolkit)** — 本仓库保留作为历史存档，后续更新在 monorepo 的 `patches/` 目录。

# P100 / GP100 (sm_60) llama.cpp 补丁集

两个在 Tesla P100 16GB（GP100, sm_60, CUDA 12.4, Windows）上实测过的独立补丁。每个都可单独 cherry-pick 到对应版本的 llama.cpp。

## 补丁清单

### 0001 — sm_60 fp16 carve-out

**目标版本**：llama.cpp `03bcbd530`（2026-11-18 基线）
**文件**：`patches/0001-sm60-fp16-carveout.patch`

上游 `common.cuh` 把 Pascal 全系当作「有 FP16 但无快速 FP16」。本补丁把 sm_60 (P100) 从 `FAST_FP16_AVAILABLE` / `fast_fp16_hardware_available` 里排除，与 sm_61 (GP104) 同等对待。

**实测**：某 35B MoE 上 79.71 → 78.65 t/s（需重新核对方向）。注意：在 Inkling2 上观察到 -6.6%，因此**该补丁未合入日常构建，仅作研究留档**。

### 0002 — GP100 MMVQ half2 矩阵向量内核（移植）

**目标版本**：llama.cpp `f4f6f2e7`（2026-11-23 master）
**文件**：`patches/0002-gp100-mmvq-kernel.patch` + `patches/mmvq-gp100.cu` + `patches/mmvq-gp100.cuh`

sm_60 没有 `__dp4a`，通用 MMVQ 用 4 个标量乘模拟 int8 点积，受指令发射限制（约 HBM2 带宽的 45%）。GP100 有全速 half2 FMA。此内核：

- 权重解包成 subnormal half 位模式（`q * 2^-24`，精确、无需 magic-number 减法），HFMA2 在 sm_60 全速处理 subnormal；
- 激活一次转换到 fp16，按 32 block 用 2 的幂预缩放避免溢出；
- 每个 sub-block 16 项 fp16 累加后 flush 到 fp32。

**精度**：相比其替换的 q8_1 路径，对 fp64 参考的误差约 1/10（激活量化到 8 bit 是原路径的主要误差源）。
**实测**：需在 P100 上自行复测后才能用于生产。

> **来源与许可**：这两个 `.cu/.cuh` 文件来自 **[qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp)** 的补丁 `0007-CUDA-GP100-sm_60-fp16-matrix-vector-path-for-batch-1-8.patch`，作者 qr0n，MIT 许可。本仓库只做了向 2026-11-23 master 的移植适配（`mmvq.cu` 的改动），**原始内核版权归原作者所有**。完整 18 个补丁见原仓库。

## 使用

```bash
# 0001
cd llama.cpp && git checkout 03bcbd530
git apply /path/to/0001-sm60-fp16-carveout.patch

# 0002：先复制新增文件，再应用 diff
cd llama.cpp && git checkout f4f6f2e7
cp mmvq-gp100.cu mmvq-gp100.cuh ggml/src/ggml-cuda/
git apply /path/to/0002-gp100-mmvq-kernel.patch
```

编译需 CUDA 12.x（13.x 不支持 sm_60），`-DGGML_CUDA=ON`，并显式指定 sm_60 架构。

## 许可

本仓库中原创的移植适配部分为 MIT，见 [LICENSE](LICENSE)。文件 `patches/mmvq-gp100.cu` 与 `.cuh` 的版权归 [qr0n](https://github.com/qr0n) 所有，同样适用 MIT 许可；使用时请保留原作者署名。上游 llama.cpp 为 MIT。
