# 13 · 正式小表：oracle / wall-clock / 真实图 / OOD

日期：2026-07-04

## 目的

回应正式投稿实验的五个关键要求：

1. 横轴从 iteration 扩展到 oracle 与 wall-clock；
2. 加入真实标准图像分布；
3. 扫描 bounded-variation 强度 `c in {0.25, 0.5, 0.9}`；
4. 测试集扩到 `ntest=32`；
5. 增加 OOD blur/noise。

## 数据与配置

- GPU：远端 3080 Ti
- 图像尺寸：`128 x 128`
- 测试样本：`ntest=32`
- 长曲线：`K_eval=1000`
- 参考解：`ref_iters=4000`
- 真实图源：公开标准图 `camera/cameraman`, `coffee`, `astronaut`, `coins`, `page`, `moon`, `chelsea`
- 主 checkpoint：`wbv_seed11.pt`

本地结果目录：

- `trackB_训练实验/final_experiment_package/formal_pilot/`
- `trackB_训练实验/final_experiment_package/formal_long/`

## 结论总览

**可以主打的结论：**

- 在真实图主分布、synthetic 主分布、wide blur OOD 中，learned-warped-bv 在**同 oracle / 同迭代预算**下优于 fixed-precond。
- `c` 扫描稳定：`0.25, 0.5, 0.9` 全部优于 fixed-precond，且 `c=0.5/0.9` 略优。
- line-search-FBHF 有回溯成本，oracle 明显多于 fixed / learned；主分布上效果也不如 learned-warped。

**不能主打的结论：**

- 目前不能声称 learned-warped 在 wall-clock 上普遍更快。网络前向使 learned 每步约慢 `2.4-2.6x`，同等 wall-clock 早期预算下 fixed-precond 更快。
- motion blur + high noise 是明显 OOD 失败点：line-search/plain-FBHF 优于 learned-warped，fixed-precond 在该设定下也明显失稳。

## 长曲线结果：err@1000

| data | blur/noise | method | err@1000 | wall-clock | oracle |
|---|---|---|---:|---:|---:|
| real | train/train | plain-FBHF | `9.51e-04` | `1.636s` | `4000` |
| real | train/train | line-search-FBHF | `8.89e-04` | `2.249s` | `4526` |
| real | train/train | fixed-precond | `2.62e-04` | `1.202s` | `4000` |
| real | train/train | learned-warped-bv | **`1.47e-04`** | `2.920s` | `4000` |
| synthetic | train/train | fixed-precond | `1.67e-03` | `1.165s` | `4000` |
| synthetic | train/train | learned-warped-bv | **`1.12e-03`** | `2.913s` | `4000` |
| real | wide/high | fixed-precond | `4.09e-04` | `1.141s` | `4000` |
| real | wide/high | learned-warped-bv | **`2.36e-04`** | `2.914s` | `4000` |
| real | motion/high | plain-FBHF | `2.83e-04` | `1.089s` | `4000` |
| real | motion/high | line-search-FBHF | **`1.75e-04`** | `1.581s` | `4526` |
| real | motion/high | learned-warped-bv | `1.40e-02` | `2.918s` | `4000` |

## Same wall-clock 预算

以 fixed-precond 的完整运行时间作为预算：

| data | blur/noise | fixed-precond err | learned-warped err at same wall-clock | 结论 |
|---|---|---:|---:|---|
| real | train/train | **`2.62e-04`** | `6.13e-04` | learned 不赢 |
| synthetic | train/train | **`1.67e-03`** | `4.19e-03` | learned 不赢 |
| real | wide/high | **`4.09e-04`** | `1.43e-03` | learned 不赢 |
| real | motion/high | `3.60e-03` | `2.94e-03` | learned 优于 fixed，但仍弱于 line-search/plain |

解释：

- learned-warped 的数值质量更高，但 Python/PyTorch 网络前向在逐迭代调用时成本明显。
- 因此正文不能写“wall-clock 加速”；只能写“oracle/fixed-budget accuracy improves”。
- wall-clock 可以作为诚实负面或工程优化空间：若将网络前向融合/编译、或用更轻量参数化，才可能转成真实时间优势。

## Same oracle 预算

在 `oracle=4000` 附近：

- real train：learned `1.47e-04` vs fixed `2.62e-04`
- synthetic train：learned `1.12e-03` vs fixed `1.67e-03`
- real wide/high：learned `2.36e-04` vs fixed `4.09e-04`
- real motion/high：learned `1.40e-02`，失败，line-search `1.75e-04`

主张边界：

> learned-warped-bv improves fixed-oracle reconstruction quality on in-distribution Gaussian deblurring and wide Gaussian blur, but does not yet give a wall-clock speedup and does not generalize to motion blur without retraining or blur-aware features.

## c 扫描

真实图主分布 `real/train/train`, `K_eval=300`：

| c | learned err@300 | fixed-precond err@300 |
|---:|---:|---:|
| `0.25` | `9.50e-04` | `1.31e-03` |
| `0.5` | **`8.85e-04`** | `1.31e-03` |
| `0.9` | **`8.80e-04`** | `1.31e-03` |

结论：

- 不是 `c=0.5` 的偶然结果。
- `c=0.5` 足够稳，可作为默认；`c=0.9` 略好但差距很小。

## 对论文实验口径的调整

主图建议改成：

1. **error vs oracle**：主分布真实图 + synthetic，learned 明确赢 fixed-precond。
2. **c 扫描小图**：展示 bounded-variation 参数不敏感。
3. **OOD 小表**：wide blur 仍赢，motion blur 失败，诚实标注为限制。
4. **wall-clock 附表**：不主打，只说明当前 PyTorch 实现的网络前向成本抵消了算法迭代收益。

不建议写：

- “wall-clock faster”；
- “wins line-search on all image restoration tasks”；
- “distribution-free acceleration”。

建议写：

- “improves fixed-oracle accuracy under a provably convergent learned warped metric”；
- “the provable bounded-variation version retains nearly all of the unconstrained metric benefit”；
- “motion blur exposes a distribution-shift limitation and motivates blur-aware training/features”。
