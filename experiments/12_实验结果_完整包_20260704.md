# 12 · 实验结果完整包（2026-07-04）

## 结论

本轮完整实验支持最终实验叙事：

1. **Track B 主算法 learned-warped-bv 有效**：在 TV 去模糊原始-对偶 FBHF 台架上，理论可证的 bounded-variation learned metric 稳定优于 plain-FBHF 与 fixed-precond FBHF。
2. **可证约束代价很小**：free metric 不满足 bounded-variation 证明条件，但它相对 bv 只有极小提升，说明主数值价值来自 warped/preconditioned 结构，而不是不可证自由度。
3. **Track A 的锚定选择性可见**：在非唯一解三算子例子中，普通 FBHF 保留初始自由坐标，Halpern-FBHF 收敛到极小范数解。

## 实验 1：TV 去模糊 learned-warped FBHF

设置：

- 图像尺寸：`96 x 96`
- 训练展开：`K=40`
- 训练步数：`3000`
- batch：`16`
- 测试展开：`K_eval=400`
- 测试批次：主图 `test_seed=20240704`，额外稳健性 `20240705/20240706/20240707`
- 源脚本：`trackB_训练实验/final_experiment_package/warped_main/trackB_warped_train.py`

主图测试批次结果（median relative primal error）：

| mode | method | err@200 | err@400 |
|---|---|---:|---:|
| bv | plain-FBHF | `3.69e-02` | `1.75e-02` |
| bv | fixed-precond | `6.78e-03` | `3.54e-03` |
| bv | learned-warped | `4.48e-03` | **`2.35e-03`** |
| free | plain-FBHF | `3.69e-02` | `1.75e-02` |
| free | fixed-precond | `6.78e-03` | `3.54e-03` |
| free | learned-warped | `4.54e-03` | **`2.30e-03`** |

4 个 held-out test seed 的 err@400 均值 ± 标准差：

| mode | method | mean err@400 | std |
|---|---|---:|---:|
| bv | plain-FBHF | `1.46e-02` | `6.91e-03` |
| bv | fixed-precond | `3.27e-03` | `3.65e-04` |
| bv | learned-warped | **`2.18e-03`** | `3.01e-04` |
| free | learned-warped | `2.18e-03` | `3.11e-04` |

解读：

- learned-warped-bv 相对 fixed-precond 的平均误差降低约 `33%`。
- free 与 bv 的平均 `err@400` 基本相同，free 没有形成实质优势。
- 因此论文主算法应写 bv，free 只作为消融/经验上界。

## 实验 2：非唯一解极小范数选择

有限维三算子包含：

```text
0 in N_{z3>=0}(z) + Bz + Cz,
```

其中 `B` 是 `(z1,z2)` 上的斜对称旋转，`C` 阻尼 `z3`，`z4` 是自由解坐标。解集为

```text
Z = {(0,0,0,t): t in R},
```

极小范数解为 `0`。

结果：

| method | final distance to min-norm solution, mean |
|---|---:|
| plain-FBHF | `2.00e+00` |
| Halpern-FBHF | **`6.66e-04`** |

解读：

- plain-FBHF 收敛到依赖初值的解，保留初始自由坐标。
- Halpern-FBHF 由锚点 `u=0` 选择 `P_Z(0)`，即极小范数解。
- 该实验不比速度，只展示锚定层在非唯一解问题中的数值意义。

## 论文图

Nature-style 图已生成，均含 SVG/PDF/TIFF/PNG：

- `trackB_训练实验/final_experiment_package/figures_nature/fig1_warped_metric.*`
- `trackB_训练实验/final_experiment_package/figures_nature/fig2_halpern_selection.*`

图形 contract：

- `trackB_训练实验/final_experiment_package/figures_nature/FIGURE_CONTRACT.md`

## 源数据与可复现文件

- warped 主实验：`trackB_训练实验/final_experiment_package/warped_main/`
- 非唯一解实验：`trackB_训练实验/final_experiment_package/nonunique_selection/`
- 制图脚本：`trackB_训练实验/make_nature_figures.py`
- 非唯一解脚本：`trackB_训练实验/run_nonunique_selection.py`

## 边界

当前结果已经足够作为论文实验第一版主干，但正式投稿前仍建议补一个小型扩展表：

- 多 blur/noise 组合；
- 至少 2-3 个自然图像或更丰富的 synthetic image family；
- 若目标期刊偏数值优化，增加 QP/portfolio 例子作外部任务验证。

不建议再把 learned-γ 标量学习放回主线；它已经由探针排除。
