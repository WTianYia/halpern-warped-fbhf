# 07 提交包说明

本目录是论文 `manuscript_v1.tex` 的独立提交包，按“论文源码 / 编译图片 / 实验复现资产”三类整理。

## 根目录

- `manuscript_v1.tex`: LaTeX 主文件。
- `manuscript_v1.pdf`: 当前编译得到的论文 PDF。
- `manuscript_v1.aux`, `manuscript_v1.log`, `manuscript_v1.out`, `manuscript_v1.fls`, `manuscript_v1.fdb_latexmk`: 当前编译辅助文件，保留用于排查引用和编译状态。

从本目录直接运行以下命令即可重新编译：

```bash
pdflatex -interaction=nonstopmode manuscript_v1.tex
pdflatex -interaction=nonstopmode manuscript_v1.tex
```

## figures

存放 LaTeX 编译直接调用的图片与图源数据。

- `fig0_halpern_selection.pdf`: 非唯一解选择实验图。
- `fig1_oracle_curves.pdf`: oracle 曲线主图。
- `fig2_cscan.pdf`: bounded-variation 参数扫描图。
- `fig3_ood.pdf`: 分布外测试图。
- `fig4_reconstruction_plate.*`: 图像恢复对比图，含 PDF/PNG/SVG/TIFF。
- `source_data_fig4_reconstruction_plate.csv`: 图像恢复图的数据说明。

## experiments

每个子目录对应一类实验，均包含源代码、原始数据、图片或生成图片所需的元数据。

### 01_nonunique_selection

非唯一解与 Halpern 极小范数选择实验。

主要内容：
- `run_nonunique_selection.py`: 实验源代码。
- `nonunique_curves.csv`, `nonunique_finals.csv`, `nonunique_summary.json`: 原始结果数据。
- `fig0_halpern_selection.pdf`, `fig2_halpern_selection.*`: 生成图片。

### 02_main_oracle_real_synthetic

真实图像与合成图像上的主 oracle-budget 实验。

主要内容：
- `trackB_formal_train.py`, `trackB_formal_eval.py`: 训练与评测源代码。
- `make_oracle_std_figures.py`: 作图代码。
- `main_real_synth/`: 原始 CSV/JSON 结果。
- `fig1_oracle_curves_meanstd.*`: 主 oracle 曲线。
- `source_data_fig1_oracle_curves_meanstd.csv`, `table_mean_std_all.csv`, `table_mean_std_selected.csv`: 作图与表格源数据。
- `standard_images/`: 真实图像输入集。
- `wbv_seed11.pt`: learned warped metric checkpoint。

### 03_safeguard_cscan

bounded-variation 强度参数扫描实验。

主要内容：
- `trackB_formal_eval.py`, `make_oracle_std_figures.py`: 评测与作图代码。
- `cscan_real/`: 原始 CSV/JSON 结果。
- `fig2_cscan_meanstd.*`: 参数扫描图。
- `source_data_fig2_cscan_meanstd.csv`: 作图源数据。
- `standard_images/`: 真实图像输入集。

### 04_distribution_shift_ood

低噪声、高噪声、宽模糊、小模糊和 motion blur 分布外实验。

主要内容：
- `trackB_formal_train.py`, `trackB_formal_eval.py`: 训练与评测源代码。
- `ood_noise_real/`, `ood_blur_real/`, `motion_adapt/`: 原始 CSV/JSON 结果。
- `fig3_ood_ratios.*`: 分布外结果图。
- `source_data_fig3_ood_ratios.csv`: 作图源数据。
- `standard_images/`: 真实图像输入集。
- `wbv_seed11.pt`, `wbv_motion_seed11.pt`: Gaussian-trained 与 motion-trained checkpoints。

### 05_reconstruction_plate

图像恢复可视化对比实验。

主要内容：
- `make_reconstruction_plate.py`: 作图代码。
- `fig4_reconstruction_plate.*`: PDF/PNG/SVG/TIFF 图片。
- `source_data_fig4_reconstruction_plate.csv`: 图像面板源数据。
- `standard_images/`: 可视化输入图像。

### 06_warped_training_runs

learned warped metric 训练与远程运行记录。

主要内容：
- `warped_main/`: 正式 warped 训练、评测曲线、metrics、checkpoint、日志。
- `remote_warped_20260704_2055/`: 云端 warped 实验完整返回包。
- `remote_sigma_sweep_20260704_1957/`: sigma sweep 云端实验返回包。

### 07_supporting_probes

路线筛选探针实验，用于说明哪些学习方向被排除。

主要内容：
- `probe_lib.py`, `probe_stepsize_run.py`, `probe2_vector_run.py`, `probe3_trackB_run.py`: 探针代码。
- `probe_curves.png`, `probe2_vector.png`, `probe3_trackB.png`: 探针图。
- `05_探针结果_learning_gamma_红灯.md`, `06_探针2_向量校正也红灯_与最终结论.md`, `07_双轨定稿_ℓ1选择_ℓ2加速_三探针结论.md`: 探针结论与原始运行摘要。

## experiments 根目录的记录文件

- `RESULTS.md`: 最终实验包总结果。
- `12_实验结果_完整包_20260704.md`, `13_正式小表_oracle_wallclock_真实图_20260704.md`, `14_正式实验_oracle_only_20260704.md`, `15_正式实验_meanstd_经典图_20260705.md`: 实验记录文档。
- `05_探针结果_learning_gamma_红灯.md`, `06_探针2_向量校正也红灯_与最终结论.md`, `07_双轨定稿_ℓ1选择_ℓ2加速_三探针结论.md`: 探针结论记录。
