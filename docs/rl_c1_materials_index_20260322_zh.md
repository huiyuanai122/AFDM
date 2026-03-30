# RL C1 当前最佳结果材料索引

日期：2026-03-22

本文档按 4 类材料整理当前 `RL 优化 C1` 最佳结果相关产物，方便直接转发、继续诊断或做汇报。

当前推荐主结果版本：

- `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl`

当前推荐主 checkpoint：

- `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt`


## 1. 结果图和日志

### 1.1 最重要的结果图

当前最推荐先看的图有这些：

1. 优化 C1 vs 不优化 C1 的主对比图
   - `figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/static_dynamic_baselines_tsv2seq_vdop_ctrl_mix_r1_tinyrl.png`
   - 对应数据：
     `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/static_dynamic_baselines_tsv2seq_vdop_ctrl_mix_r1_tinyrl.csv`

2. 主 BER-SNR 曲线图
   - `figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/fig3_ber_vs_snr_main.png`
   - 对应数据：
     `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/fig3_ber_vs_snr_main.csv`

3. 动作分布图
   - `figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/fig6_action_distribution.png`
   - 对应数据：
     `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/fig6_action_distribution.csv`
   - 频率版表：
     `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/fig6_action_distribution_freq.csv`

4. 低中 SNR 下的切换行为与 oracle 偏差图
   - `figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/fig11_switch_oracle_gap.png`
   - 对应数据：
     `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/fig11_switch_oracle_gap.csv`

5. 序列级 BER over time 图
   - 原始数据评测：
     `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/sequence_tsv2seq_vdop_ctrl_mix_r1_tinyrl_orig_ber_vs_time.png`
   - `dagger_r1` 评测：
     `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/sequence_tsv2seq_vdop_ctrl_mix_r1_tinyrl_dagger_ber_vs_time.png`

6. 序列级动作轨迹图
   - 原始数据评测：
     `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/sequence_tsv2seq_vdop_ctrl_mix_r1_tinyrl_orig_action_traj_seq127.png`
   - `dagger_r1` 评测：
     `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/sequence_tsv2seq_vdop_ctrl_mix_r1_tinyrl_dagger_action_traj_seq47.png`

7. RL 训练收敛图
   - `figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/fig5_rl_convergence.png`
   - 对应数据：
     `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/fig5_rl_convergence.csv`

8. 训练曲线图
   - RL policy：
     `figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/training_curves_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl_paper.png`
   - OAMPNet：
     `figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/training_curves_oampnet_tsv2seq_vdop_ctrl_mix_r1_tinyrl_paper.png`


### 1.2 结果表和结构化日志

当前最佳结果没有单独保存“混合训练这一轮的原始终端 stdout/stderr 文本日志”，
但保存了完整的结构化日志和结果表，可以直接用于分析：

- RL 训练摘要：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/train_summary_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl.json`
- RL 训练全过程：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/train_history_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl.json`
- RL epoch 曲线表：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/training_curves_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl_paper.csv`
- OAMPNet 训练曲线表：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/training_curves_oampnet_tsv2seq_vdop_ctrl_mix_r1_tinyrl_paper.csv`
- OAMPNet step loss 表：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/loss_step_oampnet_tsv2seq_vdop_ctrl_mix_r1_tinyrl.csv`
- 原始数据评测摘要：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/sequence_tsv2seq_vdop_ctrl_mix_r1_tinyrl_orig_summary.json`
- `dagger_r1` 评测摘要：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/sequence_tsv2seq_vdop_ctrl_mix_r1_tinyrl_dagger_summary.json`
- 自适应行为诊断：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/adaptivity_diagnostics_tsv2seq_vdop_ctrl_mix_r1_tinyrl.csv`
- 图导出 manifest：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/figure_export_manifest.json`


### 1.3 原始终端日志

现有仓库里能找到的原始 stdout/stderr 文本日志，主要来自 baseline 这条 paper pipeline 的重跑：

- `results/paper_tsv2seq_vdop_ctrl/rerun_logs/pipeline_stdout_20260320_094424.log`
- `results/paper_tsv2seq_vdop_ctrl/rerun_logs/pipeline_stderr_20260320_094424.log`
- `results/paper_tsv2seq_vdop_ctrl/rerun_logs/pipeline_stdout_20260320_094521.log`
- `results/paper_tsv2seq_vdop_ctrl/rerun_logs/pipeline_stderr_20260320_094521.log`
- `results/paper_tsv2seq_vdop_ctrl/rerun_logs/continue_stdout_20260320_165536.log`
- `results/paper_tsv2seq_vdop_ctrl/rerun_logs/continue_stderr_20260320_165536.log`

这些日志里能看到：

- detector dataset 生成进度
- oracle policy dataset 导出进度
- 部分 paper pipeline 延续运行信息

如果只是为了分析当前最佳结果，优先看结构化日志即可；如果要追溯完整流水线执行顺序，这些 stdout/stderr 也有用。


### 1.4 当前没有的结果

这条 `RL-C1` 最佳结果线当前主要围绕：

- BER
- 动作分布
- 切换率
- 恢复率
- 时变信道物理指标

当前没有现成导出的：

- BLER 曲线
- SER 曲线

也就是说，这条线目前是“BER + policy behavior diagnostics”为主，不是完整的 BER/BLER/SER 三件套。


## 2. 实验配置

### 2.1 这次主链路实际用到的脚本

当前最佳版本 `mix_r1_tinyrl` 的主链路，核心脚本是：

1. 原始 offline policy dataset 导出
   - `matlab/detectors/run_export_oracle_policy_paper_tsv2seq_vdop_ctrl.m`
   - `matlab/detectors/export_oracle_dataset_for_policy.m`

2. 构建 `dagger_r1`
   - `python/rl_c1/build_dagger_rollout_dataset.py`

3. 构建 mixed dataset
   - `python/rl_c1/build_mixed_policy_dataset.py`

4. 训练当前最佳 policy
   - `python/rl_c1/train_reinforce.py`

5. 序列级评估
   - `python/rl_c1/eval_sequence_policy.py`

6. 标准图导出
   - `python/rl_c1/export_standard_figures.py`


### 2.2 推荐复现命令

以下命令是按当前结果目录和结构化日志还原出来的可复现主链路。

解释器：

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe
```

1. 构建 `dagger_r1`

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe python\rl_c1\build_dagger_rollout_dataset.py `
  --input data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat `
  --checkpoint results\paper_tsv2seq_vdop_ctrl\best_reinforce_policy_tsv2seq_vdop_ctrl_paper.pt `
  --output data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz `
  --reward_key reward_relbase_proxy `
  --device cuda
```

2. 构建 mixed dataset

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe python\rl_c1\build_mixed_policy_dataset.py `
  --original data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat `
  --onpolicy data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz `
  --output data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1.npz `
  --reward_key reward_relbase_proxy
```

3. 训练当前最佳 `mix_r1_tinyrl`

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe python\rl_c1\train_reinforce.py `
  --data data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1.npz `
  --output_dir results\paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --reward_key reward_relbase_proxy `
  --split_mode sequence `
  --batch_size 256 `
  --epochs 20 `
  --reinforce_lr 1e-4 `
  --imitation_preserve_coef 0.5 `
  --checkpoint_name best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt `
  --history_name train_history_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl.json `
  --summary_name train_summary_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl.json `
  --device cuda
```

4. 在原始数据上做序列级评估

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe python\rl_c1\eval_sequence_policy.py `
  --data data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat `
  --checkpoint results\paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl\best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt `
  --reward_key reward_relbase_proxy `
  --output_dir results\paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --prefix sequence_tsv2seq_vdop_ctrl_mix_r1_tinyrl_orig `
  --select_snr 14 `
  --device cuda
```

5. 在 `dagger_r1` 数据上做序列级评估

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe python\rl_c1\eval_sequence_policy.py `
  --data data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz `
  --checkpoint results\paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl\best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt `
  --reward_key reward_relbase_proxy `
  --output_dir results\paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --prefix sequence_tsv2seq_vdop_ctrl_mix_r1_tinyrl_dagger `
  --select_snr 14 `
  --device cuda
```

6. 导出标准图

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe python\rl_c1\export_standard_figures.py `
  --data data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat `
  --checkpoint results\paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl\best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt `
  --train_history results\paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl\train_history_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl.json `
  --oampnet_history data\training_history_v4_tsv2seq_vdop_ctrl_paper.json `
  --reward_key reward_relbase_proxy `
  --results_dir results\paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --figures_dir figures\paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --paper_tag tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --device cuda
```


### 2.3 系统与信道关键参数

这部分来自：

- `matlab/simulation/run_generate_oampnet_paper_tsv2seq_vdop_ctrl.m`
- `matlab/detectors/run_export_oracle_policy_paper_tsv2seq_vdop_ctrl.m`
- `matlab/simulation/generate_dataset_timescaling_n256.m`

当前 paper profile 的关键参数如下：

- `N = 256`
- `Q = 0`
- `N_eff = 256`
- 调制：`QPSK`
- `Delta_f = 4`
- `fc = 12e3`
- `alpha_max = 1e-4`
- `ell_max = 16`
- 路径数：`P = 6`
- `Lcpp = 17`
- `Lcps = 1`
- 基线 `C1 = 0.015335728702672606`
- 固定 `C2 = sqrt(2)/N = 0.005524271728019903`
- `C1` 离散动作数：`21`
- `C1` 网格范围：
  - `min = 0.009201437221603563`
  - `max = 0.021470020183741646`
- `base_action index = 10`，对应网格中间位置
- `mode = timevary_sequence`
- 每条序列帧数：`40`
- 序列数：
  - export/oracle 数据：`500`
  - detector train dataset：train `500 seq`，val `100 seq`
- SNR 范围：`0:2:20`
- `doppler_mode = common_with_path_residual`
- `motion_profile = maneuver_heave`
- `path_projection_mode = symmetric_linear`
- `beta_min = 0.45`
- `beta_max = 1.65`
- `target_track_gain = 0.85`
- `target_blend = 0.85`
- `profile_v_peak = 0.98`
- `profile_heave_amp = 0.20`
- `profile_secondary_amp = 0.10`
- `rho_h = 0.98`
- `rho_acc = 0.95`
- `sigma_acc = 0.03`
- `rho_delta = 0.90`
- `sigma_delta = 0.05`
- `ell_mode = static`
- `pdp_mode = exp_fixed_per_sequence`


### 2.4 检测器配置

#### LMMSE

文件：

- `matlab/detectors/lmmse_detector.m`

当前就是标准线性 LMMSE：

- `x_hat = (H^H H + sigma^2 I)^(-1) H^H y`


#### OAMP

文件：

- `matlab/detectors/oamp_detector.m`

当前配置：

- 版本：MATLAB 版 `oamp_detector`
- 迭代次数：`10`
- 阻尼：`0.9`
- 调制：`QPSK`

这些参数在 `export_oracle_dataset_for_policy.m` 中用于 reward/oracle label 评估。


#### OAMP-Net

文件：

- `matlab/detectors/oampnet_detector.m`
- `python/training/train_oampnet_v4.py`
- `python/training/losses.py`
- `python/models/oampnet_v4.py`

当前这条 RL 最佳结果线用的是预训练 OAMPNet，不是这轮同时重训出来的。

对应参数文件：

- `data/oampnet_v4_tsv2seq_vdop_ctrl_paper_params.mat`

对应训练历史：

- `data/training_history_v4_tsv2seq_vdop_ctrl_paper.json`
- `data/training_history_v4_tsv2seq_vdop_ctrl_paper.npz`

预训练 OAMPNet 关键配置：

- 版本：`OAMPNetV4`
- `num_layers = 10`
- 训练 epoch：`30`
- batch size：`64`
- learning rate：`1e-2`
- OAMP baseline 用于对照：`10` 迭代，阻尼 `0.9`
- loss：`MSEPlusCrossEntropyLoss`
  - 形式：`alpha * MSE + beta * CE`
  - 默认：
    - `ce_alpha = 1.0`
    - `ce_beta = 0.05`
    - `ce_warmup_epochs = 15`

`training_history_v4_tsv2seq_vdop_ctrl_paper.json` 记录的最佳 epoch：

- `best_epoch = 19`
- `val_ber = 0.032174731057787696`
- `val_ber_oamp = 0.034169514973958336`
- `val_ber_lmmse = 0.03495376829117063`


### 2.5 当前最佳 RL policy 配置

这部分来自：

- `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/train_summary_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl.json`
- `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/train_history_rl_tsv2seq_vdop_ctrl_mix_r1_tinyrl.json`
- 最佳 checkpoint 本身

当前最佳 `mix_r1_tinyrl` 的关键配置：

- 状态维度：`20`
- 动作数：`21`
- `base_action = 10`
- 网络：`MLPPolicy`
- hidden dims：`128,64`
- dropout：当前 checkpoint 未显式保存，训练脚本默认 `0.0`
- reward key：`reward_relbase_proxy`
- imitation label：`hybrid_reward_margin(auto)`
- `reward_teacher_margin = 0.005`
- `reward_teacher_frac = 0.5055`
- `rl_mode = imitation_then_reinforce`
- `reinforce_lr = 1e-4`
- `imitation_preserve_coef = 0.5`
- `entropy_coef = 0.01`
- `reward_scale = 1000.0`
- `baseline_momentum = 0.95`
- `slot_ema_momentum = 0.98`
- `split_mode = sequence`
- `train_size = 32000`
- `val_size = 8000`
- RL 曲线里记录的 epoch 数：`20`
- batch size：`256`

注：

- `train_summary` 和 `train_history` 没有把所有 CLI 参数逐项原样回写。
- 上面这份配置是用“结构化日志 + checkpoint 元数据 + 脚本默认值”共同还原的。


## 3. 我现在最不满意的点

当前最想解决的问题，优先级如下。

### 3.1 第一优先级：训练集和测试集不匹配

这是这轮已经被证明存在、但还没完全解决干净的问题。

具体表现：

- 原 baseline 在原始数据上 `ber_gain_vs_base = 1.121%`
- 同一个 checkpoint 直接测 `dagger_r1` 时掉到 `0.247%`

说明：

- offline 数据中的历史特征来自 fixed/base action 轨迹
- online 部署时历史来自 policy 自己 rollout
- `prev_action_norm / prev_reward / recent_switch_rate` 的分布发生了明显偏移

这一点是当前最核心的问题来源。


### 3.2 第二优先级：`C1` 优化效果还不够稳定

虽然 `mix_r1_tinyrl` 已经明显优于 pure DAgger，也明显比“旧模型直接上新 history”稳定，
但如果只看原始 benchmark，它仍然没有真正超过当前 baseline。

关键数：

- baseline：`1.121%`
- `mix_r1_tinyrl`：`1.074%`

也就是说：

- `C1` 动态优化是有用的
- 但当前学到的策略还没有把这部分动态增益稳定兑现成最终 BER 优势


### 3.3 第三优先级：策略仍然偏静态，而且动作偏向 action 1

当前最佳 `mix_r1_tinyrl` 虽然把 action 11 塌缩大幅压下来了，
但新的偏置转到了 action 1。

动作频率：

- action 1：`51.28%`
- action 11：`20.445%`

同时：

- 原始数据评测的 `switch_rate_policy = 0.003663`
- `dagger_r1` 评测的 `switch_rate_policy = 0.008547`

这说明当前 policy 整体还是偏静态。


### 3.4 现在不是最主要矛盾的点

下面这些不是当前最优先要打的矛盾：

- OAMP-Net 比传统方法提升不大：
  当前 OAMPNet 预训练本身是有效的，不是当前瓶颈中心。
- 高 SNR error floor：
  这条线当前在 `18/20 dB` 已经接近 0，不是首要卡点。
- 多帧增益不明显：
  现在问题不在“有没有多帧”，而在“多帧 history 怎么进入 state 并被 policy 正确利用”。
- 代码结构混乱：
  现在代码虽然多，但主流程已经能明确追踪，不是最紧急问题。


## 4. 对应代码

下面是这次结果真正直接用到的核心文件。

### 4.1 主仿真和数据生成入口

- detector dataset 生成主脚本：
  `matlab/simulation/run_generate_oampnet_paper_tsv2seq_vdop_ctrl.m`
- detector dataset 生成核心实现：
  `matlab/simulation/generate_dataset_timescaling_n256.m`
- offline oracle/policy dataset 导出入口：
  `matlab/detectors/run_export_oracle_policy_paper_tsv2seq_vdop_ctrl.m`
- offline oracle/policy dataset 导出核心实现：
  `matlab/detectors/export_oracle_dataset_for_policy.m`
- online policy rollout 入口：
  `matlab/detectors/run_online_policy_oamp_oampnet_paper_tsv2seq_vdop_ctrl.m`
- online policy rollout 核心实现：
  `matlab/detectors/run_online_policy_oamp_oampnet.m`


### 4.2 信道生成和时变信道状态演化

- 时变信道默认参数：
  `matlab/common/get_timevary_defaults.m`
- 初始化时变信道状态：
  `matlab/common/init_timevary_channel_state.m`
- 逐帧推进时变信道状态：
  `matlab/common/step_timevary_channel_state.m`
- 在线 state 特征提取：
  `matlab/common/extract_online_state_features.m`


### 4.3 AFDM 等效信道与构造部分

这部分分散在以下文件里：

- `matlab/simulation/generate_dataset_timescaling_n256.m`
  - `precompute_idaf_basis`
  - `add_cpp_cps_matrix`
  - `chirp1 / chirp2`
  - AFDM 发射/扩展基构造
- `matlab/detectors/export_oracle_dataset_for_policy.m`
  - `afdm_demod_matrix`
  - 不同 `C1` 下的 `Heff` 构造和 BER 表导出


### 4.4 `C1` 选择 / 扫描 / oracle 标注

- 当前 policy 学习数据导出与 reward/oracle 标注：
  `matlab/detectors/export_oracle_dataset_for_policy.m`
- 静态/时变 sanity check：
  `matlab/detectors/sanity_check_c1_sweep.m`
  `matlab/detectors/sanity_check_c1_sweep_refine.m`
  `matlab/detectors/sanity_check_c1_sweep_timevary.m`


### 4.5 检测器代码

- LMMSE：
  `matlab/detectors/lmmse_detector.m`
- OAMP：
  `matlab/detectors/oamp_detector.m`
- OAMPNet：
  `matlab/detectors/oampnet_detector.m`


### 4.6 RL policy 训练、评估和数据构造

- state 特征定义与 feature name 工具：
  `python/rl_c1/features.py`
- offline dataset loader：
  `python/rl_c1/env_c1_bandit.py`
- DAgger rollout 数据构造：
  `python/rl_c1/build_dagger_rollout_dataset.py`
- mixed dataset 构造：
  `python/rl_c1/build_mixed_policy_dataset.py`
- policy 网络：
  `python/rl_c1/policy_net.py`
- RL 训练：
  `python/rl_c1/train_reinforce.py`
- policy 评估：
  `python/rl_c1/eval_policy.py`
- 序列级评估：
  `python/rl_c1/eval_sequence_policy.py`
- 标准图导出：
  `python/rl_c1/export_standard_figures.py`


### 4.7 OAMPNet 训练和 loss

- OAMPNet 训练脚本：
  `python/training/train_oampnet_v4.py`
- loss 定义：
  `python/training/losses.py`
- OAMPNetV4 模型：
  `python/models/oampnet_v4.py`


## 5. 最关键的结论数字

当前推荐主模型：

- `mix_r1_tinyrl`

关键数：

- 原始数据上：
  `ber_gain_vs_base = 1.074%`
- `dagger_r1` 数据上：
  `ber_gain_vs_base = 1.110%`
- `SNR_12dB policy_recovery_fraction = 9.16%`
- action 11 频率：
  `20.445%`
- action 1 频率：
  `51.28%`

对照：

- 原 baseline 在原始数据上：
  `1.121%`
- 原 baseline 直接测 `dagger_r1`：
  `0.247%`

这说明：

- 这轮已经证明 history mismatch 的判断是正确的
- mixed training 是有效方向
- 但当前最好版本仍然没有把动态 C1 增益完整兑现成稳定超过 baseline 的最终 BER 提升

