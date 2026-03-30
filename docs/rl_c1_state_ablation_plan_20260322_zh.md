# RL C1 第一轮 State Ablation 计划清单

日期：2026-03-22

## 1. 目标

这一轮不追求立刻刷新最好 BER，而是先验证一个更基础的问题：

当前策略是不是主要依赖了脆弱的 history shortcut，
从而在训练分布和 on-policy rollout 分布不一致时失真。

核心判断标准不是单看某一个绝对 BER，而是看：

- `orig` 和 `dagger_r1` 之间的性能 gap 是否缩小
- 动作分布是否仍然明显塌向某一个动作
- `switch_rate_policy` 是否回到更合理的区间
- `policy_recovery_fraction` 是否还能保持


## 2. 这轮只做 4 组 state 变体

### 2.1 Control：`full_state`

定义：

- 使用当前完整 20 维 state
- 即当前 best run 的同一套 state schema

作用：

- 作为对照组
- 直接复用已有结果，不重复训练

对应模型：

- `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl`


### 2.2 Variant A：`prev_action_only`

保留：

- 所有当前帧可观测特征
- `prev_action_norm`
- `prev_residual_proxy`
- 所有 temporal delta

去掉：

- `prev_reward`
- `recent_switch_rate`

目的：

- 验证是否仅保留“上一步动作”就足以维持主要收益
- 检查真正导致 shortcut 的是不是 reward/switch history


### 2.3 Variant B：`no_history`

保留：

- 所有当前帧可观测特征
- `prev_residual_proxy`
- 所有 temporal delta

去掉：

- `prev_action_norm`
- `prev_reward`
- `recent_switch_rate`

目的：

- 直接测试在不喂 policy-history 的情况下，`orig` 和 `dagger` gap 会不会显著缩小
- 这是验证 history shortcut 最关键的一组


### 2.4 Variant C：`prev_action_only_plus_physical_delta`

在 `prev_action_only` 基础上，再加入：

- `alpha_com`
- `v_norm`
- `delta_alpha_rms`
- `abs_alpha_com`
- `delta_alpha_com`
- `delta_v_norm`
- `delta_delta_alpha_rms`

目的：

- 测试“减少脆弱 history 依赖”之后，补充物理 Doppler 变化量能否把收益再拉回来


## 3. 数据与训练口径

### 3.1 数据源

统一使用：

- 原始数据：
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat`
- 固定的 shifted probe：
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz`

说明：

- 这一轮把 `dagger_r1` 当作统一的 on-policy 分布探针
- 目的是先做诊断，不重新定义 rollout fixed point
- 因此不先为每个变体重做各自的 on-policy rollout


### 3.2 训练配方

`full_state` 对照组直接复用现有最佳 mixed recipe：

- mixed dataset：original + `dagger_r1`
- reward：`reward_relbase_proxy`
- split：`sequence`
- batch size：`256`
- epochs：`20`
- `reinforce_lr = 1e-4`
- `imitation_preserve_coef = 0.5`

其余 3 个变体使用完全相同的训练配方，只改变 state。


## 4. 具体执行步骤

### Step 1. 补齐 state-variant builder 的 feature 级选列能力

状态：

- 已完成

目的：

- 让脚本支持按 feature 名精确构建
  `prev_action_only / no_history / prev_action_only_plus_physical_delta`


### Step 2. 构建 3 组新 variant 数据

每组都构建：

- `orig variant`
- `dagger variant`
- `mixed variant`

命名约定：

- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_<variant>.npz`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1_<variant>.npz`
- `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1_<variant>.npz`


### Step 3. 训练 3 个 mixed model

输出目录：

- `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only`
- `results/paper_tsv2seq_vdop_ctrl_mix_r1_no_history`
- `results/paper_tsv2seq_vdop_ctrl_mix_r1_prev_action_only_physical_delta`


### Step 4. 统一评估

每个变体都在两个评测集上评估：

- variant original
- variant `dagger_r1`

主要收集：

- `ber_gain_vs_base`
- `switch_rate_policy`
- `match_rate`
- `SNR_12dB policy_recovery_fraction`
- 动作分布 top-1


### Step 5. 汇总结论

最终要回答 3 个问题：

1. 去掉 `prev_reward / recent_switch_rate` 后，`orig-dagger` gap 是否显著缩小
2. 完全去掉 policy history 后，绝对性能还剩多少
3. 物理 Doppler delta 能否部分替代被移除的 history 信息


## 5. 成功 / 失败判据

### 5.1 本轮算成功

满足以下任一情况即可认为“诊断有效”：

- `no_history` 的 `orig-dagger` gap 明显小于 `full_state`
- `prev_action_only` 明显优于 `full_state` 的 shifted robustness
- `prev_action_only_plus_physical_delta` 在保持 robustness 的同时追回部分 BER


### 5.2 本轮若出现以下结果，则直接推进第二轮

- `no_history` 基本抹平 `orig-dagger` gap，但绝对 BER 降得较多

则下一轮优先做：

- GRU / temporal policy
- 或 detector uncertainty 特征增强


### 5.3 本轮若出现以下结果，则 reward 方向优先

- 变体之间 robustness 差异不大，但都偏静态、都不超过 baseline

则下一轮优先做：

- 按 `oracle gap` 加权
- 更直接的序列级收益目标

