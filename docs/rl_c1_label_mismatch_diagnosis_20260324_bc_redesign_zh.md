# RL C1 Label Mismatch Diagnosis

- Date: `20260324_bc_redesign`
- Device: `cuda`
- Train data: `data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r3_prev_action_only_physical_delta.npz`
- Rollout data: `data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_prev_action_only_physical_delta_dagger_r3.npz`
- Focus SNR: `8, 10, 12, 14`

## A1 Oracle action distribution

- `oracle_action=1` share on all train samples: 47.41%
- `oracle_action=1` share on positive-only train samples: 10.59%
- `oracle_action=1` share at `12 dB`: 58.15%
- `oracle_action=1` share at `14 dB`: 93.15%

## A2 gap / margin

- Train positive rate: 52.22%
- `gap_static` median on all train samples: `3.255208e-04`
- `gap_static` median on positive train samples: `1.171875e-02`
- `margin_oracle` median on all train samples: `0.000000e+00`
- `margin_oracle` median on positive train samples: `1.953125e-03`

## A3 checkpoint alignment

- Existing B1 artifact has per-epoch history: `False`
- Best epoch by `val_ber`: `13`
- Best epoch by rollout `closure_8to14`: `20`
- Best epoch by rollout `rl_match_8to14`: `18`
- Corr(`val_ber`, rollout `closure_8to14`): `-0.4539`
- Corr(`val_ber`, rollout `c12`): `-0.0451`

## A4 probe comparison

- `probe_oracle_action`: val_acc=46.14%, top3=55.20%, majority=46.50%, lift=-0.36%
- `probe_oracle_beats_static_best`: val_bal_acc=93.22%, focus_bal_acc=84.46%, majority_bal_acc=50.00%, lift=43.22%

## Diagnosis

- `argmin-BER` 标签存在明显静态偏置，尤其在 `12/14 dB` 上会强烈收缩到低编号动作，`action 1` 仍占主导。
- 这种偏置在正样本里没有消失，只是被削弱；说明“有收益的样本”并不等价于“需要学成全局多数类”。
- `gap_static` 和 `margin_oracle` 的中位数都偏小，很多样本虽然存在硬标签，但 headroom 或可分离性并不大，普通 CE 很容易退化成先验拟合。
- 现有 B1 artifact 没有保存逐 epoch 口径，工具链本身不足以直接回答 checkpoint 对齐问题；因此这里用同配置 replay 恢复 epoch 曲线。
- 在 replay 中，`val_ber` 的最优 epoch 与 rollout `closure/c10/c12` 的最优 epoch 不一致，说明按 `val_ber` 选 checkpoint 会错过真正关心的 downstream 指标。
- probe 结果用于比较“切不切”和“切到哪”两个目标；如果 binary probe 更稳，下一阶段应该先强化 worth-switch 信号，再学习具体动作分布。

## Artifacts

- Core table: `results\label_mismatch_diagnosis_20260324_bc_redesign\label_mismatch_core_table.csv`
- Oracle distribution: `results\label_mismatch_diagnosis_20260324_bc_redesign\oracle_action_distribution.csv`
- Gap/margin summary: `results\label_mismatch_diagnosis_20260324_bc_redesign\gap_margin_distribution.csv`
- Checkpoint alignment: `results\label_mismatch_diagnosis_20260324_bc_redesign\checkpoint_metric_alignment.csv`
- Probe results: `results\label_mismatch_diagnosis_20260324_bc_redesign\probe_results.csv`