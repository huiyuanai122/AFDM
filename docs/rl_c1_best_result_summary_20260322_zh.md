# RL C1 当前最佳结果总结

日期：2026-03-22

## 1. 结论先行

当前这条 RL 优化 C1 的实验线，有两个“最好”的答案，需要区分评价口径：

1. 如果只看原始离线基准集
   `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat`，
   目前分数最高的仍然是原始 baseline policy：
   `results/paper_tsv2seq_vdop_ctrl/best_reinforce_policy_tsv2seq_vdop_ctrl_paper.pt`

   对应指标：
   `ber_gain_vs_base = 1.121%`

2. 如果看“原始基准集表现 + 对 on-policy history mismatch 的鲁棒性”这个综合标准，
   当前最值得继续推进的是：
   `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt`

   对应指标：
   - 在原始 `paper.mat` 上：`ber_gain_vs_base = 1.074%`
   - 在 `dagger_r1` rollout 数据上：`ber_gain_vs_base = 1.110%`
   - `SNR_12dB policy_recovery_fraction = 9.16%`
   - action 11 塌缩从 baseline 的 `46.88%` 降到 `20.445%`

当前建议：

- 保留 `mix_r1_tinyrl` 作为后续继续迭代的主模型。
- 不建议继续盲目做 `r3 / r4` 这种纯 rollout 深化。
- `mix_r2_tinyrl` 已验证，没有超过 `mix_r1_tinyrl`。


## 2. 为什么要改这条线

这轮改动针对的是最核心的 train-test mismatch：

- 离线导出的 state history 来自固定 `base_action` 轨迹。
- 在线部署时，history 是策略自己动作滚出来的。
- 因此这 3 个历史特征在训练和部署时分布不一致：
  - `prev_action_norm`
  - `prev_reward`
  - `recent_switch_rate`

这会导致：

- 训练时学到的策略，在真实 online history 下失真。
- 直接把原 policy 放到 on-policy rollout 状态分布上时，性能明显掉点，并出现动作塌缩。

最直接的证据是：

- 原 baseline checkpoint 在原始数据上：
  `ber_gain_vs_base = 1.121%`
- 同一个 checkpoint 直接放到 `dagger_r1` 数据上：
  `ber_gain_vs_base = 0.247%`

这说明 history mismatch 的判断是对的，而且影响很大。


## 3. 这轮具体做了什么

### 3.1 加了 feature name 到列号的索引工具

文件：

- `python/rl_c1/features.py`

新增能力：

- `canonicalize_feature_names(...)`
- `feature_name_to_index(...)`
- `feature_index(...)`

目的：

- 不再硬编码 state 列号。
- 后续脚本可以通过名字准确定位
  `prev_action_norm / prev_reward / recent_switch_rate`。


### 3.2 让离线数据 loader 能读取 `feature_names`

文件：

- `python/rl_c1/env_c1_bandit.py`

新增能力：

- `.npz` 和 `.mat` 都能读出 `feature_names`
- `OfflineBanditData` 增加 `feature_names` 字段

目的：

- 让后续数据重写逻辑安全、可复用，不依赖列号硬编码。


### 3.3 新增 DAgger rollout 数据构建脚本

文件：

- `python/rl_c1/build_dagger_rollout_dataset.py`

作用：

- 输入原始 offline dataset 和当前 policy checkpoint
- 按 `sequence_id`、`time_index` 顺序 rollout
- 只重写 state 里的 3 个历史特征：
  - `prev_action_norm`
  - `prev_reward`
  - `recent_switch_rate`
- reward table、oracle label、BER 表、其他物理特征全部复用原始数据

每条序列的初始化方式：

- `prev_action = base_action`
- `prev_reward = 0`
- `switch_hist = 0`

每一帧的更新逻辑：

1. 先把当前历史值写入 state
2. 用当前 checkpoint 对这一帧做 greedy action
3. 用 `rewards[t, action]` 更新 `prev_reward`
4. 用 `action != prev_action` 更新 switch history
5. 进入下一帧


### 3.4 先做 pure DAgger，再做 mixed training

第一轮先直接在 `dagger_r1` 上训练两档：

- `imitation_only`
- `tiny RL`

结果发现：

- history mismatch 被明显修到了
- action 11 的塌缩也被打掉了
- 但 pure on-policy 数据把分布推得太猛
- 新偏置转移到了 action 1
- 最终 BER 没有超过原 baseline

因此第二轮改成 mixed training：

- 把 original dataset 和 `dagger_r1` dataset 混合
- 再训练：
  - `mix_r1_imonly`
  - `mix_r1_tinyrl`

为此新增脚本：

- `python/rl_c1/build_mixed_policy_dataset.py`

它的作用是：

- 对齐 original / on-policy 数据 schema
- 拼成一个新的 mixed dataset
- 保持 sequence split 兼容
- 额外记录样本来源：
  - `sample_source_code`
  - `sample_source_name`


### 3.5 做了 r2 验证，但没有继续变好

后续继续基于 `mix_r1_tinyrl` 做了：

- `dagger_r2`
- `mix_r2_tinyrl`

结果：

- `mix_r2_tinyrl` 比 `mix_r1_tinyrl` 更差
- 所以当前不建议继续堆 rollout 轮数


## 4. 关键结果对比

| 方案 | 评测集 | ber_gain_vs_base |
| --- | --- | --- |
| 原 baseline policy | 原始 `paper.mat` | `1.121%` |
| baseline policy 直接测 `dagger_r1` | `dagger_r1` | `0.247%` |
| `dagger_r1_imonly` | `dagger_r1` | `0.690%` |
| `dagger_r1_tinyrl` | `dagger_r1` | `0.840%` |
| `dagger_r1_tinyrl` | 原始 `paper.mat` | `0.458%` |
| `mix_r1_imonly` | 原始 `paper.mat` | `0.902%` |
| `mix_r1_tinyrl` | 原始 `paper.mat` | `1.074%` |
| `mix_r1_tinyrl` | `dagger_r1` | `1.110%` |
| `mix_r2_tinyrl` | 原始 `paper.mat` | `0.866%` |
| `mix_r2_tinyrl` | `dagger_r2` | `0.902%` |

可以看到：

- pure DAgger 证明了问题方向是对的，但单独使用不够好。
- mixed training 明显比 pure DAgger 更稳。
- `mix_r1_tinyrl` 是当前综合最优。
- `r2` 继续 rollout 后没有提升，反而退步。


## 5. 当前最好版本为什么是 `mix_r1_tinyrl`

`mix_r1_tinyrl` 虽然在原始 benchmark 上还略低于最初 baseline，
但它是当前最合理的折中点，原因有 4 个：

1. 在原始数据上，它已经基本追平 baseline
   `1.074%` vs `1.121%`

2. 在 on-policy 数据上，它能保持住收益
   `1.110%`

3. 它显著修复了 history mismatch 带来的退化

4. 它比 pure DAgger 更稳，没有把模型完全推到新的单一动作模式

同时，它还有两个明确的残留问题：

- 仍然偏向 action 1
- 整体 switch rate 偏低，策略仍偏静态


## 6. 动作分布和行为指标怎么看

### 6.1 action 11 塌缩被明显缓解

从动作分布看：

- 原 baseline：action 11 频率 `46.88%`
- baseline 直接测 `dagger_r1`：action 11 频率 `87.955%`
- `mix_r1_tinyrl`：action 11 频率 `20.445%`

这说明：

- 旧模型在新 history 分布下会严重塌向 action 11
- mixed training 已经把这个问题显著修掉


### 6.2 但新的偏置转到了 action 1

`mix_r1_tinyrl` 的 top action 变成了 action 1：

- action 1 频率 `51.28%`

所以现在不是“完全不塌缩”，而是：

- 从原先对 action 11 的塌缩
- 变成了对 action 1 的偏置更强

只是这个偏置比原先更可控，BER 结果也更稳。


### 6.3 12 dB 的恢复率明显提高

`SNR_12dB policy_recovery_fraction`：

- baseline：`1.71%`
- `mix_r1_tinyrl`：`9.16%`

这说明在关键中低 SNR 段，策略的自适应恢复能力有明显改善。


## 7. 现在最值得看的图

### 7.1 “优化 C1 和不优化的对比图”

最推荐直接看这张：

- `figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/static_dynamic_baselines_tsv2seq_vdop_ctrl_mix_r1_tinyrl.png`

它把以下几条曲线放在一起：

- `fixed_base`
  不优化 C1，始终用固定 base action
- `best_static_snr`
  每个 SNR 选一个固定最优动作
- `best_static_sequence`
  每条序列选一个固定最优动作
- `policy_dynamic`
  当前 RL policy 动态选 C1
- `oracle_dynamic`
  每帧选最优动作的 oracle 上界

其中最关键的是：

- `fixed_base`：不优化 C1
- `policy_dynamic`：当前最好 RL 优化 C1


### 7.2 主 BER 曲线图

- `figures/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/fig3_ber_vs_snr_main.png`

这张图可以看最终 BER-SNR 走势，但需要注意：

- 当前这版图在没有 MATLAB online detector CSV 的情况下，
  是基于已有 policy gain 做的映射图
- 可以用于结果展示和趋势判断
- 但不应把它当成完整在线联调后的最终 detector-side 实测结论


## 8. 关键产物路径

### 8.1 当前推荐 checkpoint

- `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl/best_reinforce_policy_tsv2seq_vdop_ctrl_mix_r1_tinyrl.pt`


### 8.2 关键数据

- 原始数据：
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat`
- 第一轮 rollout 数据：
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz`
- mixed 数据：
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1.npz`
- 第二轮 rollout 数据：
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r2.npz`
- 第二轮 mixed 数据：
  `data/oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r2.npz`


### 8.3 关键结果目录

- baseline：
  `results/paper_tsv2seq_vdop_ctrl`
- pure DAgger：
  `results/paper_tsv2seq_vdop_ctrl_dagger_r1_imonly`
  `results/paper_tsv2seq_vdop_ctrl_dagger_r1_tinyrl`
- mixed：
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_imonly`
  `results/paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl`
  `results/paper_tsv2seq_vdop_ctrl_mix_r2_tinyrl`


## 9. 复现链路

### 9.1 构建 `dagger_r1`

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe python\rl_c1\build_dagger_rollout_dataset.py `
  --input data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat `
  --checkpoint results\paper_tsv2seq_vdop_ctrl\best_reinforce_policy_tsv2seq_vdop_ctrl_paper.pt `
  --output data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz `
  --reward_key reward_relbase_proxy
```


### 9.2 训练 pure DAgger

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe python\rl_c1\train_reinforce.py `
  --dataset data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz `
  --reward-key reward_relbase_proxy `
  --outdir results\paper_tsv2seq_vdop_ctrl_dagger_r1_imonly `
  --run-name tsv2seq_vdop_ctrl_dagger_r1_imonly `
  --split-mode sequence `
  --imitation-only
```

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe python\rl_c1\train_reinforce.py `
  --dataset data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz `
  --reward-key reward_relbase_proxy `
  --outdir results\paper_tsv2seq_vdop_ctrl_dagger_r1_tinyrl `
  --run-name tsv2seq_vdop_ctrl_dagger_r1_tinyrl `
  --split-mode sequence `
  --tiny-rl
```


### 9.3 构建 mixed dataset 并训练当前最佳版

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe python\rl_c1\build_mixed_policy_dataset.py `
  --original data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper.mat `
  --onpolicy data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_dagger_r1.npz `
  --output data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1.npz
```

```powershell
C:\Users\MYCZ\.conda\envs\pytorch\python.exe python\rl_c1\train_reinforce.py `
  --dataset data\oracle_policy_dataset_tsv2seq_vdop_ctrl_paper_mix_r1.npz `
  --reward-key reward_relbase_proxy `
  --outdir results\paper_tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --run-name tsv2seq_vdop_ctrl_mix_r1_tinyrl `
  --split-mode sequence `
  --tiny-rl
```


## 10. 最终建议

当前阶段，最合理的结论是：

- 这轮 work 已经证明 history mismatch 是主要问题之一。
- pure DAgger 只能证明方向对，不能直接作为最好结果。
- 把 original 和 on-policy 数据混合后，效果明显更稳。
- `mix_r1_tinyrl` 是当前最适合继续推进、继续做后续 ablation 和图表汇报的版本。

如果下一步还要继续做，我建议优先方向是：

- 调整 mixed dataset 的采样比例
- 对动作分布加防塌缩约束
- 提升 switch 行为，而不是继续盲目加 rollout 轮数

