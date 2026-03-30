# RL-C1 Detector Headroom Diagnostic (2026-03-23)

## 1. 目的

在继续优化 RL 之前，先回答一个更关键的问题：

- `RL 没学到`，还是
- `当前 detector 下 C1 的可优化空间本来就很小`

为此，补齐并检查两组 detector-specific 的 4 条 BER-SNR 曲线：

- static baseline C1
- static best single C1
- oracle dynamic C1
- RL dynamic C1

对应输出文件：

- `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/c1_detector_diagnostic_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin.csv`
- `figures/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/c1_detector_diagnostic_oamp_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin.png`
- `figures/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/c1_detector_diagnostic_oampnet_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin.png`

## 2. 结论先行

这次诊断给出的结论很明确：

1. `不是 C1 完全没有 headroom`。
2. `在 OAMP 和 OAMPNet 下，oracle dynamic 相对 static best 都还有明显空间`，尤其在 `8-14 dB` 中高 SNR 段。
3. `当前 RL dynamic 几乎全段都没追上 static best`，很多 SNR 点甚至比 `static best` 更差，部分点还差于 `static baseline`。
4. 因此，当前问题更接近 `RL 没有把已有的动态 headroom 学出来`，而不是 `这个 detector 下动态 C1 根本不值得做`。
5. 但在 `16-20 dB`，两类 detector 都开始进入近似饱和区，dynamic C1 的额外价值快速缩小。

## 3. OAMP 观察

`oracle dynamic vs static best` 仍有明显空间：

- `8 dB`: `0.015332 -> 0.006348`
- `10 dB`: `0.002490 -> 0.000187`
- `12 dB`: `0.000580 -> 0.000004`
- `14 dB`: `0.000137 -> 0.000014`

对应地，`oracle dynamic` 相对 `static best` 的收益在 `8-14 dB` 大约是：

- `58.6%`
- `92.5%`
- `99.3%`
- `90.0%`

但当前 `RL dynamic` 相对 `static best` 反而为负：

- `8 dB`: `-11.8%`
- `10 dB`: `-11.1%`
- `12 dB`: `-9.8%`
- `14 dB`: `-5.7%`

这说明在 OAMP 下，`dynamic C1 值得做`，但 `当前 RL policy 还没有学到对的切换策略`。

## 4. OAMPNet 观察

`oracle dynamic vs static best` 的 headroom 同样不小，而且在关键中高 SNR 段更明显：

- `8 dB`: `0.010571 -> 0.003687`
- `10 dB`: `0.001554 -> 0.000024`
- `12 dB`: `0.000182 -> 0.000000`
- `14 dB`: `0.000021 -> 0.000000`

对应 `oracle dynamic` 相对 `static best` 的收益在 `8-14 dB` 约为：

- `65.1%`
- `98.4%`
- `100%`
- `100%`

而当前 `RL dynamic` 相对 `static best` 仍显著为负：

- `8 dB`: `-23.1%`
- `10 dB`: `-19.4%`
- `12 dB`: `-32.3%`
- `14 dB`: `-154.5%`

所以 OAMPNet 这边也不是 `headroom 太小`，而是 `RL 远未逼近 oracle dynamic`。

## 5. 对后续方向的直接影响

这一步已经足够决定后续主线：

- 不应该把当前结果解释成 `动态 C1 在 OAMP / OAMPNet 下没有价值`
- 更合理的解释是 `已有动态价值，但平均 BER 目标下，policy 没把关键样本学出来`

因此下一步优先级应该放在：

1. 强化 `RL dynamic` 对 `static best` 的超越，而不是继续加复杂 policy 结构。
2. 针对 `8-14 dB` 这段真正有 headroom 的区间做更有针对性的训练/评估。
3. 重新检查 reward / sample weighting / per-SNR emphasis，避免高占比但低价值样本把真正有价值的切换样本冲淡。

## 6. 备注

- 本次结果来自 `mix_r3_prev_action_only_physical_delta_gapw_margin` 当前 best checkpoint 的正式 online detector evaluation。
- figure manifest 已同步更新：
  - `results/paper_tsv2seq_vdop_ctrl_mix_r3_prev_action_only_physical_delta_gapw_margin/figure_export_manifest.json`
