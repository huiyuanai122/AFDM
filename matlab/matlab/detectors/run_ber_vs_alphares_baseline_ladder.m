function out = run_ber_vs_alphares_baseline_ladder(user_cfg)
%RUN_BER_VS_ALPHARES_BASELINE_LADDER
% BER vs alpha_max_res with a stronger baseline ladder under raw/residual
% time-scaling. Policies:
% 1) dynamic_oracle
% 2) dynamic_formula
% 3) dynamic_formula_quantized
% 4) static_oracle_best
% 5) static_mean
% 6) static_max
% 7) static_min

if nargin < 1
    user_cfg = struct();
end

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
common_dir = fullfile(project_root, 'matlab', 'common');
addpath(this_dir);
addpath(common_dir);

cfg = default_cfg(project_root);
cfg = merge_struct(cfg, user_cfg);
cfg.alpha_raw_list = cfg.alpha_raw_scale * cfg.alpha_res_list;
if ~exist(cfg.output_dir, 'dir')
    mkdir(cfg.output_dir);
end

fprintf('==== BER vs alpha_max_res: baseline ladder experiment ====\n');
fprintf('snr_db=%d, num_seq=%d, num_frames=%d, detector=%s\n', ...
    cfg.snr_db, cfg.num_seq, cfg.num_frames, cfg.detector_type);

num_alpha = numel(cfg.alpha_res_list);
ber_dynamic_oracle = zeros(num_alpha, 1);
ber_dynamic_formula = zeros(num_alpha, 1);
ber_dynamic_formula_quantized = zeros(num_alpha, 1);
ber_static_oracle_best = zeros(num_alpha, 1);
ber_static_mean = zeros(num_alpha, 1);
ber_static_max = zeros(num_alpha, 1);
ber_static_min = zeros(num_alpha, 1);

seq_level_policy_stats = repmat(struct( ...
    'alpha_max_res_cfg', 0.0, ...
    'alpha_max_raw_cfg', 0.0, ...
    'cfg_action_grid', [], ...
    'frame_alpha_eff', [], ...
    'seq_alpha_eff_max', [], ...
    'seq_alpha_eff_min', [], ...
    'seq_alpha_eff_mean', [], ...
    'seq_alpha_eff_std', [], ...
    'dynamic_oracle_best_idx_frame', [], ...
    'dynamic_oracle_best_c1_frame', [], ...
    'ber_dynamic_oracle_frame', [], ...
    'biterr_dynamic_oracle_frame', [], ...
    'dynamic_formula_c1_frame', [], ...
    'ber_dynamic_formula_frame', [], ...
    'biterr_dynamic_formula_frame', [], ...
    'dynamic_formula_quantized_idx_frame', [], ...
    'dynamic_formula_quantized_c1_frame', [], ...
    'ber_dynamic_formula_quantized_frame', [], ...
    'biterr_dynamic_formula_quantized_frame', [], ...
    'static_oracle_best_idx_seq', [], ...
    'static_oracle_best_c1_seq', [], ...
    'static_oracle_seq_center_c1', [], ...
    'ber_static_oracle_best_frame', [], ...
    'biterr_static_oracle_best_frame', [], ...
    'static_mean_c1_seq', [], ...
    'static_max_c1_seq', [], ...
    'static_min_c1_seq', [], ...
    'seq_ber_dynamic_oracle', [], ...
    'seq_ber_dynamic_formula', [], ...
    'seq_ber_dynamic_formula_quantized', [], ...
    'seq_ber_static_oracle_best', [], ...
    'seq_ber_static_mean', [], ...
    'seq_ber_static_max', [], ...
    'seq_ber_static_min', []), num_alpha, 1);

for ai = 1:num_alpha
    cfg_alpha = cfg;
    cfg_alpha.alpha_max_res = cfg.alpha_res_list(ai);
    cfg_alpha.alpha_max_raw = cfg.alpha_raw_list(ai);

    sys = build_system_cfg(cfg_alpha);
    tv_cfg = build_timevary_cfg(cfg_alpha);
    cfg_action_grid = compute_c1_base_from_alpha(cfg_alpha.alpha_max_res, sys) * cfg_alpha.c1_ratios(:).';

    seq_ber_dynamic = zeros(cfg_alpha.num_seq, 1);
    seq_ber_dynamic_formula = zeros(cfg_alpha.num_seq, 1);
    seq_ber_dynamic_formula_quantized = zeros(cfg_alpha.num_seq, 1);
    seq_ber_static_oracle_best = zeros(cfg_alpha.num_seq, 1);
    seq_ber_static_mean = zeros(cfg_alpha.num_seq, 1);
    seq_ber_static_max = zeros(cfg_alpha.num_seq, 1);
    seq_ber_static_min = zeros(cfg_alpha.num_seq, 1);

    frame_alpha_eff = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);
    best_idx_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);
    best_c1_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);
    ber_dynamic_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);
    biterr_dynamic_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);

    dyn_formula_c1_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);
    ber_dynamic_formula_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);
    biterr_dynamic_formula_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);

    dyn_formula_quantized_idx_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);
    dyn_formula_quantized_c1_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);
    ber_dynamic_formula_quantized_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);
    biterr_dynamic_formula_quantized_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);

    static_oracle_best_idx_seq = zeros(cfg_alpha.num_seq, 1);
    static_oracle_best_c1_seq = zeros(cfg_alpha.num_seq, 1);
    static_oracle_seq_center_c1 = zeros(cfg_alpha.num_seq, 1);
    ber_static_oracle_best_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);
    biterr_static_oracle_best_frame = zeros(cfg_alpha.num_seq, cfg_alpha.num_frames);

    static_mean_c1_seq = zeros(cfg_alpha.num_seq, 1);
    static_max_c1_seq = zeros(cfg_alpha.num_seq, 1);
    static_min_c1_seq = zeros(cfg_alpha.num_seq, 1);

    seq_alpha_max = zeros(cfg_alpha.num_seq, 1);
    seq_alpha_min = zeros(cfg_alpha.num_seq, 1);
    seq_alpha_mean = zeros(cfg_alpha.num_seq, 1);
    seq_alpha_std = zeros(cfg_alpha.num_seq, 1);

    fprintf('\nalpha_max_res=%.2e, alpha_max_raw=%.2e\n', ...
        cfg_alpha.alpha_max_res, cfg_alpha.alpha_max_raw);

    for sid = 1:cfg_alpha.num_seq
        seq_data = generate_sequence_data_light(sid, cfg_alpha, sys, tv_cfg);
        seq_summary = analyze_sequence_policy_bundle(seq_data, cfg_alpha, sys, cfg_action_grid);

        seq_ber_dynamic(sid) = seq_summary.seq_ber_dynamic_oracle;
        seq_ber_dynamic_formula(sid) = seq_summary.seq_ber_dynamic_formula;
        seq_ber_dynamic_formula_quantized(sid) = seq_summary.seq_ber_dynamic_formula_quantized;
        seq_ber_static_oracle_best(sid) = seq_summary.seq_ber_static_oracle_best;
        seq_ber_static_mean(sid) = seq_summary.seq_ber_static_mean;
        seq_ber_static_max(sid) = seq_summary.seq_ber_static_max;
        seq_ber_static_min(sid) = seq_summary.seq_ber_static_min;

        frame_alpha_eff(sid, :) = seq_summary.alpha_eff(:).';
        best_idx_frame(sid, :) = seq_summary.dynamic_oracle_best_idx_frame(:).';
        best_c1_frame(sid, :) = seq_summary.dynamic_oracle_best_c1_frame(:).';
        ber_dynamic_frame(sid, :) = seq_summary.ber_dynamic_oracle_frame(:).';
        biterr_dynamic_frame(sid, :) = seq_summary.biterr_dynamic_oracle_frame(:).';

        dyn_formula_c1_frame(sid, :) = seq_summary.dynamic_formula_c1_frame(:).';
        ber_dynamic_formula_frame(sid, :) = seq_summary.ber_dynamic_formula_frame(:).';
        biterr_dynamic_formula_frame(sid, :) = seq_summary.biterr_dynamic_formula_frame(:).';

        dyn_formula_quantized_idx_frame(sid, :) = seq_summary.dynamic_formula_quantized_idx_frame(:).';
        dyn_formula_quantized_c1_frame(sid, :) = seq_summary.dynamic_formula_quantized_c1_frame(:).';
        ber_dynamic_formula_quantized_frame(sid, :) = seq_summary.ber_dynamic_formula_quantized_frame(:).';
        biterr_dynamic_formula_quantized_frame(sid, :) = seq_summary.biterr_dynamic_formula_quantized_frame(:).';

        static_oracle_best_idx_seq(sid) = seq_summary.static_oracle_best_idx_seq;
        static_oracle_best_c1_seq(sid) = seq_summary.static_oracle_best_c1_seq;
        static_oracle_seq_center_c1(sid) = seq_summary.static_oracle_seq_center_c1;
        ber_static_oracle_best_frame(sid, :) = seq_summary.ber_static_oracle_best_frame(:).';
        biterr_static_oracle_best_frame(sid, :) = seq_summary.biterr_static_oracle_best_frame(:).';

        static_mean_c1_seq(sid) = seq_summary.static_mean_c1_seq;
        static_max_c1_seq(sid) = seq_summary.static_max_c1_seq;
        static_min_c1_seq(sid) = seq_summary.static_min_c1_seq;

        seq_alpha_max(sid) = seq_summary.seq_alpha_eff_max;
        seq_alpha_min(sid) = seq_summary.seq_alpha_eff_min;
        seq_alpha_mean(sid) = seq_summary.seq_alpha_eff_mean;
        seq_alpha_std(sid) = seq_summary.seq_alpha_eff_std;

        if mod(sid, cfg_alpha.progress_every_seq) == 0 || sid == cfg_alpha.num_seq
            fprintf(['  progress %4d/%4d | dyn_oracle=%.3e | dyn_formula=%.3e | ' ...
                     'static_oracle_best=%.3e\n'], ...
                sid, cfg_alpha.num_seq, ...
                mean(seq_ber_dynamic(1:sid)), ...
                mean(seq_ber_dynamic_formula(1:sid)), ...
                mean(seq_ber_static_oracle_best(1:sid)));
        end
    end

    ber_dynamic_oracle(ai) = mean(seq_ber_dynamic);
    ber_dynamic_formula(ai) = mean(seq_ber_dynamic_formula);
    ber_dynamic_formula_quantized(ai) = mean(seq_ber_dynamic_formula_quantized);
    ber_static_oracle_best(ai) = mean(seq_ber_static_oracle_best);
    ber_static_mean(ai) = mean(seq_ber_static_mean);
    ber_static_max(ai) = mean(seq_ber_static_max);
    ber_static_min(ai) = mean(seq_ber_static_min);

    seq_level_policy_stats(ai) = struct( ...
        'alpha_max_res_cfg', cfg_alpha.alpha_max_res, ...
        'alpha_max_raw_cfg', cfg_alpha.alpha_max_raw, ...
        'cfg_action_grid', cfg_action_grid, ...
        'frame_alpha_eff', frame_alpha_eff, ...
        'seq_alpha_eff_max', seq_alpha_max, ...
        'seq_alpha_eff_min', seq_alpha_min, ...
        'seq_alpha_eff_mean', seq_alpha_mean, ...
        'seq_alpha_eff_std', seq_alpha_std, ...
        'dynamic_oracle_best_idx_frame', best_idx_frame, ...
        'dynamic_oracle_best_c1_frame', best_c1_frame, ...
        'ber_dynamic_oracle_frame', ber_dynamic_frame, ...
        'biterr_dynamic_oracle_frame', biterr_dynamic_frame, ...
        'dynamic_formula_c1_frame', dyn_formula_c1_frame, ...
        'ber_dynamic_formula_frame', ber_dynamic_formula_frame, ...
        'biterr_dynamic_formula_frame', biterr_dynamic_formula_frame, ...
        'dynamic_formula_quantized_idx_frame', dyn_formula_quantized_idx_frame, ...
        'dynamic_formula_quantized_c1_frame', dyn_formula_quantized_c1_frame, ...
        'ber_dynamic_formula_quantized_frame', ber_dynamic_formula_quantized_frame, ...
        'biterr_dynamic_formula_quantized_frame', biterr_dynamic_formula_quantized_frame, ...
        'static_oracle_best_idx_seq', static_oracle_best_idx_seq, ...
        'static_oracle_best_c1_seq', static_oracle_best_c1_seq, ...
        'static_oracle_seq_center_c1', static_oracle_seq_center_c1, ...
        'ber_static_oracle_best_frame', ber_static_oracle_best_frame, ...
        'biterr_static_oracle_best_frame', biterr_static_oracle_best_frame, ...
        'static_mean_c1_seq', static_mean_c1_seq, ...
        'static_max_c1_seq', static_max_c1_seq, ...
        'static_min_c1_seq', static_min_c1_seq, ...
        'seq_ber_dynamic_oracle', seq_ber_dynamic, ...
        'seq_ber_dynamic_formula', seq_ber_dynamic_formula, ...
        'seq_ber_dynamic_formula_quantized', seq_ber_dynamic_formula_quantized, ...
        'seq_ber_static_oracle_best', seq_ber_static_oracle_best, ...
        'seq_ber_static_mean', seq_ber_static_mean, ...
        'seq_ber_static_max', seq_ber_static_max, ...
        'seq_ber_static_min', seq_ber_static_min);

    fprintf(['  alpha_res=%.2e | dyn_oracle=%.3e | dyn_formula=%.3e | ' ...
             'dyn_formula_q=%.3e | static_oracle_best=%.3e | ' ...
             'static_mean=%.3e | static_max=%.3e | static_min=%.3e\n'], ...
        cfg_alpha.alpha_max_res, ...
        ber_dynamic_oracle(ai), ...
        ber_dynamic_formula(ai), ...
        ber_dynamic_formula_quantized(ai), ...
        ber_static_oracle_best(ai), ...
        ber_static_mean(ai), ...
        ber_static_max(ai), ...
        ber_static_min(ai));
end

fig = figure('Color', 'w');
semilogy(cfg.alpha_res_list, ber_dynamic_oracle, '-o', 'LineWidth', 1.7, 'DisplayName', 'dynamic_oracle');
hold on;
semilogy(cfg.alpha_res_list, ber_dynamic_formula, '-s', 'LineWidth', 1.5, 'DisplayName', 'dynamic_formula');
semilogy(cfg.alpha_res_list, ber_dynamic_formula_quantized, '-d', 'LineWidth', 1.5, 'DisplayName', 'dynamic_formula_quantized');
semilogy(cfg.alpha_res_list, ber_static_oracle_best, '-^', 'LineWidth', 1.5, 'DisplayName', 'static_oracle_best');
semilogy(cfg.alpha_res_list, ber_static_mean, '-v', 'LineWidth', 1.5, 'DisplayName', 'static_mean');
semilogy(cfg.alpha_res_list, ber_static_max, '-p', 'LineWidth', 1.5, 'DisplayName', 'static_max');
semilogy(cfg.alpha_res_list, ber_static_min, '-h', 'LineWidth', 1.5, 'DisplayName', 'static_min');
grid on;
xlabel('\alpha_{max,res}^{cfg}');
ylabel('Sequence-average BER');
title('BER vs \alpha_{max,res}: baseline ladder');
legend('Location', 'best');
save_figure_pair(fig, fullfile(cfg.output_dir, 'fig_ber_vs_alphares_baseline_ladder'));

results = struct();
results.alpha_res_list = cfg.alpha_res_list(:);
results.alpha_raw_list = cfg.alpha_raw_list(:);
results.snr_db = cfg.snr_db;
results.num_seq = cfg.num_seq;
results.num_frames = cfg.num_frames;
results.detector_type = cfg.detector_type;
bits_cfg = cfg;
bits_cfg.alpha_max_res = cfg.alpha_res_list(1);
bits_cfg.alpha_max_raw = cfg.alpha_raw_list(1);
results.bits_per_frame = build_system_cfg(bits_cfg).bits_per_frame;
results.ber_dynamic_oracle = ber_dynamic_oracle;
results.ber_dynamic_formula = ber_dynamic_formula;
results.ber_dynamic_formula_quantized = ber_dynamic_formula_quantized;
results.ber_static_oracle_best = ber_static_oracle_best;
results.ber_static_mean = ber_static_mean;
results.ber_static_max = ber_static_max;
results.ber_static_min = ber_static_min;
results.seq_level_policy_stats = seq_level_policy_stats;
results.meta_cfg = cfg;

save(cfg.output_mat, 'results', '-v7.3');

fprintf('\nSummary table\n');
fprintf([' alpha_res       alpha_raw       dyn_oracle    dyn_formula   dyn_formula_q ' ...
         'static_orcl   static_mean   static_max    static_min    dyn-stat_orc  dyn-df        dyn-dfq\n']);
for ai = 1:num_alpha
    gap_static_oracle = ber_dynamic_oracle(ai) - ber_static_oracle_best(ai);
    gap_dynamic_formula = ber_dynamic_oracle(ai) - ber_dynamic_formula(ai);
    gap_dynamic_formula_q = ber_dynamic_oracle(ai) - ber_dynamic_formula_quantized(ai);
    fprintf([' %10.2e   %10.2e   %10.3e   %10.3e   %10.3e   %10.3e   ' ...
             '%10.3e   %10.3e   %10.3e   %10.3e   %10.3e   %10.3e\n'], ...
        cfg.alpha_res_list(ai), cfg.alpha_raw_list(ai), ...
        ber_dynamic_oracle(ai), ...
        ber_dynamic_formula(ai), ...
        ber_dynamic_formula_quantized(ai), ...
        ber_static_oracle_best(ai), ...
        ber_static_mean(ai), ...
        ber_static_max(ai), ...
        ber_static_min(ai), ...
        gap_static_oracle, ...
        gap_dynamic_formula, ...
        gap_dynamic_formula_q);
end

fprintf('\nSaved figure: %s\n', fullfile(cfg.output_dir, 'fig_ber_vs_alphares_baseline_ladder.png'));
fprintf('Saved result: %s\n', cfg.output_mat);

out = results;
end

function cfg = default_cfg(project_root)
cfg = struct();
cfg.output_dir = fullfile(project_root, 'outputs');
cfg.output_mat = fullfile(cfg.output_dir, 'results_ber_vs_alphares_baseline_ladder.mat');

cfg.alpha_res_list = [2 4 6 8 10 12 14 16] * 1e-5;
cfg.alpha_raw_scale = 5;

cfg.snr_db = 10;
cfg.num_seq = 200;
cfg.num_frames = 9;

cfg.enable_resampling_comp = true;
cfg.alpha_hat_mode = 'common_component';
cfg.doppler_mode = 'common_with_path_residual';
cfg.motion_profile = 'smooth_ar';

cfg.rho_h = 0.98;
cfg.rho_acc = 0.95;
cfg.sigma_acc = 0.03;
cfg.rho_delta = 0.90;
cfg.sigma_delta = 0.05;
cfg.path_projection_mode = 'ones';
cfg.beta_min = 1.0;
cfg.beta_max = 1.0;
cfg.ell_mode = 'static';
cfg.pdp_mode = 'exp_fixed_per_sequence';

cfg.detector_type = 'oamp';
cfg.oamp_iter = 10;
cfg.oamp_damping = 0.9;
cfg.num_noise = 1;
cfg.progress_every_seq = 25;

cfg.seed_base = 20260328;
cfg.c1_ratios = linspace(0.6, 1.4, 21);
cfg.static_oracle_ratios = linspace(0.5, 1.5, 41);

cfg.N = 256;
cfg.Delta_f = 4;
cfg.fc = 12e3;
cfg.ell_max = 16;
cfg.P = 6;
cfg.Q = 0;
cfg.Nv = 2;
end

function sys = build_system_cfg(cfg)
sys = struct();
sys.N = cfg.N;
sys.Delta_f = cfg.Delta_f;
sys.T_sym = 1 / cfg.Delta_f;
sys.B = cfg.N * cfg.Delta_f;
sys.dt = 1 / sys.B;
sys.fc = cfg.fc;
sys.ell_max = cfg.ell_max;
sys.P = cfg.P;
sys.Q = cfg.Q;
sys.N_eff = cfg.N - 2 * cfg.Q;
sys.data_idx = (cfg.Q + 1):(cfg.N - cfg.Q);
sys.Nv = cfg.Nv;
sys.alpha_max_res = cfg.alpha_max_res;
sys.alpha_max_raw = cfg.alpha_max_raw;
sys.Lcpp = max(1, ceil(sys.ell_max / (1 - cfg.alpha_max_res)));
sys.Lcps = max(1, ceil(cfg.alpha_max_res * sys.N / (1 + cfg.alpha_max_res)));
sys.L = sys.N + sys.Lcpp + sys.Lcps;
sys.c2 = sqrt(2) / sys.N;
sys.bits_per_frame = 2 * numel(sys.data_idx);
end

function tv_cfg = build_timevary_cfg(cfg)
tv_base = struct( ...
    'rho_alpha', 0.98, ...
    'rho_h', cfg.rho_h, ...
    'alpha_max', cfg.alpha_max_res, ...
    'alpha_max_raw', cfg.alpha_max_raw, ...
    'alpha_max_res', cfg.alpha_max_res, ...
    'num_frames', cfg.num_frames, ...
    'doppler_mode', cfg.doppler_mode, ...
    'rho_acc', cfg.rho_acc, ...
    'sigma_acc', cfg.sigma_acc, ...
    'rho_delta', cfg.rho_delta, ...
    'sigma_delta', cfg.sigma_delta, ...
    'delta_max', [], ...
    'motion_profile', cfg.motion_profile, ...
    'path_projection_mode', cfg.path_projection_mode, ...
    'beta_min', cfg.beta_min, ...
    'beta_max', cfg.beta_max, ...
    'ell_mode', cfg.ell_mode, ...
    'pdp_mode', cfg.pdp_mode, ...
    'enable_resampling_comp', cfg.enable_resampling_comp, ...
    'alpha_hat_mode', cfg.alpha_hat_mode, ...
    'clip_alpha_res', true, ...
    'log_alpha_stats', true);
tv_cfg = get_timevary_defaults(tv_base);
end

function seq_data = generate_sequence_data_light(seq_idx, cfg, sys, tv_cfg)
seq_seed = safe_rng_seed(cfg.seed_base + 100000 * seq_idx);
rng(seq_seed, 'twister');

[seq_state, ch_t] = init_timevary_channel_state(sys.P, sys.ell_max, cfg.alpha_max_res, tv_cfg);

frames = repmat(struct( ...
    'seq_idx', 0, ...
    'frame_idx', 0, ...
    'ch', [], ...
    'alpha_eff', 0.0, ...
    'noise_var', 0.0, ...
    'c1_base', 0.0, ...
    'c1_grid', [], ...
    'x_seed', 0, ...
    'noise_seed_base', 0, ...
    'x', []), cfg.num_frames, 1);
alpha_eff = zeros(cfg.num_frames, 1);

for tt = 1:cfg.num_frames
    if tt > 1
        [seq_state, ch_t] = step_timevary_channel_state(seq_state);
    end

    frame = struct();
    frame.seq_idx = seq_idx;
    frame.frame_idx = tt;
    frame.ch = ch_t;
    frame.alpha_eff = max(abs(ch_t.alpha_res(:)));
    frame.noise_var = 1 / (10 ^ (cfg.snr_db / 10));
    frame.c1_base = compute_c1_base_from_alpha(frame.alpha_eff, sys);
    frame.c1_grid = frame.c1_base * cfg.c1_ratios(:).';
    frame.x_seed = safe_rng_seed(cfg.seed_base + 100000 * seq_idx + 1000 * tt + 1);
    frame.noise_seed_base = safe_rng_seed(cfg.seed_base + 100000 * seq_idx + 1000 * tt + 100);
    frame.x = generate_qpsk_frame(frame.x_seed, sys);

    frames(tt) = frame;
    alpha_eff(tt) = frame.alpha_eff;
end

seq_data = struct();
seq_data.frames = frames;
seq_data.alpha_eff = alpha_eff;
end

function seq_summary = analyze_sequence_policy_bundle(seq_data, cfg, sys, cfg_action_grid)
num_frames = numel(seq_data.frames);
alpha_eff_vec = seq_data.alpha_eff(:);
seq_alpha_max = max(alpha_eff_vec);
seq_alpha_min = min(alpha_eff_vec);
seq_alpha_mean = mean(alpha_eff_vec);
seq_alpha_std = std(alpha_eff_vec);

c1_static_max = compute_c1_base_from_alpha(seq_alpha_max, sys);
c1_static_min = compute_c1_base_from_alpha(seq_alpha_min, sys);
c1_static_mean = compute_c1_base_from_alpha(seq_alpha_mean, sys);
c1_static_oracle_center = compute_c1_base_from_alpha(seq_alpha_mean, sys);
c1_static_oracle_grid = c1_static_oracle_center * cfg.static_oracle_ratios(:).';

cache = containers.Map('KeyType', 'char', 'ValueType', 'any');

best_idx_frame = zeros(num_frames, 1);
best_c1_frame = zeros(num_frames, 1);
ber_dynamic_frame = zeros(num_frames, 1);
biterr_dynamic_frame = zeros(num_frames, 1);

dynamic_formula_c1_frame = zeros(num_frames, 1);
ber_dynamic_formula_frame = zeros(num_frames, 1);
biterr_dynamic_formula_frame = zeros(num_frames, 1);

dynamic_formula_quantized_idx_frame = zeros(num_frames, 1);
dynamic_formula_quantized_c1_frame = zeros(num_frames, 1);
ber_dynamic_formula_quantized_frame = zeros(num_frames, 1);
biterr_dynamic_formula_quantized_frame = zeros(num_frames, 1);

err_dynamic_total = 0;
bits_dynamic_total = 0;
err_dynamic_formula_total = 0;
bits_dynamic_formula_total = 0;
err_dynamic_formula_quant_total = 0;
bits_dynamic_formula_quant_total = 0;

for tt = 1:num_frames
    frame = seq_data.frames(tt);
    sweep = run_frame_c1_sweep_cached(frame, cfg, sys, cache);

    best_idx_frame(tt) = sweep.best_idx;
    best_c1_frame(tt) = sweep.best_c1;
    ber_dynamic_frame(tt) = sweep.best_ber;
    biterr_dynamic_frame(tt) = sweep.bit_err_grid(sweep.best_idx);
    err_dynamic_total = err_dynamic_total + sweep.bit_err_grid(sweep.best_idx);
    bits_dynamic_total = bits_dynamic_total + sweep.bits_per_eval;

    dynamic_formula_c1_frame(tt) = frame.c1_base;
    ber_dynamic_formula_frame(tt) = sweep.ber_grid(sweep.base_idx);
    biterr_dynamic_formula_frame(tt) = sweep.bit_err_grid(sweep.base_idx);
    err_dynamic_formula_total = err_dynamic_formula_total + sweep.bit_err_grid(sweep.base_idx);
    bits_dynamic_formula_total = bits_dynamic_formula_total + sweep.bits_per_eval;

    [quant_idx, quant_c1] = nearest_index_and_value(cfg_action_grid, frame.c1_base);
    [err_now, bits_now] = evaluate_frame_for_c1_cached(frame, quant_c1, cfg, sys, cache);
    dynamic_formula_quantized_idx_frame(tt) = quant_idx;
    dynamic_formula_quantized_c1_frame(tt) = quant_c1;
    biterr_dynamic_formula_quantized_frame(tt) = err_now;
    ber_dynamic_formula_quantized_frame(tt) = err_now / max(bits_now, 1);
    err_dynamic_formula_quant_total = err_dynamic_formula_quant_total + err_now;
    bits_dynamic_formula_quant_total = bits_dynamic_formula_quant_total + bits_now;
end

[err_static_max_total, bits_static_max_total] = evaluate_sequence_fixed_c1(seq_data, c1_static_max, cfg, sys, cache);
[err_static_min_total, bits_static_min_total] = evaluate_sequence_fixed_c1(seq_data, c1_static_min, cfg, sys, cache);
[err_static_mean_total, bits_static_mean_total] = evaluate_sequence_fixed_c1(seq_data, c1_static_mean, cfg, sys, cache);

best_static_oracle_idx = 1;
best_static_oracle_c1 = c1_static_oracle_grid(1);
best_static_oracle_ber = inf;
best_static_oracle_err = inf;
best_static_oracle_bits = 0;
best_static_oracle_frame_ber = zeros(num_frames, 1);
best_static_oracle_frame_err = zeros(num_frames, 1);

for kk = 1:numel(c1_static_oracle_grid)
    c1_try = c1_static_oracle_grid(kk);
    [err_try, bits_try, frame_err_try, frame_ber_try] = evaluate_sequence_fixed_c1(seq_data, c1_try, cfg, sys, cache);
    ber_try = err_try / max(bits_try, 1);
    if ber_try < best_static_oracle_ber
        best_static_oracle_ber = ber_try;
        best_static_oracle_idx = kk;
        best_static_oracle_c1 = c1_try;
        best_static_oracle_err = err_try;
        best_static_oracle_bits = bits_try;
        best_static_oracle_frame_err = frame_err_try;
        best_static_oracle_frame_ber = frame_ber_try;
    end
end

seq_summary = struct();
seq_summary.alpha_eff = alpha_eff_vec;
seq_summary.seq_alpha_eff_max = seq_alpha_max;
seq_summary.seq_alpha_eff_min = seq_alpha_min;
seq_summary.seq_alpha_eff_mean = seq_alpha_mean;
seq_summary.seq_alpha_eff_std = seq_alpha_std;

seq_summary.dynamic_oracle_best_idx_frame = best_idx_frame;
seq_summary.dynamic_oracle_best_c1_frame = best_c1_frame;
seq_summary.ber_dynamic_oracle_frame = ber_dynamic_frame;
seq_summary.biterr_dynamic_oracle_frame = biterr_dynamic_frame;

seq_summary.dynamic_formula_c1_frame = dynamic_formula_c1_frame;
seq_summary.ber_dynamic_formula_frame = ber_dynamic_formula_frame;
seq_summary.biterr_dynamic_formula_frame = biterr_dynamic_formula_frame;

seq_summary.dynamic_formula_quantized_idx_frame = dynamic_formula_quantized_idx_frame;
seq_summary.dynamic_formula_quantized_c1_frame = dynamic_formula_quantized_c1_frame;
seq_summary.ber_dynamic_formula_quantized_frame = ber_dynamic_formula_quantized_frame;
seq_summary.biterr_dynamic_formula_quantized_frame = biterr_dynamic_formula_quantized_frame;

seq_summary.static_oracle_best_idx_seq = best_static_oracle_idx;
seq_summary.static_oracle_best_c1_seq = best_static_oracle_c1;
seq_summary.static_oracle_seq_center_c1 = c1_static_oracle_center;
seq_summary.ber_static_oracle_best_frame = best_static_oracle_frame_ber;
seq_summary.biterr_static_oracle_best_frame = best_static_oracle_frame_err;

seq_summary.static_mean_c1_seq = c1_static_mean;
seq_summary.static_max_c1_seq = c1_static_max;
seq_summary.static_min_c1_seq = c1_static_min;

seq_summary.seq_ber_dynamic_oracle = err_dynamic_total / max(bits_dynamic_total, 1);
seq_summary.seq_ber_dynamic_formula = err_dynamic_formula_total / max(bits_dynamic_formula_total, 1);
seq_summary.seq_ber_dynamic_formula_quantized = err_dynamic_formula_quant_total / max(bits_dynamic_formula_quant_total, 1);
seq_summary.seq_ber_static_oracle_best = best_static_oracle_err / max(best_static_oracle_bits, 1);
seq_summary.seq_ber_static_mean = err_static_mean_total / max(bits_static_mean_total, 1);
seq_summary.seq_ber_static_max = err_static_max_total / max(bits_static_max_total, 1);
seq_summary.seq_ber_static_min = err_static_min_total / max(bits_static_min_total, 1);
end

function sweep = run_frame_c1_sweep_cached(frame, cfg, sys, cache)
c1_grid = frame.c1_grid(:).';
num_c1 = numel(c1_grid);
bit_err_grid = zeros(1, num_c1);
ber_grid = zeros(1, num_c1);
bits_per_eval = 0;

for m = 1:num_c1
    [bit_err_grid(m), bits_per_eval] = evaluate_frame_for_c1_cached(frame, c1_grid(m), cfg, sys, cache);
    ber_grid(m) = bit_err_grid(m) / max(bits_per_eval, 1);
end

[best_ber, best_idx] = min(ber_grid);
[~, base_idx] = min(abs(c1_grid - frame.c1_base));

sweep = struct();
sweep.frame_index = frame.frame_idx;
sweep.alpha_eff = frame.alpha_eff;
sweep.c1_base = frame.c1_base;
sweep.c1_grid = c1_grid;
sweep.bit_err_grid = bit_err_grid;
sweep.ber_grid = ber_grid;
sweep.bits_per_eval = bits_per_eval;
sweep.base_idx = base_idx;
sweep.best_idx = best_idx;
sweep.best_c1 = c1_grid(best_idx);
sweep.best_ber = best_ber;
end

function [bit_err_total, bits_total, frame_bit_err, frame_ber] = evaluate_sequence_fixed_c1(seq_data, c1, cfg, sys, cache)
num_frames = numel(seq_data.frames);
frame_bit_err = zeros(num_frames, 1);
frame_ber = zeros(num_frames, 1);
bit_err_total = 0;
bits_total = 0;

for tt = 1:num_frames
    [err_now, bits_now] = evaluate_frame_for_c1_cached(seq_data.frames(tt), c1, cfg, sys, cache);
    frame_bit_err(tt) = err_now;
    frame_ber(tt) = err_now / max(bits_now, 1);
    bit_err_total = bit_err_total + err_now;
    bits_total = bits_total + bits_now;
end
end

function [bit_err_total, bits_total] = evaluate_frame_for_c1_cached(frame, c1, cfg, sys, cache)
key = sprintf('f%03d_c1_%0.15e', frame.frame_idx, double(c1));
if isKey(cache, key)
    val = cache(key);
    bit_err_total = val(1);
    bits_total = val(2);
    return;
end

[bit_err_total, bits_total] = evaluate_frame_for_c1(frame, c1, cfg, sys);
cache(key) = [bit_err_total, bits_total];
end

function [bit_err_total, bits_total] = evaluate_frame_for_c1(frame, c1, cfg, sys)
Heff = build_heff_for_c1(frame.ch, c1, sys);
bit_err_total = 0;
bits_total = 0;

for ni = 1:cfg.num_noise
    noise_seed = safe_rng_seed(frame.noise_seed_base + ni);
    rng(noise_seed, 'twister');
    w = sqrt(frame.noise_var / 2) * (randn(sys.N, 1) + 1j * randn(sys.N, 1));
    y = Heff * frame.x + w;

    switch lower(cfg.detector_type)
        case 'oamp'
            x_hat = oamp_detector(y, Heff, frame.noise_var, cfg.oamp_iter, cfg.oamp_damping, sys.Q);
        case 'lmmse'
            x_hat = lmmse_detector(y, Heff, frame.noise_var);
        otherwise
            error('Unsupported detector_type=%s', cfg.detector_type);
    end

    bit_err_total = bit_err_total + count_qpsk_bit_errors(x_hat(sys.data_idx), frame.x(sys.data_idx));
    bits_total = bits_total + sys.bits_per_frame;
end
end

function Heff = build_heff_for_c1(ch, c1, sys)
XTB = precompute_idaf_basis(sys.N, c1, sys.c2);
Xext = add_cpp_cps_matrix(XTB, c1, sys.Lcpp, sys.Lcps);
YT = build_timescaling_G_sparse(sys.N, sys.L, sys.Lcpp, ch, sys.fc, sys.dt) * Xext;

n = (0:sys.N - 1).';
chirp1 = exp(-1j * 2 * pi * c1 * (n .^ 2));
chirp2 = exp(-1j * 2 * pi * sys.c2 * (n .^ 2));
Heff = afdm_demod_matrix(YT, chirp1, chirp2);

Hsub = Heff(sys.data_idx, sys.data_idx);
sub_power = (norm(Hsub, 'fro') ^ 2) / sys.N_eff;
Heff = Heff / sqrt(max(sub_power, 1e-12));
end

function x = generate_qpsk_frame(seed_in, sys)
rng(seed_in, 'twister');
x = zeros(sys.N, 1);
x(sys.data_idx) = qpsk_symbols(numel(sys.data_idx));
x = x * sqrt(sys.N / sys.N_eff);
end

function c1_base = compute_c1_base_from_alpha(alpha_eff, sys)
alpha_eff = max(double(alpha_eff), 0.0);
kmax = ceil((alpha_eff * sys.fc) * sys.T_sym);
den = (1 - 4 * alpha_eff * (sys.N - 1));
if den <= 0
    error('c1 design invalid: den<=0 for alpha_eff=%.3e', alpha_eff);
end
c1_base = (2 * kmax + 2 * alpha_eff * (sys.N - 1) + 2 * sys.Nv + 1) / (2 * sys.N * den);
end

function [idx, value] = nearest_index_and_value(grid, target)
[~, idx] = min(abs(grid(:).' - target));
value = grid(idx);
end

function save_figure_pair(fig, prefix)
saveas(fig, [prefix '.png']);
savefig(fig, [prefix '.fig']);
end

function dst = merge_struct(dst, src)
if isempty(src)
    return;
end
fn = fieldnames(src);
for i = 1:numel(fn)
    dst.(fn{i}) = src.(fn{i});
end
end

function seed_u32 = safe_rng_seed(seed_in)
seed_u32 = mod(double(seed_in), 2^32 - 1);
if seed_u32 < 0
    seed_u32 = seed_u32 + (2^32 - 1);
end
end

function XTB = precompute_idaf_basis(N, c1, c2)
n = (0:N - 1).';
m = 0:N - 1;
phase_n = exp(1j * 2 * pi * c1 * (n .^ 2));
phase_m = exp(1j * 2 * pi * c2 * (m .^ 2));
W = exp(1j * 2 * pi * (n * m) / N) / sqrt(N);
XTB = (phase_n .* W) .* phase_m;
end

function Xext = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps)
N = size(XTB, 1);
if Lcpp == 0 && Lcps == 0
    Xext = XTB;
    return;
end

if Lcpp > 0
    n_pre = (-Lcpp:-1).';
    idx_pre = n_pre + N + 1;
    phase_pre = exp(-1j * 2 * pi * c1 * (N^2 + 2 * N * n_pre));
    Xpre = XTB(idx_pre, :) .* phase_pre;
else
    Xpre = zeros(0, N, 'like', XTB);
end

if Lcps > 0
    n_suf = (N:(N + Lcps - 1)).';
    idx_suf = (n_suf - N + 1);
    phase_suf = exp(+1j * 2 * pi * c1 * (N^2 + 2 * N * n_suf));
    Xsuf = XTB(idx_suf, :) .* phase_suf;
else
    Xsuf = zeros(0, N, 'like', XTB);
end

Xext = [Xpre; XTB; Xsuf];
end

function G = build_timescaling_G_sparse(N, L, Lcpp, ch, fc, dt)
P = ch.P;
ell = ch.ell;
if isfield(ch, 'alpha_res')
    alpha = ch.alpha_res;
else
    alpha = ch.alpha;
end
h = ch.h;

n = (0:N - 1).';
max_nnz = N * P * 2;
I = zeros(max_nnz, 1);
J = zeros(max_nnz, 1);
V = zeros(max_nnz, 1) + 1j * zeros(max_nnz, 1);
ptr = 0;

for i = 1:P
    idx = (1 + alpha(i)) * n - ell(i) + Lcpp;
    idx0 = floor(idx);
    frac = idx - idx0;
    idx1 = idx0 + 1;
    phase = exp(1j * 2 * pi * (alpha(i) * fc) * (n * dt));

    v0 = (idx0 >= 0) & (idx0 <= (L - 1));
    nn0 = sum(v0);
    if nn0 > 0
        rows = find(v0);
        cols = idx0(v0) + 1;
        vals = h(i) * phase(v0) .* (1 - frac(v0));
        I(ptr + 1:ptr + nn0) = rows;
        J(ptr + 1:ptr + nn0) = cols;
        V(ptr + 1:ptr + nn0) = vals;
        ptr = ptr + nn0;
    end

    v1 = (idx1 >= 0) & (idx1 <= (L - 1)) & (frac > 0);
    nn1 = sum(v1);
    if nn1 > 0
        rows = find(v1);
        cols = idx1(v1) + 1;
        vals = h(i) * phase(v1) .* frac(v1);
        I(ptr + 1:ptr + nn1) = rows;
        J(ptr + 1:ptr + nn1) = cols;
        V(ptr + 1:ptr + nn1) = vals;
        ptr = ptr + nn1;
    end
end

I = I(1:ptr);
J = J(1:ptr);
V = V(1:ptr);
G = sparse(I, J, V, N, L);
end

function Heff = afdm_demod_matrix(YT, chirp1, chirp2)
N = size(YT, 1);
tmp = fft(YT .* chirp1, [], 1) / sqrt(N);
Heff = tmp .* chirp2;
end

function s = qpsk_symbols(M)
b1 = randi([0, 1], M, 1);
b2 = randi([0, 1], M, 1);
re = 1 - 2 * b1;
im = 1 - 2 * b2;
s = (re + 1j * im) / sqrt(2);
end

function bit_err = count_qpsk_bit_errors(x_hat, x_true)
b_hat = qpsk_demod_bits_gray(x_hat);
b_true = qpsk_demod_bits_gray(x_true);
bit_err = sum(b_hat ~= b_true);
end

function bits = qpsk_demod_bits_gray(x)
x = x(:);
amp = median(abs(x));
if amp < 1e-12
    amp = 1;
end
x = x / amp;

const = [1 + 1j; -1 + 1j; -1 - 1j; 1 - 1j] / sqrt(2);
labels = [0 0; 0 1; 1 1; 1 0];

d2 = abs(x - const.').^2;
[~, idx] = min(d2, [], 2);
bits2 = labels(idx, :);
bits = reshape(bits2.', [], 1);
end
