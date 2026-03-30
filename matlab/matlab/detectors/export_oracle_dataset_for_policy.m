function export_result = export_oracle_dataset_for_policy(user_cfg)
%% export_oracle_dataset_for_policy.m
% Unified offline dataset export for C1 policy learning.
%
% New sequence-aware logic:
% - mode='timevary_sequence': one channel state per sequence, recursively evolved by frame index.
% - mode='iid': legacy frame-independent channel sampling.
%
% State feature uses online-observable schema by default.
% Set use_oracle_state=true only for debug ablation.

if nargin < 1
    user_cfg = struct();
end

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
common_dir = fullfile(project_root, 'matlab', 'common');
addpath(this_dir);
addpath(common_dir);

%% ===== 0) Export mode/config =====
cfg = struct();
cfg.run_profile = 'smoke';               % 'smoke' (legacy quick export) | 'paper' (formal paper dataset)
cfg.paper_id = 'tsv2seq';

% Legacy-compatible defaults.
cfg.mode = 'timevary_sequence';          % 'iid' | 'timevary_sequence'
cfg.output_file = fullfile(project_root, 'data', 'oracle_policy_dataset.mat');
cfg.output_csv = true;

cfg.detector_target = 'oampnet';         % 'lmmse' | 'oamp' | 'oampnet'
cfg.oamp_iter = 10;
cfg.oamp_damping = 0.9;
cfg.oampnet_param_version = 'tsv1';

cfg.num_seq = 300;
cfg.num_frames = 10;
cfg.num_samples_static = 3000;

cfg.snr_db_list = [14 18];
cfg.num_noise = 8;
cfg.reward_lambda_proxy = 1e-4;
cfg.reward_primary_key = 'reward_mix';
cfg.reward_relbase_floor = 1e-4;
cfg.reward_proxy_improve_lambda = 0.15;
cfg.reward_clip_abs = 2.0;
cfg.label_snr_mid = 10;
cfg.label_snr_high = 12;
cfg.label_eval_repeats_low = 1;
cfg.label_eval_repeats_mid = 2;
cfg.label_eval_repeats_high = 4;
cfg.use_oracle_state = false;
cfg.include_physical_doppler_state = false;
cfg.include_policy_history_state = true;
cfg.context_switch_window = 5;

cfg.seed = 3;

cfg = merge_struct(cfg, user_cfg);
run_profile = char(string(get_struct(cfg, 'run_profile', 'smoke')));
if strcmpi(run_profile, 'paper')
    if ~isfield(user_cfg, 'output_file') || isempty(user_cfg.output_file)
        cfg.output_file = fullfile(project_root, 'data', ['oracle_policy_dataset_' char(string(cfg.paper_id)) '_paper.mat']);
    end
    if ~isfield(user_cfg, 'oampnet_param_version') || isempty(user_cfg.oampnet_param_version)
        cfg.oampnet_param_version = sprintf('%s_paper', char(string(cfg.paper_id)));
    end
    if ~isfield(user_cfg, 'num_seq') || isempty(user_cfg.num_seq)
        cfg.num_seq = 2000;
    end
    if ~isfield(user_cfg, 'num_frames') || isempty(user_cfg.num_frames)
        cfg.num_frames = 10;
    end
    if ~isfield(user_cfg, 'snr_db_list') || isempty(user_cfg.snr_db_list)
        cfg.snr_db_list = 0:2:20;
    end
    if ~isfield(user_cfg, 'num_noise') || isempty(user_cfg.num_noise)
        cfg.num_noise = 1;
    end
end
if contains(lower(char(string(cfg.paper_id))), 'vdop_ctrl')
    if ~isfield(user_cfg, 'doppler_mode') || isempty(user_cfg.doppler_mode)
        cfg.doppler_mode = 'common_with_path_residual';
    end
    if ~isfield(user_cfg, 'motion_profile') || isempty(user_cfg.motion_profile)
        cfg.motion_profile = 'maneuver_heave';
    end
    if ~isfield(user_cfg, 'path_projection_mode') || isempty(user_cfg.path_projection_mode)
        cfg.path_projection_mode = 'symmetric_linear';
    end
    if ~isfield(user_cfg, 'beta_min') || isempty(user_cfg.beta_min)
        cfg.beta_min = 0.45;
    end
    if ~isfield(user_cfg, 'beta_max') || isempty(user_cfg.beta_max)
        cfg.beta_max = 1.65;
    end
    if ~isfield(user_cfg, 'num_frames') || isempty(user_cfg.num_frames)
        cfg.num_frames = 40;
    end
    if ~isfield(user_cfg, 'reward_primary_key') || isempty(user_cfg.reward_primary_key)
        cfg.reward_primary_key = 'reward_relbase_proxy';
    end
    if ~isfield(user_cfg, 'reward_proxy_improve_lambda') || isempty(user_cfg.reward_proxy_improve_lambda)
        cfg.reward_proxy_improve_lambda = 0.20;
    end
    if ~isfield(user_cfg, 'reward_clip_abs') || isempty(user_cfg.reward_clip_abs)
        cfg.reward_clip_abs = 2.0;
    end
    if ~isfield(user_cfg, 'label_eval_repeats_mid') || isempty(user_cfg.label_eval_repeats_mid)
        cfg.label_eval_repeats_mid = 3;
    end
    if ~isfield(user_cfg, 'label_eval_repeats_high') || isempty(user_cfg.label_eval_repeats_high)
        cfg.label_eval_repeats_high = 6;
    end
end
if contains(lower(char(string(cfg.paper_id))), 'vdop') && ...
        (~isfield(user_cfg, 'doppler_mode') || isempty(user_cfg.doppler_mode))
    cfg.doppler_mode = 'common_with_path_residual';
end
mode = char(string(cfg.mode));
use_timevary = strcmpi(mode, 'timevary_sequence');
if ~use_timevary && ~strcmpi(mode, 'iid')
    error('Unsupported cfg.mode=%s. Use ''iid'' or ''timevary_sequence''.', mode);
end

%% ===== 1) System/channel parameters =====
N = 256;
Delta_f = 4;
T_sym = 1 / Delta_f;
B = N * Delta_f;
dt = 1 / B;
fc = 12e3;
alpha_max_res = get_struct(cfg, 'alpha_max_res', get_struct(cfg, 'alpha_max', 1e-4));
alpha_max_raw = get_struct(cfg, 'alpha_max_raw', max(5e-4, alpha_max_res));
ell_max = 16;
P = 6;
Q = 0;
N_eff = N - 2 * Q;
data_idx = (Q + 1):(N - Q);

Lcpp = max(1, ceil(ell_max / (1 - alpha_max_res)));
Lcps = max(1, ceil(alpha_max_res * N / (1 + alpha_max_res)));
L = N + Lcpp + Lcps;

Nv = 2;
kmax = ceil((alpha_max_res * fc) * T_sym);
den = (1 - 4 * alpha_max_res * (N - 1));
c1_base = (2 * kmax + 2 * alpha_max_res * (N - 1) + 2 * Nv + 1) / (2 * N * den);
c2 = sqrt(2) / N;

ratios = linspace(0.6, 1.4, 21);
c1_grid = c1_base * ratios;
num_actions = numel(c1_grid);
[~, base_action] = min(abs(c1_grid - c1_base));

%% ===== 2) Dataset size and timevary cfg =====
if use_timevary
    num_seq = max(1, round(cfg.num_seq));
    num_frames = max(1, round(cfg.num_frames));
    num_samples = num_seq * num_frames;
else
    num_samples = max(1, round(cfg.num_samples_static));
    num_seq = num_samples;
    num_frames = 1;
end

rng(safe_rng_seed(cfg.seed));

tv_override = struct();
if isfield(cfg, 'timevary_cfg') && isstruct(cfg.timevary_cfg)
    tv_override = cfg.timevary_cfg;
elseif isfield(cfg, 'timevary') && isstruct(cfg.timevary)
    tv_override = cfg.timevary;
end

tv_base = struct( ...
    'rho_alpha', get_struct(cfg, 'rho_alpha', 0.98), ...
    'rho_h', get_struct(cfg, 'rho_h', 0.98), ...
    'alpha_max', alpha_max_res, ...
    'alpha_max_raw', alpha_max_raw, ...
    'alpha_max_res', alpha_max_res, ...
    'num_frames', num_frames, ...
    'doppler_mode', get_struct(cfg, 'doppler_mode', 'independent_path_ar1'), ...
    'rho_acc', get_struct(cfg, 'rho_acc', 0.95), ...
    'sigma_acc', get_struct(cfg, 'sigma_acc', 0.03), ...
    'rho_delta', get_struct(cfg, 'rho_delta', 0.90), ...
    'sigma_delta', get_struct(cfg, 'sigma_delta', 0.05), ...
    'delta_max', get_struct(cfg, 'delta_max', []), ...
    'motion_profile', get_struct(cfg, 'motion_profile', 'smooth_ar'), ...
    'target_track_gain', get_struct(cfg, 'target_track_gain', 0.75), ...
    'target_blend', get_struct(cfg, 'target_blend', 0.80), ...
    'profile_v_peak', get_struct(cfg, 'profile_v_peak', 0.95), ...
    'profile_turn_range', get_struct(cfg, 'profile_turn_range', [0.32, 0.48]), ...
    'profile_recede_range', get_struct(cfg, 'profile_recede_range', [0.68, 0.86]), ...
    'profile_heave_amp', get_struct(cfg, 'profile_heave_amp', 0.18), ...
    'profile_heave_cycles', get_struct(cfg, 'profile_heave_cycles', 1.35), ...
    'profile_secondary_amp', get_struct(cfg, 'profile_secondary_amp', 0.08), ...
    'profile_secondary_cycles', get_struct(cfg, 'profile_secondary_cycles', 2.70), ...
    'profile_jitter_std', get_struct(cfg, 'profile_jitter_std', 0.04), ...
    'path_projection_mode', get_struct(cfg, 'path_projection_mode', 'ones'), ...
    'beta_min', get_struct(cfg, 'beta_min', 1.0), ...
    'beta_max', get_struct(cfg, 'beta_max', 1.0), ...
    'ell_mode', get_struct(cfg, 'ell_mode', 'static'), ...
    'pdp_mode', get_struct(cfg, 'pdp_mode', 'exp_fixed_per_sequence'), ...
    'enable_resampling_comp', get_struct(cfg, 'enable_resampling_comp', true), ...
    'alpha_hat_mode', get_struct(cfg, 'alpha_hat_mode', 'common_component'), ...
    'clip_alpha_res', get_struct(cfg, 'clip_alpha_res', true), ...
    'log_alpha_stats', get_struct(cfg, 'log_alpha_stats', true));
tv_cfg = get_timevary_defaults(merge_struct(tv_base, tv_override));

% TODO: Consider enabling ell slow drift after validating impact on detector stability.

data_dir = fullfile(project_root, 'data');
oampnet_params = [];
oampnet_param_path = '';
if strcmpi(cfg.detector_target, 'oampnet')
    oampnet_param_path = fullfile(data_dir, ['oampnet_v4_' cfg.oampnet_param_version '_params.mat']);
    if ~exist(oampnet_param_path, 'file')
        error('Missing OAMPNet params: %s', oampnet_param_path);
    end
    oampnet_params = load(oampnet_param_path);
end

%% ===== 3) Precompute action-dependent basis =====
Xext_all = cell(num_actions, 1);
chirp1_all = cell(num_actions, 1);
chirp2_all = cell(num_actions, 1);
for a = 1:num_actions
    c1 = c1_grid(a);
    XTB = precompute_idaf_basis(N, c1, c2);
    Xext_all{a} = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps);

    idx_n = (0:N-1).';
    chirp1_all{a} = exp(-1j * 2 * pi * c1 * (idx_n .^ 2));
    chirp2_all{a} = exp(-1j * 2 * pi * c2 * (idx_n .^ 2));
end

%% ===== 4) Allocate =====
dummy_ctx = struct( ...
    'prev_action_norm', 0, ...
    'prev_reward', 0, ...
    'prev_residual_proxy', 0, ...
    'recent_switch_rate', 0, ...
    'frame_index_norm', 0, ...
    'prev_offdiag_ratio', 0, ...
    'prev_band_energy_ratio', 0, ...
    'prev_frob_norm', 0);
dummy_ch = struct('alpha', zeros(P, 1), 'ell', zeros(P, 1), 'h', ones(P, 1));
[dummy_feat, feature_names] = extract_online_state_features( ...
    eye(N), [], [], 1.0, data_idx, dummy_ctx, cfg.use_oracle_state, dummy_ch, ...
    cfg.include_physical_doppler_state, cfg.include_policy_history_state);
feat_dim = numel(dummy_feat);

state = zeros(num_samples, feat_dim, 'single');
reward = zeros(num_samples, num_actions, 'single');
reward_ber = zeros(num_samples, num_actions, 'single');
reward_proxy = zeros(num_samples, num_actions, 'single');
reward_mix = zeros(num_samples, num_actions, 'single');
reward_relbase = zeros(num_samples, num_actions, 'single');
reward_proxy_gain = zeros(num_samples, num_actions, 'single');
reward_relbase_proxy = zeros(num_samples, num_actions, 'single');
ber_actions = zeros(num_samples, num_actions, 'single');
mse_proxy_actions = zeros(num_samples, num_actions, 'single');

oracle_action = zeros(num_samples, 1, 'int32');
oracle_action_reward = zeros(num_samples, 1, 'int32');
chosen_action_fixed = int32(base_action) * ones(num_samples, 1, 'int32');
chosen_action_oracle = zeros(num_samples, 1, 'int32');

snr_db = zeros(num_samples, 1, 'single');
sequence_id = zeros(num_samples, 1, 'int32');
time_index = zeros(num_samples, 1, 'int32');
alpha_com = zeros(num_samples, 1, 'single');
v_norm = zeros(num_samples, 1, 'single');
delta_alpha_rms = zeros(num_samples, 1, 'single');
alpha_hat_t = zeros(num_samples, 1, 'single');
alpha_raw_t = zeros(num_samples, P, 'single');
alpha_res_t = zeros(num_samples, P, 'single');
alpha_raw_max = zeros(num_samples, 1, 'single');
alpha_res_max = zeros(num_samples, 1, 'single');

sequence_alpha_stats = struct();
sequence_alpha_stats.sequence_id = zeros(num_seq, 1, 'int32');
sequence_alpha_stats.alpha_raw_abs_max = zeros(num_seq, 1, 'single');
sequence_alpha_stats.alpha_raw_p99_abs = zeros(num_seq, 1, 'single');
sequence_alpha_stats.alpha_raw_p995_abs = zeros(num_seq, 1, 'single');
sequence_alpha_stats.alpha_res_abs_max = zeros(num_seq, 1, 'single');
sequence_alpha_stats.alpha_res_p99_abs = zeros(num_seq, 1, 'single');
sequence_alpha_stats.alpha_res_p995_abs = zeros(num_seq, 1, 'single');

sample_idx = 0;
for sid = 1:num_seq
    if use_timevary
        [seq_state, ch_t] = init_timevary_channel_state(P, ell_max, alpha_max_res, tv_cfg);
        snr_pick_seq = cfg.snr_db_list(randi(numel(cfg.snr_db_list)));
    end

    seq_alpha_raw_abs = zeros(num_frames, P);
    seq_alpha_res_abs = zeros(num_frames, P);

    prev_action_norm = 0.0;
    prev_reward = 0.0;
    prev_residual_proxy = 0.0;
    prev_offdiag_ratio = 0.0;
    prev_band_energy_ratio = 0.0;
    prev_frob_norm = 0.0;
    switch_hist = zeros(cfg.context_switch_window, 1);
    switch_count = 0;
    prev_behavior_action = base_action;

    for tt = 1:num_frames
        sample_idx = sample_idx + 1;

        if use_timevary
            if tt > 1
                [seq_state, ch_t] = step_timevary_channel_state(seq_state);
            end
            ch = ch_t;
            snr_pick = snr_pick_seq;
        else
            [~, ch] = init_timevary_channel_state(P, ell_max, alpha_max_res, tv_cfg);
            snr_pick = cfg.snr_db_list(randi(numel(cfg.snr_db_list)));
        end

        noise_var = 1 / (10 ^ (snr_pick / 10));
        G = build_timescaling_G_sparse(N, L, Lcpp, ch, fc, dt);

        x = zeros(N, 1);
        x(data_idx) = qpsk_symbols(numel(data_idx));
        x = x * sqrt(N / N_eff);

        ber_vec = zeros(num_actions, 1);
        proxy_vec = zeros(num_actions, 1);
        Heff_base = [];
        y_base_obs = [];
        x_base_obs = [];
        base_residual_proxy = 0.0;

        for a = 1:num_actions
            YT = G * Xext_all{a};
            Heff = afdm_demod_matrix(YT, chirp1_all{a}, chirp2_all{a});

            Hsub = Heff(data_idx, data_idx);
            sub_power = (norm(Hsub, 'fro') ^ 2) / N_eff;
            Heff = Heff / sqrt(max(sub_power, 1e-12));

            if a == base_action
                Heff_base = Heff;
            end

            err = 0;
            bits = 0;
            proxy_acc = 0;
            num_label_eval = get_label_eval_repeats(snr_pick, cfg);
            for ni = 1:num_label_eval
                seed = 1000000 * sample_idx + 100 * a + ni;
                rng(safe_rng_seed(seed));
                w = sqrt(noise_var / 2) * (randn(N, 1) + 1j * randn(N, 1));
                y = Heff * x + w;
                x_det = run_target_detector(y, Heff, noise_var, cfg.detector_target, cfg.oamp_iter, cfg.oamp_damping, oampnet_params, Q);

                err = err + count_qpsk_bit_errors(x_det(data_idx), x(data_idx));
                bits = bits + 2 * numel(data_idx);

                x_hard = qpsk_hard_projection(x_det);
                resid = y - Heff * x_hard;
                proxy_now = mean(abs(resid) .^ 2) / max(noise_var, 1e-12);
                proxy_acc = proxy_acc + proxy_now;

                if a == base_action && ni == 1
                    y_base_obs = y;
                    x_base_obs = x_det;
                    base_residual_proxy = proxy_now;
                end
            end

            ber_vec(a) = err / max(bits, 1);
            proxy_vec(a) = proxy_acc / max(num_label_eval, 1);
        end

        base_ber = ber_vec(base_action);
        base_proxy = proxy_vec(base_action);
        proxy_norm = proxy_vec / max(median(proxy_vec), 1e-12);
        reward_ber_vec = -ber_vec;
        reward_proxy_vec = -proxy_norm;
        reward_mix_vec = -(ber_vec + cfg.reward_lambda_proxy * proxy_norm);
        reward_relbase_vec = (base_ber - ber_vec) / max(base_ber, cfg.reward_relbase_floor);
        reward_proxy_gain_vec = (base_proxy - proxy_vec) / max(base_proxy, 1e-12);
        reward_relbase_proxy_vec = reward_relbase_vec + cfg.reward_proxy_improve_lambda * reward_proxy_gain_vec;
        reward_relbase_vec = clip_reward_vec(reward_relbase_vec, cfg.reward_clip_abs);
        reward_proxy_gain_vec = clip_reward_vec(reward_proxy_gain_vec, cfg.reward_clip_abs);
        reward_relbase_proxy_vec = clip_reward_vec(reward_relbase_proxy_vec, cfg.reward_clip_abs);

        [~, a_star_ber] = min(ber_vec);
        reward_primary_vec = select_primary_reward(cfg.reward_primary_key, reward_ber_vec, reward_mix_vec, reward_relbase_vec, reward_relbase_proxy_vec);
        [~, a_star_reward] = max(reward_primary_vec);

        ctx = struct();
        ctx.prev_action_norm = prev_action_norm;
        ctx.prev_reward = prev_reward;
        ctx.prev_residual_proxy = prev_residual_proxy;
        if switch_count > 0
            ctx.recent_switch_rate = mean(switch_hist(1:switch_count));
        else
            ctx.recent_switch_rate = 0.0;
        end
        ctx.frame_index_norm = (tt - 1) / max(num_frames - 1, 1);
        ctx.prev_offdiag_ratio = prev_offdiag_ratio;
        ctx.prev_band_energy_ratio = prev_band_energy_ratio;
        ctx.prev_frob_norm = prev_frob_norm;

        feat = extract_online_state_features( ...
            Heff_base, y_base_obs, x_base_obs, noise_var, data_idx, ctx, ...
            cfg.use_oracle_state, ch, cfg.include_physical_doppler_state, cfg.include_policy_history_state);

        state(sample_idx, :) = single(feat);
        reward(sample_idx, :) = single(reward_primary_vec(:).');
        reward_ber(sample_idx, :) = single(reward_ber_vec(:).');
        reward_proxy(sample_idx, :) = single(reward_proxy_vec(:).');
        reward_mix(sample_idx, :) = single(reward_mix_vec(:).');
        reward_relbase(sample_idx, :) = single(reward_relbase_vec(:).');
        reward_proxy_gain(sample_idx, :) = single(reward_proxy_gain_vec(:).');
        reward_relbase_proxy(sample_idx, :) = single(reward_relbase_proxy_vec(:).');
        ber_actions(sample_idx, :) = single(ber_vec(:).');
        mse_proxy_actions(sample_idx, :) = single(proxy_norm(:).');

        oracle_action(sample_idx) = int32(a_star_ber);
        oracle_action_reward(sample_idx) = int32(a_star_reward);
        chosen_action_oracle(sample_idx) = int32(a_star_ber);

        snr_db(sample_idx) = single(snr_pick);
        sequence_id(sample_idx) = int32(sid);
        time_index(sample_idx) = int32(tt);
        if isfield(ch, 'alpha_com')
            alpha_com(sample_idx) = single(ch.alpha_com);
        else
            alpha_com(sample_idx) = single(mean(ch.alpha));
        end
        if isfield(ch, 'v_norm')
            v_norm(sample_idx) = single(ch.v_norm);
        else
            v_norm(sample_idx) = single(alpha_com(sample_idx) / max(alpha_max_raw, 1e-12));
        end
        if isfield(ch, 'delta_alpha_rms')
            delta_alpha_rms(sample_idx) = single(ch.delta_alpha_rms);
        else
            delta_alpha_rms(sample_idx) = single(sqrt(mean(abs(ch.alpha - mean(ch.alpha)).^2)));
        end
        if isfield(ch, 'alpha_hat')
            alpha_hat_t(sample_idx) = single(ch.alpha_hat);
        end
        if isfield(ch, 'alpha_raw')
            alpha_raw_now = reshape(double(ch.alpha_raw(:)), 1, []);
        else
            alpha_raw_now = reshape(double(ch.alpha(:)), 1, []);
        end
        if isfield(ch, 'alpha_res')
            alpha_res_now = reshape(double(ch.alpha_res(:)), 1, []);
        else
            alpha_res_now = reshape(double(ch.alpha(:)), 1, []);
        end
        alpha_raw_t(sample_idx, :) = single(alpha_raw_now);
        alpha_res_t(sample_idx, :) = single(alpha_res_now);
        alpha_raw_max(sample_idx) = single(max(abs(alpha_raw_now)));
        alpha_res_max(sample_idx) = single(max(abs(alpha_res_now)));
        seq_alpha_raw_abs(tt, :) = abs(alpha_raw_now);
        seq_alpha_res_abs(tt, :) = abs(alpha_res_now);

        % Context update for next frame (behavior policy = fixed action, no oracle leakage).
        behavior_action = base_action;
        switch_now = double(tt > 1 && behavior_action ~= prev_behavior_action);
        if switch_count < cfg.context_switch_window
            switch_count = switch_count + 1;
            switch_hist(switch_count) = switch_now;
        else
            switch_hist = [switch_hist(2:end); switch_now]; %#ok<AGROW>
        end

        prev_action_norm = (behavior_action - 1) / max(num_actions - 1, 1);
        prev_reward = reward_primary_vec(behavior_action);
        prev_residual_proxy = base_residual_proxy;
        prev_frob_norm = feat(1);
        prev_offdiag_ratio = feat(3);
        prev_band_energy_ratio = feat(6);
        prev_behavior_action = behavior_action;

        if mod(sample_idx, 100) == 0
            fprintf('progress %d/%d\n', sample_idx, num_samples);
        end
    end

    raw_seq_vals = seq_alpha_raw_abs(:);
    res_seq_vals = seq_alpha_res_abs(:);
    sequence_alpha_stats.sequence_id(sid) = int32(sid);
    sequence_alpha_stats.alpha_raw_abs_max(sid) = single(max(raw_seq_vals));
    sequence_alpha_stats.alpha_raw_p99_abs(sid) = single(prctile(raw_seq_vals, 99));
    sequence_alpha_stats.alpha_raw_p995_abs(sid) = single(prctile(raw_seq_vals, 99.5));
    sequence_alpha_stats.alpha_res_abs_max(sid) = single(max(res_seq_vals));
    sequence_alpha_stats.alpha_res_p99_abs(sid) = single(prctile(res_seq_vals, 99));
    sequence_alpha_stats.alpha_res_p995_abs(sid) = single(prctile(res_seq_vals, 99.5));
end

dataset_mode = mode;
dataset_mode_code = int32(use_timevary);
doppler_mode = tv_cfg.doppler_mode;
doppler_mode_code = tv_cfg.doppler_mode_code;
alpha_raw_stats = build_abs_stats_struct(abs(double(alpha_raw_t)));
alpha_res_stats = build_abs_stats_struct(abs(double(alpha_res_t)));
alpha_max_raw_meta = tv_cfg.alpha_max_raw;
alpha_max_res_meta = tv_cfg.alpha_max_res;
enable_resampling_comp = logical(tv_cfg.enable_resampling_comp);
alpha_hat_mode = tv_cfg.alpha_hat_mode;

timevary_hparams = struct();
timevary_hparams.rho_alpha = tv_cfg.rho_alpha;
timevary_hparams.rho_h = tv_cfg.rho_h;
timevary_hparams.alpha_max = tv_cfg.alpha_max;
timevary_hparams.alpha_max_raw = tv_cfg.alpha_max_raw;
timevary_hparams.alpha_max_res = tv_cfg.alpha_max_res;
timevary_hparams.doppler_mode_code = tv_cfg.doppler_mode_code;
timevary_hparams.rho_acc = tv_cfg.rho_acc;
timevary_hparams.sigma_acc = tv_cfg.sigma_acc;
timevary_hparams.rho_delta = tv_cfg.rho_delta;
timevary_hparams.sigma_delta = tv_cfg.sigma_delta;
timevary_hparams.delta_max = tv_cfg.delta_max;
timevary_hparams.motion_profile = tv_cfg.motion_profile;
timevary_hparams.target_track_gain = tv_cfg.target_track_gain;
timevary_hparams.target_blend = tv_cfg.target_blend;
timevary_hparams.profile_v_peak = tv_cfg.profile_v_peak;
timevary_hparams.profile_turn_range = tv_cfg.profile_turn_range;
timevary_hparams.profile_recede_range = tv_cfg.profile_recede_range;
timevary_hparams.profile_heave_amp = tv_cfg.profile_heave_amp;
timevary_hparams.profile_heave_cycles = tv_cfg.profile_heave_cycles;
timevary_hparams.profile_secondary_amp = tv_cfg.profile_secondary_amp;
timevary_hparams.profile_secondary_cycles = tv_cfg.profile_secondary_cycles;
timevary_hparams.profile_jitter_std = tv_cfg.profile_jitter_std;
timevary_hparams.path_projection_mode = tv_cfg.path_projection_mode;
timevary_hparams.beta_min = tv_cfg.beta_min;
timevary_hparams.beta_max = tv_cfg.beta_max;
timevary_hparams.ell_mode = tv_cfg.ell_mode;
timevary_hparams.pdp_mode = tv_cfg.pdp_mode;
timevary_hparams.num_frames = int32(num_frames);
timevary_hparams.enable_resampling_comp = logical(tv_cfg.enable_resampling_comp);
timevary_hparams.alpha_hat_mode = tv_cfg.alpha_hat_mode;
timevary_hparams.clip_alpha_res = logical(tv_cfg.clip_alpha_res);
timevary_hparams.alpha_raw_abs_max = alpha_raw_stats.abs_max;
timevary_hparams.alpha_raw_abs_mean = alpha_raw_stats.abs_mean;
timevary_hparams.alpha_raw_p99_abs = alpha_raw_stats.p99_abs;
timevary_hparams.alpha_raw_p995_abs = alpha_raw_stats.p995_abs;
timevary_hparams.alpha_res_abs_max = alpha_res_stats.abs_max;
timevary_hparams.alpha_res_abs_mean = alpha_res_stats.abs_mean;
timevary_hparams.alpha_res_p99_abs = alpha_res_stats.p99_abs;
timevary_hparams.alpha_res_p995_abs = alpha_res_stats.p995_abs;
timevary_hparams.reward_primary_key = cfg.reward_primary_key;
timevary_hparams.reward_relbase_floor = cfg.reward_relbase_floor;
timevary_hparams.reward_proxy_improve_lambda = cfg.reward_proxy_improve_lambda;
timevary_hparams.reward_clip_abs = cfg.reward_clip_abs;
timevary_hparams.include_physical_doppler_state = logical(cfg.include_physical_doppler_state);
timevary_hparams.include_policy_history_state = logical(cfg.include_policy_history_state);
timevary_hparams.label_snr_mid = cfg.label_snr_mid;
timevary_hparams.label_snr_high = cfg.label_snr_high;
timevary_hparams.label_eval_repeats_low = int32(cfg.label_eval_repeats_low);
timevary_hparams.label_eval_repeats_mid = int32(cfg.label_eval_repeats_mid);
timevary_hparams.label_eval_repeats_high = int32(cfg.label_eval_repeats_high);
detector_target = cfg.detector_target;
rl_problem_type = 'contextual_bandit';
reward_primary_key = cfg.reward_primary_key;

save(cfg.output_file, ...
    'state', 'reward', 'reward_ber', 'reward_proxy', 'reward_mix', ...
    'reward_relbase', 'reward_proxy_gain', 'reward_relbase_proxy', 'reward_primary_key', ...
    'ber_actions', 'mse_proxy_actions', ...
    'oracle_action', 'oracle_action_reward', ...
    'chosen_action_fixed', 'chosen_action_oracle', ...
    'snr_db', 'sequence_id', 'time_index', ...
    'alpha_com', 'v_norm', 'delta_alpha_rms', ...
    'alpha_hat_t', 'alpha_raw_t', 'alpha_res_t', 'alpha_raw_max', 'alpha_res_max', ...
    'c1_grid', 'c1_base', 'base_action', ...
    'dataset_mode', 'dataset_mode_code', 'doppler_mode', 'doppler_mode_code', 'detector_target', 'rl_problem_type', ...
    'num_seq', 'num_frames', 'timevary_hparams', ...
    'alpha_max_raw_meta', 'alpha_max_res_meta', 'enable_resampling_comp', 'alpha_hat_mode', ...
    'alpha_raw_stats', 'alpha_res_stats', 'sequence_alpha_stats', ...
    'feature_names', ...
    'cfg', ...
    'oampnet_param_path', ...
    '-v7.3');

if cfg.output_csv
    csv_path = replace(cfg.output_file, '.mat', '.csv');
    fid = fopen(csv_path, 'w');
    fprintf(fid, 'sequence_id,time_index,chosen_action_oracle,chosen_action_fixed,snr_db,dataset_mode,doppler_mode,alpha_com,v_norm,delta_alpha_rms,alpha_hat_t,alpha_raw_max,alpha_res_max\n');
    for i = 1:num_samples
        fprintf(fid, '%d,%d,%d,%d,%.6f,%s,%s,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n', sequence_id(i), time_index(i), ...
            oracle_action(i) - 1, chosen_action_fixed(i) - 1, snr_db(i), dataset_mode, doppler_mode, ...
            alpha_com(i), v_norm(i), delta_alpha_rms(i), alpha_hat_t(i), alpha_raw_max(i), alpha_res_max(i));
    end
    fclose(fid);
    fprintf('Saved csv index: %s\n', csv_path);
end

fprintf('\nSaved offline dataset to %s\n', cfg.output_file);
fprintf('mode=%s, samples=%d, num_actions=%d, state_dim=%d\n', dataset_mode, num_samples, num_actions, feat_dim);
fprintf('detector_target=%s\n', cfg.detector_target);
fprintf('alpha stats | raw max=%.3e p99=%.3e p995=%.3e | res max=%.3e p99=%.3e p995=%.3e\n', ...
    alpha_raw_stats.abs_max, alpha_raw_stats.p99_abs, alpha_raw_stats.p995_abs, ...
    alpha_res_stats.abs_max, alpha_res_stats.p99_abs, alpha_res_stats.p995_abs);
if strcmpi(cfg.detector_target, 'oampnet')
    fprintf('oampnet_params=%s\n', oampnet_param_path);
end

if nargout > 0
    export_result = struct();
    export_result.output_file = cfg.output_file;
    export_result.dataset_mode = dataset_mode;
    export_result.doppler_mode = doppler_mode;
    export_result.num_samples = num_samples;
    export_result.state_dim = feat_dim;
    export_result.feature_names = feature_names;
    export_result.alpha_raw_stats = alpha_raw_stats;
    export_result.alpha_res_stats = alpha_res_stats;
end
end

%% ===== helper functions =====
function out = get_struct(s, name, default_v)
if isfield(s, name)
    out = s.(name);
else
    out = default_v;
end
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

function x_det = run_target_detector(y, H, noise_var, detector_target, oamp_iter, oamp_damping, oampnet_params, Q)
if strcmpi(detector_target, 'lmmse')
    x_det = lmmse_detector(y, H, noise_var);
elseif strcmpi(detector_target, 'oamp')
    x_det = oamp_detector(y, H, noise_var, oamp_iter, oamp_damping, Q);
elseif strcmpi(detector_target, 'oampnet')
    if isempty(oampnet_params)
        error('oampnet_params is empty while detector_target=oampnet');
    end
    x_det = oampnet_detector(y, H, noise_var, oampnet_params, Q);
else
    error('Unsupported detector_target=%s', detector_target);
end
end

function XTB = precompute_idaf_basis(N, c1, c2)
n = (0:N-1).';
m = 0:N-1;
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

n = (0:N-1).';
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

function stats = build_abs_stats_struct(abs_mat)
vals = double(abs(abs_mat(:)));
if isempty(vals)
    vals = 0.0;
end
stats = struct();
stats.abs_max = max(vals);
stats.abs_mean = mean(vals);
stats.p99_abs = prctile(vals, 99);
stats.p995_abs = prctile(vals, 99.5);
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

d2 = abs(x - const.') .^ 2;
[~, idx] = min(d2, [], 2);
bits2 = labels(idx, :);
bits = reshape(bits2.', [], 1);
end

function x_hard = qpsk_hard_projection(x)
re = sign(real(x));
im = sign(imag(x));
re(re == 0) = 1;
im(im == 0) = 1;
x_hard = (re + 1j * im) / sqrt(2);
end

function seed_u32 = safe_rng_seed(seed_in)
seed_u32 = mod(double(seed_in), 2^32 - 1);
if seed_u32 < 0
    seed_u32 = seed_u32 + (2^32 - 1);
end
seed_u32 = floor(seed_u32);
end

function num_eval = get_label_eval_repeats(snr_pick, cfg)
if snr_pick >= cfg.label_snr_high
    num_eval = cfg.label_eval_repeats_high;
elseif snr_pick >= cfg.label_snr_mid
    num_eval = cfg.label_eval_repeats_mid;
else
    num_eval = cfg.label_eval_repeats_low;
end
num_eval = max(1, round(double(num_eval)));
end

function reward_vec = select_primary_reward(primary_key, reward_ber_vec, reward_mix_vec, reward_relbase_vec, reward_relbase_proxy_vec)
switch lower(char(string(primary_key)))
    case 'reward_ber'
        reward_vec = reward_ber_vec;
    case 'reward_relbase'
        reward_vec = reward_relbase_vec;
    case 'reward_relbase_proxy'
        reward_vec = reward_relbase_proxy_vec;
    otherwise
        reward_vec = reward_mix_vec;
end
end

function x = clip_reward_vec(x, clip_abs)
clip_abs = max(double(clip_abs), 0.0);
if clip_abs > 0
    x = min(max(x, -clip_abs), clip_abs);
end
end
