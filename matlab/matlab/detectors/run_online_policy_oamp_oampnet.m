function result = run_online_policy_oamp_oampnet(user_cfg)
%% run_online_policy_oamp_oampnet.m
% Online measured evaluation with true sequence rollout:
% - fixed base action
% - RL-selected action
% - oracle action (evaluation only)
%
% Legacy note:
% Previous version sampled each channel independently by ch_k.
% New version uses sequence-level recursive channel state.

if nargin < 1
    user_cfg = struct();
end

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
results_dir = fullfile(project_root, 'results');
data_dir = fullfile(project_root, 'data');
common_dir = fullfile(project_root, 'matlab', 'common');
if ~exist(results_dir, 'dir'); mkdir(results_dir); end
addpath(this_dir);
addpath(common_dir);

%% ===== Config =====
cfg = struct();
cfg.run_profile = 'smoke';  % 'smoke' (legacy quick check) | 'paper' (formal BER-SNR sweep)
cfg.paper_id = 'tsv2seq';
cfg.version = 'tsv1';
cfg.snr_db_list = [14 18];
cfg.num_seq = 80;
cfg.num_frames = 10;
cfg.num_noise = 4;
cfg.seed_base = 20260310;

cfg.detector_target = 'oampnet';
cfg.oamp_iter = 10;
cfg.oamp_damping = 0.9;
cfg.Q = 0;
cfg.eval_detector_mode = 'both';   % 'both' (legacy) | 'target_only' (fast smoke)

cfg.use_oracle_state = false;
cfg.include_physical_doppler_state = false;
cfg.include_policy_history_state = true;
cfg.context_switch_window = 5;

cfg.policy_path = fullfile(results_dir, 'rl_c1_policy_matlab_params.mat');
cfg.params_path = fullfile(data_dir, ['oampnet_v4_' cfg.version '_params.mat']);
cfg.output_csv = fullfile(results_dir, 'policy_online_detector_frame_rollout.csv');
cfg.output_mat = fullfile(results_dir, 'policy_online_detector_eval_result.mat');
cfg.output_csv_wide = fullfile(results_dir, 'ber_results_policy_online_oamp_oampnet.csv');
cfg.output_csv_fig3 = fullfile(results_dir, 'fig3_ber_vs_snr_main_online.csv');
cfg.output_csv_fig4 = fullfile(results_dir, 'fig4_ablation_gain_online.csv');
cfg.output_csv_c1_diag = fullfile(results_dir, 'c1_detector_diagnostic_online.csv');
cfg.figure_png_main = fullfile(project_root, 'figures', 'fig3_ber_vs_snr_main_online.png');
cfg.figure_png_gain = fullfile(project_root, 'figures', 'fig4_ablation_gain_online.png');
cfg.figure_png_c1_diag_oamp = fullfile(project_root, 'figures', 'c1_detector_diagnostic_oamp_online.png');
cfg.figure_png_c1_diag_oampnet = fullfile(project_root, 'figures', 'c1_detector_diagnostic_oampnet_online.png');

cfg.paper_snr_db_list = 0:2:20;
cfg.paper_target_error_bits_low = 300;
cfg.paper_target_error_bits_mid = 200;
cfg.paper_target_error_bits_high = 100;
cfg.paper_max_bits_low = 2e5;
cfg.paper_max_bits_mid = 5e5;
cfg.paper_max_bits_high = 2e6;

cfg = merge_struct(cfg, user_cfg);
run_profile = char(string(get_struct(cfg, 'run_profile', 'smoke')));
paper_tag = char(string(cfg.paper_id));
if contains(lower(paper_tag), 'vdop') && (~isfield(user_cfg, 'doppler_mode') || isempty(user_cfg.doppler_mode))
    cfg.doppler_mode = 'common_with_path_residual';
end
if contains(lower(paper_tag), 'vdop_ctrl')
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
end

paper_results_dir = fullfile(results_dir, ['paper_' paper_tag]);
paper_figures_dir = fullfile(project_root, 'figures', ['paper_' paper_tag]);
if strcmpi(run_profile, 'paper')
    if ~isfield(user_cfg, 'version') || isempty(user_cfg.version)
        cfg.version = sprintf('%s_paper', paper_tag);
    end
    if ~isfield(user_cfg, 'snr_db_list') || isempty(user_cfg.snr_db_list)
        cfg.snr_db_list = cfg.paper_snr_db_list;
    end
    if ~isfield(user_cfg, 'num_noise') || isempty(user_cfg.num_noise)
        % Paper profile favors more channel evolution and bit-budget control
        % over repeated noise draws on the exact same frame.
        cfg.num_noise = 1;
    end
    if ~exist(paper_results_dir, 'dir'); mkdir(paper_results_dir); end
    if ~exist(paper_figures_dir, 'dir'); mkdir(paper_figures_dir); end
    cfg.output_csv = fullfile(paper_results_dir, ['policy_online_detector_frame_rollout_' paper_tag '.csv']);
    cfg.output_mat = fullfile(paper_results_dir, ['policy_online_detector_eval_result_' paper_tag '.mat']);
    cfg.output_csv_wide = fullfile(paper_results_dir, ['ber_results_policy_online_oamp_oampnet_' paper_tag '.csv']);
    cfg.output_csv_fig3 = fullfile(paper_results_dir, ['ber_comparison_matlab_' paper_tag '.csv']);
    cfg.output_csv_fig4 = fullfile(paper_results_dir, ['gain_vs_snr_' paper_tag '.csv']);
    cfg.output_csv_c1_diag = fullfile(paper_results_dir, ['c1_detector_diagnostic_' paper_tag '.csv']);
    cfg.figure_png_main = fullfile(paper_figures_dir, ['ber_comparison_matlab_' paper_tag '.png']);
    cfg.figure_png_gain = fullfile(paper_figures_dir, ['gain_vs_snr_' paper_tag '.png']);
    cfg.figure_png_c1_diag_oamp = fullfile(paper_figures_dir, ['c1_detector_diagnostic_oamp_' paper_tag '.png']);
    cfg.figure_png_c1_diag_oampnet = fullfile(paper_figures_dir, ['c1_detector_diagnostic_oampnet_' paper_tag '.png']);
end
if ~isfield(user_cfg, 'params_path') || isempty(user_cfg.params_path)
    cfg.params_path = fullfile(data_dir, ['oampnet_v4_' cfg.version '_params.mat']);
end
if strcmpi(run_profile, 'paper') && (~isfield(user_cfg, 'policy_path') || isempty(user_cfg.policy_path))
    cfg.policy_path = fullfile(paper_results_dir, ['rl_c1_policy_matlab_params_' paper_tag '_paper.mat']);
end

%% ===== Load RL policy =====
if ~exist(cfg.policy_path, 'file')
    error('Missing policy mat: %s. Run python/rl_c1/export_policy_to_matlab.py first.', cfg.policy_path);
end
policy = load(cfg.policy_path);
policy_feature_names = {};
if isfield(policy, 'feature_names')
    policy_feature_names = normalize_feature_names(policy.feature_names);
end

%% ===== Load OAMPNet params =====
detector_target_pre = char(string(get_struct(cfg, 'detector_target', 'oampnet')));
eval_mode_pre = char(string(get_struct(cfg, 'eval_detector_mode', 'both')));
need_oampnet_params = strcmpi(eval_mode_pre, 'both') || strcmpi(detector_target_pre, 'oampnet');
if need_oampnet_params
    if ~exist(cfg.params_path, 'file')
        error('Missing OAMPNet params: %s', cfg.params_path);
    end
    oampnet_params = load(cfg.params_path);
else
    oampnet_params = [];
end

%% ===== AFDM/System params =====
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
Nv = 2;
Q = cfg.Q;
N_eff = N - 2 * Q;
data_idx = (Q + 1):(N - Q);
Lcpp = max(1, ceil(ell_max / (1 - alpha_max_res)));
Lcps = max(1, ceil(alpha_max_res * N / (1 + alpha_max_res)));
L = N + Lcpp + Lcps;

if isfield(policy, 'c1_grid')
    c1_grid = double(policy.c1_grid(:));
else
    kmax = ceil((alpha_max_res * fc) * T_sym);
    den = (1 - 4 * alpha_max_res * (N - 1));
    c1_base = (2 * kmax + 2 * alpha_max_res * (N - 1) + 2 * Nv + 1) / (2 * N * den);
    ratios = linspace(0.6, 1.4, 21);
    c1_grid = c1_base * ratios(:);
end
num_actions = numel(c1_grid);

kmax = ceil((alpha_max_res * fc) * T_sym);
den = (1 - 4 * alpha_max_res * (N - 1));
c1_base_formula = (2 * kmax + 2 * alpha_max_res * (N - 1) + 2 * Nv + 1) / (2 * N * den);
[~, base_idx] = min(abs(c1_grid - c1_base_formula));
c2 = sqrt(2) / N;

fprintf('==== Online Policy + Detector Evaluation (Sequence Rollout) ====\n');
fprintf('run_profile: %s\n', run_profile);
fprintf('policy: %s\n', cfg.policy_path);
if need_oampnet_params
    fprintf('oampnet params: %s\n', cfg.params_path);
else
    fprintf('oampnet params: not required for detector_target=%s, eval_mode=%s\n', detector_target_pre, eval_mode_pre);
end
fprintf('snr list: %s\n', mat2str(cfg.snr_db_list));
fprintf('num_seq=%d, num_frames=%d, num_noise=%d, num_actions=%d\n', ...
    cfg.num_seq, cfg.num_frames, cfg.num_noise, num_actions);
fprintf('use_oracle_state=%d (default should be 0)\n', cfg.use_oracle_state);
fprintf('alpha_max_raw=%.1e, alpha_max_res=%.1e, enable_resampling_comp=%d, alpha_hat_mode=%s\n', ...
    alpha_max_raw, alpha_max_res, get_struct(cfg, 'enable_resampling_comp', true), ...
    char(string(get_struct(cfg, 'alpha_hat_mode', 'common_component'))));

%% ===== Precompute per-action basis =====
Xext_all = cell(num_actions, 1);
chirp1_all = cell(num_actions, 1);
chirp2_all = cell(num_actions, 1);
for a = 1:num_actions
    c1 = c1_grid(a);
    XTB = precompute_idaf_basis(N, c1, c2);
    Xext_all{a} = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps);

    n = (0:N-1).';
    chirp1_all{a} = exp(-1j * 2 * pi * c1 * (n .^ 2));
    chirp2_all{a} = exp(-1j * 2 * pi * c2 * (n .^ 2));
end

%% ===== Feature dim check =====
dummy_ctx = struct( ...
    'prev_action_norm', 0, ...
    'prev_reward', 0, ...
    'prev_residual_proxy', 0, ...
    'recent_switch_rate', 0, ...
    'frame_index_norm', 0, ...
    'prev_offdiag_ratio', 0, ...
    'prev_band_energy_ratio', 0, ...
    'prev_frob_norm', 0, ...
    'prev_alpha_com', 0, ...
    'prev_v_norm', 0, ...
    'prev_delta_alpha_rms', 0);
dummy_ch = struct('alpha', zeros(P, 1), 'ell', zeros(P, 1), 'h', ones(P, 1));
if ~isempty(policy_feature_names)
    [dummy_feat, feature_names] = extract_online_state_features_named( ...
        eye(N), [], [], 1.0, data_idx, dummy_ctx, cfg.use_oracle_state, dummy_ch, policy_feature_names);
else
    [dummy_feat, feature_names] = extract_online_state_features( ...
        eye(N), [], [], 1.0, data_idx, dummy_ctx, cfg.use_oracle_state, dummy_ch, ...
        cfg.include_physical_doppler_state, cfg.include_policy_history_state);
end
state_dim_expected = numel(dummy_feat);
if isfield(policy, 'state_dim')
    state_dim_policy = round(double(policy.state_dim(1)));
else
    state_dim_policy = numel(policy.state_mean(:));
end
if state_dim_policy ~= state_dim_expected
    error(['Policy state_dim mismatch. policy=%d, expected=%d. ' ...
        'If use_oracle_state/include_physical_doppler_state changed, retrain/re-export policy with matching feature schema.'], ...
        state_dim_policy, state_dim_expected);
end

%% ===== Time-vary config =====
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
    'num_frames', cfg.num_frames, ...
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

dataset_mode = 'timevary_sequence';
dataset_mode_code = int32(1);
doppler_mode = tv_cfg.doppler_mode;
doppler_mode_code = tv_cfg.doppler_mode_code;
detector_target = char(string(cfg.detector_target));
eval_detector_mode = char(string(cfg.eval_detector_mode));
compute_oamp = strcmpi(eval_detector_mode, 'both') || strcmpi(detector_target, 'oamp');
compute_oampnet = strcmpi(eval_detector_mode, 'both') || strcmpi(detector_target, 'oampnet');
if ~compute_oamp && ~compute_oampnet && ~strcmpi(detector_target, 'lmmse')
    error('Invalid eval mode / detector_target combination.');
end

%% ===== Containers =====
num_snr = numel(cfg.snr_db_list);
bits_per_frame = 2 * numel(data_idx) * cfg.num_noise;
if strcmpi(run_profile, 'paper')
    max_bits_global = max([cfg.paper_max_bits_low, cfg.paper_max_bits_mid, cfg.paper_max_bits_high]);
    rows_per_snr_cap = max(1, ceil(max_bits_global / max(bits_per_frame, 1))) + cfg.num_frames;
else
    rows_per_snr_cap = cfg.num_seq * cfg.num_frames;
end
total_rows = rows_per_snr_cap * num_snr;
seq_per_snr_cap = max(1, ceil(rows_per_snr_cap / max(cfg.num_frames, 1))) + 1;
total_seq_summary = seq_per_snr_cap * num_snr;

sequence_id = zeros(total_rows, 1, 'int32');
time_index = zeros(total_rows, 1, 'int32');
snr_db = zeros(total_rows, 1, 'single');

chosen_action_rl = zeros(total_rows, 1, 'int32');
chosen_action_oracle = zeros(total_rows, 1, 'int32');
chosen_action_fixed = int32(base_idx) * ones(total_rows, 1, 'int32');

reward_rl = zeros(total_rows, 1, 'single');
reward_oracle = zeros(total_rows, 1, 'single');
reward_fixed = zeros(total_rows, 1, 'single');

ber_rl = zeros(total_rows, 1, 'single');
ber_oracle = zeros(total_rows, 1, 'single');
ber_fixed = zeros(total_rows, 1, 'single');

ber_rl_oamp = zeros(total_rows, 1, 'single');
ber_oracle_oamp = zeros(total_rows, 1, 'single');
ber_fixed_oamp = zeros(total_rows, 1, 'single');
ber_rl_oampnet = zeros(total_rows, 1, 'single');
ber_oracle_oampnet = zeros(total_rows, 1, 'single');
ber_fixed_oampnet = zeros(total_rows, 1, 'single');

regret_rl = zeros(total_rows, 1, 'single');
switch_flag_rl = zeros(total_rows, 1, 'int32');
switch_flag_oracle = zeros(total_rows, 1, 'int32');

alpha_rms = zeros(total_rows, 1, 'single');
h_delta_norm = zeros(total_rows, 1, 'single');
alpha_com = zeros(total_rows, 1, 'single');
v_norm = zeros(total_rows, 1, 'single');
delta_alpha_rms = zeros(total_rows, 1, 'single');
alpha_hat_t = zeros(total_rows, 1, 'single');
alpha_raw_max = zeros(total_rows, 1, 'single');
alpha_res_max = zeros(total_rows, 1, 'single');

% Per-sequence summary.
sequence_summary = struct();
sequence_summary.global_sequence_id = zeros(total_seq_summary, 1, 'int32');
sequence_summary.snr_db = zeros(total_seq_summary, 1, 'single');
sequence_summary.switch_rate_rl = zeros(total_seq_summary, 1, 'single');
sequence_summary.switch_rate_oracle = zeros(total_seq_summary, 1, 'single');
sequence_summary.avg_regret_rl = zeros(total_seq_summary, 1, 'single');
sequence_summary.avg_ber_rl = zeros(total_seq_summary, 1, 'single');
sequence_summary.avg_ber_oracle = zeros(total_seq_summary, 1, 'single');
sequence_summary.avg_ber_fixed = zeros(total_seq_summary, 1, 'single');

row_ptr = 0;
seq_summary_ptr = 0;
error_count_fixed_oamp = zeros(num_snr, 1);
error_count_fixed_oampnet = zeros(num_snr, 1);
error_count_rl_oamp = zeros(num_snr, 1);
error_count_rl_oampnet = zeros(num_snr, 1);
error_count_oracle_oamp = zeros(num_snr, 1);
error_count_oracle_oampnet = zeros(num_snr, 1);
total_bits_snr = zeros(num_snr, 1);
num_sequence_run = zeros(num_snr, 1);
error_count_action_oamp = zeros(num_snr, num_actions);
error_count_action_oampnet = zeros(num_snr, num_actions);
total_bits_action = zeros(num_snr, num_actions);

for si = 1:num_snr
    snr_val = cfg.snr_db_list(si);
    noise_var = 1 / (10^(snr_val / 10));
    budget = paper_budget_for_snr(cfg, snr_val, run_profile);

    fprintf('SNR=%ddB rollout... target_err=%d max_bits=%.3e\n', ...
        snr_val, budget.target_error_bits, budget.max_bits);

    sid_local = 0;
    while true
        if ~strcmpi(run_profile, 'paper') && sid_local >= cfg.num_seq
            break;
        end
        if row_ptr + cfg.num_frames > total_rows
            error('Preallocation cap exceeded. Increase rows_per_snr_cap for this profile.');
        end
        sid_local = sid_local + 1;
        global_sid = int32((si - 1) * 100000 + sid_local);

        rng(safe_rng_seed(cfg.seed_base + 100000 * si + sid_local));
        [seq_state, ch_t] = init_timevary_channel_state(P, ell_max, alpha_max_res, tv_cfg);
        h_prev = ch_t.h;

        prev_action_rl = base_idx;
        prev_action_oracle = base_idx;
        prev_reward = 0.0;
        prev_resid_proxy = 0.0;
        prev_offdiag_ratio = 0.0;
        prev_band_energy_ratio = 0.0;
        prev_frob_norm = 0.0;
        prev_alpha_com = 0.0;
        prev_v_norm = 0.0;
        prev_delta_alpha_rms = 0.0;
        switch_hist = zeros(cfg.context_switch_window, 1);
        switch_count = 0;

        seq_rows = zeros(cfg.num_frames, 1);

        for tt = 1:cfg.num_frames
            if tt > 1
                [seq_state, ch_t] = step_timevary_channel_state(seq_state);
            end

            row_ptr = row_ptr + 1;
            seq_rows(tt) = row_ptr;

            sequence_id(row_ptr) = global_sid;
            time_index(row_ptr) = int32(tt);
            snr_db(row_ptr) = single(snr_val);

            alpha_rms(row_ptr) = single(sqrt(mean(abs(ch_t.alpha).^2)));
            h_delta_norm(row_ptr) = single(norm(ch_t.h - h_prev));
            if isfield(ch_t, 'alpha_com')
                alpha_com(row_ptr) = single(ch_t.alpha_com);
            else
                alpha_com(row_ptr) = single(mean(ch_t.alpha));
            end
            if isfield(ch_t, 'v_norm')
                v_norm(row_ptr) = single(ch_t.v_norm);
            else
                v_norm(row_ptr) = single(alpha_com(row_ptr) / max(alpha_max_raw, 1e-12));
            end
            if isfield(ch_t, 'delta_alpha_rms')
                delta_alpha_rms(row_ptr) = single(ch_t.delta_alpha_rms);
            else
                delta_alpha_rms(row_ptr) = single(sqrt(mean(abs(ch_t.alpha - mean(ch_t.alpha)).^2)));
            end
            if isfield(ch_t, 'alpha_hat')
                alpha_hat_t(row_ptr) = single(ch_t.alpha_hat);
            end
            if isfield(ch_t, 'alpha_raw')
                alpha_raw_max(row_ptr) = single(max(abs(ch_t.alpha_raw)));
            else
                alpha_raw_max(row_ptr) = single(max(abs(ch_t.alpha)));
            end
            if isfield(ch_t, 'alpha_res')
                alpha_res_max(row_ptr) = single(max(abs(ch_t.alpha_res)));
            else
                alpha_res_max(row_ptr) = single(max(abs(ch_t.alpha)));
            end
            h_prev = ch_t.h;
            if get_struct(cfg, 'log_alpha_frames', false)
                fprintf('  sid=%d frame=%d | alpha_hat=%.3e raw_max=%.3e res_max=%.3e\n', ...
                    sid_local, tt, alpha_hat_t(row_ptr), alpha_raw_max(row_ptr), alpha_res_max(row_ptr));
            end

            x = zeros(N, 1);
            x(data_idx) = qpsk_symbols(numel(data_idx));
            x = x * sqrt(N / N_eff);

            G = build_timescaling_G_sparse(N, L, Lcpp, ch_t, fc, dt);

            ber_action_oamp = nan(num_actions, 1);
            ber_action_oampnet = nan(num_actions, 1);
            ber_action_target = zeros(num_actions, 1);
            err_action_oamp = nan(num_actions, 1);
            err_action_oampnet = nan(num_actions, 1);

            Heff_base = [];
            y_base_obs = [];
            x_base_obs = [];
            base_residual_proxy = 0.0;
            bits = 0;

            for a = 1:num_actions
                YT = G * Xext_all{a};
                Heff = afdm_demod_matrix(YT, chirp1_all{a}, chirp2_all{a});

                Hsub = Heff(data_idx, data_idx);
                sub_power = (norm(Hsub, 'fro')^2) / N_eff;
                Heff = Heff / sqrt(max(sub_power, 1e-12));

                if a == base_idx
                    Heff_base = Heff;
                end

                err_o = 0;
                err_n = 0;
                err_t = 0;
                bits_local = 0;
                for ni = 1:cfg.num_noise
                    rng(safe_rng_seed(cfg.seed_base + 100000 * si + 1000 * sid_local + 100 * tt + 10 * a + ni));
                    w = sqrt(noise_var / 2) * (randn(N, 1) + 1j * randn(N, 1));
                    y = Heff * x + w;

                    x_o = [];
                    x_n = [];
                    if compute_oamp
                        x_o = oamp_detector(y, Heff, noise_var, cfg.oamp_iter, cfg.oamp_damping, Q);
                        err_o = err_o + count_qpsk_bit_errors(x_o(data_idx), x(data_idx));
                    end
                    if compute_oampnet
                        x_n = oampnet_detector(y, Heff, noise_var, oampnet_params, Q);
                        err_n = err_n + count_qpsk_bit_errors(x_n(data_idx), x(data_idx));
                    end

                    if strcmpi(detector_target, 'lmmse')
                        x_t = lmmse_detector(y, Heff, noise_var);
                    elseif strcmpi(detector_target, 'oamp')
                        if isempty(x_o)
                            x_o = oamp_detector(y, Heff, noise_var, cfg.oamp_iter, cfg.oamp_damping, Q);
                        end
                        x_t = x_o;
                    else
                        if isempty(x_n)
                            x_n = oampnet_detector(y, Heff, noise_var, oampnet_params, Q);
                        end
                        x_t = x_n;
                    end
                    err_t = err_t + count_qpsk_bit_errors(x_t(data_idx), x(data_idx));
                    bits_local = bits_local + 2 * numel(data_idx);

                    if a == base_idx && ni == 1
                        x_for_state = x_t;
                        y_base_obs = y;
                        x_base_obs = x_for_state;
                        x_hard = qpsk_hard_projection(x_for_state);
                        resid = y - Heff * x_hard;
                        base_residual_proxy = mean(abs(resid).^2) / max(noise_var, 1e-12);
                    end
                end
                bits = max(bits, bits_local);
                total_bits_action(si, a) = total_bits_action(si, a) + bits_local;

                if compute_oamp
                    err_action_oamp(a) = err_o;
                    ber_action_oamp(a) = err_o / max(bits_local, 1);
                    error_count_action_oamp(si, a) = error_count_action_oamp(si, a) + err_o;
                end
                if compute_oampnet
                    err_action_oampnet(a) = err_n;
                    ber_action_oampnet(a) = err_n / max(bits_local, 1);
                    error_count_action_oampnet(si, a) = error_count_action_oampnet(si, a) + err_n;
                end
                ber_action_target(a) = err_t / max(bits_local, 1);
            end

            if compute_oamp
                [~, oracle_idx_oamp] = min(ber_action_oamp);
            else
                oracle_idx_oamp = base_idx;
            end
            if compute_oampnet
                [~, oracle_idx_oampnet] = min(ber_action_oampnet);
            else
                oracle_idx_oampnet = base_idx;
            end
            [~, oracle_idx_target] = min(ber_action_target);
            ber_target = ber_action_target;
            reward_target = -ber_target;

            ctx = struct();
            ctx.prev_action_norm = (prev_action_rl - 1) / max(num_actions - 1, 1);
            ctx.prev_reward = prev_reward;
            ctx.prev_residual_proxy = prev_resid_proxy;
            if switch_count > 0
                ctx.recent_switch_rate = mean(switch_hist(1:switch_count));
            else
                ctx.recent_switch_rate = 0.0;
            end
            ctx.frame_index_norm = (tt - 1) / max(cfg.num_frames - 1, 1);
            ctx.prev_offdiag_ratio = prev_offdiag_ratio;
            ctx.prev_band_energy_ratio = prev_band_energy_ratio;
            ctx.prev_frob_norm = prev_frob_norm;
            ctx.prev_alpha_com = prev_alpha_com;
            ctx.prev_v_norm = prev_v_norm;
            ctx.prev_delta_alpha_rms = prev_delta_alpha_rms;

            if ~isempty(policy_feature_names)
                state_feat = extract_online_state_features_named( ...
                    Heff_base, y_base_obs, x_base_obs, noise_var, data_idx, ctx, ...
                    cfg.use_oracle_state, ch_t, policy_feature_names);
            else
                state_feat = extract_online_state_features( ...
                    Heff_base, y_base_obs, x_base_obs, noise_var, data_idx, ctx, ...
                    cfg.use_oracle_state, ch_t, cfg.include_physical_doppler_state, cfg.include_policy_history_state);
            end
            rl_idx = policy_greedy_from_state(state_feat, policy);

            chosen_action_rl(row_ptr) = int32(rl_idx);
            chosen_action_oracle(row_ptr) = int32(oracle_idx_target);

            reward_rl(row_ptr) = single(reward_target(rl_idx));
            reward_oracle(row_ptr) = single(reward_target(oracle_idx_target));
            reward_fixed(row_ptr) = single(reward_target(base_idx));

            ber_rl(row_ptr) = single(ber_target(rl_idx));
            ber_oracle(row_ptr) = single(ber_target(oracle_idx_target));
            ber_fixed(row_ptr) = single(ber_target(base_idx));

            if compute_oamp
                ber_rl_oamp(row_ptr) = single(ber_action_oamp(rl_idx));
                ber_oracle_oamp(row_ptr) = single(ber_action_oamp(oracle_idx_oamp));
                ber_fixed_oamp(row_ptr) = single(ber_action_oamp(base_idx));
                error_count_fixed_oamp(si) = error_count_fixed_oamp(si) + err_action_oamp(base_idx);
                error_count_rl_oamp(si) = error_count_rl_oamp(si) + err_action_oamp(rl_idx);
                error_count_oracle_oamp(si) = error_count_oracle_oamp(si) + err_action_oamp(oracle_idx_oamp);
            else
                ber_rl_oamp(row_ptr) = single(nan);
                ber_oracle_oamp(row_ptr) = single(nan);
                ber_fixed_oamp(row_ptr) = single(nan);
            end
            if compute_oampnet
                ber_rl_oampnet(row_ptr) = single(ber_action_oampnet(rl_idx));
                ber_oracle_oampnet(row_ptr) = single(ber_action_oampnet(oracle_idx_oampnet));
                ber_fixed_oampnet(row_ptr) = single(ber_action_oampnet(base_idx));
                error_count_fixed_oampnet(si) = error_count_fixed_oampnet(si) + err_action_oampnet(base_idx);
                error_count_rl_oampnet(si) = error_count_rl_oampnet(si) + err_action_oampnet(rl_idx);
                error_count_oracle_oampnet(si) = error_count_oracle_oampnet(si) + err_action_oampnet(oracle_idx_oampnet);
            else
                ber_rl_oampnet(row_ptr) = single(nan);
                ber_oracle_oampnet(row_ptr) = single(nan);
                ber_fixed_oampnet(row_ptr) = single(nan);
            end
            total_bits_snr(si) = total_bits_snr(si) + bits;

            regret_rl(row_ptr) = single(ber_target(rl_idx) - ber_target(oracle_idx_target));

            sw_rl = int32(tt > 1 && rl_idx ~= prev_action_rl);
            sw_or = int32(tt > 1 && oracle_idx_target ~= prev_action_oracle);
            switch_flag_rl(row_ptr) = sw_rl;
            switch_flag_oracle(row_ptr) = sw_or;

            if switch_count < cfg.context_switch_window
                switch_count = switch_count + 1;
                switch_hist(switch_count) = double(sw_rl);
            else
                switch_hist = [switch_hist(2:end); double(sw_rl)]; %#ok<AGROW>
            end

            prev_action_rl = rl_idx;
            prev_action_oracle = oracle_idx_target;
            prev_reward = reward_target(rl_idx);
            prev_resid_proxy = base_residual_proxy;
            prev_frob_norm = feature_value_by_name_or_default(feature_names, state_feat, 'frob_norm', prev_frob_norm);
            prev_offdiag_ratio = feature_value_by_name_or_default(feature_names, state_feat, 'offdiag_energy_ratio', prev_offdiag_ratio);
            prev_band_energy_ratio = feature_value_by_name_or_default(feature_names, state_feat, 'band_energy_ratio', prev_band_energy_ratio);
            prev_alpha_com = feature_value_by_name_or_default(feature_names, state_feat, 'alpha_com', alpha_com(row_ptr));
            prev_v_norm = feature_value_by_name_or_default(feature_names, state_feat, 'v_norm', v_norm(row_ptr));
            prev_delta_alpha_rms = feature_value_by_name_or_default(feature_names, state_feat, 'delta_alpha_rms', delta_alpha_rms(row_ptr));
        end

        seq_summary_ptr = seq_summary_ptr + 1;
        rows = seq_rows;
        sequence_summary.global_sequence_id(seq_summary_ptr) = sequence_id(rows(1));
        sequence_summary.snr_db(seq_summary_ptr) = single(snr_val);
        if numel(rows) > 1
            sequence_summary.switch_rate_rl(seq_summary_ptr) = single(mean(double(switch_flag_rl(rows(2:end)))));
            sequence_summary.switch_rate_oracle(seq_summary_ptr) = single(mean(double(switch_flag_oracle(rows(2:end)))));
        else
            sequence_summary.switch_rate_rl(seq_summary_ptr) = single(0.0);
            sequence_summary.switch_rate_oracle(seq_summary_ptr) = single(0.0);
        end
        sequence_summary.avg_regret_rl(seq_summary_ptr) = single(mean(double(regret_rl(rows))));
        sequence_summary.avg_ber_rl(seq_summary_ptr) = single(mean(double(ber_rl(rows))));
        sequence_summary.avg_ber_oracle(seq_summary_ptr) = single(mean(double(ber_oracle(rows))));
        sequence_summary.avg_ber_fixed(seq_summary_ptr) = single(mean(double(ber_fixed(rows))));
        num_sequence_run(si) = sid_local;

        if strcmpi(run_profile, 'paper')
            if paper_should_stop( ...
                    budget, total_bits_snr(si), ...
                    [error_count_fixed_oamp(si), error_count_fixed_oampnet(si), error_count_rl_oamp(si), error_count_rl_oampnet(si)], ...
                    [compute_oamp, compute_oampnet, compute_oamp, compute_oampnet])
                break;
            end
        elseif sid_local >= cfg.num_seq
            break;
        end
    end

    fprintf('SNR=%ddB finished | sequences=%d total_bits=%.3e\n', snr_val, sid_local, total_bits_snr(si));
end

% Trim unused preallocated entries.
sequence_id = sequence_id(1:row_ptr);
time_index = time_index(1:row_ptr);
snr_db = snr_db(1:row_ptr);
chosen_action_rl = chosen_action_rl(1:row_ptr);
chosen_action_oracle = chosen_action_oracle(1:row_ptr);
chosen_action_fixed = chosen_action_fixed(1:row_ptr);
reward_rl = reward_rl(1:row_ptr);
reward_oracle = reward_oracle(1:row_ptr);
reward_fixed = reward_fixed(1:row_ptr);
ber_rl = ber_rl(1:row_ptr);
ber_oracle = ber_oracle(1:row_ptr);
ber_fixed = ber_fixed(1:row_ptr);
ber_rl_oamp = ber_rl_oamp(1:row_ptr);
ber_oracle_oamp = ber_oracle_oamp(1:row_ptr);
ber_fixed_oamp = ber_fixed_oamp(1:row_ptr);
ber_rl_oampnet = ber_rl_oampnet(1:row_ptr);
ber_oracle_oampnet = ber_oracle_oampnet(1:row_ptr);
ber_fixed_oampnet = ber_fixed_oampnet(1:row_ptr);
regret_rl = regret_rl(1:row_ptr);
switch_flag_rl = switch_flag_rl(1:row_ptr);
switch_flag_oracle = switch_flag_oracle(1:row_ptr);
alpha_rms = alpha_rms(1:row_ptr);
h_delta_norm = h_delta_norm(1:row_ptr);
alpha_com = alpha_com(1:row_ptr);
v_norm = v_norm(1:row_ptr);
delta_alpha_rms = delta_alpha_rms(1:row_ptr);
alpha_hat_t = alpha_hat_t(1:row_ptr);
alpha_raw_max = alpha_raw_max(1:row_ptr);
alpha_res_max = alpha_res_max(1:row_ptr);
sequence_summary.global_sequence_id = sequence_summary.global_sequence_id(1:seq_summary_ptr);
sequence_summary.snr_db = sequence_summary.snr_db(1:seq_summary_ptr);
sequence_summary.switch_rate_rl = sequence_summary.switch_rate_rl(1:seq_summary_ptr);
sequence_summary.switch_rate_oracle = sequence_summary.switch_rate_oracle(1:seq_summary_ptr);
sequence_summary.avg_regret_rl = sequence_summary.avg_regret_rl(1:seq_summary_ptr);
sequence_summary.avg_ber_rl = sequence_summary.avg_ber_rl(1:seq_summary_ptr);
sequence_summary.avg_ber_oracle = sequence_summary.avg_ber_oracle(1:seq_summary_ptr);
sequence_summary.avg_ber_fixed = sequence_summary.avg_ber_fixed(1:seq_summary_ptr);
total_rows = row_ptr;

alpha_raw_stats = build_abs_stats_struct(abs(double(alpha_raw_max)));
alpha_res_stats = build_abs_stats_struct(abs(double(alpha_res_max)));

%% ===== Aggregate by SNR =====
summary_by_snr = struct();
summary_by_snr.snr_db = cfg.snr_db_list(:);
summary_by_snr.avg_ber_fixed = zeros(num_snr, 1);
summary_by_snr.avg_ber_rl = zeros(num_snr, 1);
summary_by_snr.avg_ber_oracle = zeros(num_snr, 1);
summary_by_snr.avg_regret_rl = zeros(num_snr, 1);
summary_by_snr.avg_switch_rl = zeros(num_snr, 1);
summary_by_snr.avg_switch_oracle = zeros(num_snr, 1);
summary_by_snr.avg_ber_fixed_oamp = zeros(num_snr, 1);
summary_by_snr.avg_ber_rl_oamp = zeros(num_snr, 1);
summary_by_snr.avg_ber_oracle_oamp = zeros(num_snr, 1);
summary_by_snr.avg_ber_fixed_oampnet = zeros(num_snr, 1);
summary_by_snr.avg_ber_rl_oampnet = zeros(num_snr, 1);
summary_by_snr.avg_ber_oracle_oampnet = zeros(num_snr, 1);
summary_by_snr.num_sequence_run = zeros(num_snr, 1);
summary_by_snr.num_total_bits = total_bits_snr;
summary_by_snr.base_action = repmat(int32(base_idx - 1), num_snr, 1);
summary_by_snr.base_c1 = repmat(c1_grid(base_idx), num_snr, 1);

summary_by_snr.fixed_oamp_num_error_bits = error_count_fixed_oamp;
summary_by_snr.fixed_oampnet_num_error_bits = error_count_fixed_oampnet;
summary_by_snr.rl_oamp_num_error_bits = error_count_rl_oamp;
summary_by_snr.rl_oampnet_num_error_bits = error_count_rl_oampnet;
summary_by_snr.oracle_oamp_num_error_bits = error_count_oracle_oamp;
summary_by_snr.oracle_oampnet_num_error_bits = error_count_oracle_oampnet;
summary_by_snr.best_static_oamp_num_error_bits = zeros(num_snr, 1);
summary_by_snr.best_static_oampnet_num_error_bits = zeros(num_snr, 1);
summary_by_snr.best_static_oamp_action = -ones(num_snr, 1, 'int32');
summary_by_snr.best_static_oampnet_action = -ones(num_snr, 1, 'int32');
summary_by_snr.best_static_oamp_c1 = nan(num_snr, 1);
summary_by_snr.best_static_oampnet_c1 = nan(num_snr, 1);

summary_by_snr.fixed_oamp_ber_raw = nan(num_snr, 1);
summary_by_snr.fixed_oampnet_ber_raw = nan(num_snr, 1);
summary_by_snr.rl_oamp_ber_raw = nan(num_snr, 1);
summary_by_snr.rl_oampnet_ber_raw = nan(num_snr, 1);
summary_by_snr.oracle_oamp_ber_raw = nan(num_snr, 1);
summary_by_snr.oracle_oampnet_ber_raw = nan(num_snr, 1);
summary_by_snr.best_static_oamp_ber_raw = nan(num_snr, 1);
summary_by_snr.best_static_oampnet_ber_raw = nan(num_snr, 1);
summary_by_snr.fixed_oamp_ber_plot = nan(num_snr, 1);
summary_by_snr.fixed_oampnet_ber_plot = nan(num_snr, 1);
summary_by_snr.rl_oamp_ber_plot = nan(num_snr, 1);
summary_by_snr.rl_oampnet_ber_plot = nan(num_snr, 1);
summary_by_snr.oracle_oamp_ber_plot = nan(num_snr, 1);
summary_by_snr.oracle_oampnet_ber_plot = nan(num_snr, 1);
summary_by_snr.best_static_oamp_ber_plot = nan(num_snr, 1);
summary_by_snr.best_static_oampnet_ber_plot = nan(num_snr, 1);
summary_by_snr.fixed_oamp_is_upper_bound = zeros(num_snr, 1);
summary_by_snr.fixed_oampnet_is_upper_bound = zeros(num_snr, 1);
summary_by_snr.rl_oamp_is_upper_bound = zeros(num_snr, 1);
summary_by_snr.rl_oampnet_is_upper_bound = zeros(num_snr, 1);
summary_by_snr.oracle_oamp_is_upper_bound = zeros(num_snr, 1);
summary_by_snr.oracle_oampnet_is_upper_bound = zeros(num_snr, 1);
summary_by_snr.best_static_oamp_is_upper_bound = zeros(num_snr, 1);
summary_by_snr.best_static_oampnet_is_upper_bound = zeros(num_snr, 1);
summary_by_snr.oamp_static_headroom_raw = nan(num_snr, 1);
summary_by_snr.oamp_dynamic_headroom_raw = nan(num_snr, 1);
summary_by_snr.oamp_rl_gap_to_best_static_raw = nan(num_snr, 1);
summary_by_snr.oamp_rl_gap_to_oracle_raw = nan(num_snr, 1);
summary_by_snr.oamp_rl_gain_vs_best_static_pct = nan(num_snr, 1);
summary_by_snr.oamp_oracle_gain_vs_best_static_pct = nan(num_snr, 1);
summary_by_snr.oampnet_static_headroom_raw = nan(num_snr, 1);
summary_by_snr.oampnet_dynamic_headroom_raw = nan(num_snr, 1);
summary_by_snr.oampnet_rl_gap_to_best_static_raw = nan(num_snr, 1);
summary_by_snr.oampnet_rl_gap_to_oracle_raw = nan(num_snr, 1);
summary_by_snr.oampnet_rl_gain_vs_best_static_pct = nan(num_snr, 1);
summary_by_snr.oampnet_oracle_gain_vs_best_static_pct = nan(num_snr, 1);

for si = 1:num_snr
    m = (snr_db == cfg.snr_db_list(si));
    summary_by_snr.avg_ber_fixed(si) = safe_mean(ber_fixed(m));
    summary_by_snr.avg_ber_rl(si) = safe_mean(ber_rl(m));
    summary_by_snr.avg_ber_oracle(si) = safe_mean(ber_oracle(m));
    summary_by_snr.avg_regret_rl(si) = safe_mean(regret_rl(m));
    summary_by_snr.avg_ber_fixed_oamp(si) = safe_mean(ber_fixed_oamp(m));
    summary_by_snr.avg_ber_rl_oamp(si) = safe_mean(ber_rl_oamp(m));
    summary_by_snr.avg_ber_oracle_oamp(si) = safe_mean(ber_oracle_oamp(m));
    summary_by_snr.avg_ber_fixed_oampnet(si) = safe_mean(ber_fixed_oampnet(m));
    summary_by_snr.avg_ber_rl_oampnet(si) = safe_mean(ber_rl_oampnet(m));
    summary_by_snr.avg_ber_oracle_oampnet(si) = safe_mean(ber_oracle_oampnet(m));
    summary_by_snr.num_sequence_run(si) = num_sequence_run(si);

    seq_m = (sequence_summary.snr_db == cfg.snr_db_list(si));
    summary_by_snr.avg_switch_rl(si) = safe_mean(sequence_summary.switch_rate_rl(seq_m));
    summary_by_snr.avg_switch_oracle(si) = safe_mean(sequence_summary.switch_rate_oracle(seq_m));

    [summary_by_snr.fixed_oamp_ber_raw(si), summary_by_snr.fixed_oamp_ber_plot(si), summary_by_snr.fixed_oamp_is_upper_bound(si)] = ...
        ber_from_counts(summary_by_snr.fixed_oamp_num_error_bits(si), total_bits_snr(si));
    [summary_by_snr.fixed_oampnet_ber_raw(si), summary_by_snr.fixed_oampnet_ber_plot(si), summary_by_snr.fixed_oampnet_is_upper_bound(si)] = ...
        ber_from_counts(summary_by_snr.fixed_oampnet_num_error_bits(si), total_bits_snr(si));
    [summary_by_snr.rl_oamp_ber_raw(si), summary_by_snr.rl_oamp_ber_plot(si), summary_by_snr.rl_oamp_is_upper_bound(si)] = ...
        ber_from_counts(summary_by_snr.rl_oamp_num_error_bits(si), total_bits_snr(si));
    [summary_by_snr.rl_oampnet_ber_raw(si), summary_by_snr.rl_oampnet_ber_plot(si), summary_by_snr.rl_oampnet_is_upper_bound(si)] = ...
        ber_from_counts(summary_by_snr.rl_oampnet_num_error_bits(si), total_bits_snr(si));
    [summary_by_snr.oracle_oamp_ber_raw(si), summary_by_snr.oracle_oamp_ber_plot(si), summary_by_snr.oracle_oamp_is_upper_bound(si)] = ...
        ber_from_counts(summary_by_snr.oracle_oamp_num_error_bits(si), total_bits_snr(si));
    [summary_by_snr.oracle_oampnet_ber_raw(si), summary_by_snr.oracle_oampnet_ber_plot(si), summary_by_snr.oracle_oampnet_is_upper_bound(si)] = ...
        ber_from_counts(summary_by_snr.oracle_oampnet_num_error_bits(si), total_bits_snr(si));

    fprintf('SNR=%ddB | BER fixed=%.3e rl=%.3e oracle=%.3e | regret=%.3e | switch(rl)=%.3f\n', ...
        cfg.snr_db_list(si), summary_by_snr.avg_ber_fixed(si), summary_by_snr.avg_ber_rl(si), ...
        summary_by_snr.avg_ber_oracle(si), summary_by_snr.avg_regret_rl(si), summary_by_snr.avg_switch_rl(si));

    if compute_oamp
        valid_oamp = total_bits_action(si, :) > 0;
        oamp_static_ber = inf(1, num_actions);
        oamp_static_ber(valid_oamp) = error_count_action_oamp(si, valid_oamp) ./ max(total_bits_action(si, valid_oamp), 1);
        [~, best_idx_oamp] = min(oamp_static_ber);
        summary_by_snr.best_static_oamp_action(si) = int32(best_idx_oamp - 1);
        summary_by_snr.best_static_oamp_c1(si) = c1_grid(best_idx_oamp);
        summary_by_snr.best_static_oamp_num_error_bits(si) = error_count_action_oamp(si, best_idx_oamp);
        [summary_by_snr.best_static_oamp_ber_raw(si), summary_by_snr.best_static_oamp_ber_plot(si), summary_by_snr.best_static_oamp_is_upper_bound(si)] = ...
            ber_from_counts(summary_by_snr.best_static_oamp_num_error_bits(si), total_bits_action(si, best_idx_oamp));
        summary_by_snr.oamp_static_headroom_raw(si) = max(summary_by_snr.fixed_oamp_ber_raw(si) - summary_by_snr.best_static_oamp_ber_raw(si), 0.0);
        summary_by_snr.oamp_dynamic_headroom_raw(si) = max(summary_by_snr.best_static_oamp_ber_raw(si) - summary_by_snr.oracle_oamp_ber_raw(si), 0.0);
        summary_by_snr.oamp_rl_gap_to_best_static_raw(si) = summary_by_snr.rl_oamp_ber_raw(si) - summary_by_snr.best_static_oamp_ber_raw(si);
        summary_by_snr.oamp_rl_gap_to_oracle_raw(si) = summary_by_snr.rl_oamp_ber_raw(si) - summary_by_snr.oracle_oamp_ber_raw(si);
        summary_by_snr.oamp_rl_gain_vs_best_static_pct(si) = safe_relative_gain_pct(summary_by_snr.best_static_oamp_ber_raw(si), summary_by_snr.rl_oamp_ber_raw(si));
        summary_by_snr.oamp_oracle_gain_vs_best_static_pct(si) = safe_relative_gain_pct(summary_by_snr.best_static_oamp_ber_raw(si), summary_by_snr.oracle_oamp_ber_raw(si));
        fprintf('  OAMP    | base=%.3e best_static=%.3e rl=%.3e oracle=%.3e | dyn_headroom=%.3e rl-best=%.3e\n', ...
            summary_by_snr.fixed_oamp_ber_raw(si), summary_by_snr.best_static_oamp_ber_raw(si), ...
            summary_by_snr.rl_oamp_ber_raw(si), summary_by_snr.oracle_oamp_ber_raw(si), ...
            summary_by_snr.oamp_dynamic_headroom_raw(si), summary_by_snr.oamp_rl_gap_to_best_static_raw(si));
    end
    if compute_oampnet
        valid_oampnet = total_bits_action(si, :) > 0;
        oampnet_static_ber = inf(1, num_actions);
        oampnet_static_ber(valid_oampnet) = error_count_action_oampnet(si, valid_oampnet) ./ max(total_bits_action(si, valid_oampnet), 1);
        [~, best_idx_oampnet] = min(oampnet_static_ber);
        summary_by_snr.best_static_oampnet_action(si) = int32(best_idx_oampnet - 1);
        summary_by_snr.best_static_oampnet_c1(si) = c1_grid(best_idx_oampnet);
        summary_by_snr.best_static_oampnet_num_error_bits(si) = error_count_action_oampnet(si, best_idx_oampnet);
        [summary_by_snr.best_static_oampnet_ber_raw(si), summary_by_snr.best_static_oampnet_ber_plot(si), summary_by_snr.best_static_oampnet_is_upper_bound(si)] = ...
            ber_from_counts(summary_by_snr.best_static_oampnet_num_error_bits(si), total_bits_action(si, best_idx_oampnet));
        summary_by_snr.oampnet_static_headroom_raw(si) = max(summary_by_snr.fixed_oampnet_ber_raw(si) - summary_by_snr.best_static_oampnet_ber_raw(si), 0.0);
        summary_by_snr.oampnet_dynamic_headroom_raw(si) = max(summary_by_snr.best_static_oampnet_ber_raw(si) - summary_by_snr.oracle_oampnet_ber_raw(si), 0.0);
        summary_by_snr.oampnet_rl_gap_to_best_static_raw(si) = summary_by_snr.rl_oampnet_ber_raw(si) - summary_by_snr.best_static_oampnet_ber_raw(si);
        summary_by_snr.oampnet_rl_gap_to_oracle_raw(si) = summary_by_snr.rl_oampnet_ber_raw(si) - summary_by_snr.oracle_oampnet_ber_raw(si);
        summary_by_snr.oampnet_rl_gain_vs_best_static_pct(si) = safe_relative_gain_pct(summary_by_snr.best_static_oampnet_ber_raw(si), summary_by_snr.rl_oampnet_ber_raw(si));
        summary_by_snr.oampnet_oracle_gain_vs_best_static_pct(si) = safe_relative_gain_pct(summary_by_snr.best_static_oampnet_ber_raw(si), summary_by_snr.oracle_oampnet_ber_raw(si));
        fprintf('  OAMPNet | base=%.3e best_static=%.3e rl=%.3e oracle=%.3e | dyn_headroom=%.3e rl-best=%.3e\n', ...
            summary_by_snr.fixed_oampnet_ber_raw(si), summary_by_snr.best_static_oampnet_ber_raw(si), ...
            summary_by_snr.rl_oampnet_ber_raw(si), summary_by_snr.oracle_oampnet_ber_raw(si), ...
            summary_by_snr.oampnet_dynamic_headroom_raw(si), summary_by_snr.oampnet_rl_gap_to_best_static_raw(si));
    end
end

%% ===== Save long CSV =====
fid = fopen(cfg.output_csv, 'w');
fprintf(fid, ['sequence_id,time_index,chosen_action_rl,chosen_action_oracle,chosen_action_fixed,' ...
    'reward_rl,reward_oracle,reward_fixed,ber_rl,ber_oracle,ber_fixed,regret_rl,' ...
    'switch_flag_rl,switch_flag_oracle,snr_db,detector_target,dataset_mode,doppler_mode,alpha_com,v_norm,delta_alpha_rms,' ...
    'alpha_hat_t,alpha_raw_max,alpha_res_max\n']);
for i = 1:total_rows
    fprintf(fid, '%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%d,%.6f,%s,%s,%s,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n', ...
        sequence_id(i), time_index(i), ...
        chosen_action_rl(i) - 1, chosen_action_oracle(i) - 1, chosen_action_fixed(i) - 1, ...
        reward_rl(i), reward_oracle(i), reward_fixed(i), ...
        ber_rl(i), ber_oracle(i), ber_fixed(i), regret_rl(i), ...
        switch_flag_rl(i), switch_flag_oracle(i), snr_db(i), detector_target, dataset_mode, doppler_mode, ...
        alpha_com(i), v_norm(i), delta_alpha_rms(i), alpha_hat_t(i), alpha_raw_max(i), alpha_res_max(i));
end
fclose(fid);

%% ===== Save MAT =====
result = struct();
result.sequence_id = sequence_id;
result.time_index = time_index;
result.chosen_action_rl = chosen_action_rl - 1;
result.chosen_action_oracle = chosen_action_oracle - 1;
result.chosen_action_fixed = chosen_action_fixed - 1;
result.reward_rl = reward_rl;
result.reward_oracle = reward_oracle;
result.reward_fixed = reward_fixed;
result.ber_rl = ber_rl;
result.ber_oracle = ber_oracle;
result.ber_fixed = ber_fixed;
result.regret_rl = regret_rl;
result.switch_flag_rl = switch_flag_rl;
result.switch_flag_oracle = switch_flag_oracle;
result.snr_db = snr_db;
result.detector_target = detector_target;
result.dataset_mode = dataset_mode;
result.dataset_mode_code = dataset_mode_code;

result.ber_rl_oamp = ber_rl_oamp;
result.ber_oracle_oamp = ber_oracle_oamp;
result.ber_fixed_oamp = ber_fixed_oamp;
result.ber_rl_oampnet = ber_rl_oampnet;
result.ber_oracle_oampnet = ber_oracle_oampnet;
result.ber_fixed_oampnet = ber_fixed_oampnet;
result.alpha_rms = alpha_rms;
result.h_delta_norm = h_delta_norm;
result.alpha_com = alpha_com;
result.v_norm = v_norm;
result.delta_alpha_rms = delta_alpha_rms;
result.alpha_hat_t = alpha_hat_t;
result.alpha_raw_max = alpha_raw_max;
result.alpha_res_max = alpha_res_max;
result.alpha_max_raw = alpha_max_raw;
result.alpha_max_res = alpha_max_res;
result.enable_resampling_comp = logical(tv_cfg.enable_resampling_comp);
result.alpha_hat_mode = tv_cfg.alpha_hat_mode;
result.alpha_raw_stats = alpha_raw_stats;
result.alpha_res_stats = alpha_res_stats;
result.doppler_mode = doppler_mode;
result.doppler_mode_code = doppler_mode_code;
result.c1_grid = c1_grid;
result.base_action = int32(base_idx - 1);
result.error_count_action_oamp = error_count_action_oamp;
result.error_count_action_oampnet = error_count_action_oampnet;
result.total_bits_action = total_bits_action;

result.summary_by_snr = summary_by_snr;
result.sequence_summary = sequence_summary;
result.feature_names = feature_names;
result.config = cfg;
result.timevary_hparams = tv_cfg;
result.run_profile = run_profile;

save(cfg.output_mat, 'result', '-v7.3');

%% ===== Legacy-compatible aggregate CSVs =====
fid = fopen(cfg.output_csv_wide, 'w');
fprintf(fid, ['snr_db,fixed_oamp,fixed_oampnet,best_static_oamp,best_static_oampnet,rl_oamp,rl_oampnet,oracle_oamp,oracle_oampnet,' ...
    'avg_regret_rl,avg_switch_rl,avg_switch_oracle,num_seq_run,num_frames,num_noise,dataset_mode,detector_target,' ...
    'doppler_mode,' ...
    'fixed_oamp_ber_raw,fixed_oampnet_ber_raw,best_static_oamp_ber_raw,best_static_oampnet_ber_raw,rl_oamp_ber_raw,rl_oampnet_ber_raw,oracle_oamp_ber_raw,oracle_oampnet_ber_raw,' ...
    'fixed_oamp_num_error_bits,fixed_oampnet_num_error_bits,best_static_oamp_num_error_bits,best_static_oampnet_num_error_bits,rl_oamp_num_error_bits,rl_oampnet_num_error_bits,oracle_oamp_num_error_bits,oracle_oampnet_num_error_bits,' ...
    'fixed_oamp_num_total_bits,fixed_oampnet_num_total_bits,best_static_oamp_num_total_bits,best_static_oampnet_num_total_bits,rl_oamp_num_total_bits,rl_oampnet_num_total_bits,oracle_oamp_num_total_bits,oracle_oampnet_num_total_bits,' ...
    'fixed_oamp_is_upper_bound,fixed_oampnet_is_upper_bound,best_static_oamp_is_upper_bound,best_static_oampnet_is_upper_bound,rl_oamp_is_upper_bound,rl_oampnet_is_upper_bound,oracle_oamp_is_upper_bound,oracle_oampnet_is_upper_bound,' ...
    'base_action,base_c1,best_static_oamp_action,best_static_oampnet_action,best_static_oamp_c1,best_static_oampnet_c1\n']);
for si = 1:num_snr
    total_bits_now = total_bits_snr(si);
    fprintf(fid, ['%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6f,%d,%d,%d,%s,%s,%s,' ...
        '%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%d,%d,%d,%d,%d,%d,%d,%d,%.0f,%.6e,%d,%d,%.6e,%.6e\n'], ...
        cfg.snr_db_list(si), ...
        summary_by_snr.fixed_oamp_ber_plot(si), summary_by_snr.fixed_oampnet_ber_plot(si), ...
        summary_by_snr.best_static_oamp_ber_plot(si), summary_by_snr.best_static_oampnet_ber_plot(si), ...
        summary_by_snr.rl_oamp_ber_plot(si), summary_by_snr.rl_oampnet_ber_plot(si), ...
        summary_by_snr.oracle_oamp_ber_plot(si), summary_by_snr.oracle_oampnet_ber_plot(si), ...
        summary_by_snr.avg_regret_rl(si), summary_by_snr.avg_switch_rl(si), summary_by_snr.avg_switch_oracle(si), ...
        num_sequence_run(si), cfg.num_frames, cfg.num_noise, dataset_mode, detector_target, doppler_mode, ...
        summary_by_snr.fixed_oamp_ber_raw(si), summary_by_snr.fixed_oampnet_ber_raw(si), ...
        summary_by_snr.best_static_oamp_ber_raw(si), summary_by_snr.best_static_oampnet_ber_raw(si), ...
        summary_by_snr.rl_oamp_ber_raw(si), summary_by_snr.rl_oampnet_ber_raw(si), ...
        summary_by_snr.oracle_oamp_ber_raw(si), summary_by_snr.oracle_oampnet_ber_raw(si), ...
        summary_by_snr.fixed_oamp_num_error_bits(si), summary_by_snr.fixed_oampnet_num_error_bits(si), ...
        summary_by_snr.best_static_oamp_num_error_bits(si), summary_by_snr.best_static_oampnet_num_error_bits(si), ...
        summary_by_snr.rl_oamp_num_error_bits(si), summary_by_snr.rl_oampnet_num_error_bits(si), ...
        summary_by_snr.oracle_oamp_num_error_bits(si), summary_by_snr.oracle_oampnet_num_error_bits(si), ...
        total_bits_now, total_bits_now, total_bits_now, total_bits_now, total_bits_now, total_bits_now, total_bits_now, total_bits_now, ...
        summary_by_snr.fixed_oamp_is_upper_bound(si), summary_by_snr.fixed_oampnet_is_upper_bound(si), ...
        summary_by_snr.best_static_oamp_is_upper_bound(si), summary_by_snr.best_static_oampnet_is_upper_bound(si), ...
        summary_by_snr.rl_oamp_is_upper_bound(si), summary_by_snr.rl_oampnet_is_upper_bound(si), ...
        summary_by_snr.oracle_oamp_is_upper_bound(si), summary_by_snr.oracle_oampnet_is_upper_bound(si), ...
        summary_by_snr.base_action(si), summary_by_snr.base_c1(si), ...
        summary_by_snr.best_static_oamp_action(si), summary_by_snr.best_static_oampnet_action(si), ...
        summary_by_snr.best_static_oamp_c1(si), summary_by_snr.best_static_oampnet_c1(si));
end
fclose(fid);

fid = fopen(cfg.output_csv_fig3, 'w');
fprintf(fid, 'snr_db,method,ber_plot,ber_raw,num_error_bits,num_total_bits,is_upper_bound\n');
for si = 1:num_snr
    s = cfg.snr_db_list(si);
    tb = total_bits_snr(si);
    fprintf(fid, '%d,fixed_oamp,%.6e,%.6e,%.0f,%.0f,%d\n', s, summary_by_snr.fixed_oamp_ber_plot(si), summary_by_snr.fixed_oamp_ber_raw(si), summary_by_snr.fixed_oamp_num_error_bits(si), tb, summary_by_snr.fixed_oamp_is_upper_bound(si));
    fprintf(fid, '%d,fixed_oampnet,%.6e,%.6e,%.0f,%.0f,%d\n', s, summary_by_snr.fixed_oampnet_ber_plot(si), summary_by_snr.fixed_oampnet_ber_raw(si), summary_by_snr.fixed_oampnet_num_error_bits(si), tb, summary_by_snr.fixed_oampnet_is_upper_bound(si));
    fprintf(fid, '%d,best_static_oamp,%.6e,%.6e,%.0f,%.0f,%d\n', s, summary_by_snr.best_static_oamp_ber_plot(si), summary_by_snr.best_static_oamp_ber_raw(si), summary_by_snr.best_static_oamp_num_error_bits(si), tb, summary_by_snr.best_static_oamp_is_upper_bound(si));
    fprintf(fid, '%d,best_static_oampnet,%.6e,%.6e,%.0f,%.0f,%d\n', s, summary_by_snr.best_static_oampnet_ber_plot(si), summary_by_snr.best_static_oampnet_ber_raw(si), summary_by_snr.best_static_oampnet_num_error_bits(si), tb, summary_by_snr.best_static_oampnet_is_upper_bound(si));
    fprintf(fid, '%d,rl_oamp,%.6e,%.6e,%.0f,%.0f,%d\n', s, summary_by_snr.rl_oamp_ber_plot(si), summary_by_snr.rl_oamp_ber_raw(si), summary_by_snr.rl_oamp_num_error_bits(si), tb, summary_by_snr.rl_oamp_is_upper_bound(si));
    fprintf(fid, '%d,rl_oampnet,%.6e,%.6e,%.0f,%.0f,%d\n', s, summary_by_snr.rl_oampnet_ber_plot(si), summary_by_snr.rl_oampnet_ber_raw(si), summary_by_snr.rl_oampnet_num_error_bits(si), tb, summary_by_snr.rl_oampnet_is_upper_bound(si));
    fprintf(fid, '%d,oracle_oamp,%.6e,%.6e,%.0f,%.0f,%d\n', s, summary_by_snr.oracle_oamp_ber_plot(si), summary_by_snr.oracle_oamp_ber_raw(si), summary_by_snr.oracle_oamp_num_error_bits(si), tb, summary_by_snr.oracle_oamp_is_upper_bound(si));
    fprintf(fid, '%d,oracle_oampnet,%.6e,%.6e,%.0f,%.0f,%d\n', s, summary_by_snr.oracle_oampnet_ber_plot(si), summary_by_snr.oracle_oampnet_ber_raw(si), summary_by_snr.oracle_oampnet_num_error_bits(si), tb, summary_by_snr.oracle_oampnet_is_upper_bound(si));
end
fclose(fid);

fid = fopen(cfg.output_csv_fig4, 'w');
fprintf(fid, 'snr_db,gain_vs_fixed_oampnet,oracle_gap_oampnet\n');
for si = 1:num_snr
    s = cfg.snr_db_list(si);
    ref = max(summary_by_snr.fixed_oampnet_ber_plot(si), 1e-12);
    gain = 1.0 - summary_by_snr.rl_oampnet_ber_plot(si) / ref;
    oracle_gap = summary_by_snr.rl_oampnet_ber_plot(si) - summary_by_snr.oracle_oampnet_ber_plot(si);
    fprintf(fid, '%d,%.6e,%.6e\n', s, gain, oracle_gap);
end
fclose(fid);

fid = fopen(cfg.output_csv_c1_diag, 'w');
fprintf(fid, ['snr_db,detector,base_action,base_c1,best_static_action,best_static_c1,' ...
    'static_baseline_ber_plot,static_baseline_ber_raw,static_best_single_ber_plot,static_best_single_ber_raw,' ...
    'rl_dynamic_ber_plot,rl_dynamic_ber_raw,oracle_dynamic_ber_plot,oracle_dynamic_ber_raw,' ...
    'static_headroom_raw,dynamic_headroom_raw,rl_gap_to_best_static_raw,rl_gap_to_oracle_raw,' ...
    'rl_gain_vs_best_static_pct,oracle_gain_vs_best_static_pct\n']);
for si = 1:num_snr
    fprintf(fid, ['%d,oamp,%d,%.6e,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,' ...
        '%.6e,%.6e,%.6e,%.6e,%.6f,%.6f\n'], ...
        cfg.snr_db_list(si), summary_by_snr.base_action(si), summary_by_snr.base_c1(si), ...
        summary_by_snr.best_static_oamp_action(si), summary_by_snr.best_static_oamp_c1(si), ...
        summary_by_snr.fixed_oamp_ber_plot(si), summary_by_snr.fixed_oamp_ber_raw(si), ...
        summary_by_snr.best_static_oamp_ber_plot(si), summary_by_snr.best_static_oamp_ber_raw(si), ...
        summary_by_snr.rl_oamp_ber_plot(si), summary_by_snr.rl_oamp_ber_raw(si), ...
        summary_by_snr.oracle_oamp_ber_plot(si), summary_by_snr.oracle_oamp_ber_raw(si), ...
        summary_by_snr.oamp_static_headroom_raw(si), summary_by_snr.oamp_dynamic_headroom_raw(si), ...
        summary_by_snr.oamp_rl_gap_to_best_static_raw(si), summary_by_snr.oamp_rl_gap_to_oracle_raw(si), ...
        summary_by_snr.oamp_rl_gain_vs_best_static_pct(si), summary_by_snr.oamp_oracle_gain_vs_best_static_pct(si));
    fprintf(fid, ['%d,oampnet,%d,%.6e,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,' ...
        '%.6e,%.6e,%.6e,%.6e,%.6f,%.6f\n'], ...
        cfg.snr_db_list(si), summary_by_snr.base_action(si), summary_by_snr.base_c1(si), ...
        summary_by_snr.best_static_oampnet_action(si), summary_by_snr.best_static_oampnet_c1(si), ...
        summary_by_snr.fixed_oampnet_ber_plot(si), summary_by_snr.fixed_oampnet_ber_raw(si), ...
        summary_by_snr.best_static_oampnet_ber_plot(si), summary_by_snr.best_static_oampnet_ber_raw(si), ...
        summary_by_snr.rl_oampnet_ber_plot(si), summary_by_snr.rl_oampnet_ber_raw(si), ...
        summary_by_snr.oracle_oampnet_ber_plot(si), summary_by_snr.oracle_oampnet_ber_raw(si), ...
        summary_by_snr.oampnet_static_headroom_raw(si), summary_by_snr.oampnet_dynamic_headroom_raw(si), ...
        summary_by_snr.oampnet_rl_gap_to_best_static_raw(si), summary_by_snr.oampnet_rl_gap_to_oracle_raw(si), ...
        summary_by_snr.oampnet_rl_gain_vs_best_static_pct(si), summary_by_snr.oampnet_oracle_gain_vs_best_static_pct(si));
end
fclose(fid);

plot_paper_summary(summary_by_snr, cfg, run_profile);
plot_c1_detector_diagnostic(summary_by_snr.snr_db, summary_by_snr.fixed_oamp_ber_plot, ...
    summary_by_snr.best_static_oamp_ber_plot, summary_by_snr.rl_oamp_ber_plot, ...
    summary_by_snr.oracle_oamp_ber_plot, 'OAMP', cfg.figure_png_c1_diag_oamp, run_profile);
plot_c1_detector_diagnostic(summary_by_snr.snr_db, summary_by_snr.fixed_oampnet_ber_plot, ...
    summary_by_snr.best_static_oampnet_ber_plot, summary_by_snr.rl_oampnet_ber_plot, ...
    summary_by_snr.oracle_oampnet_ber_plot, 'OAMPNet', cfg.figure_png_c1_diag_oampnet, run_profile);

fprintf('\nSaved:\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n', ...
    cfg.output_csv, cfg.output_mat, cfg.output_csv_wide, cfg.output_csv_fig3, cfg.output_csv_fig4, ...
    cfg.output_csv_c1_diag, cfg.figure_png_c1_diag_oamp, cfg.figure_png_c1_diag_oampnet);
fprintf('alpha stats | raw max=%.3e p99=%.3e p995=%.3e | res max=%.3e p99=%.3e p995=%.3e\n', ...
    alpha_raw_stats.abs_max, alpha_raw_stats.p99_abs, alpha_raw_stats.p995_abs, ...
    alpha_res_stats.abs_max, alpha_res_stats.p99_abs, alpha_res_stats.p995_abs);

if nargout == 0
    clear result;
end
end

%% ===== Helpers =====
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

function a_idx = policy_greedy_from_state(state_feat, policy)
x = (state_feat(:) - policy.state_mean(:)) ./ max(policy.state_std(:), 1e-6);
L = round(double(policy.num_linear_layers(1)));
h = x;
for li = 1:L
    W = policy.(sprintf('W%d', li));
    b = policy.(sprintf('b%d', li));
    h = W * h + b;
    if li < L
        h = max(h, 0);
    end
end
[~, a_idx] = max(h);
end

function names = normalize_feature_names(raw_names)
names = {};
if isempty(raw_names)
    return;
end
if iscell(raw_names)
    flat = raw_names(:);
    names = cell(1, numel(flat));
    for i = 1:numel(flat)
        names{i} = strtrim(char(string(flat{i})));
    end
else
    flat = raw_names;
    if isstring(flat)
        flat = cellstr(flat(:));
    elseif ischar(flat)
        flat = cellstr(flat);
    end
    flat = flat(:);
    names = cell(1, numel(flat));
    for i = 1:numel(flat)
        names{i} = strtrim(char(string(flat(i))));
    end
end
names = names(~cellfun(@isempty, names));
end

function [f, feature_names] = extract_online_state_features_named( ...
    Heff_base, y_obs, x_hat_obs, noise_var, data_idx, ctx, use_oracle_state, ch_oracle, requested_feature_names)
feature_names = normalize_feature_names(requested_feature_names);
if isempty(feature_names)
    f = [];
    return;
end

physical_names = {'alpha_com', 'v_norm', 'delta_alpha_rms', ...
    'abs_alpha_com', 'delta_alpha_com', 'delta_v_norm', 'delta_delta_alpha_rms'};
history_names = {'prev_action_norm', 'prev_reward', 'recent_switch_rate'};
oracle_names = {'oracle_max_alpha', 'oracle_mean_alpha', 'oracle_delay_spread', 'oracle_h_l4_over_l2'};

need_physical = any(ismember(feature_names, physical_names));
need_history = any(ismember(feature_names, history_names));
need_oracle = any(ismember(feature_names, oracle_names));

[base_feat, base_names] = extract_online_state_features( ...
    Heff_base, y_obs, x_hat_obs, noise_var, data_idx, ctx, ...
    use_oracle_state || need_oracle, ch_oracle, need_physical, need_history);

alpha_com_now = feature_value_by_name_or_default(base_names, base_feat, 'alpha_com', 0.0);
v_norm_now = feature_value_by_name_or_default(base_names, base_feat, 'v_norm', 0.0);
delta_alpha_rms_now = feature_value_by_name_or_default(base_names, base_feat, 'delta_alpha_rms', 0.0);
prev_alpha_com = ctx_value(ctx, 'prev_alpha_com', 0.0);
prev_v_norm = ctx_value(ctx, 'prev_v_norm', 0.0);
prev_delta_alpha_rms = ctx_value(ctx, 'prev_delta_alpha_rms', 0.0);

f = zeros(1, numel(feature_names));
for i = 1:numel(feature_names)
    name = feature_names{i};
    idx = find(strcmp(base_names, name), 1);
    if ~isempty(idx)
        f(i) = base_feat(idx);
        continue;
    end
    switch name
        case 'abs_alpha_com'
            f(i) = abs(alpha_com_now);
        case 'delta_alpha_com'
            f(i) = alpha_com_now - prev_alpha_com;
        case 'delta_v_norm'
            f(i) = v_norm_now - prev_v_norm;
        case 'delta_delta_alpha_rms'
            f(i) = delta_alpha_rms_now - prev_delta_alpha_rms;
        otherwise
            error('Requested policy feature not supported online: %s', name);
    end
end
end

function v = feature_value_by_name_or_default(feature_names, feat, target_name, default_v)
v = default_v;
if isempty(feature_names) || isempty(feat)
    return;
end
idx = find(strcmp(feature_names, target_name), 1);
if isempty(idx)
    return;
end
v = double(feat(idx));
if ~isfinite(v)
    v = default_v;
end
end

function v = ctx_value(ctx, name, default_v)
if isstruct(ctx) && isfield(ctx, name)
    v = double(ctx.(name));
else
    v = double(default_v);
end
if ~isfinite(v)
    v = double(default_v);
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

function stats = build_abs_stats_struct(abs_vals)
flat = abs_vals(:);
flat = flat(isfinite(flat));
if isempty(flat)
    stats = struct('abs_max', 0.0, 'abs_mean', 0.0, 'p99_abs', 0.0, 'p995_abs', 0.0);
    return;
end
stats = struct();
stats.abs_max = max(flat);
stats.abs_mean = mean(flat);
stats.p99_abs = quantile(flat, 0.99);
stats.p995_abs = quantile(flat, 0.995);
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

function x_hard = qpsk_hard_projection(x)
re = sign(real(x));
im = sign(imag(x));
re(re == 0) = 1;
im(im == 0) = 1;
x_hard = (re + 1j * im) / sqrt(2);
end

function m = safe_mean(x)
x = double(x(:));
v = ~isnan(x);
if any(v)
    m = mean(x(v));
else
    m = nan;
end
end

function budget = paper_budget_for_snr(cfg, snr_db, run_profile)
if ~strcmpi(run_profile, 'paper')
    budget.target_error_bits = inf;
    budget.max_bits = inf;
    return;
end
if snr_db <= 8
    budget.target_error_bits = cfg.paper_target_error_bits_low;
    budget.max_bits = cfg.paper_max_bits_low;
elseif snr_db <= 14
    budget.target_error_bits = cfg.paper_target_error_bits_mid;
    budget.max_bits = cfg.paper_max_bits_mid;
else
    budget.target_error_bits = cfg.paper_target_error_bits_high;
    budget.max_bits = cfg.paper_max_bits_high;
end
end

function stop_now = paper_should_stop(budget, total_bits_now, error_counts, is_active)
stop_now = false;
if total_bits_now >= budget.max_bits
    stop_now = true;
    return;
end
active_err = error_counts(logical(is_active));
if isempty(active_err)
    return;
end
stop_now = all(active_err >= budget.target_error_bits);
end

function [ber_raw, ber_plot, is_upper_bound] = ber_from_counts(num_err, total_bits)
if total_bits <= 0
    ber_raw = nan;
    ber_plot = nan;
    is_upper_bound = 0;
    return;
end
ber_raw = num_err / total_bits;
if num_err <= 0
    ber_plot = 3 / total_bits;
    is_upper_bound = 1;
else
    ber_plot = ber_raw;
    is_upper_bound = 0;
end
end

function gain_pct = safe_relative_gain_pct(ref_ber, candidate_ber)
if isnan(ref_ber) || isnan(candidate_ber)
    gain_pct = nan;
    return;
end
if ref_ber <= 0
    if candidate_ber <= 0
        gain_pct = 0.0;
    else
        gain_pct = nan;
    end
    return;
end
gain_pct = 100.0 * (1.0 - candidate_ber / ref_ber);
end

function plot_paper_summary(summary_by_snr, cfg, run_profile)
fig_dir = fileparts(cfg.figure_png_main);
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

fig = figure('Visible', 'off');
semilogy(summary_by_snr.snr_db, summary_by_snr.fixed_oamp_ber_plot, '-o', 'LineWidth', 1.6, 'DisplayName', 'Fixed C1 + OAMP'); hold on;
semilogy(summary_by_snr.snr_db, summary_by_snr.fixed_oampnet_ber_plot, '-s', 'LineWidth', 1.6, 'DisplayName', 'Fixed C1 + OAMPNet');
semilogy(summary_by_snr.snr_db, summary_by_snr.rl_oamp_ber_plot, '--o', 'LineWidth', 1.6, 'DisplayName', 'RL C1 + OAMP');
semilogy(summary_by_snr.snr_db, summary_by_snr.rl_oampnet_ber_plot, '--s', 'LineWidth', 1.6, 'DisplayName', 'RL C1 + OAMPNet');
semilogy(summary_by_snr.snr_db, summary_by_snr.oracle_oampnet_ber_plot, ':^', 'LineWidth', 1.8, 'DisplayName', 'Oracle C1 + OAMPNet');
grid on;
xlabel('SNR (dB)');
ylabel('BER');
if strcmpi(run_profile, 'paper')
    title('BER vs SNR under Time-Varying Sequences');
else
    title('Online BER vs SNR');
end
legend('Location', 'southwest');
saveas(fig, cfg.figure_png_main);
saveas(fig, replace(cfg.figure_png_main, '.png', '.pdf'));
close(fig);

fig = figure('Visible', 'off');
gain = 100 * (1.0 - summary_by_snr.rl_oampnet_ber_plot ./ max(summary_by_snr.fixed_oampnet_ber_plot, 1e-12));
plot(summary_by_snr.snr_db, gain, '-o', 'LineWidth', 1.8);
grid on;
xlabel('SNR (dB)');
ylabel('Gain vs Fixed OAMPNet (%)');
title('RL C1 Gain vs SNR');
saveas(fig, cfg.figure_png_gain);
saveas(fig, replace(cfg.figure_png_gain, '.png', '.pdf'));
close(fig);
end

function plot_c1_detector_diagnostic(snr_db, ber_base, ber_best_static, ber_rl, ber_oracle, detector_name, output_png, run_profile)
fig_dir = fileparts(output_png);
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

fig = figure('Visible', 'off');
semilogy(snr_db, ber_base, '-o', 'LineWidth', 1.7, 'DisplayName', 'Static baseline C1'); hold on;
semilogy(snr_db, ber_best_static, '-s', 'LineWidth', 1.7, 'DisplayName', 'Static best single C1');
semilogy(snr_db, ber_rl, '--d', 'LineWidth', 1.7, 'DisplayName', 'RL dynamic C1');
semilogy(snr_db, ber_oracle, ':^', 'LineWidth', 1.9, 'DisplayName', 'Oracle dynamic C1');
grid on;
xlabel('SNR (dB)');
ylabel('BER');
if strcmpi(run_profile, 'paper')
    title(sprintf('C1 Headroom Diagnostic: %s', detector_name));
else
    title(sprintf('Online C1 Diagnostic: %s', detector_name));
end
legend('Location', 'southwest');
saveas(fig, output_png);
saveas(fig, replace(output_png, '.png', '.pdf'));
close(fig);
end

function seed_u32 = safe_rng_seed(seed_in)
seed_u32 = mod(double(seed_in), 2^32 - 1);
if seed_u32 < 0
    seed_u32 = seed_u32 + (2^32 - 1);
end
seed_u32 = floor(seed_u32);
end
