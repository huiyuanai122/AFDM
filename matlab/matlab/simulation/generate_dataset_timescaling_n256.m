function generate_dataset_timescaling_n256(dataset_type, num_samples, batch_size, version, cfg)
%% generate_dataset_timescaling_n256.m
% Generate AFDM detector-training dataset with two modes:
%   1) mode='iid' (legacy frame-independent channel)
%   2) mode='timevary_sequence' (new sequence-level recursive channel)
%
% Output compatibility:
% - keeps legacy fields: H_dataset, x_dataset, snr_dataset, system_params
% - adds metadata fields: sequence_id, time_index, dataset_mode, dataset_mode_code, timevary_hparams

if nargin < 1 || isempty(dataset_type)
    dataset_type = 'train';
end
if nargin < 2 || isempty(num_samples)
    if strcmpi(dataset_type, 'train')
        num_samples = 5000;
    elseif strcmpi(dataset_type, 'val')
        num_samples = 500;
    else
        num_samples = 5500;
    end
end
if nargin < 3 || isempty(batch_size)
    batch_size = 2000;
end
if nargin < 4
    version = '';
end
if nargin < 5
    cfg = struct();
end

script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..');
common_dir = fullfile(project_root, 'matlab', 'common');
addpath(common_dir);

%% ===== Defaults =====
N = getfield_with_default(cfg, 'N', 256);
assert(N == 256, 'Current Python training scripts expect N=256.');

Delta_f = getfield_with_default(cfg, 'Delta_f', 4);
T = 1 / Delta_f;
B = N * Delta_f;
dt = 1 / B;

fc = getfield_with_default(cfg, 'fc', 12e3);
alpha_max_res = getfield_with_default(cfg, 'alpha_max_res', getfield_with_default(cfg, 'alpha_max', 1e-4));
alpha_max_raw = getfield_with_default(cfg, 'alpha_max_raw', max(5e-4, alpha_max_res));
ell_max = getfield_with_default(cfg, 'ell_max', 16);
P = getfield_with_default(cfg, 'P', 6);
Q = getfield_with_default(cfg, 'Q', 0);
N_eff = N - 2 * Q;
if N_eff <= 0
    error('Q too large: N_eff <= 0');
end

Lcpp = getfield_with_default(cfg, 'Lcpp', max(1, ceil(ell_max / (1 - alpha_max_res))));
Lcps = getfield_with_default(cfg, 'Lcps', max(1, ceil(alpha_max_res * N / (1 + alpha_max_res))));

Nv = getfield_with_default(cfg, 'Nv', 2);
kmax = ceil((alpha_max_res * fc) * T);
den = (1 - 4 * alpha_max_res * (N - 1));
if den <= 0
    error('Invalid c1 design: denominator <= 0');
end
c1 = (2 * kmax + 2 * alpha_max_res * (N - 1) + 2 * Nv + 1) / (2 * N * den);
c2 = sqrt(2) / N;

mode = char(string(getfield_with_default(cfg, 'mode', 'iid')));
if ~strcmpi(mode, 'iid') && ~strcmpi(mode, 'timevary_sequence')
    error('Unsupported cfg.mode=%s. Use ''iid'' or ''timevary_sequence''.', mode);
end
if contains(lower(char(string(version))), 'vdop') && ~isfield(cfg, 'doppler_mode')
    cfg.doppler_mode = 'common_with_path_residual';
end
if contains(lower(char(string(version))), 'vdop_ctrl')
    if ~isfield(cfg, 'doppler_mode')
        cfg.doppler_mode = 'common_with_path_residual';
    end
    if ~isfield(cfg, 'motion_profile')
        cfg.motion_profile = 'maneuver_heave';
    end
    if ~isfield(cfg, 'path_projection_mode')
        cfg.path_projection_mode = 'symmetric_linear';
    end
    if ~isfield(cfg, 'beta_min')
        cfg.beta_min = 0.45;
    end
    if ~isfield(cfg, 'beta_max')
        cfg.beta_max = 1.65;
    end
    if ~isfield(cfg, 'num_frames')
        cfg.num_frames = 40;
    end
end
use_timevary = strcmpi(mode, 'timevary_sequence');
run_profile = char(string(getfield_with_default(cfg, 'run_profile', 'default')));
if strcmpi(run_profile, 'paper') && use_timevary
    if ~isfield(cfg, 'num_frames')
        num_frames_cfg = 10;
    end
    if ~isfield(cfg, 'num_seq')
        num_seq_cfg = ceil(num_samples / max(num_frames_cfg, 1));
    end
end

num_frames_cfg = max(1, round(getfield_with_default(cfg, 'num_frames', 10)));
num_seq_cfg = max(1, round(getfield_with_default(cfg, 'num_seq', ceil(num_samples / num_frames_cfg))));

if use_timevary
    num_seq = num_seq_cfg;
    num_frames = num_frames_cfg;
    num_samples = num_seq * num_frames;
else
    num_seq = num_samples;
    num_frames = 1;
end

tv_override = struct();
if isfield(cfg, 'timevary_cfg') && isstruct(cfg.timevary_cfg)
    tv_override = cfg.timevary_cfg;
elseif isfield(cfg, 'timevary') && isstruct(cfg.timevary)
    tv_override = cfg.timevary;
end
tv_base = struct( ...
    'rho_alpha', getfield_with_default(cfg, 'rho_alpha', 0.98), ...
    'rho_h', getfield_with_default(cfg, 'rho_h', 0.98), ...
    'alpha_max', alpha_max_res, ...
    'alpha_max_raw', alpha_max_raw, ...
    'alpha_max_res', alpha_max_res, ...
    'num_frames', num_frames, ...
    'doppler_mode', getfield_with_default(cfg, 'doppler_mode', 'independent_path_ar1'), ...
    'rho_acc', getfield_with_default(cfg, 'rho_acc', 0.95), ...
    'sigma_acc', getfield_with_default(cfg, 'sigma_acc', 0.03), ...
    'rho_delta', getfield_with_default(cfg, 'rho_delta', 0.90), ...
    'sigma_delta', getfield_with_default(cfg, 'sigma_delta', 0.05), ...
    'delta_max', getfield_with_default(cfg, 'delta_max', []), ...
    'motion_profile', getfield_with_default(cfg, 'motion_profile', 'smooth_ar'), ...
    'target_track_gain', getfield_with_default(cfg, 'target_track_gain', 0.75), ...
    'target_blend', getfield_with_default(cfg, 'target_blend', 0.80), ...
    'profile_v_peak', getfield_with_default(cfg, 'profile_v_peak', 0.95), ...
    'profile_turn_range', getfield_with_default(cfg, 'profile_turn_range', [0.32, 0.48]), ...
    'profile_recede_range', getfield_with_default(cfg, 'profile_recede_range', [0.68, 0.86]), ...
    'profile_heave_amp', getfield_with_default(cfg, 'profile_heave_amp', 0.18), ...
    'profile_heave_cycles', getfield_with_default(cfg, 'profile_heave_cycles', 1.35), ...
    'profile_secondary_amp', getfield_with_default(cfg, 'profile_secondary_amp', 0.08), ...
    'profile_secondary_cycles', getfield_with_default(cfg, 'profile_secondary_cycles', 2.70), ...
    'profile_jitter_std', getfield_with_default(cfg, 'profile_jitter_std', 0.04), ...
    'path_projection_mode', getfield_with_default(cfg, 'path_projection_mode', 'ones'), ...
    'beta_min', getfield_with_default(cfg, 'beta_min', 1.0), ...
    'beta_max', getfield_with_default(cfg, 'beta_max', 1.0), ...
    'ell_mode', getfield_with_default(cfg, 'ell_mode', 'static'), ...
    'pdp_mode', getfield_with_default(cfg, 'pdp_mode', 'exp_fixed_per_sequence'), ...
    'enable_resampling_comp', getfield_with_default(cfg, 'enable_resampling_comp', true), ...
    'alpha_hat_mode', getfield_with_default(cfg, 'alpha_hat_mode', 'common_component'), ...
    'clip_alpha_res', getfield_with_default(cfg, 'clip_alpha_res', true), ...
    'log_alpha_stats', getfield_with_default(cfg, 'log_alpha_stats', true));
tv_cfg = get_timevary_defaults(merge_struct(tv_base, tv_override));

fprintf('========================================\n');
fprintf('AFDM dataset generation (mode=%s)\n', mode);
fprintf('========================================\n');
fprintf('type=%s, samples=%d, batch=%d, version=%s\n', dataset_type, num_samples, batch_size, version);
fprintf('N=%d, N_eff=%d, Q=%d\n', N, N_eff, Q);
fprintf('fc=%.3gHz, alpha_max_raw=%.1e, alpha_max_res=%.1e, ell_max=%d, P=%d\n', ...
    fc, tv_cfg.alpha_max_raw, tv_cfg.alpha_max_res, ell_max, P);
fprintf('Lcpp=%d, Lcps=%d, c1=%.6g, c2=%.6g | resampling=%d (%s)\n', ...
    Lcpp, Lcps, c1, c2, tv_cfg.enable_resampling_comp, tv_cfg.alpha_hat_mode);
fprintf('num_seq=%d, num_frames=%d\n', num_seq, num_frames);

%% ===== Precompute =====
XTB = precompute_idaf_basis(N, c1, c2);
Xext = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps);
L = size(Xext, 1);

n = (0:N-1).';
chirp1 = exp(-1j * 2 * pi * c1 * (n .^ 2));
chirp2 = exp(-1j * 2 * pi * c2 * (n .^ 2));

SNR_range = 0:2:20;
if strcmpi(dataset_type, 'test')
    snr_per_point = floor(num_samples / numel(SNR_range));
end

%% ===== Batch generation =====
num_batches = ceil(num_samples / batch_size);
total_generated = 0;
all_files = {};

current_sid = -1;
seq_state = [];
ch_t = [];
seq_snr = 0;

for batch_idx = 1:num_batches
    start_idx = (batch_idx - 1) * batch_size + 1;
    end_idx = min(batch_idx * batch_size, num_samples);
    cur_bs = end_idx - start_idx + 1;

    fprintf('\nBatch %d/%d: %d~%d (%d samples)\n', batch_idx, num_batches, start_idx, end_idx, cur_bs);

    H_dataset = zeros(N, N, cur_bs, 'like', 1j);
    x_dataset = zeros(N, 1, cur_bs, 'like', 1j);
    snr_dataset = zeros(cur_bs, 1);
    sequence_id = zeros(cur_bs, 1, 'int32');
    time_index = zeros(cur_bs, 1, 'int32');
    alpha_com_dataset = zeros(cur_bs, 1, 'single');
    v_norm_dataset = zeros(cur_bs, 1, 'single');
    delta_alpha_rms_dataset = zeros(cur_bs, 1, 'single');
    alpha_hat_dataset = zeros(cur_bs, 1, 'single');
    alpha_raw_max_dataset = zeros(cur_bs, 1, 'single');
    alpha_res_max_dataset = zeros(cur_bs, 1, 'single');
    alpha_raw_abs_paths = zeros(cur_bs, P, 'single');
    alpha_res_abs_paths = zeros(cur_bs, P, 'single');

    for local_idx = 1:cur_bs
        global_idx = start_idx + local_idx - 1;

        if mod(local_idx, 200) == 0 || local_idx == 1
            fprintf('  progress: %d/%d (%.1f%%)\n', local_idx, cur_bs, 100 * local_idx / cur_bs);
        end

        if use_timevary
            sid = floor((global_idx - 1) / num_frames) + 1;
            tt = mod(global_idx - 1, num_frames) + 1;

            if sid ~= current_sid
                [seq_state, ch_t] = init_timevary_channel_state(P, ell_max, alpha_max_res, tv_cfg);
                seq_snr = pick_sequence_snr(sid, num_seq, dataset_type, SNR_range);
                current_sid = sid;
            elseif tt > 1
                [seq_state, ch_t] = step_timevary_channel_state(seq_state);
            end

            snr_db = seq_snr;
        else
            sid = global_idx;
            tt = 1;
            [~, ch_t] = init_timevary_channel_state(P, ell_max, alpha_max_res, tv_cfg);

            if strcmpi(dataset_type, 'test')
                snr_i = floor((global_idx - 1) / max(1, snr_per_point)) + 1;
                snr_i = min(snr_i, numel(SNR_range));
                snr_db = SNR_range(snr_i);
            else
                snr_db = SNR_range(randi(numel(SNR_range)));
            end
        end

        snr_dataset(local_idx) = snr_db;
        sequence_id(local_idx) = int32(sid);
        time_index(local_idx) = int32(tt);
        if isfield(ch_t, 'alpha_com')
            alpha_com_dataset(local_idx) = single(ch_t.alpha_com);
        else
            alpha_com_dataset(local_idx) = single(mean(ch_t.alpha));
        end
        if isfield(ch_t, 'v_norm')
            v_norm_dataset(local_idx) = single(ch_t.v_norm);
        else
            v_norm_dataset(local_idx) = single(alpha_com_dataset(local_idx) / max(tv_cfg.alpha_max_raw, 1e-12));
        end
        if isfield(ch_t, 'delta_alpha_rms')
            delta_alpha_rms_dataset(local_idx) = single(ch_t.delta_alpha_rms);
        else
            delta_alpha_rms_dataset(local_idx) = single(sqrt(mean(abs(ch_t.alpha - mean(ch_t.alpha)).^2)));
        end
        if isfield(ch_t, 'alpha_hat')
            alpha_hat_dataset(local_idx) = single(ch_t.alpha_hat);
        end
        if isfield(ch_t, 'alpha_raw')
            alpha_raw_vec = abs(double(ch_t.alpha_raw(:)));
        else
            alpha_raw_vec = abs(double(ch_t.alpha(:)));
        end
        if isfield(ch_t, 'alpha_res')
            alpha_res_vec = abs(double(ch_t.alpha_res(:)));
        else
            alpha_res_vec = abs(double(ch_t.alpha(:)));
        end
        alpha_raw_abs_paths(local_idx, :) = single(reshape(alpha_raw_vec, 1, []));
        alpha_res_abs_paths(local_idx, :) = single(reshape(alpha_res_vec, 1, []));
        alpha_raw_max_dataset(local_idx) = single(max(alpha_raw_vec));
        alpha_res_max_dataset(local_idx) = single(max(alpha_res_vec));

        G = build_timescaling_G_sparse(N, L, Lcpp, ch_t, fc, dt);
        YT = G * Xext;
        Heff = afdm_demod_matrix(YT, chirp1, chirp2);

        data_idx = (Q + 1):(N - Q);
        H_sub = Heff(data_idx, data_idx);
        sub_power = (norm(H_sub, 'fro')^2) / N_eff;
        Heff = Heff / sqrt(max(sub_power, 1e-12));

        x = zeros(N, 1);
        x(data_idx) = qpsk_symbols(numel(data_idx));
        x = x * sqrt(N / N_eff);

        H_dataset(:, :, local_idx) = Heff;
        x_dataset(:, 1, local_idx) = x;
    end

    data_dir = fullfile(project_root, 'data');
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end

    version_str = '';
    if ~isempty(version)
        version_str = sprintf('_%s', version);
    end

    if num_batches == 1
        filename = fullfile(data_dir, sprintf('afdm_n%d_%s%s.mat', N, lower(dataset_type), version_str));
    else
        filename = fullfile(data_dir, sprintf('afdm_n%d_%s%s_part%d.mat', N, lower(dataset_type), version_str, batch_idx));
    end

    dataset_mode = mode;
    dataset_mode_code = int32(use_timevary);
    doppler_mode = tv_cfg.doppler_mode;
    doppler_mode_code = tv_cfg.doppler_mode_code;
    alpha_raw_stats = build_abs_stats_struct(alpha_raw_abs_paths);
    alpha_res_stats = build_abs_stats_struct(alpha_res_abs_paths);
    alpha_max_raw_meta = tv_cfg.alpha_max_raw;
    alpha_max_res_meta = tv_cfg.alpha_max_res;
    enable_resampling_comp = logical(tv_cfg.enable_resampling_comp);
    alpha_hat_mode = tv_cfg.alpha_hat_mode;
    timevary_hparams = struct(...
        'rho_alpha', tv_cfg.rho_alpha, ...
        'rho_h', tv_cfg.rho_h, ...
        'alpha_max', tv_cfg.alpha_max, ...
        'alpha_max_raw', tv_cfg.alpha_max_raw, ...
        'alpha_max_res', tv_cfg.alpha_max_res, ...
        'doppler_mode_code', tv_cfg.doppler_mode_code, ...
        'rho_acc', tv_cfg.rho_acc, ...
        'sigma_acc', tv_cfg.sigma_acc, ...
        'rho_delta', tv_cfg.rho_delta, ...
        'sigma_delta', tv_cfg.sigma_delta, ...
        'delta_max', tv_cfg.delta_max, ...
        'motion_profile', tv_cfg.motion_profile, ...
        'target_track_gain', tv_cfg.target_track_gain, ...
        'target_blend', tv_cfg.target_blend, ...
        'profile_v_peak', tv_cfg.profile_v_peak, ...
        'profile_turn_range', tv_cfg.profile_turn_range, ...
        'profile_recede_range', tv_cfg.profile_recede_range, ...
        'profile_heave_amp', tv_cfg.profile_heave_amp, ...
        'profile_heave_cycles', tv_cfg.profile_heave_cycles, ...
        'profile_secondary_amp', tv_cfg.profile_secondary_amp, ...
        'profile_secondary_cycles', tv_cfg.profile_secondary_cycles, ...
        'profile_jitter_std', tv_cfg.profile_jitter_std, ...
        'path_projection_mode', tv_cfg.path_projection_mode, ...
        'beta_min', tv_cfg.beta_min, ...
        'beta_max', tv_cfg.beta_max, ...
        'ell_mode', tv_cfg.ell_mode, ...
        'pdp_mode', tv_cfg.pdp_mode, ...
        'num_frames', int32(num_frames), ...
        'enable_resampling_comp', logical(tv_cfg.enable_resampling_comp), ...
        'alpha_hat_mode', tv_cfg.alpha_hat_mode, ...
        'clip_alpha_res', logical(tv_cfg.clip_alpha_res), ...
        'alpha_raw_abs_max', alpha_raw_stats.abs_max, ...
        'alpha_raw_abs_mean', alpha_raw_stats.abs_mean, ...
        'alpha_raw_p99_abs', alpha_raw_stats.p99_abs, ...
        'alpha_raw_p995_abs', alpha_raw_stats.p995_abs, ...
        'alpha_res_abs_max', alpha_res_stats.abs_max, ...
        'alpha_res_abs_mean', alpha_res_stats.abs_mean, ...
        'alpha_res_p99_abs', alpha_res_stats.p99_abs, ...
        'alpha_res_p995_abs', alpha_res_stats.p995_abs);

    system_params = struct(...
        'N', N, ...
        'N_eff', N_eff, ...
        'Q', Q, ...
        'Delta_f', Delta_f, ...
        'T', T, ...
        'B', B, ...
        'dt', dt, ...
        'fc', fc, ...
        'alpha_max', alpha_max_res, ...
        'alpha_max_raw', tv_cfg.alpha_max_raw, ...
        'alpha_max_res', tv_cfg.alpha_max_res, ...
        'ell_max', ell_max, ...
        'P', P, ...
        'Lcpp', Lcpp, ...
        'Lcps', Lcps, ...
        'c1', c1, ...
        'c2', c2, ...
        'Nv', Nv, ...
        'kmax', kmax, ...
        'dataset_type', dataset_type, ...
        'dataset_mode', dataset_mode, ...
        'dataset_mode_code', dataset_mode_code, ...
        'doppler_mode', doppler_mode, ...
        'doppler_mode_code', doppler_mode_code, ...
        'batch_idx', batch_idx, ...
        'num_batches', num_batches, ...
        'total_samples', num_samples, ...
        'num_seq', num_seq, ...
        'num_frames', num_frames, ...
        'version', version);

    save(filename, ...
        'H_dataset', 'x_dataset', 'snr_dataset', ...
        'sequence_id', 'time_index', ...
        'alpha_com_dataset', 'v_norm_dataset', 'delta_alpha_rms_dataset', ...
        'alpha_hat_dataset', 'alpha_raw_max_dataset', 'alpha_res_max_dataset', ...
        'dataset_mode', 'dataset_mode_code', 'doppler_mode', 'doppler_mode_code', 'timevary_hparams', ...
        'alpha_max_raw_meta', 'alpha_max_res_meta', 'enable_resampling_comp', 'alpha_hat_mode', ...
        'alpha_raw_stats', 'alpha_res_stats', ...
        'system_params', '-v7.3');

    info = dir(filename);
    fprintf('  saved: %s (%.1f MB)\n', filename, info.bytes / 1024^2);
    fprintf('  alpha stats | raw max=%.3e p99=%.3e p995=%.3e | res max=%.3e p99=%.3e p995=%.3e\n', ...
        alpha_raw_stats.abs_max, alpha_raw_stats.p99_abs, alpha_raw_stats.p995_abs, ...
        alpha_res_stats.abs_max, alpha_res_stats.p99_abs, alpha_res_stats.p995_abs);

    all_files{end + 1} = filename; %#ok<AGROW>
    total_generated = total_generated + cur_bs;

    clear H_dataset x_dataset snr_dataset sequence_id time_index alpha_com_dataset v_norm_dataset delta_alpha_rms_dataset;
    clear alpha_hat_dataset alpha_raw_max_dataset alpha_res_max_dataset alpha_raw_abs_paths alpha_res_abs_paths;
end

fprintf('\n========================================\n');
fprintf('Done. total_samples=%d, files=%d\n', total_generated, numel(all_files));
for i = 1:numel(all_files)
    fprintf('  %d) %s\n', i, all_files{i});
end
fprintf('========================================\n');
end

%% ===== Local helpers =====
function snr_db = pick_sequence_snr(sid, num_seq, dataset_type, snr_range)
if strcmpi(dataset_type, 'test')
    sid = max(1, min(num_seq, sid));
    idx = mod(sid - 1, numel(snr_range)) + 1;
    snr_db = snr_range(idx);
else
    snr_db = snr_range(randi(numel(snr_range)));
end
end

function v = getfield_with_default(s, name, default)
if isstruct(s) && isfield(s, name)
    v = s.(name);
else
    v = default;
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

function ch = gen_channel_paper_aligned(P, ell_max, alpha_max)
% Legacy helper retained for backward compatibility in other scripts.
[~, ch] = init_timevary_channel_state(P, ell_max, alpha_max, ...
    struct('rho_alpha', 0, 'rho_h', 0, 'enable_resampling_comp', false));
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
b1 = randi([0,1], M, 1);
b2 = randi([0,1], M, 1);
re = 1 - 2*b1;
im = 1 - 2*b2;
s = (re + 1j*im)/sqrt(2);
end
