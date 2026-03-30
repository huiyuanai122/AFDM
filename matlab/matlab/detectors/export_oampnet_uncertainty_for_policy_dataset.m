function out = export_oampnet_uncertainty_for_policy_dataset(user_cfg)
% Export OAMPNet posterior / uncertainty features for the exact
% tsv2seq_vdop_ctrl paper-style policy dataset replay.

if nargin < 1
    user_cfg = struct();
end

this_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(this_dir, '..', '..');
common_dir = fullfile(project_root, 'matlab', 'common');
addpath(this_dir);
addpath(common_dir);

cfg = struct();
cfg.paper_id = 'tsv2seq_vdop_ctrl';
cfg.output_csv = fullfile(project_root, 'results', 'oampnet_uncertainty_policy_dataset_tsv2seq_vdop_ctrl.csv');
cfg.output_mat = fullfile(project_root, 'results', 'oampnet_uncertainty_policy_dataset_tsv2seq_vdop_ctrl.mat');
cfg.seed = 3;
cfg.num_seq = 500;
cfg.num_frames = 40;
cfg.snr_db_list = 0:2:20;
cfg.doppler_mode = 'common_with_path_residual';
cfg.motion_profile = 'maneuver_heave';
cfg.path_projection_mode = 'symmetric_linear';
cfg.beta_min = 0.45;
cfg.beta_max = 1.65;
cfg.target_track_gain = 0.85;
cfg.target_blend = 0.85;
cfg.profile_v_peak = 0.98;
cfg.profile_heave_amp = 0.20;
cfg.profile_secondary_amp = 0.10;
cfg.oampnet_param_version = 'tsv2seq_vdop_ctrl_paper';
cfg.label_snr_mid = 10;
cfg.label_snr_high = 12;
cfg.label_eval_repeats_low = 1;
cfg.label_eval_repeats_mid = 3;
cfg.label_eval_repeats_high = 6;
cfg.progress_every = 100;
cfg.include_extra_quantiles = true;
cfg = merge_struct(cfg, user_cfg);

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
[~, base_action] = min(abs(c1_grid - c1_base));
base_c1 = c1_grid(base_action);

tv_base = struct( ...
    'rho_alpha', 0.98, ...
    'rho_h', 0.98, ...
    'alpha_max', alpha_max_res, ...
    'alpha_max_raw', alpha_max_raw, ...
    'alpha_max_res', alpha_max_res, ...
    'num_frames', cfg.num_frames, ...
    'doppler_mode', cfg.doppler_mode, ...
    'rho_acc', 0.95, ...
    'sigma_acc', 0.03, ...
    'rho_delta', 0.90, ...
    'sigma_delta', 0.05, ...
    'delta_max', [], ...
    'motion_profile', cfg.motion_profile, ...
    'target_track_gain', cfg.target_track_gain, ...
    'target_blend', cfg.target_blend, ...
    'profile_v_peak', cfg.profile_v_peak, ...
    'profile_turn_range', [0.32, 0.48], ...
    'profile_recede_range', [0.68, 0.86], ...
    'profile_heave_amp', cfg.profile_heave_amp, ...
    'profile_heave_cycles', 1.35, ...
    'profile_secondary_amp', cfg.profile_secondary_amp, ...
    'profile_secondary_cycles', 2.70, ...
    'profile_jitter_std', 0.04, ...
    'path_projection_mode', cfg.path_projection_mode, ...
    'beta_min', cfg.beta_min, ...
    'beta_max', cfg.beta_max, ...
    'ell_mode', 'static', ...
    'pdp_mode', 'exp_fixed_per_sequence', ...
    'enable_resampling_comp', get_struct(cfg, 'enable_resampling_comp', true), ...
    'alpha_hat_mode', get_struct(cfg, 'alpha_hat_mode', 'common_component'), ...
    'clip_alpha_res', get_struct(cfg, 'clip_alpha_res', true), ...
    'log_alpha_stats', get_struct(cfg, 'log_alpha_stats', true));
tv_cfg = get_timevary_defaults(tv_base);

param_path = fullfile(project_root, 'data', ['oampnet_v4_' cfg.oampnet_param_version '_params.mat']);
if ~exist(param_path, 'file')
    error('Missing OAMPNet params: %s', param_path);
end
params = load(param_path);

XTB = precompute_idaf_basis(N, base_c1, c2);
Xext = add_cpp_cps_matrix(XTB, base_c1, Lcpp, Lcps);
idx_n = (0:N-1).';
chirp1 = exp(-1j * 2 * pi * base_c1 * (idx_n .^ 2));
chirp2 = exp(-1j * 2 * pi * c2 * (idx_n .^ 2));

num_samples = cfg.num_seq * cfg.num_frames;
sample_id = zeros(num_samples, 1, 'int32');
sequence_id = zeros(num_samples, 1, 'int32');
time_index = zeros(num_samples, 1, 'int32');
snr_db = zeros(num_samples, 1, 'single');
alpha_com = zeros(num_samples, 1, 'single');
v_norm = zeros(num_samples, 1, 'single');
delta_alpha_rms = zeros(num_samples, 1, 'single');
action_index = int32(base_action - 1) * ones(num_samples, 1, 'int32');
action_c1 = single(base_c1) * ones(num_samples, 1, 'single');

posterior_entropy_mean = zeros(num_samples, 1, 'single');
posterior_entropy_std = zeros(num_samples, 1, 'single');
posterior_entropy_p90 = zeros(num_samples, 1, 'single');
posterior_margin_mean = zeros(num_samples, 1, 'single');
posterior_margin_p10 = zeros(num_samples, 1, 'single');
posterior_margin_p25 = zeros(num_samples, 1, 'single');
maxprob_mean = zeros(num_samples, 1, 'single');
maxprob_p10 = zeros(num_samples, 1, 'single');
maxprob_std = zeros(num_samples, 1, 'single');
tau_mean = zeros(num_samples, 1, 'single');
tau_std = zeros(num_samples, 1, 'single');
tau_p90 = zeros(num_samples, 1, 'single');
tau_p95 = zeros(num_samples, 1, 'single');
tau_max = zeros(num_samples, 1, 'single');

rng(safe_rng_seed(cfg.seed));
sample_idx = 0;
for sid = 1:cfg.num_seq
    [seq_state, ch_t] = init_timevary_channel_state(P, ell_max, alpha_max_res, tv_cfg);
    snr_pick_seq = cfg.snr_db_list(randi(numel(cfg.snr_db_list)));

    for tt = 1:cfg.num_frames
        sample_idx = sample_idx + 1;
        if tt > 1
            [seq_state, ch_t] = step_timevary_channel_state(seq_state); %#ok<NASGU>
        end
        ch = ch_t;
        snr_pick = snr_pick_seq;
        noise_var = 1 / (10 ^ (snr_pick / 10));

        G = build_timescaling_G_sparse(N, L, Lcpp, ch, fc, dt);
        YT = G * Xext;
        Heff = afdm_demod_matrix(YT, chirp1, chirp2);

        Hsub = Heff(data_idx, data_idx);
        sub_power = (norm(Hsub, 'fro') ^ 2) / N_eff;
        Heff = Heff / sqrt(max(sub_power, 1e-12));

        x = zeros(N, 1);
        x(data_idx) = qpsk_symbols(numel(data_idx));
        x = x * sqrt(N / N_eff);

        seed = 1000000 * sample_idx + 100 * base_action + 1;
        rng(safe_rng_seed(seed));
        w = sqrt(noise_var / 2) * (randn(N, 1) + 1j * randn(N, 1));
        y = Heff * x + w;

        [~, diag] = oampnet_detector(y, Heff, noise_var, params, Q);
        prob = diag.prob_last(data_idx, :);
        tau = real(diag.tau_vec_last(data_idx));

        entropy_val = -sum(prob .* log(max(prob, 1e-12)), 2);
        prob_sorted = sort(prob, 2, 'descend');
        margin_val = prob_sorted(:, 1) - prob_sorted(:, 2);
        maxprob_val = prob_sorted(:, 1);

        sample_id(sample_idx) = int32(sample_idx - 1);
        sequence_id(sample_idx) = int32(sid);
        time_index(sample_idx) = int32(tt);
        snr_db(sample_idx) = single(snr_pick);
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

        posterior_entropy_mean(sample_idx) = single(mean(entropy_val));
        posterior_entropy_std(sample_idx) = single(std(entropy_val, 0));
        posterior_entropy_p90(sample_idx) = single(prctile(entropy_val, 90));
        posterior_margin_mean(sample_idx) = single(mean(margin_val));
        posterior_margin_p10(sample_idx) = single(prctile(margin_val, 10));
        posterior_margin_p25(sample_idx) = single(prctile(margin_val, 25));
        maxprob_mean(sample_idx) = single(mean(maxprob_val));
        maxprob_p10(sample_idx) = single(prctile(maxprob_val, 10));
        maxprob_std(sample_idx) = single(std(maxprob_val, 0));
        tau_mean(sample_idx) = single(mean(tau));
        tau_std(sample_idx) = single(std(tau, 0));
        tau_p90(sample_idx) = single(prctile(tau, 90));
        tau_p95(sample_idx) = single(prctile(tau, 95));
        tau_max(sample_idx) = single(max(tau));

        advance_export_rng_tail(sample_idx, snr_pick, numel(c1_grid), N, cfg);

        if mod(sample_idx, cfg.progress_every) == 0
            fprintf('oampnet uncertainty progress %d/%d\n', sample_idx, num_samples);
        end
    end
end

out_dir = fileparts(cfg.output_csv);
if ~isempty(out_dir) && ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
write_uncertainty_csv(cfg.output_csv, ...
    sample_id, sequence_id, time_index, snr_db, alpha_com, v_norm, delta_alpha_rms, ...
    action_index, action_c1, ...
    posterior_entropy_mean, posterior_entropy_std, posterior_entropy_p90, ...
    posterior_margin_mean, posterior_margin_p10, posterior_margin_p25, ...
    maxprob_mean, maxprob_p10, maxprob_std, ...
    tau_mean, tau_std, tau_p90, tau_p95, tau_max);

if ~isempty(cfg.output_mat)
    mat_dir = fileparts(cfg.output_mat);
    if ~isempty(mat_dir) && ~exist(mat_dir, 'dir')
        mkdir(mat_dir);
    end
    save(cfg.output_mat, ...
        'sample_id', 'sequence_id', 'time_index', 'snr_db', ...
        'alpha_com', 'v_norm', 'delta_alpha_rms', ...
        'action_index', 'action_c1', ...
        'posterior_entropy_mean', 'posterior_entropy_std', 'posterior_entropy_p90', ...
        'posterior_margin_mean', 'posterior_margin_p10', 'posterior_margin_p25', ...
        'maxprob_mean', 'maxprob_p10', 'maxprob_std', ...
        'tau_mean', 'tau_std', 'tau_p90', 'tau_p95', 'tau_max', ...
        'base_action', 'base_c1', 'param_path', 'cfg', ...
        'alpha_max_raw', 'alpha_max_res', '-v7.3');
end

out = struct();
out.output_csv = cfg.output_csv;
out.output_mat = cfg.output_mat;
out.num_samples = num_samples;
out.base_action = base_action - 1;
out.base_c1 = base_c1;
out.param_path = param_path;

end


function write_uncertainty_csv(path_out, ...
    sample_id, sequence_id, time_index, snr_db, alpha_com, v_norm, delta_alpha_rms, ...
    action_index, action_c1, ...
    posterior_entropy_mean, posterior_entropy_std, posterior_entropy_p90, ...
    posterior_margin_mean, posterior_margin_p10, posterior_margin_p25, ...
    maxprob_mean, maxprob_p10, maxprob_std, ...
    tau_mean, tau_std, tau_p90, tau_p95, tau_max)

fid = fopen(path_out, 'w');
fprintf(fid, ['sample_id,sequence_id,time_index,snr_db,alpha_com,v_norm,delta_alpha_rms,' ...
    'action_index,action_c1,' ...
    'posterior_entropy_mean,posterior_entropy_std,posterior_entropy_p90,' ...
    'posterior_margin_mean,posterior_margin_p10,posterior_margin_p25,' ...
    'maxprob_mean,maxprob_p10,maxprob_std,' ...
    'tau_mean,tau_std,tau_p90,tau_p95,tau_max\n']);
for i = 1:numel(sample_id)
    fprintf(fid, ...
        '%d,%d,%d,%.6f,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n', ...
        sample_id(i), sequence_id(i), time_index(i), snr_db(i), alpha_com(i), v_norm(i), delta_alpha_rms(i), ...
        action_index(i), action_c1(i), ...
        posterior_entropy_mean(i), posterior_entropy_std(i), posterior_entropy_p90(i), ...
        posterior_margin_mean(i), posterior_margin_p10(i), posterior_margin_p25(i), ...
        maxprob_mean(i), maxprob_p10(i), maxprob_std(i), ...
        tau_mean(i), tau_std(i), tau_p90(i), tau_p95(i), tau_max(i));
end
fclose(fid);

end


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


function seed_u32 = safe_rng_seed(seed_in)
seed_u32 = mod(double(seed_in), 2^32 - 1);
if seed_u32 < 0
    seed_u32 = seed_u32 + (2^32 - 1);
end
seed_u32 = floor(seed_u32);
end


function advance_export_rng_tail(sample_idx, snr_pick, num_actions, N, cfg)
num_eval = get_label_eval_repeats(snr_pick, cfg);
seed_last = 1000000 * sample_idx + 100 * num_actions + num_eval;
rng(safe_rng_seed(seed_last));
randn(N, 1);
randn(N, 1);
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
