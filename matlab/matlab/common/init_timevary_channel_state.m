function [state, ch_t] = init_timevary_channel_state(P, ell_max, alpha_max_legacy, cfg_in)
%INIT_TIMEVARY_CHANNEL_STATE Initialize one sequence-level time-varying channel state.
%
% New unified logic:
% - Draw one initial {ell, alpha_raw, h} per sequence.
% - Support residual-scale AFDM design after coarse resampling compensation.
% - Keep legacy alpha_t as the residual/design-scale alias for compatibility.

if nargin < 4
    cfg_in = struct();
end
if nargin < 3
    alpha_max_legacy = [];
end

cfg_in = inject_legacy_alpha_cfg(cfg_in, alpha_max_legacy);
cfg = get_timevary_defaults(cfg_in);

P = round(double(P));
ell_max = round(double(ell_max));

ell = sort(randi([0, ell_max], P, 1));
ell = ell - ell(1);

if isempty(cfg.ell_rms)
    ell_rms = max(1, ell_max / 3);
else
    ell_rms = max(1e-12, double(cfg.ell_rms));
end

switch lower(cfg.pdp_mode)
    case 'exp_fixed_per_sequence'
        pwr = exp(-double(ell) / ell_rms);
    otherwise
        % TODO: add alternative PDP options if needed.
        pwr = exp(-double(ell) / ell_rms);
end

h0 = (randn(P, 1) + 1j * randn(P, 1)) / sqrt(2);
h0 = h0 .* sqrt(pwr);
h0 = h0 / max(norm(h0), cfg.h_norm_eps);

beta_path = init_beta_path(P, cfg);
[motion_state, v_norm0, acc0, target_v0] = init_motion_state(cfg);
[alpha_raw0, alpha_com0, delta_alpha0] = init_alpha_state(P, cfg, beta_path, v_norm0);
alpha_hat0 = compute_alpha_hat(cfg, alpha_com0, alpha_raw0);
alpha_res0 = compute_alpha_residual(cfg, alpha_raw0, alpha_hat0);
[alpha_raw_stats0, alpha_res_stats0] = compute_alpha_frame_stats(alpha_raw0, alpha_res0);

state = struct();
state.P = P;
state.ell_max = ell_max;
state.alpha_max_raw = cfg.alpha_max_raw;
state.alpha_max_res = cfg.alpha_max_res;
state.alpha_max = cfg.alpha_max_res;  % legacy alias
state.cfg = cfg;
state.frame_index = int32(1);
state.ell = ell;
state.alpha_raw_t = alpha_raw0;
state.alpha_res_t = alpha_res0;
state.alpha_hat_t = alpha_hat0;
state.alpha_t = alpha_res0;  % legacy alias
state.alpha_com_t = alpha_com0;
state.delta_alpha_t = delta_alpha0;
state.delta_alpha_rms_t = compute_path_spread_rms(alpha_raw0, alpha_com0);
state.alpha_raw_stats_t = alpha_raw_stats0;
state.alpha_res_stats_t = alpha_res_stats0;
state.v_norm_t = v_norm0;
state.acc_t = acc0;
state.target_v_norm_t = target_v0;
state.motion_state = motion_state;
state.beta_path = beta_path;
state.h_t = h0;
state.pdp_power = pwr;

state.alpha_log_count = 1;
state.alpha_raw_abs_max_seen = alpha_raw_stats0.abs_max;
state.alpha_res_abs_max_seen = alpha_res_stats0.abs_max;
state.alpha_raw_abs_mean_acc = alpha_raw_stats0.abs_mean;
state.alpha_res_abs_mean_acc = alpha_res_stats0.abs_mean;

ch_t = compose_channel(state);
end

function cfg_in = inject_legacy_alpha_cfg(cfg_in, alpha_max_legacy)
if isempty(alpha_max_legacy)
    return;
end
if ~isfield(cfg_in, 'alpha_max') || isempty(cfg_in.alpha_max)
    cfg_in.alpha_max = alpha_max_legacy;
end
if ~isfield(cfg_in, 'alpha_max_res') || isempty(cfg_in.alpha_max_res)
    cfg_in.alpha_max_res = alpha_max_legacy;
end
end

function [alpha_raw_t, alpha_com_t, delta_alpha_t] = init_alpha_state(P, cfg, beta_path, v_norm_t)
switch lower(cfg.doppler_mode)
    case 'independent_path_ar1'
        alpha_raw_t = (2 * rand(P, 1) - 1) * cfg.alpha_max_raw;
        alpha_com_t = mean(alpha_raw_t);
        delta_alpha_t = alpha_raw_t - alpha_com_t;
    case 'common_only'
        alpha_com_t = cfg.alpha_max_raw * v_norm_t;
        delta_alpha_t = zeros(P, 1);
        alpha_raw_t = beta_path .* alpha_com_t;
    case 'common_with_path_residual'
        alpha_com_t = cfg.alpha_max_raw * v_norm_t;
        delta_alpha_t = cfg.sigma_delta * cfg.alpha_max_raw * randn(P, 1);
        delta_alpha_t = clip_delta(delta_alpha_t, cfg.delta_max);
        alpha_raw_t = beta_path .* alpha_com_t + delta_alpha_t;
        if cfg.clip_alpha
            alpha_raw_t = min(max(alpha_raw_t, -cfg.alpha_max_raw), cfg.alpha_max_raw);
        end
    otherwise
        error('Unsupported doppler_mode=%s.', cfg.doppler_mode);
end
end

function alpha_hat_t = compute_alpha_hat(cfg, alpha_com_t, alpha_raw_t)
switch lower(cfg.alpha_hat_mode)
    case 'common_component'
        alpha_hat_t = double(alpha_com_t);
    case 'none'
        alpha_hat_t = 0.0;
    otherwise
        error('Unsupported alpha_hat_mode=%s.', cfg.alpha_hat_mode);
end
if ~isscalar(alpha_hat_t)
    alpha_hat_t = mean(double(alpha_hat_t(:)));
end
if ~isfinite(alpha_hat_t)
    if isempty(alpha_raw_t)
        alpha_hat_t = 0.0;
    else
        alpha_hat_t = mean(double(alpha_raw_t(:)));
    end
end
end

function alpha_res_t = compute_alpha_residual(cfg, alpha_raw_t, alpha_hat_t)
if ~cfg.enable_resampling_comp
    alpha_res_t = alpha_raw_t;
else
    den = 1 + double(alpha_hat_t);
    if abs(den) < 1e-12
        den = sign(den) * 1e-12;
        if den == 0
            den = 1e-12;
        end
    end
    alpha_res_t = (1 + alpha_raw_t) ./ den - 1;
end
if cfg.clip_alpha_res
    alpha_res_t = min(max(alpha_res_t, -cfg.alpha_max_res), cfg.alpha_max_res);
end
end

function [raw_stats, res_stats] = compute_alpha_frame_stats(alpha_raw_t, alpha_res_t)
raw_stats = build_abs_stats(alpha_raw_t);
res_stats = build_abs_stats(alpha_res_t);
end

function stats = build_abs_stats(x)
abs_x = abs(double(x(:)));
if isempty(abs_x)
    abs_x = 0.0;
end
stats = struct();
stats.abs_max = max(abs_x);
stats.abs_mean = mean(abs_x);
stats.abs_rms = sqrt(mean(abs_x .^ 2));
end

function ch = compose_channel(state)
ch = struct();
ch.P = state.P;
ch.ell = state.ell;
ch.alpha = state.alpha_res_t;
ch.alpha_raw = state.alpha_raw_t;
ch.alpha_res = state.alpha_res_t;
ch.alpha_hat = state.alpha_hat_t;
ch.alpha_com = state.alpha_com_t;
ch.delta_alpha = state.delta_alpha_t;
ch.delta_alpha_rms = state.delta_alpha_rms_t;
ch.alpha_raw_abs_max = state.alpha_raw_stats_t.abs_max;
ch.alpha_res_abs_max = state.alpha_res_stats_t.abs_max;
ch.alpha_raw_abs_mean = state.alpha_raw_stats_t.abs_mean;
ch.alpha_res_abs_mean = state.alpha_res_stats_t.abs_mean;
ch.h = state.h_t;
ch.v_norm = state.v_norm_t;
ch.acc = state.acc_t;
ch.target_v_norm = state.target_v_norm_t;
ch.beta_path = state.beta_path;
ch.motion_profile = state.cfg.motion_profile;
ch.doppler_mode = state.cfg.doppler_mode;
ch.doppler_mode_code = state.cfg.doppler_mode_code;
ch.alpha_max_raw = state.alpha_max_raw;
ch.alpha_max_res = state.alpha_max_res;
ch.alpha_hat_mode = state.cfg.alpha_hat_mode;
ch.enable_resampling_comp = state.cfg.enable_resampling_comp;
end

function beta_path = init_beta_path(P, cfg)
switch lower(cfg.path_projection_mode)
    case 'ones'
        beta_path = ones(P, 1);
    case 'symmetric_linear'
        if P <= 1
            beta_path = ones(P, 1);
        else
            beta_path = linspace(cfg.beta_min, cfg.beta_max, P).';
            if cfg.beta_random_permute
                beta_path = beta_path(randperm(P));
            end
        end
    case 'random_uniform'
        beta_path = cfg.beta_min + (cfg.beta_max - cfg.beta_min) * rand(P, 1);
    otherwise
        beta_path = ones(P, 1);
end
if strcmpi(cfg.doppler_mode, 'common_only')
    beta_path = ones(P, 1);
end
end

function [motion_state, v_norm_t, acc_t, target_v_t] = init_motion_state(cfg)
motion_state = struct();
motion_state.profile_name = cfg.motion_profile;
motion_state.turn_tau = rand_in_range(cfg.profile_turn_range);
motion_state.recede_tau = rand_in_range(cfg.profile_recede_range);
motion_state.heave_phase_1 = 2 * pi * rand();
motion_state.heave_phase_2 = 2 * pi * rand();
motion_state.profile_sign = sign(rand() - 0.5);
if motion_state.profile_sign == 0
    motion_state.profile_sign = 1;
end
motion_state.profile_gain = 0.85 + 0.15 * rand();

switch lower(cfg.motion_profile)
    case 'smooth_ar'
        target_v_t = (2 * rand() - 1) * cfg.v_init_range;
        v_norm_t = target_v_t;
        acc_t = cfg.acc_init_std * randn();
    otherwise
        target_v_t = compute_profile_target(cfg, motion_state, 1);
        v_norm_t = target_v_t + 0.15 * cfg.profile_jitter_std * randn();
        v_norm_t = clip_unit(v_norm_t);
        acc_t = 0.0;
end
end

function target_v = compute_profile_target(cfg, motion_state, frame_index)
tau = (double(frame_index) - 1) / max(double(cfg.num_frames) - 1, 1);
peak = cfg.profile_v_peak;
switch lower(cfg.motion_profile)
    case 'smooth_ar'
        target_v = motion_state.profile_gain * motion_state.profile_sign * (2 * tau - 1) * cfg.v_init_range;
    case 'approach_turn_recede'
        t1 = motion_state.turn_tau;
        if tau <= t1
            base = -peak + 2 * peak * (tau / max(t1, 1e-6));
        else
            base = peak - 2 * peak * ((tau - t1) / max(1 - t1, 1e-6));
        end
        heave = cfg.profile_heave_amp * sin(2 * pi * cfg.profile_heave_cycles * tau + motion_state.heave_phase_1);
        heave = heave + cfg.profile_secondary_amp * sin(2 * pi * cfg.profile_secondary_cycles * tau + motion_state.heave_phase_2);
        target_v = motion_state.profile_sign * motion_state.profile_gain * base + heave;
    case 'maneuver_heave'
        t1 = motion_state.turn_tau;
        t2 = max(motion_state.recede_tau, t1 + 0.08);
        if tau <= t1
            base = -0.35 * peak + 1.15 * peak * (tau / max(t1, 1e-6));
        elseif tau <= t2
            base = 0.80 * peak - 1.75 * peak * ((tau - t1) / max(t2 - t1, 1e-6));
        else
            base = -0.95 * peak + 1.55 * peak * ((tau - t2) / max(1 - t2, 1e-6));
        end
        heave = cfg.profile_heave_amp * sin(2 * pi * cfg.profile_heave_cycles * tau + motion_state.heave_phase_1);
        heave = heave + cfg.profile_secondary_amp * sin(2 * pi * cfg.profile_secondary_cycles * tau + motion_state.heave_phase_2);
        target_v = motion_state.profile_sign * motion_state.profile_gain * base + heave;
    otherwise
        target_v = (2 * rand() - 1) * cfg.v_init_range;
end
target_v = clip_unit(target_v);
end

function r = rand_in_range(x)
x = double(x(:).');
if numel(x) < 2
    r = x(1);
else
    r = x(1) + (x(2) - x(1)) * rand();
end
end

function y = clip_unit(x)
y = min(max(x, -1.0), 1.0);
end

function rms_val = compute_path_spread_rms(alpha_t, alpha_com_t)
rms_val = sqrt(mean(abs(alpha_t - alpha_com_t).^2));
end

function x = clip_delta(x, delta_max)
if delta_max <= 0
    x(:) = 0;
else
    x = min(max(x, -delta_max), delta_max);
end
end
