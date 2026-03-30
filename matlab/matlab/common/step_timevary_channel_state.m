function [state, ch_t] = step_timevary_channel_state(state)
%STEP_TIMEVARY_CHANNEL_STATE Advance one frame in a sequence-level channel state.
%
% Unified logic:
% - evolve raw wideband time-scaling first
% - apply coarse compensation to produce residual design-scale alpha
% - keep legacy alpha_t as the residual/design-scale alias

cfg = state.cfg;
P = state.P;

switch lower(cfg.doppler_mode)
    case 'independent_path_ar1'
        ra = cfg.rho_alpha;
        alpha_raw_t = ra * state.alpha_raw_t + sqrt(max(0.0, 1 - ra^2)) * state.alpha_max_raw * randn(P, 1);
        if cfg.clip_alpha
            alpha_raw_t = min(max(alpha_raw_t, -state.alpha_max_raw), state.alpha_max_raw);
        end
        alpha_com_t = mean(alpha_raw_t);
        delta_alpha_t = alpha_raw_t - alpha_com_t;
        v_norm_t = alpha_com_t / max(state.alpha_max_raw, 1e-12);
        v_norm_t = min(max(v_norm_t, -1.0), 1.0);
        acc_t = 0.0;
        target_v_norm_t = v_norm_t;
    case 'common_only'
        [v_norm_t, acc_t, target_v_norm_t] = advance_motion_state(state, cfg);
        alpha_com_t = state.alpha_max_raw * v_norm_t;
        delta_alpha_t = zeros(P, 1);
        alpha_raw_t = state.beta_path .* alpha_com_t;
    case 'common_with_path_residual'
        [v_norm_t, acc_t, target_v_norm_t] = advance_motion_state(state, cfg);
        alpha_com_t = state.alpha_max_raw * v_norm_t;
        delta_alpha_t = cfg.rho_delta * state.delta_alpha_t + cfg.sigma_delta * state.alpha_max_raw * randn(P, 1);
        delta_alpha_t = clip_delta(delta_alpha_t, cfg.delta_max);
        alpha_raw_t = state.beta_path .* alpha_com_t + delta_alpha_t;
        if cfg.clip_alpha
            alpha_raw_t = min(max(alpha_raw_t, -state.alpha_max_raw), state.alpha_max_raw);
        end
    otherwise
        error('Unsupported doppler_mode=%s.', cfg.doppler_mode);
end

alpha_hat_t = compute_alpha_hat(cfg, alpha_com_t, alpha_raw_t);
alpha_res_t = compute_alpha_residual(cfg, alpha_raw_t, alpha_hat_t, state.alpha_max_res);
[alpha_raw_stats_t, alpha_res_stats_t] = compute_alpha_frame_stats(alpha_raw_t, alpha_res_t);

% h recursion with PDP weighting and normalization.
rh = cfg.rho_h;
innov = (randn(P, 1) + 1j * randn(P, 1)) / sqrt(2);
h_t = rh * state.h_t + sqrt(max(0.0, 1 - rh^2)) * innov;
h_t = h_t .* sqrt(state.pdp_power);
h_t = h_t / max(norm(h_t), cfg.h_norm_eps);

% ell evolution (default static).
ell = state.ell;
switch lower(cfg.ell_mode)
    case 'static'
        % keep unchanged
    case 'slow_drift'
        if rand() < cfg.ell_slow_drift_prob
            drift = randi([-cfg.ell_slow_drift_step, cfg.ell_slow_drift_step], P, 1);
            ell = ell + drift;
            ell = min(max(ell, 0), state.ell_max);
            ell = sort(ell);
            ell = ell - min(ell);
        end
    otherwise
        % unknown mode -> fallback to static for safety.
end

state.alpha_raw_t = alpha_raw_t;
state.alpha_res_t = alpha_res_t;
state.alpha_hat_t = alpha_hat_t;
state.alpha_t = alpha_res_t;  % legacy alias
state.alpha_com_t = alpha_com_t;
state.delta_alpha_t = delta_alpha_t;
state.delta_alpha_rms_t = compute_path_spread_rms(alpha_raw_t, alpha_com_t);
state.alpha_raw_stats_t = alpha_raw_stats_t;
state.alpha_res_stats_t = alpha_res_stats_t;
state.v_norm_t = v_norm_t;
state.acc_t = acc_t;
state.target_v_norm_t = target_v_norm_t;
state.h_t = h_t;
state.ell = ell;
state.frame_index = int32(double(state.frame_index) + 1);

if cfg.log_alpha_stats
    state.alpha_log_count = state.alpha_log_count + 1;
    state.alpha_raw_abs_max_seen = max(state.alpha_raw_abs_max_seen, alpha_raw_stats_t.abs_max);
    state.alpha_res_abs_max_seen = max(state.alpha_res_abs_max_seen, alpha_res_stats_t.abs_max);
    state.alpha_raw_abs_mean_acc = state.alpha_raw_abs_mean_acc + alpha_raw_stats_t.abs_mean;
    state.alpha_res_abs_mean_acc = state.alpha_res_abs_mean_acc + alpha_res_stats_t.abs_mean;
end

ch_t = compose_channel(state);
end

function [v_norm_t, acc_t, target_v_norm_t] = advance_motion_state(state, cfg)
switch lower(cfg.motion_profile)
    case 'smooth_ar'
        acc_t = cfg.rho_acc * state.acc_t + cfg.sigma_acc * randn();
        v_norm_t = state.v_norm_t + acc_t;
        target_v_norm_t = v_norm_t;
    otherwise
        next_frame = double(state.frame_index) + 1;
        target_v_norm_t = compute_profile_target(cfg, state.motion_state, next_frame);
        acc_raw = cfg.rho_acc * state.acc_t ...
            + cfg.target_track_gain * (target_v_norm_t - state.v_norm_t) ...
            + cfg.profile_jitter_std * randn();
        v_pred = state.v_norm_t + acc_raw;
        v_norm_t = (1 - cfg.target_blend) * v_pred + cfg.target_blend * target_v_norm_t;
        acc_t = v_norm_t - state.v_norm_t;
end
if cfg.clip_v_norm
    v_norm_t = clip_unit(v_norm_t);
    acc_t = v_norm_t - state.v_norm_t;
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
        target_v = 0.0;
end
target_v = clip_unit(target_v);
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
    alpha_hat_t = mean(double(alpha_raw_t(:)));
end
end

function alpha_res_t = compute_alpha_residual(cfg, alpha_raw_t, alpha_hat_t, alpha_max_res)
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
    alpha_res_t = min(max(alpha_res_t, -alpha_max_res), alpha_max_res);
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
