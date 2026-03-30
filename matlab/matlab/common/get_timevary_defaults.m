function cfg = get_timevary_defaults(cfg_in)
%GET_TIMEVARY_DEFAULTS Default hyper-parameters for sequence-level time-varying channel state.
%
% Unified logic:
% - One sequence has one state object.
% - Per-frame channel is produced by state recursion.
% - doppler_mode selects the time-varying Doppler model:
%   * independent_path_ar1         : legacy baseline
%   * common_only                  : one motion-driven common Doppler term
%   * common_with_path_residual    : common Doppler + small per-path residual
%
% Legacy note:
% Older scripts had local and inconsistent channel evolution code.
% This helper centralizes those defaults so export/online/training share one definition.

if nargin < 1 || isempty(cfg_in)
    cfg_in = struct();
end

has_alpha_max_legacy = isfield(cfg_in, 'alpha_max');
has_alpha_max_raw = isfield(cfg_in, 'alpha_max_raw');
has_alpha_max_res = isfield(cfg_in, 'alpha_max_res');

cfg = struct();

% Legacy compatibility:
% - alpha_max is retained as the residual/design-scale alias.
cfg.alpha_max_raw = 5e-4;
cfg.alpha_max_res = 1e-4;
cfg.alpha_max = cfg.alpha_max_res;

cfg.enable_resampling_comp = true;
cfg.alpha_hat_mode = 'common_component';  % 'common_component' | 'none'
cfg.clip_alpha_res = true;
cfg.log_alpha_stats = true;

% Legacy path-wise recursion parameters.
cfg.rho_alpha = 0.98;
cfg.rho_h = 0.98;

% Common Doppler motion model.
cfg.doppler_mode = 'independent_path_ar1';
cfg.doppler_mode_code = int32(0);
cfg.rho_acc = 0.95;
cfg.sigma_acc = 0.03;
cfg.rho_delta = 0.90;
cfg.sigma_delta = 0.05;
cfg.delta_max = [];                   % [] -> 0.15 * alpha_max
cfg.v_init_range = 0.35;              % initial v_norm ~ U[-range, range]
cfg.acc_init_std = [];                % [] -> sigma_acc
cfg.clip_v_norm = true;

% Motion-profile controls for common Doppler.
% Legacy note:
% - 'smooth_ar' reproduces the previous common-Doppler recursion.
% New logic:
% - profiled motion modes intentionally create regime changes so that
%   dynamic C1 control can matter in the same SNR range where BER is
%   sensitive to C1.
cfg.motion_profile = 'smooth_ar';      % 'smooth_ar' | 'approach_turn_recede' | 'maneuver_heave'
cfg.target_track_gain = 0.75;          % how strongly v_norm tracks the profile target
cfg.target_blend = 0.80;               % blend target into recursive velocity update
cfg.profile_v_peak = 0.95;             % max normalized radial speed before clipping
cfg.profile_turn_range = [0.32, 0.48]; % normalized turn time range
cfg.profile_recede_range = [0.68, 0.86];
cfg.profile_heave_amp = 0.18;          % low-frequency oscillation
cfg.profile_heave_cycles = 1.35;
cfg.profile_secondary_amp = 0.08;      % secondary oscillation
cfg.profile_secondary_cycles = 2.70;
cfg.profile_jitter_std = 0.04;         % additional motion jitter

% Path-wise projection of the common motion term.
cfg.path_projection_mode = 'ones';     % 'ones' | 'symmetric_linear' | 'random_uniform'
cfg.beta_min = 1.0;
cfg.beta_max = 1.0;
cfg.beta_random_permute = true;

% Sequence settings.
cfg.num_frames = 10;
cfg.ell_mode = 'static';              % 'static' | 'slow_drift'
cfg.ell_slow_drift_prob = 0.0;        % used when ell_mode='slow_drift'
cfg.ell_slow_drift_step = 1;

% Path power-delay profile.
cfg.pdp_mode = 'exp_fixed_per_sequence';
cfg.ell_rms = [];                     % [] -> auto from ell_max

% Stability and clipping.
cfg.clip_alpha = true;
cfg.h_norm_eps = 1e-12;

% Merge user overrides.
fields_in = fieldnames(cfg_in);
for k = 1:numel(fields_in)
    cfg.(fields_in{k}) = cfg_in.(fields_in{k});
end

if has_alpha_max_legacy && ~has_alpha_max_res
    cfg.alpha_max_res = cfg_in.alpha_max;
end
if has_alpha_max_legacy && ~has_alpha_max_raw
    cfg.alpha_max_raw = max(cfg.alpha_max_raw, double(cfg.alpha_max_res));
end

% Safety clamps.
cfg.rho_alpha = min(max(double(cfg.rho_alpha), 0.0), 0.999999);
cfg.rho_h = min(max(double(cfg.rho_h), 0.0), 0.999999);
cfg.alpha_max_raw = max(double(cfg.alpha_max_raw), 0.0);
cfg.alpha_max_res = max(double(cfg.alpha_max_res), 0.0);
cfg.alpha_max = cfg.alpha_max_res;
cfg.rho_acc = min(max(double(cfg.rho_acc), 0.0), 0.999999);
cfg.sigma_acc = max(double(cfg.sigma_acc), 0.0);
cfg.rho_delta = min(max(double(cfg.rho_delta), 0.0), 0.999999);
cfg.sigma_delta = max(double(cfg.sigma_delta), 0.0);
cfg.num_frames = max(1, round(double(cfg.num_frames)));
cfg.ell_slow_drift_prob = min(max(double(cfg.ell_slow_drift_prob), 0.0), 1.0);
cfg.ell_slow_drift_step = max(1, round(double(cfg.ell_slow_drift_step)));
cfg.h_norm_eps = max(double(cfg.h_norm_eps), 1e-15);
cfg.v_init_range = min(max(double(cfg.v_init_range), 0.0), 1.0);
cfg.clip_v_norm = logical(cfg.clip_v_norm);
cfg.enable_resampling_comp = logical(cfg.enable_resampling_comp);
cfg.clip_alpha_res = logical(cfg.clip_alpha_res);
cfg.log_alpha_stats = logical(cfg.log_alpha_stats);
cfg.target_track_gain = max(double(cfg.target_track_gain), 0.0);
cfg.target_blend = min(max(double(cfg.target_blend), 0.0), 1.0);
cfg.profile_v_peak = min(max(double(cfg.profile_v_peak), 0.0), 1.5);
cfg.profile_heave_amp = max(double(cfg.profile_heave_amp), 0.0);
cfg.profile_heave_cycles = max(double(cfg.profile_heave_cycles), 0.0);
cfg.profile_secondary_amp = max(double(cfg.profile_secondary_amp), 0.0);
cfg.profile_secondary_cycles = max(double(cfg.profile_secondary_cycles), 0.0);
cfg.profile_jitter_std = max(double(cfg.profile_jitter_std), 0.0);
cfg.beta_min = double(cfg.beta_min);
cfg.beta_max = double(cfg.beta_max);
if cfg.beta_min > cfg.beta_max
    tmp = cfg.beta_min;
    cfg.beta_min = cfg.beta_max;
    cfg.beta_max = tmp;
end
cfg.beta_random_permute = logical(cfg.beta_random_permute);

cfg.profile_turn_range = normalize_range(cfg.profile_turn_range, [0.15, 0.60]);
cfg.profile_recede_range = normalize_range(cfg.profile_recede_range, [0.55, 0.95]);
if cfg.profile_recede_range(1) <= cfg.profile_turn_range(2)
    cfg.profile_recede_range(1) = min(0.98, cfg.profile_turn_range(2) + 0.08);
end

if isempty(cfg.delta_max)
    cfg.delta_max = 0.15 * cfg.alpha_max_raw;
else
    cfg.delta_max = max(double(cfg.delta_max), 0.0);
end
if isempty(cfg.acc_init_std)
    cfg.acc_init_std = cfg.sigma_acc;
else
    cfg.acc_init_std = max(double(cfg.acc_init_std), 0.0);
end

cfg.ell_mode = char(string(cfg.ell_mode));
cfg.pdp_mode = char(string(cfg.pdp_mode));
cfg.doppler_mode = char(string(cfg.doppler_mode));
cfg.motion_profile = char(string(cfg.motion_profile));
cfg.path_projection_mode = char(string(cfg.path_projection_mode));
cfg.alpha_hat_mode = char(string(cfg.alpha_hat_mode));

switch lower(cfg.doppler_mode)
    case 'independent_path_ar1'
        cfg.doppler_mode_code = int32(0);
    case 'common_only'
        cfg.doppler_mode_code = int32(1);
    case 'common_with_path_residual'
        cfg.doppler_mode_code = int32(2);
    otherwise
        error('Unsupported doppler_mode=%s.', cfg.doppler_mode);
end

switch lower(cfg.motion_profile)
    case {'smooth_ar', 'approach_turn_recede', 'maneuver_heave'}
        % supported
    otherwise
        error('Unsupported motion_profile=%s.', cfg.motion_profile);
end

switch lower(cfg.path_projection_mode)
    case {'ones', 'symmetric_linear', 'random_uniform'}
        % supported
    otherwise
        error('Unsupported path_projection_mode=%s.', cfg.path_projection_mode);
end

switch lower(cfg.alpha_hat_mode)
    case {'common_component', 'none'}
        % supported
    otherwise
        error('Unsupported alpha_hat_mode=%s.', cfg.alpha_hat_mode);
end
end

function r = normalize_range(x, fallback)
x = double(x(:).');
if numel(x) < 2
    x = fallback;
else
    x = x(1:2);
end
x = sort(x);
x(1) = min(max(x(1), 0.0), 0.98);
x(2) = min(max(x(2), x(1) + 1e-3), 0.99);
r = x;
end
