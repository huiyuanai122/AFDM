function [f, feature_names] = extract_online_state_features(Heff_base, y_obs, x_hat_obs, noise_var, data_idx, ctx, use_oracle_state, ch_oracle, include_physical_doppler_state, include_policy_history_state)
%EXTRACT_ONLINE_STATE_FEATURES Unified online-observable feature schema.
%
% Formal paper schema (default, no leakage):
%  1)  frob_norm
%  2)  diag_energy_ratio
%  3)  offdiag_energy_ratio
%  4)  cond_log10_proxy
%  5)  col_energy_cv
%  6)  band_energy_ratio
%  7)  residual_energy
%  8)  residual_over_noise
%  9)  soft_symbol_confidence
% 10)  proj_consistency
%
% Optional policy-history block (enabled by default):
%   prev_action_norm, prev_reward, recent_switch_rate
%
% Always-kept temporal context:
%   prev_residual_proxy, frame_index_norm, snr_norm,
%   delta_residual_proxy, delta_offdiag_ratio,
%   delta_band_energy_ratio, delta_frob_norm
%
% Optional physical Doppler extension (disabled by default):
%   alpha_com, v_norm, delta_alpha_rms
%
% Optional debug-only oracle extension (disabled by default):
%   oracle_max_alpha, oracle_mean_alpha, oracle_delay_spread, oracle_h_l4_over_l2

if nargin < 6 || isempty(ctx)
    ctx = struct();
end
if nargin < 7 || isempty(use_oracle_state)
    use_oracle_state = false;
end
if nargin < 8
    ch_oracle = [];
end
if nargin < 9 || isempty(include_physical_doppler_state)
    include_physical_doppler_state = false;
end
if nargin < 10 || isempty(include_policy_history_state)
    include_policy_history_state = true;
end

eps0 = 1e-12;

if nargin >= 5 && ~isempty(data_idx)
    Hsub = Heff_base(data_idx, data_idx);
else
    Hsub = Heff_base;
end

n_eff = size(Hsub, 1);
frob_norm = norm(Hsub, 'fro') / sqrt(max(n_eff, 1));

energy_total = sum(abs(Hsub(:)).^2);
if energy_total < eps0
    energy_total = eps0;
end

diag_e = sum(abs(diag(Hsub)).^2);
diag_ratio = diag_e / energy_total;
offdiag_ratio = max(0.0, 1.0 - diag_ratio);

svals = svd(Hsub);
if isempty(svals)
    cond_proxy = 0.0;
else
    cond_proxy = log10((max(svals) + eps0) / (min(svals) + eps0));
end

col_energy = sum(abs(Hsub).^2, 1);
col_energy_cv = std(col_energy) / max(mean(col_energy), eps0);

band_w = max(1, min(8, floor(n_eff / 16)));
[rr, cc] = ndgrid(1:n_eff, 1:n_eff);
band_mask = abs(rr - cc) <= band_w;
band_ratio = sum(abs(Hsub(band_mask)).^2) / energy_total;

residual_e = 0.0;
resid_over_noise = 0.0;
soft_conf = 0.0;
proj_consistency = 0.0;
if ~isempty(y_obs) && ~isempty(x_hat_obs)
    resid = y_obs - Heff_base * x_hat_obs;
    residual_e = mean(abs(resid).^2);
    resid_over_noise = residual_e / max(noise_var, eps0);

    soft_conf = mean(abs(real(x_hat_obs)) + abs(imag(x_hat_obs))) / sqrt(2);
    proj_consistency = 1.0 - (norm(resid)^2 / max(norm(y_obs)^2, eps0));
end

prev_action_norm = get_ctx(ctx, 'prev_action_norm', 0.0);
prev_reward = get_ctx(ctx, 'prev_reward', 0.0);
prev_residual = get_ctx(ctx, 'prev_residual_proxy', 0.0);
recent_switch_rate = get_ctx(ctx, 'recent_switch_rate', 0.0);
frame_index_norm = get_ctx(ctx, 'frame_index_norm', 0.0);
prev_offdiag_ratio = get_ctx(ctx, 'prev_offdiag_ratio', 0.0);
prev_band_ratio = get_ctx(ctx, 'prev_band_energy_ratio', 0.0);
prev_frob_norm = get_ctx(ctx, 'prev_frob_norm', 0.0);

snr_db = -10 * log10(max(noise_var, eps0));
snr_norm = snr_db / 30.0;

delta_residual = residual_e - prev_residual;
delta_offdiag = offdiag_ratio - prev_offdiag_ratio;
delta_band_ratio = band_ratio - prev_band_ratio;
delta_frob = frob_norm - prev_frob_norm;

f = [frob_norm, diag_ratio, offdiag_ratio, cond_proxy, col_energy_cv, band_ratio, ...
     residual_e, resid_over_noise, soft_conf, proj_consistency];

feature_names = {
    'frob_norm', 'diag_energy_ratio', 'offdiag_energy_ratio', 'cond_log10_proxy', ...
    'col_energy_cv', 'band_energy_ratio', 'residual_energy', 'residual_over_noise', ...
    'soft_symbol_confidence', 'proj_consistency'
};

if include_policy_history_state
    f = [f, prev_action_norm, prev_reward];
    feature_names = [feature_names, {'prev_action_norm', 'prev_reward'}]; %#ok<AGROW>
end

f = [f, prev_residual];
feature_names = [feature_names, {'prev_residual_proxy'}]; %#ok<AGROW>

if include_policy_history_state
    f = [f, recent_switch_rate];
    feature_names = [feature_names, {'recent_switch_rate'}]; %#ok<AGROW>
end

f = [f, frame_index_norm, snr_norm, delta_residual, delta_offdiag, delta_band_ratio, delta_frob];
feature_names = [feature_names, { ...
    'frame_index_norm', 'snr_norm', 'delta_residual_proxy', ...
    'delta_offdiag_ratio', 'delta_band_energy_ratio', 'delta_frob_norm'}]; %#ok<AGROW>

if include_physical_doppler_state
    alpha_com = 0.0;
    v_norm = 0.0;
    delta_alpha_rms = 0.0;
    if ~isempty(ch_oracle)
        if isstruct(ch_oracle)
            if isfield(ch_oracle, 'alpha_com')
                alpha_com = double(ch_oracle.alpha_com);
            elseif isfield(ch_oracle, 'alpha')
                alpha_com = mean(double(ch_oracle.alpha(:)));
            end
            if isfield(ch_oracle, 'v_norm')
                v_norm = double(ch_oracle.v_norm);
            else
                v_norm = alpha_com;
            end
            if isfield(ch_oracle, 'delta_alpha_rms')
                delta_alpha_rms = double(ch_oracle.delta_alpha_rms);
            elseif isfield(ch_oracle, 'alpha')
                delta_alpha_rms = sqrt(mean(abs(double(ch_oracle.alpha(:)) - alpha_com).^2));
            end
        end
    end

    f = [f, alpha_com, v_norm, delta_alpha_rms];
    feature_names = [feature_names, {'alpha_com', 'v_norm', 'delta_alpha_rms'}]; %#ok<AGROW>
end

if use_oracle_state
    if isempty(ch_oracle)
        warning('use_oracle_state=true but ch_oracle missing; oracle extension not appended.');
    else
        max_alpha = max(abs(ch_oracle.alpha));
        mean_alpha = mean(abs(ch_oracle.alpha));
        delay_spread = std(double(ch_oracle.ell)) / max(double(max(ch_oracle.ell) + 1), 1.0);
        h_l4_over_l2 = norm(ch_oracle.h, 4) / max(norm(ch_oracle.h, 2), eps0);

        f = [f, max_alpha, mean_alpha, delay_spread, h_l4_over_l2];
        feature_names = [feature_names, {'oracle_max_alpha', 'oracle_mean_alpha', 'oracle_delay_spread', 'oracle_h_l4_over_l2'}]; %#ok<AGROW>
    end
end

f = double(f(:).');
end

function v = get_ctx(ctx, name, default_v)
if isstruct(ctx) && isfield(ctx, name)
    v = double(ctx.(name));
else
    v = double(default_v);
end
if ~isfinite(v)
    v = double(default_v);
end
end
