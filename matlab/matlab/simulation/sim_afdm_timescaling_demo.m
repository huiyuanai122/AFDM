%% sim_afdm_timescaling_demo.m
% ============================================================
% AFDM over Wideband Doubly-Dispersive Channels with Time-Scaling
% Paper-aligned simulation: (4)(5)(11)(13)(21)
% Outputs: Heff (DAF-domain), y = Heff*x + w (unitary-noise)
% ============================================================
clear; clc; close all;

%% ------------------- Physical/System Params -------------------
N       = 256;          % number of AFDM symbols per frame
Delta_f = 4;            % subcarrier spacing (Hz), paper often uses 4Hz
T       = 1/Delta_f;    % symbol duration (s)
B       = N*Delta_f;    % bandwidth (Hz)
dt      = 1/B;          % sampling interval (s)

fc      = 12e3;         % carrier (Hz) e.g., underwater acoustic ~12kHz
alpha_max = 1e-4;       % Doppler scale factor bound (paper example scale)

% Delay spread in samples (tau_max = ell_max*dt)
ell_max = 16;           % you can set from your channel profile
P       = 6;            % number of paths

% CPP/CPS lengths from paper (6) discussion:
% Tcpp > tau_max/(1-alpha_max), Tcps > alpha_max*T/(1+alpha_max)
Lcpp = max(1, ceil( (ell_max) / (1 - alpha_max) ));
Lcps = max(1, ceil( alpha_max*N / (1 + alpha_max) ));

% Guard bins (optional, to match your existing BER counting style)
Q_guard = 0;            % if you want exclude edges: set e.g. 4 or 8

% Chirp parameter design (paper approx eq(45) + c2 irrational)
Nv   = 2; % “significant magnitude” width used in paper sparsity design
kmax = ceil((alpha_max*fc) * T); % k_max ≈ nu_max/Delta_f, nu_max=alpha_max*fc
den  = (1 - 4*alpha_max*(N-1));
if den <= 0
    error("c1 design invalid: 1-4*alpha_max*(N-1) <= 0. Reduce N or alpha_max.");
end
c1 = (2*kmax + 2*alpha_max*(N-1) + 2*Nv + 1) / (2*N*den); % paper eq(45)
c2 = sqrt(2)/N;  % arbitrary irrational number (paper mentions irrational c2)

fprintf("N=%d, dt=%.3eus, T=%.3fs, B=%.1fHz\n", N, dt*1e6, T, B);
fprintf("alpha_max=%.1e, ell_max=%d (tau_max=%.3fms)\n", alpha_max, ell_max, ell_max*dt*1e3);
fprintf("Lcpp=%d, Lcps=%d, c1=%.6g, c2=%.6g\n", Lcpp, Lcps, c1, c2);

%% ------------------- Monte Carlo BER Demo -------------------
SNR_dB     = 0:2:20;
num_frames = 500;  % demo; you can increase
ber        = zeros(size(SNR_dB));

for s = 1:numel(SNR_dB)
    snr_lin   = 10^(SNR_dB(s)/10);
    noise_var = 1/snr_lin;  % consistent with your existing scripts: Es=1

    bit_err = 0;
    bit_tot = 0;

    for frm = 1:num_frames
        % ---- Generate paper-aligned channel (delays+time-scaling) ----
        ch = gen_channel_paper_aligned(P, ell_max, alpha_max);
        ch.N    = N;
        ch.dt   = dt;
        ch.fc   = fc;
        ch.Lcpp = Lcpp;
        ch.Lcps = Lcps;
        ch.c1   = c1;
        ch.c2   = c2;

        % ---- Build DAF-domain equivalent channel by probing (physics-consistent) ----
        Heff = build_heff_probe(ch);

        % % --- Sanity check: Heff matches physical chain ---
        % x_test = (randn(N,1)+1j*randn(N,1))/sqrt(2);  % random CN(0,1)
        % y1 = Heff * x_test;
        % 
        % % direct physical chain without probing
        % yT_test = afdm_mod(x_test, c1, c2);
        % x_ext   = add_cpp_cps(yT_test, c1, Lcpp, Lcps);
        % yT_win  = timescaling_channel_observe_window(x_ext, ch);
        % y2      = afdm_demod(yT_win, c1, c2);
        % 
        % rel_err = norm(y1 - y2) / max(1e-12, norm(y2));
        % fprintf("Heff self-check rel_err = %.3e\n", rel_err);

        % ---- Generate DAF-domain data symbols ----
        x = zeros(N,1);
        data_idx = (1+Q_guard):(N-Q_guard);
        x(data_idx) = qpsk_symbols(numel(data_idx)); % unit power QPSK

        % ---- Receive in DAF domain (unitary DAFT => AWGN stays AWGN) ----
        w = sqrt(noise_var/2) * (randn(N,1) + 1j*randn(N,1));
        y = Heff*x + w;

        % ---- Detect (example: your existing LMMSE detector) ----
        x_hat = lmmse_detector(y, Heff, noise_var);

        % ---- BER (QPSK hard decision) ----
        bit_err = bit_err + count_qpsk_bit_errors(x_hat(data_idx), x(data_idx));
        bit_tot = bit_tot + 2*numel(data_idx);
    end

    ber(s) = bit_err/bit_tot;
    fprintf("SNR=%2ddB, BER=%.3e\n", SNR_dB(s), ber(s));
end

figure; semilogy(SNR_dB, ber, '-o'); grid on;
xlabel('SNR (dB)'); ylabel('BER'); title('AFDM (paper-aligned time-scaling wideband DD channel)');

%% ============================================================
%% Local functions
%% ============================================================

function ch = gen_channel_paper_aligned(P, ell_max, alpha_max)
% Make earliest path delay = 0 to match paper's "1st path as time reference"
    ell = sort(randi([0, ell_max], P, 1));
    ell = ell - ell(1);

    alpha = (2*rand(P,1)-1) * alpha_max;  % real-valued time-scaling factors

    % simple exponential power delay profile (you can replace with your own UWA PDP)
    ell_rms = max(1, ell_max/3);
    pwr = exp(-ell/ell_rms);

    h = (randn(P,1) + 1j*randn(P,1))/sqrt(2);
    h = h .* sqrt(pwr);
    h = h / norm(h); % normalize total power

    ch.P     = P;
    ch.ell   = ell;     % integer sample delays (can be fractional if you want)
    ch.alpha = alpha;   % Doppler scale factors
    ch.h     = h;       % complex path gains
end

function H = build_heff_probe(ch)
% Build DAF-domain Heff such that y = Heff*x (noise-free),
% by physical modulation->CPP/CPS->time-scaling channel->DAFT demod.
    N   = ch.N;  c1 = ch.c1; c2 = ch.c2;
    Lcpp = ch.Lcpp; Lcps = ch.Lcps;

    % Precompute IDAF basis columns for speed (N=256 is fine)
    XTB = precompute_idaf_basis(N, c1, c2); % [N x N], column m is xT for x=e_m

    H = zeros(N,N);
    for m = 1:N
        xT    = XTB(:,m);
        x_ext = add_cpp_cps(xT, c1, Lcpp, Lcps);
        yT    = timescaling_channel_observe_window(x_ext, ch); % length N (after removing CPP/CPS)
        y     = afdm_demod(yT, c1, c2);
        H(:,m)= y;
    end
end

function XTB = precompute_idaf_basis(N, c1, c2)
% Column m corresponds to x = e_m in DAF domain:
% xT[n] = (1/sqrt(N)) exp(j2π(c1 n^2 + (m-1)n/N + c2(m-1)^2))
    n = (0:N-1).';
    m = (0:N-1);
    phase_n = exp(1j*2*pi*c1*(n.^2));             % [N x 1]
    phase_m = exp(1j*2*pi*c2*(m.^2));             % [1 x N]
    W = exp(1j*2*pi*(n*m)/N) / sqrt(N);           % [N x N]
    XTB = (phase_n .* W) .* phase_m;              % broadcasting: [N x N]
end

function xT = afdm_mod(x, c1, c2)
% xT = U^H x, U = Λc2 F Λc1 (paper eq(21)), IDAF matches paper eq(4)
    N = length(x);
    n = (0:N-1).'; m = n;
    x1  = x .* exp(1j*2*pi*c2*(m.^2));
    tmp = sqrt(N) * ifft(x1);                     % matches (1/sqrt(N)) sum exp(+j2π mn/N)
    xT  = tmp .* exp(1j*2*pi*c1*(n.^2));
end

function x = afdm_demod(yT, c1, c2)
% x = U yT, U = Λc2 F Λc1 (paper eq(21))
    N = length(yT);
    n = (0:N-1).'; m = n;
    tmp = (1/sqrt(N)) * fft(yT .* exp(-1j*2*pi*c1*(n.^2)));
    x   = tmp .* exp(-1j*2*pi*c2*(m.^2));
end

function x_ext = add_cpp_cps(xT, c1, Lcpp, Lcps)
% CPP/CPS per paper eq(5a)(5b)
    N = length(xT);

    n_pre = (-Lcpp:-1).';
    x_pre = xT(n_pre + N + 1) .* exp(-1j*2*pi*c1*(N^2 + 2*N*n_pre));

    n_suf = (N:(N+Lcps-1)).';
    x_suf = xT(n_suf - N + 1) .* exp(+1j*2*pi*c1*(N^2 + 2*N*n_suf));

    x_ext = [x_pre; xT; x_suf];
end

function yT = timescaling_channel_observe_window(x_ext, ch)
% Physics channel (paper eq(11)), sampled to discrete time (paper eq(13)):
% yT[n] = sum_i h_i * xT(((1+alpha_i)n - ell_i)*dt) * exp(j2π nu_i n dt) + w
% Here x_ext provides xT(t) samples over [-Lcpp, N+Lcps-1] * dt,
% and we observe n=0..N-1 (after removing CPP/CPS).
    N    = ch.N;
    P    = ch.P;
    ell  = ch.ell;
    alpha= ch.alpha;
    h    = ch.h;
    fc   = ch.fc;
    dt   = ch.dt;
    Lcpp = ch.Lcpp;

    L = length(x_ext);
    grid = (0:L-1).';         % 0-based index for interp
    n = (0:N-1).';            % observation window samples

    yT = zeros(N,1);

    for i = 1:P
        % idx in x_ext (0-based): (1+alpha)*n - ell + Lcpp
        idx = (1 + alpha(i)) * n - ell(i) + Lcpp;

        % linear interpolation for fractional sample positions
        xi = interp1(grid, x_ext, idx, 'linear', 0);

        % phase term exp(j2π nu_i t), nu_i = alpha_i * fc
        phase = exp(1j*2*pi*(alpha(i)*fc) * (n*dt));

        yT = yT + h(i) * (xi .* phase);
    end
end

function s = qpsk_symbols(M)
% Unit-power QPSK: {±1±j}/sqrt(2)
    b1 = randi([0,1], M, 1);
    b2 = randi([0,1], M, 1);
    re = 1 - 2*b1;
    im = 1 - 2*b2;
    s = (re + 1j*im)/sqrt(2);
end

function e = count_qpsk_bit_errors(x_hat, x_true)
% Hard decision by sign(Re/Im) and count bit errors (2 bits/symbol)
    xh = x_hat(:);
    xt = x_true(:);

    b1h = real(xh) < 0;
    b2h = imag(xh) < 0;

    b1t = real(xt) < 0;
    b2t = imag(xt) < 0;

    e = sum(b1h ~= b1t) + sum(b2h ~= b2t);
end
function x_hat = lmmse_detector(y, H, noise_var)
% LMMSE for y = Hx + w,  w ~ CN(0, noise_var * I)
% Assume E[xx^H] = I (unit symbol power), consistent with unit-power QPSK.
    N = size(H,2);
    A = (H' * H) + noise_var * eye(N);
    b = (H' * y);
    x_hat = A \ b;
end
