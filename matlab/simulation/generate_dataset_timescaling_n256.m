%% =========================================================================
%% AFDM 深度学习数据集生成（时间尺度宽带双色散水声信道，论文对齐版）
%% =========================================================================
% 目的：生成与 sim_afdm_timescaling_multiframe_demo.m 同一物理信道/AFDM 体制下
%       的 (H, x, SNR) 数据集，供 Python 端 OAMP / OAMPNet 展开网络训练使用。
%
% 输出（MATLAB -v7.3，Python utils.AFDMDataset 可直接读取）：
%   H_dataset   : [N x N x S]  复数，DAF 域等效信道矩阵 Heff，使 y = H*x + w
%   x_dataset   : [N x 1 x S]  复数，DAF 域发送符号（QPSK，含可选 guard）
%   snr_dataset : [S x 1]      实数，SNR 标签（dB）
%   system_params (struct)     至少包含 N, N_eff, Q（Python 端会读取）
%
% 使用示例：
%   generate_dataset_timescaling_n256('train', 10000, 2000, 'tsv1');
%   generate_dataset_timescaling_n256('val',   1000,  1000, 'tsv1');
%   generate_dataset_timescaling_n256('test', 11000,  2000, 'tsv1');
%
% 说明：
%   1) 本脚本使用“单帧隔离模型”构建 Heff（与论文中 CPP/CPS 的等效线性模型一致），
%      因此与 Python 数据集接口 y=H*x+w 完全对齐。
%   2) 若你要评估“CPP/CPS=0 的跨帧干扰”，应另外用连续流仿真得到 y，并比较失配影响；
%      训练/验证数据集建议使用 CPP/CPS 按论文取值的线性等效模型。
% =========================================================================

function generate_dataset_timescaling_n256(dataset_type, num_samples, batch_size, version, cfg)

    %% ==================== 参数与默认值 ====================
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

    % --- 固定 N=256（与现有 Python 训练代码一致） ---
    N = getfield_with_default(cfg, 'N', 256);
    assert(N == 256, '当前工程训练脚本默认 N=256，请保持一致。');

    % --- 物理/系统参数（与 sim_afdm_timescaling_multiframe_demo.m 对齐） ---
    Delta_f   = getfield_with_default(cfg, 'Delta_f', 4);        % Hz
    T         = 1/Delta_f;                                       % s
    B         = N*Delta_f;                                       % Hz
    dt        = 1/B;                                             % s

    fc        = getfield_with_default(cfg, 'fc', 12e3);           % Hz
    alpha_max = getfield_with_default(cfg, 'alpha_max', 1e-4);    % time-scaling bound
    ell_max   = getfield_with_default(cfg, 'ell_max', 16);        % max delay (samples)
    P         = getfield_with_default(cfg, 'P', 6);               % #paths

    % guard（训练/BER统计时会忽略边缘 Q）
    Q         = getfield_with_default(cfg, 'Q', 0);
    N_eff     = N - 2*Q;
    if N_eff <= 0
        error('Q 过大导致 N_eff<=0');
    end

    % --- CPP/CPS（按论文条件离散化估计） ---
    Lcpp = getfield_with_default(cfg, 'Lcpp', max(1, ceil( ell_max/(1-alpha_max) )));
    Lcps = getfield_with_default(cfg, 'Lcps', max(1, ceil( alpha_max*N/(1+alpha_max) )));

    % --- c1/c2 设计（与 demo 一致） ---
    Nv   = getfield_with_default(cfg, 'Nv', 2);
    kmax = ceil((alpha_max*fc) * T); % nu_max/Delta_f
    den  = (1 - 4*alpha_max*(N-1));
    if den <= 0
        error('c1 设计无效：1-4*alpha_max*(N-1)<=0，减小 N 或 alpha_max。');
    end
    c1 = (2*kmax + 2*alpha_max*(N-1) + 2*Nv + 1) / (2*N*den);
    c2 = sqrt(2)/N;

    fprintf('========================================\n');
    fprintf('AFDM 数据集生成（time-scaling, paper-aligned）\n');
    fprintf('========================================\n');
    fprintf('type=%s, samples=%d, batch=%d, version=%s\n', dataset_type, num_samples, batch_size, version);
    fprintf('N=%d, N_eff=%d, Q=%d\n', N, N_eff, Q);
    fprintf('Delta_f=%.3gHz, T=%.3gs, B=%.3gHz, dt=%.3gus\n', Delta_f, T, B, dt*1e6);
    fprintf('fc=%.3gHz, alpha_max=%.1e, ell_max=%d (tau_max=%.3gms), P=%d\n', fc, alpha_max, ell_max, ell_max*dt*1e3, P);
    fprintf('Lcpp=%d, Lcps=%d, c1=%.6g, c2=%.6g\n', Lcpp, Lcps, c1, c2);

    %% ==================== 预计算（一次即可） ====================
    fprintf('\n[预计算] IDAF 基 & 扩展帧矩阵...\n');
    XTB   = precompute_idaf_basis(N, c1, c2);              % [N x N]
    Xext  = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps);       % [L x N]
    L     = size(Xext, 1);

    % demod 所需 chirp
    n = (0:N-1).';
    chirp1 = exp(-1j*2*pi*c1*(n.^2));
    chirp2 = exp(-1j*2*pi*c2*(n.^2));

    %% ==================== SNR 设计 ====================
    SNR_range = 0:2:20;
    if strcmpi(dataset_type, 'test')
        snr_per_point = floor(num_samples / numel(SNR_range));
    end

    %% ==================== 分批生成并保存 ====================
    num_batches = ceil(num_samples / batch_size);
    total_generated = 0;
    all_files = {};

    for batch_idx = 1:num_batches
        start_idx = (batch_idx - 1) * batch_size + 1;
        end_idx   = min(batch_idx * batch_size, num_samples);
        cur_bs    = end_idx - start_idx + 1;

        fprintf('\n----------------------------------------\n');
        fprintf('Batch %d/%d: %d~%d (%d samples)\n', batch_idx, num_batches, start_idx, end_idx, cur_bs);
        fprintf('----------------------------------------\n');

        H_dataset   = zeros(N, N, cur_bs, 'like', 1j);
        x_dataset   = zeros(N, 1, cur_bs, 'like', 1j);
        snr_dataset = zeros(cur_bs, 1);

        for local_idx = 1:cur_bs
            global_idx = start_idx + local_idx - 1;

            if mod(local_idx, 200) == 0 || local_idx == 1
                fprintf('  progress: %d/%d (%.1f%%)\n', local_idx, cur_bs, 100*local_idx/cur_bs);
            end

            % --- SNR label ---
            if strcmpi(dataset_type, 'test')
                snr_i = floor((global_idx-1) / max(1, snr_per_point)) + 1;
                snr_i = min(snr_i, numel(SNR_range));
                snr_db = SNR_range(snr_i);
            else
                snr_db = SNR_range(randi(numel(SNR_range)));
            end
            snr_dataset(local_idx) = snr_db;

            % --- 随机信道（论文对齐：连续 alpha，离散 delay，指数功率谱） ---
            ch = gen_channel_paper_aligned(P, ell_max, alpha_max);

            % --- 构造稀疏 G: yT = G * x_ext ---
            G = build_timescaling_G_sparse(N, L, Lcpp, ch, fc, dt);

            % --- 一次性得到 N×N 的时域响应矩阵: YT = G * Xext ---
            YT = G * Xext; % [N x N]

            % --- DAFT 解调得到等效 Heff（DAF 域） ---
            Heff = afdm_demod_matrix(YT, chirp1, chirp2);

            % --- 归一化：使有效子载波子矩阵平均能量为 1 ---
            data_idx = (Q+1):(N-Q);
            H_sub = Heff(data_idx, data_idx);
            sub_power = (norm(H_sub, 'fro')^2) / N_eff;
            Heff = Heff / sqrt(max(sub_power, 1e-12));

            % --- 生成 DAF 域发送符号 x（QPSK，按 N/N_eff 缩放） ---
            x = zeros(N,1);
            x(data_idx) = qpsk_symbols(numel(data_idx));
            x = x * sqrt(N / N_eff);

            % --- 写入数据集 ---
            H_dataset(:,:,local_idx) = Heff;
            x_dataset(:,1,local_idx) = x;

            % （可选）自检：偶尔抽查 y=H*x 是否与直接仿真一致
            % if local_idx == 1 && batch_idx == 1
            %     y1 = Heff * x;
            %     y2 = afdm_demod_vector(G * (add_cpp_cps_vector(afdm_mod_fast(XTB, x), c1, Lcpp, Lcps)), c1, c2);
            %     fprintf('  [check] rel_err = %.3e\n', norm(y1-y2)/max(1e-12,norm(y2)));
            % end
        end

        %% ---- 保存当前 batch ----
        script_dir = fileparts(mfilename('fullpath'));

        % 兼容不同工程结构：优先寻找“最近的包含 data/ 的上级目录”
        project_root = script_dir;
        for k = 1:4
            cand = project_root;
            if exist(fullfile(cand, 'data'), 'dir') || exist(fullfile(cand, 'train_oampnet_v4.py'), 'file')
                project_root = cand;
                break;
            end
            project_root = fileparts(project_root);
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

        system_params = struct(...
            'N', N, ...
            'N_eff', N_eff, ...
            'Q', Q, ...
            'Delta_f', Delta_f, ...
            'T', T, ...
            'B', B, ...
            'dt', dt, ...
            'fc', fc, ...
            'alpha_max', alpha_max, ...
            'ell_max', ell_max, ...
            'P', P, ...
            'Lcpp', Lcpp, ...
            'Lcps', Lcps, ...
            'c1', c1, ...
            'c2', c2, ...
            'Nv', Nv, ...
            'kmax', kmax, ...
            'dataset_type', dataset_type, ...
            'batch_idx', batch_idx, ...
            'num_batches', num_batches, ...
            'total_samples', num_samples, ...
            'version', version ...
        );

        save(filename, 'H_dataset', 'x_dataset', 'snr_dataset', 'system_params', '-v7.3');
        info = dir(filename);
        fprintf('  saved: %s (%.1f MB)\n', filename, info.bytes/1024^2);

        all_files{end+1} = filename; %#ok<AGROW>
        total_generated = total_generated + cur_bs;

        clear H_dataset x_dataset snr_dataset;
    end

    fprintf('\n========================================\n');
    fprintf('Done. total_samples=%d, files=%d\n', total_generated, numel(all_files));
    for i = 1:numel(all_files)
        fprintf('  %d) %s\n', i, all_files{i});
    end
    fprintf('========================================\n');
end


%% ==================== Local helpers ====================

function v = getfield_with_default(s, name, default)
    if isstruct(s) && isfield(s, name)
        v = s.(name);
    else
        v = default;
    end
end

function ch = gen_channel_paper_aligned(P, ell_max, alpha_max)
    % delay: integer samples, first path referenced to 0
    ell = sort(randi([0, ell_max], P, 1));
    ell = ell - ell(1);

    % time-scaling: continuous uniform
    alpha = (2*rand(P,1)-1) * alpha_max;

    % exponential power delay profile
    ell_rms = max(1, ell_max/3);
    pwr = exp(-ell/ell_rms);

    h = (randn(P,1) + 1j*randn(P,1))/sqrt(2);
    h = h .* sqrt(pwr);
    h = h / norm(h);

    ch.P = P;
    ch.ell = ell;
    ch.alpha = alpha;
    ch.h = h;
end

function XTB = precompute_idaf_basis(N, c1, c2)
    n = (0:N-1).';
    m = 0:N-1;
    phase_n = exp(1j*2*pi*c1*(n.^2));
    phase_m = exp(1j*2*pi*c2*(m.^2));
    W = exp(1j*2*pi*(n*m)/N) / sqrt(N);
    XTB = (phase_n .* W) .* phase_m; % implicit expansion
end

function Xext = add_cpp_cps_matrix(XTB, c1, Lcpp, Lcps)
    % XTB: [N x N], each column is xT when x=e_m
    N = size(XTB, 1);
    if Lcpp==0 && Lcps==0
        Xext = XTB;
        return;
    end

    if Lcpp > 0
        n_pre = (-Lcpp:-1).';
        idx_pre = n_pre + N + 1; % MATLAB 1-based
        phase_pre = exp(-1j*2*pi*c1*(N^2 + 2*N*n_pre));
        Xpre = XTB(idx_pre, :) .* phase_pre;
    else
        Xpre = zeros(0, N, 'like', XTB);
    end

    if Lcps > 0
        n_suf = (N:(N+Lcps-1)).';
        idx_suf = (n_suf - N + 1);
        phase_suf = exp(+1j*2*pi*c1*(N^2 + 2*N*n_suf));
        Xsuf = XTB(idx_suf, :) .* phase_suf;
    else
        Xsuf = zeros(0, N, 'like', XTB);
    end

    Xext = [Xpre; XTB; Xsuf];
end

function G = build_timescaling_G_sparse(N, L, Lcpp, ch, fc, dt)
    % Build sparse linear operator: yT[n] = sum_k G(n,k) * x_ext[k]
    % n=0..N-1, k=0..L-1, where k=0 corresponds to -Lcpp.
    P = ch.P;
    ell = ch.ell;
    alpha = ch.alpha;
    h = ch.h;

    n = (0:N-1).';

    % allocate upper bound nnz
    max_nnz = N * P * 2;
    I = zeros(max_nnz, 1);
    J = zeros(max_nnz, 1);
    V = zeros(max_nnz, 1) + 1j*zeros(max_nnz, 1);
    ptr = 0;

    for i = 1:P
        idx = (1 + alpha(i)) * n - ell(i) + Lcpp; % local continuous index
        idx0 = floor(idx);
        frac = idx - idx0;
        idx1 = idx0 + 1;

        phase = exp(1j*2*pi*(alpha(i)*fc) * (n*dt));

        % k0 contribution
        v0 = (idx0 >= 0) & (idx0 <= (L-1));
        nn0 = sum(v0);
        if nn0 > 0
            rows = find(v0);
            cols = idx0(v0) + 1; % to 1-based
            vals = h(i) * phase(v0) .* (1 - frac(v0));
            I(ptr+1:ptr+nn0) = rows;
            J(ptr+1:ptr+nn0) = cols;
            V(ptr+1:ptr+nn0) = vals;
            ptr = ptr + nn0;
        end

        % k1 contribution
        v1 = (idx1 >= 0) & (idx1 <= (L-1)) & (frac > 0);
        nn1 = sum(v1);
        if nn1 > 0
            rows = find(v1);
            cols = idx1(v1) + 1;
            vals = h(i) * phase(v1) .* frac(v1);
            I(ptr+1:ptr+nn1) = rows;
            J(ptr+1:ptr+nn1) = cols;
            V(ptr+1:ptr+nn1) = vals;
            ptr = ptr + nn1;
        end
    end

    I = I(1:ptr);
    J = J(1:ptr);
    V = V(1:ptr);

    G = sparse(I, J, V, N, L);
end

function Heff = afdm_demod_matrix(YT, chirp1, chirp2)
    % Vectorized DAFT demod for matrix input (each column is one waveform)
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
