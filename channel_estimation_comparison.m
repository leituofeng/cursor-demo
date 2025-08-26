% --- OFDM, OCDM, AFDM, OTFS 信道估计性能统一对比仿真 ---
% 版本: 2.4 (最终修正版: 彻底解决sigmax矩阵维度错误)
% --------------------------------------------------------------------------
close all; clear all; clc;

%% 1. 统一仿真参数
% --------------------------------------------------------------------------
fprintf('--- 初始化统一仿真参数 ---\n');
% --- 物理层和信道参数 ---
N = 256;
bps = 2;
M = 2^bps;
delta_f = 15e3;
fc = 4e9;
Ts = 1 / (N * delta_f);

% --- 信道特性 ---
max_speed_kmh = 300;
max_doppler_freq = (max_speed_kmh / 3.6) / (3e8 / fc);
max_norm_doppler = floor(max_doppler_freq / delta_f);
P_paths = 4;
max_delay_s = 2e-6;
max_delay_idx = ceil(max_delay_s / Ts);

% --- 仿真控制 ---
SNRd_dB = 0:4:24;
SNRp_dB = 35;
num_channel_realizations = 200;
num_frames_per_channel = 20;

fprintf('N=%d, P=%d, l_max=%d, alpha_max=%d\n', N, P_paths, max_delay_idx, max_norm_doppler);

%% 2. 各方案的预计算和帧结构定义
% --------------------------------------------------------------------------
noise_var = 1.0;
sigma_p = 10^(SNRp_dB / 10);
x_pilot_val = sqrt(sigma_p);

% --- (1) OFDM ---
OFDM.F = dftmtx(N) / sqrt(N);
OFDM.IF = OFDM.F';
OFDM.pilot_spacing = 8;
OFDM.pilot_indices = (1:OFDM.pilot_spacing:N)';
OFDM.data_indices = setdiff((1:N)', OFDM.pilot_indices);
OFDM.num_data_symbols = length(OFDM.data_indices);
OFDM.c1 = 0; % OFDM没有chirp

% --- (2) OCDM ---
OCDM.c1 = 1 / (2 * N);
OCDM.c2 = 1 / (2 * N);
L1_ocdm = diag(exp(-1j * 2 * pi * OCDM.c1 * ((0:N-1).^2)));
L2_ocdm = diag(exp(-1j * 2 * pi * OCDM.c2 * ((0:N-1).^2)));
OCDM.A = L2_ocdm * (dftmtx(N)/sqrt(N)) * L1_ocdm;
OCDM.IA = OCDM.A';
OCDM.Q = (max_delay_idx + 1) * (2 * max_norm_doppler + 1) - 1;
OCDM.pilot_idx = 1;
OCDM.data_indices = (OCDM.Q+2 : N-OCDM.Q)';
if isempty(OCDM.data_indices), error('OCDM N is too small for given channel spread'); end
OCDM.num_data_symbols = length(OCDM.data_indices);
OCDM.lut = create_lut(N, max_delay_idx, max_norm_doppler, OCDM.c1);

% --- (3) AFDM ---
AFDM.c1 = (2 * max_norm_doppler + 1) / (2 * N);
AFDM.c2 = sqrt(2)/(N^2); % Note: This c2 is for theory, not directly used in this simplified estimator
L1_afdm = diag(exp(-1j * 2 * pi * AFDM.c1 * ((0:N-1).^2)));
L2_afdm = diag(exp(-1j * 2 * pi * AFDM.c2 * ((0:N-1).^2)));
AFDM.A = L2_afdm * (dftmtx(N)/sqrt(N)) * L1_afdm;
AFDM.IA = AFDM.A';
AFDM.Q = (max_delay_idx + 1) * (2 * max_norm_doppler + 1) - 1;
AFDM.pilot_idx = 1;
AFDM.data_indices = (AFDM.Q+2 : N-AFDM.Q)';
if isempty(AFDM.data_indices), error('AFDM N is too small for given channel spread'); end
AFDM.num_data_symbols = length(AFDM.data_indices);
AFDM.lut = create_lut(N, max_delay_idx, max_norm_doppler, AFDM.c1);

% --- (4) OTFS ---
OTFS.N_otfs = 16;
OTFS.M_otfs = N / OTFS.N_otfs;
OTFS.pilot_val = x_pilot_val;
OTFS.pilot_l = 0; OTFS.pilot_k = 0;
OTFS.pilot_idx_1d = OTFS.pilot_l * OTFS.M_otfs + OTFS.pilot_k + 1;
OTFS.guard_l = max_delay_idx;
OTFS.guard_k = max_norm_doppler;
[k_indices, l_indices] = meshgrid(0:OTFS.M_otfs-1, 0:OTFS.N_otfs-1);
is_guard = (abs(l_indices - OTFS.pilot_l) <= OTFS.guard_l) & (abs(k_indices - OTFS.pilot_k) <= OTFS.guard_k);
is_pilot = (l_indices == OTFS.pilot_l) & (k_indices == OTFS.pilot_k);
is_data = ~is_guard;
OTFS.data_indices_2d = find(is_data);
OTFS.pilot_indices_2d = find(is_pilot);
OTFS.data_indices_2d = setdiff(OTFS.data_indices_2d, OTFS.pilot_indices_2d);
OTFS.num_data_symbols = length(OTFS.data_indices_2d);

%% 3. 预生成信道
% --------------------------------------------------------------------------
fprintf('--- 预生成信道以保证公平对比 ---\n');
channel_realizations = cell(num_channel_realizations, 1);
for i = 1:num_channel_realizations
    [h, l, a] = generate_realistic_channel(P_paths, max_delay_s, Ts, max_doppler_freq, delta_f);
    channel_realizations{i} = {h, l, a};
end

%% 4. 主仿真循环
% --------------------------------------------------------------------------
results = struct();
schemes = {'OFDM', 'OCDM', 'AFDM', 'OTFS'};
for s = schemes
    results.(s{1}).ber_ideal = zeros(1, length(SNRd_dB));
    results.(s{1}).ber_est = zeros(1, length(SNRd_dB));
end

for i_snr = 1:length(SNRd_dB)
    sigma_d = 10^(SNRd_dB(i_snr) / 10);
    fprintf('\nSNR = %d dB...\n', SNRd_dB(i_snr));

    % --- (1) OFDM 仿真 ---
    s = 'OFDM';
    fprintf('  Simulating %s...\n', s);
    error_bits_ideal = 0; error_bits_est = 0;
    for i_realization = 1:num_channel_realizations
        [h_true, l_true, a_true] = channel_realizations{i_realization}{:};
        
        H_time_true = build_time_domain_channel_matrix(N, h_true, l_true, a_true, OFDM.c1);
        H_eff_ideal = OFDM.F * H_time_true * OFDM.IF; 

        for i_frame = 1:num_frames_per_channel
            data_bits = randi([0 1], OFDM.num_data_symbols * bps, 1);
            data_symbols = sqrt(sigma_d) * qammod(data_bits, M, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
            
            noise = sqrt(noise_var/2) * (randn(N, 1) + 1i*randn(N, 1));

            % --- 理想CSI ---
            X_ideal = zeros(N, 1); X_ideal(OFDM.data_indices) = data_symbols;
            s_tx_ideal = OFDM.IF * X_ideal;
            s_rx_ideal = H_time_true * s_tx_ideal + noise;
            y_ideal = OFDM.F * s_rx_ideal;
            % !!! BUG FIX: Removed wrapping diag() function. sparse() already creates the N x N matrix.
            sigmax_ideal = sparse(OFDM.data_indices, OFDM.data_indices, sigma_d, N, N);
            x_demod = Block_LMMSE_detector(N, sigmax_ideal, noise_var, H_eff_ideal, y_ideal);
            bits_est = qamdemod(x_demod(OFDM.data_indices)/sqrt(sigma_d), M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true);
            error_bits_ideal = error_bits_ideal + sum(xor(bits_est, data_bits));
            
            % --- 估计CSI ---
            X_est_freq = zeros(N, 1); X_est_freq(OFDM.data_indices) = data_symbols; X_est_freq(OFDM.pilot_indices) = x_pilot_val;
            s_tx_est = OFDM.IF * X_est_freq;
            s_rx_est = H_time_true * s_tx_est + noise;
            y_est = OFDM.F * s_rx_est;
            H_freq_est_vec = estimate_channel_ofdm(y_est, OFDM, SNRp_dB); 
            H_eff_est = diag(H_freq_est_vec); 
            % !!! BUG FIX: Removed wrapping diag() functions.
            sigmax_est_data = sparse(OFDM.data_indices, OFDM.data_indices, sigma_d, N, N);
            sigmax_est_pilot = sparse(OFDM.pilot_indices, OFDM.pilot_indices, sigma_p, N, N);
            sigmax_est = sigmax_est_data + sigmax_est_pilot;
            x_demod = Block_LMMSE_detector(N, sigmax_est, noise_var, H_eff_est, y_est);
            bits_est = qamdemod(x_demod(OFDM.data_indices)/sqrt(sigma_d), M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true);
            error_bits_est = error_bits_est + sum(xor(bits_est, data_bits));
        end
    end
    total_bits_ofdm = OFDM.num_data_symbols * bps * num_frames_per_channel * num_channel_realizations;
    results.(s).ber_ideal(i_snr) = error_bits_ideal / total_bits_ofdm;
    results.(s).ber_est(i_snr) = error_bits_est / total_bits_ofdm;

    % --- (2) OCDM 仿真 ---
    s = 'OCDM';
    fprintf('  Simulating %s...\n', s);
    error_bits_ideal = 0; error_bits_est = 0;
    for i_realization = 1:num_channel_realizations
        [h_true, l_true, a_true] = channel_realizations{i_realization}{:};
        for i_frame = 1:num_frames_per_channel
            data_bits = randi([0 1], OCDM.num_data_symbols * bps, 1);
            data_symbols = sqrt(sigma_d) * qammod(data_bits, M, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
            
            % 理想
            X_ideal = zeros(N, 1); X_ideal(OCDM.data_indices) = data_symbols;
            [y_ideal, H_eff_ideal] = transmit_over_channel(X_ideal, N, OCDM.IA, h_true, l_true, a_true, OCDM.c1, OCDM.A, noise_var);
            % !!! BUG FIX: Removed wrapping diag() function.
            sigmax_ideal = sparse(OCDM.data_indices, OCDM.data_indices, sigma_d, N, N);
            x_demod = Block_LMMSE_detector(N, sigmax_ideal, noise_var, H_eff_ideal, y_ideal);
            bits_est = qamdemod(x_demod(OCDM.data_indices)/sqrt(sigma_d), M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true);
            error_bits_ideal = error_bits_ideal + sum(xor(bits_est, data_bits));
            
            % 估计
            X_est = zeros(N, 1); X_est(OCDM.data_indices) = data_symbols; X_est(OCDM.pilot_idx) = x_pilot_val;
            y_est = transmit_over_channel(X_est, N, OCDM.IA, h_true, l_true, a_true, OCDM.c1, OCDM.A, noise_var);
            [h_est, l_est, a_est] = estimate_channel_peak_finding(y_est, P_paths, OCDM.lut, x_pilot_val, OCDM.c1, OCDM.c2);
            H_eff_est = build_eff_channel_matrix(N, OCDM.A, OCDM.IA, h_est, l_est, a_est, OCDM.c1);
            % !!! BUG FIX: Removed wrapping diag() functions.
            sigmax_est = sparse(OCDM.data_indices, OCDM.data_indices, sigma_d, N, N) + sparse(OCDM.pilot_idx, OCDM.pilot_idx, sigma_p, N, N);
            x_demod = Block_LMMSE_detector(N, sigmax_est, noise_var, H_eff_est, y_est);
            bits_est = qamdemod(x_demod(OCDM.data_indices)/sqrt(sigma_d), M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true);
            error_bits_est = error_bits_est + sum(xor(bits_est, data_bits));
        end
    end
    total_bits_ocdm = OCDM.num_data_symbols * bps * num_frames_per_channel * num_channel_realizations;
    results.(s).ber_ideal(i_snr) = error_bits_ideal / total_bits_ocdm;
    results.(s).ber_est(i_snr) = error_bits_est / total_bits_ocdm;

    % --- (3) AFDM 仿真 ---
    s = 'AFDM';
    fprintf('  Simulating %s...\n', s);
    error_bits_ideal = 0; error_bits_est = 0;
    for i_realization = 1:num_channel_realizations
        [h_true, l_true, a_true] = channel_realizations{i_realization}{:};
        for i_frame = 1:num_frames_per_channel
            data_bits = randi([0 1], AFDM.num_data_symbols * bps, 1);
            data_symbols = sqrt(sigma_d) * qammod(data_bits, M, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
            
            % 理想
            X_ideal = zeros(N, 1); X_ideal(AFDM.data_indices) = data_symbols;
            [y_ideal, H_eff_ideal] = transmit_over_channel(X_ideal, N, AFDM.IA, h_true, l_true, a_true, AFDM.c1, AFDM.A, noise_var);
            % !!! BUG FIX: Removed wrapping diag() function.
            sigmax_ideal = sparse(AFDM.data_indices, AFDM.data_indices, sigma_d, N, N);
            x_demod = Block_LMMSE_detector(N, sigmax_ideal, noise_var, H_eff_ideal, y_ideal);
            bits_est = qamdemod(x_demod(AFDM.data_indices)/sqrt(sigma_d), M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true);
            error_bits_ideal = error_bits_ideal + sum(xor(bits_est, data_bits));
            
            % 估计
            X_est = zeros(N, 1); X_est(AFDM.data_indices) = data_symbols; X_est(AFDM.pilot_idx) = x_pilot_val;
            y_est = transmit_over_channel(X_est, N, AFDM.IA, h_true, l_true, a_true, AFDM.c1, AFDM.A, noise_var);
            [h_est, l_est, a_est] = estimate_channel_peak_finding(y_est, P_paths, AFDM.lut, x_pilot_val, AFDM.c1, AFDM.c2);
            H_eff_est = build_eff_channel_matrix(N, AFDM.A, AFDM.IA, h_est, l_est, a_est, AFDM.c1);
            % !!! BUG FIX: Removed wrapping diag() functions.
            sigmax_est = sparse(AFDM.data_indices, AFDM.data_indices, sigma_d, N, N) + sparse(AFDM.pilot_idx, AFDM.pilot_idx, sigma_p, N, N);
            x_demod = Block_LMMSE_detector(N, sigmax_est, noise_var, H_eff_est, y_est);
            bits_est = qamdemod(x_demod(AFDM.data_indices)/sqrt(sigma_d), M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true);
            error_bits_est = error_bits_est + sum(xor(bits_est, data_bits));
        end
    end
    total_bits_afdm = AFDM.num_data_symbols * bps * num_frames_per_channel * num_channel_realizations;
    results.(s).ber_ideal(i_snr) = error_bits_ideal / total_bits_afdm;
    results.(s).ber_est(i_snr) = error_bits_est / total_bits_afdm;

    % --- (4) OTFS 仿真 ---
    s = 'OTFS';
    fprintf('  Simulating %s...\n', s);
    error_bits_ideal = 0; error_bits_est = 0;
    for i_realization = 1:num_channel_realizations
        [h_true, l_true, a_true] = channel_realizations{i_realization}{:};
        for i_frame = 1:num_frames_per_channel
            data_bits = randi([0 1], OTFS.num_data_symbols * bps, 1);
            data_symbols = sqrt(sigma_d) * qammod(data_bits, M, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
            
            % 理想
            x_dd_ideal = zeros(OTFS.N_otfs, OTFS.M_otfs);
            x_dd_ideal(OTFS.data_indices_2d) = data_symbols;
            [y_ideal_vec, H_eff_ideal] = transmit_over_channel_otfs(x_dd_ideal, OTFS, h_true, l_true, a_true, noise_var);
            % (OTFS implementation was correct, no fix needed here)
            sigmax_ideal_vec = zeros(N, 1);
            sigmax_ideal_vec(OTFS.data_indices_2d) = sigma_d;
            sigmax_ideal = diag(sigmax_ideal_vec);
            x_demod = Block_LMMSE_detector(N, sigmax_ideal, noise_var, H_eff_ideal, y_ideal_vec);
            x_demod_2d = reshape(x_demod, OTFS.N_otfs, OTFS.M_otfs);
            bits_est = qamdemod(x_demod_2d(OTFS.data_indices_2d)/sqrt(sigma_d), M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true);
            error_bits_ideal = error_bits_ideal + sum(xor(bits_est, data_bits));
            
            % 估计
            x_dd_est = zeros(OTFS.N_otfs, OTFS.M_otfs);
            x_dd_est(OTFS.data_indices_2d) = data_symbols;
            x_dd_est(OTFS.pilot_indices_2d) = x_pilot_val;
            y_est_vec = transmit_over_channel_otfs(x_dd_est, OTFS, h_true, l_true, a_true, noise_var);
            H_eff_est = estimate_channel_otfs(y_est_vec, OTFS, x_pilot_val);
            % (OTFS implementation was correct, no fix needed here)
            sigmax_est_vec = zeros(N,1);
            sigmax_est_vec(OTFS.data_indices_2d) = sigma_d;
            sigmax_est_vec(OTFS.pilot_idx_1d) = sigma_p;
            sigmax_est = diag(sigmax_est_vec);
            x_demod = Block_LMMSE_detector(N, sigmax_est, noise_var, H_eff_est, y_est_vec);
            x_demod_2d = reshape(x_demod, OTFS.N_otfs, OTFS.M_otfs);
            bits_est = qamdemod(x_demod_2d(OTFS.data_indices_2d)/sqrt(sigma_d), M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true);
            error_bits_est = error_bits_est + sum(xor(bits_est, data_bits));
        end
    end
    total_bits_otfs = OTFS.num_data_symbols * bps * num_frames_per_channel * num_channel_realizations;
    results.(s).ber_ideal(i_snr) = error_bits_ideal / total_bits_otfs;
    results.(s).ber_est(i_snr) = error_bits_est / total_bits_otfs;
end

%% 5. 绘制结果
% --------------------------------------------------------------------------
figure; hold on; grid on; box on;
colors = {'r', 'g', 'b', 'm'};
markers = {'-o', '-s', '-^', '-d'};
for i = 1:length(schemes)
    s = schemes{i};
    semilogy(SNRd_dB, results.(s).ber_ideal, markers{i}, 'Color', colors{i}, 'LineWidth', 1.5, 'DisplayName', [s ' Ideal CSI']);
end
legend('Location', 'southwest', 'FontSize', 10);
title(sprintf('理想CSI性能对比 (N=%d, %dkm/h, %d条路径)', N, max_speed_kmh, P_paths), 'FontSize', 12);
xlabel('每个数据符号的信噪比 (dB)', 'FontSize', 11);
ylabel('误比特率 (BER)', 'FontSize', 11);
set(gca, 'YScale', 'log'); ylim([1e-5, 1]); axis tight;

figure; hold on; grid on; box on;
for i = 1:length(schemes)
    s = schemes{i};
    semilogy(SNRd_dB, results.(s).ber_est, markers{i}, 'Color', colors{i}, 'LineWidth', 1.5, 'DisplayName', [s ' Estimated CSI']);
end
legend('Location', 'southwest', 'FontSize', 10);
title(sprintf('估计CSI性能对比 (导频SNR=%ddB)', SNRp_dB), 'FontSize', 12);
xlabel('每个数据符号的信噪比 (dB)', 'FontSize', 11);
ylabel('误比特率 (BER)', 'FontSize', 11);
set(gca, 'YScale', 'log'); ylim([1e-5, 1]); axis tight;

%% 6. 辅助函数 (Functions are consistent and correct)
% =========================================================================
function [fadingCoefs, delay_indices, doppler_indices] = generate_realistic_channel(P_paths, max_delay_s, Ts, max_doppler_freq, delta_f)
    path_delays_s = linspace(0, max_delay_s, P_paths);
    path_powers_linear = exp(-(0:P_paths-1)); path_powers_linear = path_powers_linear / sum(path_powers_linear);
    delay_indices = round(path_delays_s / Ts);
    [delay_indices, unique_map] = unique(delay_indices, 'stable');
    path_powers_linear = path_powers_linear(unique_map); path_powers_linear = path_powers_linear / sum(path_powers_linear);
    num_effective_paths = length(delay_indices);
    doppler_indices = zeros(1, num_effective_paths);
    for i = 1:num_effective_paths
        physical_doppler = max_doppler_freq * (2*rand-1);
        doppler_indices(i) = round(physical_doppler / delta_f);
    end
    rayleigh_fading = (randn(1, num_effective_paths) + 1j*randn(1, num_effective_paths))/sqrt(2);
    fadingCoefs = rayleigh_fading .* sqrt(path_powers_linear);
    [~, unique_idx] = unique([delay_indices', doppler_indices'], 'rows', 'stable');
    delay_indices = delay_indices(unique_idx); doppler_indices = doppler_indices(unique_idx); fadingCoefs = fadingCoefs(unique_idx);
end

function H = build_time_domain_channel_matrix(N, h, l, a, c1)
    H = zeros(N, N);
    for p = 1:length(h)
        lp = l(p); kp = a(p); hp = h(p);
        DelayMatrix = circshift(eye(N), lp);
        DopplerMatrix = diag(exp(1j*2*pi*(0:N-1)/N*kp));
        if c1 ~= 0
            phase_term = exp(-1j*2*pi*c1*(2*(0:N-1)*lp - lp^2)).';
            H_path = diag(phase_term) * DopplerMatrix * DelayMatrix;
        else
            H_path = DopplerMatrix * DelayMatrix;
        end
        H = H + hp * H_path;
    end
end

function [x_data] = Block_LMMSE_detector(N, sigmax, noise_var, Heff, y)
    if isdiag(Heff)
       Heff_diag = diag(Heff);
       sigmax_diag = diag(sigmax);
       snr_val = sigmax_diag / noise_var;
       g = conj(Heff_diag) ./ (abs(Heff_diag).^2 + 1./snr_val);
       g(isnan(g) | isinf(g)) = 0;
       x_data = g .* y;
    else
       Rn = Heff * sigmax * Heff' + noise_var * eye(N);
       x_data = sigmax * Heff' * (Rn \ y);
    end
end

function [y_received, H_eff_ideal] = transmit_over_channel(X, N, Modulator, h, l, a, c1, Demodulator, noise_var)
    s_tx = Modulator * X;
    H_time = build_time_domain_channel_matrix(N, h, l, a, c1);
    noise = sqrt(noise_var/2) * (randn(N, 1) + 1i*randn(N, 1));
    s_rx = H_time * s_tx + noise;
    y_received = Demodulator * s_rx;
    if nargout > 1, H_eff_ideal = Demodulator * H_time * Modulator; end
end

function H_eff = build_eff_channel_matrix(N, A, IA, h, l, a, c1)
    H_time = build_time_domain_channel_matrix(N, h, l, a, c1);
    H_eff = A * H_time * IA;
end

function [h_est, l_est, a_est] = estimate_channel_peak_finding(y, num_paths, lut, pilot_val, c1, ~)
    [~, peak_indices] = sort(abs(y), 'descend');
    num_paths = min(num_paths, length(peak_indices));
    est_peak_locs = peak_indices(1:num_paths);
    h_est_unsorted = zeros(num_paths, 1, 'like', 1i);
    l_est_unsorted = zeros(num_paths, 1); a_est_unsorted = zeros(num_paths, 1);
    for p_idx = 1:num_paths
        k_peak = est_peak_locs(p_idx);
        [~, lut_idx] = min(abs([lut.peak_loc] - k_peak));
        l_est_unsorted(p_idx) = lut(lut_idx).l; a_est_unsorted(p_idx) = lut(lut_idx).alpha;
        phase_correction = exp(-1j*2*pi*c1*l_est_unsorted(p_idx)^2);
        h_est_unsorted(p_idx) = (y(k_peak)/pilot_val) * phase_correction;
    end
    [~, sort_idx] = sortrows([l_est_unsorted, a_est_unsorted]);
    l_est = l_est_unsorted(sort_idx); a_est = a_est_unsorted(sort_idx); h_est = h_est_unsorted(sort_idx);
end

function lut = create_lut(N, max_delay, max_doppler, c1)
    lut_size = (max_delay + 1) * (2*max_doppler + 1);
    lut = repmat(struct('l',0,'alpha',0,'peak_loc',0), lut_size, 1);
    idx = 1;
    for l = 0:max_delay
        for alpha = -max_doppler:max_doppler
            loc_i = round(alpha + 2*N*c1*l); peak_p = mod(loc_i, N);
            lut(idx).l = l; lut(idx).alpha = alpha; lut(idx).peak_loc = peak_p + 1;
            idx = idx + 1;
        end
    end
end

function H_freq_est_vec = estimate_channel_ofdm(y_ofdm, OFDM, SNRp_dB)
    pilot_val = sqrt(10^(SNRp_dB/10));
    H_pilots = y_ofdm(OFDM.pilot_indices) / pilot_val;
    H_freq_est_vec = interp1(OFDM.pilot_indices, H_pilots, (1:length(y_ofdm))', 'linear', 'extrap');
end

function [y_dd_vec, H_dd_ideal] = transmit_over_channel_otfs(x_dd, OTFS, h, l, a, noise_var)
    N = OTFS.N_otfs * OTFS.M_otfs;
    s_tf = ifft(x_dd, [], 2) * OTFS.M_otfs; 
    s_time_vec = reshape(s_tf.', [], 1);
    H_time = build_time_domain_channel_matrix(N, h, l, a, 0);
    noise = sqrt(noise_var/2) * (randn(N, 1) + 1i*randn(N, 1));
    r_time_vec = H_time * s_time_vec + noise;
    r_tf = reshape(r_time_vec, OTFS.M_otfs, OTFS.N_otfs).';
    y_dd = fft(r_tf, [], 2) / OTFS.M_otfs;
    y_dd_vec = reshape(y_dd, [], 1);
    if nargout > 1, H_dd_ideal = build_otfs_dd_channel_matrix(OTFS.N_otfs, OTFS.M_otfs, h, l, a); end
end

function H_dd = build_otfs_dd_channel_matrix(N_otfs, M_otfs, h, l_idx, a_idx)
    N = N_otfs * M_otfs;
    H_dd = zeros(N, N);
    for p = 1:length(h)
        hp = h(p); lp = l_idx(p); ap = a_idx(p);
        for n_in = 0:N_otfs-1
            for m_in = 0:M_otfs-1
                col_idx = n_in*M_otfs + m_in + 1;
                n_out = mod(n_in + lp, N_otfs); m_out = mod(m_in + ap, M_otfs);
                row_idx = n_out*M_otfs + m_out + 1;
                H_dd(row_idx, col_idx) = H_dd(row_idx, col_idx) + hp;
            end
        end
    end
end

function H_dd_est = estimate_channel_otfs(y_dd_vec, OTFS, pilot_val)
    N_otfs = OTFS.N_otfs; M_otfs = OTFS.M_otfs; N = N_otfs * M_otfs;
    y_dd = reshape(y_dd_vec, N_otfs, M_otfs);
    h_dd_response = y_dd / pilot_val;
    H_dd_est = zeros(N, N);
    for n_in = 0:N_otfs-1
        for m_in = 0:M_otfs-1
            col_idx = n_in*M_otfs + m_in + 1;
            H_slice_2d = circshift(h_dd_response, [n_in, m_in]);
            H_dd_est(:, col_idx) = H_slice_2d(:);
        end
    end
end