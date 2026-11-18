% === USRP 收发双端数据绝对物理比对验证脚本 (读取 TX Dat vs RX Dict) ===
clear; clc;

%% 1. 配置你想验证的目标与路径
rx_dict_filename = 'rx_mySig57_dict.mat';          % 接收端生成的字典文件
tx_dat_filename  = 'all_signals_tx_continuous.dat'; % 【发送端】送入 USRP 的真实物理数据
tx_meta_filename = 'tx_metadata.mat';               % 发送端生成的标签地图

fprintf('正在加载接收端字典 %s ...\n', rx_dict_filename);
load(rx_dict_filename, 'mod_snr_data');

fprintf('正在检索发送端物理数据 %s ...\n', tx_dat_filename);
load(tx_meta_filename, 'tx_metadata');

%% 1:OOK, 24:QPSK, 34:256QAM, 57:tone (对照发送端的 mods 列表)
target_mod_idx = 24;    % 建议先用 24(QPSK) 或 57(Tone) 测试
target_snr = 46;       % 想查看的 SNR
sample_to_plot = 5;    % 提取该类别的第几个样本来画图 (比如第 1 个)

% 2. 提取 RX (接收端) 数据
num_blocks = size(mod_snr_data, 1);
found_rx_idx = -1;
for k = 1:num_blocks
    meta = mod_snr_data{k, 1}; 
    if meta{1} == target_mod_idx && abs(meta{2} - target_snr) < 1e-6
        found_rx_idx = k;
        break;
    end
end

if found_rx_idx == -1
    error('接收字典中未找到 Mod_Idx=%d, SNR=%d 的数据块！', target_mod_idx, target_snr);
end

rx_iq_block = mod_snr_data{found_rx_idx, 2}; 
rx_sample = squeeze(rx_iq_block(sample_to_plot, :, :));
rx_I = rx_sample(1, :);
rx_Q = rx_sample(2, :);

fprintf('成功提取 RX 接收端数据。\n');

% 3. 提取 TX (发送端) 真实发射数据
% 【关键修复】：利用 metadata 中的上帝视角绝对索引，直接跳过所有的 0 和 帧头！
found_tx_idx = -1;

for k = 1:length(tx_metadata)
    if tx_metadata(k).mod_idx == target_mod_idx && abs(tx_metadata(k).snr - target_snr) < 1e-6
        found_tx_idx = k;
        break;
    end
end

if found_tx_idx == -1
    error('发送端元数据中未找到 Mod_Idx=%d, SNR=%d 的数据！', target_mod_idx, target_snr);
end

iq_len = tx_metadata(found_tx_idx).iq_length;

% 直接获取当前块有效数据的物理起点 (绝对复数样本索引)
base_tx_sample_idx = tx_metadata(found_tx_idx).data_start_idx;

% 加上你需要看的是该块内的第几个样本
% (-1 是因为 MATLAB 索引从 1 开始，但物理偏移计算从 0 起步)
target_tx_sample_idx = (base_tx_sample_idx - 1) + (sample_to_plot - 1) * iq_len;

% 计算字节偏移量: 1 个复数 = 2 个 float32 = 8 个字节
byte_offset = target_tx_sample_idx * 8; 
floats_to_read = 2 * iq_len;

% 精准读取 TX 连续文件
fileID_tx = fopen(tx_dat_filename, 'rb');
if fileID_tx == -1
    error('无法打开发送端数据文件 %s', tx_dat_filename);
end
fseek(fileID_tx, byte_offset, 'bof');
tx_raw = fread(fileID_tx, floats_to_read, 'float32', 0, 'ieee-le');
fclose(fileID_tx);

if length(tx_raw) < floats_to_read
    error('读取发送端数据越界！文件可能不完整。');
end

% 恢复为复数
tx_complex = tx_raw(1:2:end) + 1i * tx_raw(2:2:end);
tx_I = real(tx_complex).';
tx_Q = imag(tx_complex).';

% ！！！关键一步：为了视觉上好对比，我们将取出的 TX 数据也进行一次 Z-score 归一化
tx_I = (tx_I - mean(tx_I)) / std(tx_I);
tx_Q = (tx_Q - mean(tx_Q)) / std(tx_Q);

fprintf('成功提取 TX 发送端基带物理数据。\n');

% 4. 互相关精确对齐检测 
% 使用复数包络计算互相关，找出物理链路的样本延迟
[corr_val, lags] = xcorr(abs(rx_I + 1i*rx_Q), abs(tx_I + 1i*tx_Q));
[~, max_idx] = max(corr_val);
sample_offset = lags(max_idx);

fprintf('\n==================================================\n');
if sample_offset == 0
    fprintf('🎯 完美对齐！误差为 0 个样本点 (证明动态双同步生效！)。\n');
elseif sample_offset > 0
    fprintf('⚠️ 警告：接收数据【向左偏移】 %d 个样本点。\n', abs(sample_offset));
    fprintf('   -> 接收端切片程序的帧头动态锁定存在偏差，或频偏严重破坏了波形。\n');
else
    fprintf('⚠️ 警告：接收数据【向右偏移】 %d 个样本点。\n', abs(sample_offset));
    fprintf('   -> 接收端切片程序的帧头动态锁定存在偏差，或频偏严重破坏了波形。\n');
end
fprintf('==================================================\n\n');

% 5. 频谱计算辅助函数
L = length(rx_I); 
f_axis = linspace(-0.5, 0.5, L);

% 匿名函数：计算归一化 PSD
calc_psd = @(I, Q) 10 * log10((abs(fftshift(fft(I + 1i * Q))).^2) / L + 1e-12);

tx_psd_dB = calc_psd(tx_I, tx_Q);
rx_psd_dB = calc_psd(rx_I, rx_Q);

% 6. 绘图：TX vs RX 上下硬核比对
target_mod_name = tx_metadata(found_tx_idx).mod_name; % 提取名称用于画图标题

figure('Name', sprintf('物理层收发比对验证: Mod=%s, SNR=%d', target_mod_name, target_snr), ...
       'Color', 'w', 'Position', [50, 100, 1600, 800]);

% ------------------- 第一排：TX 发送端物理数据 -------------------
subplot(2, 3, 1);
plot(tx_I, 'b', 'LineWidth', 1.2); hold on; plot(tx_Q, 'r', 'LineWidth', 1.2);
title('【TX 发送端】原始理想波形');
ylabel('归一化幅度'); grid on; xlim([1, L]);

subplot(2, 3, 2);
scatter(tx_I, tx_Q, 10, 'filled', 'MarkerFaceColor', 'b', 'MarkerFaceAlpha', 0.5);
title('【TX 发送端】发送端星座图');
axis square; grid on; 
max_lim = max([max(abs(tx_I)), max(abs(tx_Q)), max(abs(rx_I)), max(abs(rx_Q))]) * 1.2; 
if max_lim < 1e-6; max_lim = 1; end
xlim([-max_lim, max_lim]); ylim([-max_lim, max_lim]);

subplot(2, 3, 3);
plot(f_axis, tx_psd_dB, 'k', 'LineWidth', 1.2);
title('【TX 发送端】发送端基带频谱');
grid on; xlim([-0.5, 0.5]); ylim([max(tx_psd_dB)-80, max(tx_psd_dB)+5]);

% ------------------- 第二排：RX 接收端硬件数据 -------------------
subplot(2, 3, 4);
plot(rx_I, 'b', 'LineWidth', 1.2); hold on; plot(rx_Q, 'r', 'LineWidth', 1.2);
title('【RX 接收端】USRP 实际接收波形');
xlabel('样本点'); ylabel('归一化幅度'); grid on; xlim([1, L]);

subplot(2, 3, 5);
scatter(rx_I, rx_Q, 10, 'filled', 'MarkerFaceColor', [0 0.5 0.5], 'MarkerFaceAlpha', 0.5);
title('【RX 接收端】接收端星座图');
xlabel('I (同相)'); ylabel('Q (正交)');
axis square; grid on; xlim([-max_lim, max_lim]); ylim([-max_lim, max_lim]);

subplot(2, 3, 6);
plot(f_axis, rx_psd_dB, 'k', 'LineWidth', 1.2);
title('【RX 接收端】接收端基带频谱');
xlabel('归一化频率 (f / Fs)'); grid on; xlim([-0.5, 0.5]); ylim([max(rx_psd_dB)-80, max(rx_psd_dB)+5]);

sgtitle(sprintf('USRP 物理层对齐查验 (调制类别: %s, SNR: %d dB, 抽样: %d)', ...
        target_mod_name, target_snr, sample_to_plot), 'FontSize', 16, 'FontWeight', 'bold');