% === USRP 接收端：全自动动态双同步 + 标签地图引导切割 (能量嗅探+帧头锁定) ===
clear; clc; format long;

%% 1. 配置文件路径
rx_filename = 'rx_all_signals_ZC.dat';    % USRP 接收回来的二进制文件
metadata_filename = 'tx_metadata.mat';    % 发送端生成的标签地图 (含 preamble_complex)
save_filename = 'rx_mySig57_dict.mat';    % 最终输出的 Cell 文件

fprintf('1. 加载发送端标签地图与超级帧头...\n');
if ~exist(metadata_filename, 'file')
    error('找不到 %s，请确保发送端生成了该索引文件。', metadata_filename);
end
load(metadata_filename, 'tx_metadata', 'preamble_complex');

total_blocks = length(tx_metadata);
preamble_len = length(preamble_complex);
h_match = conj(fliplr(preamble_complex(:).')); 

fileID = fopen(rx_filename, 'rb');
if fileID == -1; error('未找到接收文件 %s', rx_filename); end

%% 2. 阶段一：能量嗅探 (跳过 USRP 开机盲区，寻找真实发射起点)
fprintf('2. 正在启动能量雷达，扫描物理信号爆发点...\n');

sniff_floats = 20000000; % 扫描前1000万个复数，寻找物理跳变
test_data = fread(fileID, sniff_floats, 'float32', 0, 'ieee-le');
if isempty(test_data); error('接收文件为空！'); end

rx_env = abs(test_data(1:2:end) + 1i * test_data(2:2:end));
smooth_env = movmean(rx_env, 500);
signal_peak_est = prctile(smooth_env, 99); 
threshold = 0.2 * signal_peak_est; 

start_idx = find(smooth_env > threshold, 1);
if isempty(start_idx)
    error('❌ 在扫描范围内未检测到信号爆发，请确认 USRP 是否成功发射！');
end

% 找到功率爆发点后，向前回退 50000 个点作为安全基准线
global_safe_start = max(0, start_idx - 50000);
fprintf('   🎯 成功避开开机底噪！物理跳变点在第 %d 个样本。\n', start_idx);
fprintf('   -> 锚定安全搜索起点: %d\n\n', global_safe_start);

%% 3. 阶段二：逐块精准切割 (匹配滤波 + 惯性导航)
fprintf('3. 开始流水线作业：嗅探帧头 -> 动态锁定 -> Z-Score归一化 -> 封包...\n');
mod_snr_data = cell(total_blocks, 2);

local_search_margin = 30000;  % 局部搜索容差
current_rx_sample_cursor = 0; % 物理游标

for k = 1:total_blocks
    
    current_mod_idx = tx_metadata(k).mod_idx;     
    current_mod_name = tx_metadata(k).mod_name;   
    current_snr = tx_metadata(k).snr;             
    samples_per_mod_snr = tx_metadata(k).sample_count; 
    num_length = tx_metadata(k).iq_length; 
    current_bw_array = tx_metadata(k).bw_array; 
    current_fc_array = tx_metadata(k).fc_array; 
    
    data_len = samples_per_mod_snr * num_length; 
    
    % --- A. 确定物理读取窗口 ---
    if k == 1
        % 第一块：从阶段一测出的安全起点开始找
        seek_start_idx = global_safe_start;
        read_len = 100000; % 只需要 10万 的窗口，彻底杜绝千万样本导致的伪峰值
    else
        % 后续块：理论跳跃距离 = 100000零 + 1000前缀 + 帧头长
        expected_jump = 100000 + 1000 + preamble_len;
        seek_start_idx = current_rx_sample_cursor + expected_jump - local_search_margin;
        if seek_start_idx < 0; seek_start_idx = 0; end
        read_len = local_search_margin * 2 + preamble_len; 
    end
    
    fseek(fileID, seek_start_idx * 8, 'bof'); 
    raw_window = fread(fileID, read_len * 2, 'float32', 0, 'ieee-le');
    if length(raw_window) < preamble_len * 2; break; end
    
    rx_window_complex = raw_window(1:2:end) + 1i * raw_window(2:2:end);
    rx_window_no_dc = rx_window_complex - mean(rx_window_complex);
    
    % --- B. 匹配滤波找峰值 ---
    mf_out = abs(filter(h_match, 1, rx_window_no_dc));
    [peak_val, rough_peak_idx] = max(mf_out);
    mean_noise = mean(mf_out);
    
    % --- C. 细同步：动态帧对齐计算 ---
    if peak_val < 6 * mean_noise
        fprintf('  ⚠️ 第 %d 块 ZC 峰值弱。启用【惯性导航】推算...\n', k);
        if k == 1
            error('第 1 块在小窗口内仍找不到 ZC 峰值，可能是发射端严重畸变！');
        end
        samples_to_skip_in_window = local_search_margin + preamble_len - 1;
    else
        fine_start = max(1, rough_peak_idx - preamble_len - 150);
        fine_end = min(length(rx_window_no_dc), rough_peak_idx + 150);
        rx_fine = rx_window_no_dc(fine_start:fine_end);
        
        [c, lags] = xcorr(rx_fine(:), preamble_complex(:));
        [~, max_c_idx] = max(abs(c));
        best_lag = lags(max_c_idx);
        
        preamble_start_in_fine = best_lag + 1;
        preamble_start_in_window = (fine_start - 1) + preamble_start_in_fine;
        samples_to_skip_in_window = preamble_start_in_window + preamble_len - 1;
    end
    
    % --- D. 定位真实数据并提取 ---
    absolute_data_start = seek_start_idx + samples_to_skip_in_window;
    
    fseek(fileID, absolute_data_start * 8, 'bof');
    floats_to_read = 2 * data_len;
    raw_data = fread(fileID, floats_to_read, 'float32', 0, 'ieee-le');
    if length(raw_data) < floats_to_read; break; end
    
    % --- E. 数据整形与 Z-Score 归一化 ---
    rx_complex = raw_data(1:2:end) + 1i * raw_data(2:2:end);
    temp_data = reshape(rx_complex, num_length, samples_per_mod_snr);
    
    iq_data = zeros(samples_per_mod_snr, 2, num_length, 'single');
    iq_data(:, 1, :) = permute(real(temp_data), [2, 1]); 
    iq_data(:, 2, :) = permute(imag(temp_data), [2, 1]); 
    
    for s = 1:samples_per_mod_snr
        sample_slice = squeeze(iq_data(s, :, :));
        mean_val = mean(sample_slice(:));
        std_val = std(sample_slice(:));
        if std_val < 1e-6
            iq_data(s, :, :) = zeros(size(sample_slice), 'single');
        else
            iq_data(s, :, :) = single((sample_slice - mean_val) / std_val);
        end
    end
    
    % --- F. 打包入 Cell ---
    mod_snr_data{k, 1} = {current_mod_idx, current_snr, current_bw_array, current_fc_array};
    mod_snr_data{k, 2} = iq_data;
    
    current_rx_sample_cursor = absolute_data_start + data_len;
    
    if mod(k, 50) == 0 || k == total_blocks
        fprintf('   ✅ 进度: %d / %d 数据块已封装 (当前: %-8s, SNR: %d dB)\n', k, total_blocks, current_mod_name, current_snr);
    end
end

fclose(fileID);

valid_rows = ~cellfun('isempty', mod_snr_data(:,1));
mod_snr_data = mod_snr_data(valid_rows, :);

fprintf('\n4. 开始保存至 %s ...\n', save_filename);
save(save_filename, 'mod_snr_data', '-v7.3', '-nocompression');
fprintf('   🎉 完美落盘！\n');