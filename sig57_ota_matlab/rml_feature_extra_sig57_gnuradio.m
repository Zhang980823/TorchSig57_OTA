% === HDF5 全数据集聚合为单一 GNU Radio .dat 文件 (含标签索引与帧头) ===
% 【OTA实测版：50000补零 + Zadoff-Chu同步帧头 + 绝对索引追踪】
clear; clc;

%% 1. 参数与路径配置
h5_filename = 'sig53_fast_4090.h5';
output_filename = 'all_signals_tx_continuous.dat'; % 最终输出的单一连续文件
metadata_filename = 'tx_metadata.mat';             % 保存标签与切分信息的索引文件

% 调制类型列表 (57种)
mods = ["ook", "4ask", "8ask", "16ask", "32ask", "64ask", ...
       "2fsk", "2gfsk", "2msk", "2gmsk", "4fsk", "4gfsk", ...
       "4msk", "4gmsk", "8fsk", "8gfsk", "8msk", "8gmsk", ...
       "16fsk", "16gfsk", "16msk", "16gmsk", "bpsk", "qpsk", ...
       "8psk", "16psk", "32psk", "64psk", "16qam", "32qam", ...
       "32qam_cross", "64qam", "128qam_cross", "256qam", ...
       "512qam_cross", "1024qam", "ofdm-64", "ofdm-72", ...
       "ofdm-128", "ofdm-180", "ofdm-256", "ofdm-300", ...
       "ofdm-512", "ofdm-600", "ofdm-900", "ofdm-1024", ...
       "ofdm-1200", "ofdm-2048", "fm", "am-dsb-sc", ... 
       "am-dsb", "am-lsb", "am-usb", "lfm_data", "lfm_radar", ...
       "chirpss", "tone"];

num_length = 1024; % 设定的每个IQ样本长度
target_rms = 0.3;  % 目标 RMS (-10 dBFS)，保证 USRP 发射时处于最佳线性区

%% ================= 新增：装甲级同步结构与帧头配置 =================
num_zeros = 100000; % 延长静默期：10万个点 (10MSps下为10ms)，给硬件充分的喘息时间
zeros_iq = zeros(2, num_zeros, 'single'); 

% 1. 生成“牺牲前缀” (Sacrificial Prefix)
% 作用：专扛放大器瞬态失真和 AGC 削顶。用 1000 个点的单频波作为缓冲垫
sacrificial_len = 1000;
t_sac = (0:sacrificial_len-1);
sacrificial_complex = exp(1i * 2 * pi * 0.05 * t_sac); % 低频单频波
sacrificial_rms = sqrt(mean(abs(sacrificial_complex).^2));
sacrificial_iq_norm = single(([real(sacrificial_complex); imag(sacrificial_complex)] ./ sacrificial_rms) * target_rms);

% 2. 生成增强型 Zadoff-Chu 序列作为核心帧头
% 长度提升 4 倍：4093 (更大的质数)，匹配滤波增益增加 6dB，抗噪抗频偏能力翻倍
N_zc = 4093; 
u = 31;      % 更换根索引
n_zc = 0:(N_zc-1);
preamble_complex = exp(-1i * pi * u * n_zc .* (n_zc + 1) / N_zc);

% 帧头 RMS 归一化并转为 2xN 格式
preamble_rms = sqrt(mean(abs(preamble_complex).^2)); 
preamble_iq = [real(preamble_complex); imag(preamble_complex)];
preamble_iq_norm = single((preamble_iq ./ preamble_rms) * target_rms);
preamble_len = N_zc;
%% ============================================================

%% 2. 加载标签与数据
fprintf('正在读取 H5 原始数据...\n');
data_X = h5read(h5_filename,'/X');
data_Y = h5read(h5_filename,'/Y');
data_Z = h5read(h5_filename,'/Z');
data_BW = h5read(h5_filename,'/BW');
data_FC = h5read(h5_filename,'/FC');

num_mods = numel(mods);
snrs = sort(unique(data_Z),'descend');
num_snrs = numel(snrs);
total_blocks = num_mods * num_snrs;

%% 3. 初始化索引结构体
% 新增：记录绝对样本位置，方便接收端切割
meta_info = struct('mod_idx', [], 'mod_name', "", 'snr', [], ...
                   'sample_count', [], 'iq_length', [], ...
                   'zeros_start_idx', [], 'preamble_start_idx', [], ...
                   'data_start_idx', [], 'data_end_idx', []);
tx_metadata = repmat(meta_info, total_blocks, 1);

%% 4. 打开文件准备持续写入
fileID = fopen(output_filename, 'wb');
if fileID == -1
    error('无法创建输出文件，请检查路径权限或磁盘空间！');
end

fprintf('开始整合并记录标签，预计产生 %d 个数据块...\n', total_blocks);

%% 5. 双层循环，按顺序拼接处理并记录
block_count = 0;
total_samples_written = 0;
valid_block_idx = 1; % 记录有效块的实际索引
current_tx_sample_idx = 0; % 追踪累计发送的复数样本总数（用于生成切分切片字典）

for i = 1:num_mods
    for j = 1:num_snrs        
        current_snr = snrs(j);
        
        mask = (data_Y == (i-1)) & (abs(data_Z - current_snr) < 1e-6);
        sample_idx = find(mask);
        
        if isempty(sample_idx)
            continue; 
        end
        
        % 提取当前组合的 IQ 数据并交织
        raw_iq = data_X(:, :, sample_idx); 
        current_bw_array = data_BW(sample_idx);
        current_fc_array = data_FC(sample_idx);

        iq_stream = permute(raw_iq, [2, 1, 3]);
        iq_matrix = reshape(iq_stream, 2, []); 
        
        % 计算复数 RMS 并拦截静默“黑洞”
        complex_sig = complex(iq_matrix(1,:), iq_matrix(2,:));
        current_rms = sqrt(mean(abs(complex_sig).^2)); 
        
        if isnan(current_rms) || current_rms < 1e-6
            continue; 
        end
        
        % 恒定 RMS 功率归一化 & 数据清洗
        iq_matrix_norm = (single(iq_matrix) ./ current_rms) * target_rms; 
        iq_matrix_norm(isnan(iq_matrix_norm) | isinf(iq_matrix_norm)) = 0;
        iq_matrix_norm(iq_matrix_norm > 0.99) = 0.99;  
        iq_matrix_norm(iq_matrix_norm < -0.99) = -0.99; 

        current_sample_count = numel(sample_idx);
        data_samples_len = current_sample_count * num_length;

        % 写入 100,000 个零
        tx_metadata(valid_block_idx).zeros_start_idx = current_tx_sample_idx + 1;
        fwrite(fileID, zeros_iq, 'float32', 0, 'ieee-le');
        current_tx_sample_idx = current_tx_sample_idx + num_zeros;
        

        % 写入缓冲前缀
        fwrite(fileID, sacrificial_iq_norm, 'float32', 0, 'ieee-le');
        current_tx_sample_idx = current_tx_sample_idx + sacrificial_len;
        
        % 写入 Zadoff-Chu 帧头 (纯净区)
        tx_metadata(valid_block_idx).preamble_start_idx = current_tx_sample_idx + 1;
        fwrite(fileID, preamble_iq_norm, 'float32', 0, 'ieee-le');
        current_tx_sample_idx = current_tx_sample_idx + preamble_len;
        
        % 写入有效数据
        tx_metadata(valid_block_idx).data_start_idx = current_tx_sample_idx + 1;
        fwrite(fileID, iq_matrix_norm, 'float32', 0, 'ieee-le');
        current_tx_sample_idx = current_tx_sample_idx + data_samples_len;
        tx_metadata(valid_block_idx).data_end_idx = current_tx_sample_idx;

        % 记录元数据其余信息
        tx_metadata(valid_block_idx).mod_idx = i; 
        tx_metadata(valid_block_idx).mod_name = mods(i);
        tx_metadata(valid_block_idx).snr = current_snr;
        tx_metadata(valid_block_idx).sample_count = current_sample_count;
        tx_metadata(valid_block_idx).iq_length = num_length;
        tx_metadata(valid_block_idx).bw_array = current_bw_array;
        tx_metadata(valid_block_idx).fc_array = current_fc_array;
        
        total_samples_written = current_tx_sample_idx; 
        
        valid_block_idx = valid_block_idx + 1;
        block_count = block_count + 1;
        
        if mod(block_count, 50) == 0 || block_count == total_blocks
            fprintf('进度: %d / %d 数据块已写入...\n', block_count, total_blocks);
        end
    end
end

fclose(fileID);

% 裁剪掉未使用的结构体空间
tx_metadata = tx_metadata(1:valid_block_idx-1);

% ================= 新增：保存帧头供接收端匹配滤波使用 =================
% 将 preamble_complex 一并打包存入 mat 文件中，接收端代码直接读取它做互相关（xcorr）
save(metadata_filename, 'tx_metadata', 'preamble_complex', 'target_rms');

fprintf('\n发送端处理完毕！\n');
fprintf('生成信号文件: %s\n', output_filename);
fprintf('生成标签地图: %s\n', metadata_filename);
fprintf('包含同步帧头，ZC序列长度: %d\n', preamble_len);
fprintf('总有效发射复数样本数 (含补零与帧头): %d\n', total_samples_written);