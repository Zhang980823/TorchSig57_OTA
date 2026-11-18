import torch
import numpy as np
import os
import h5py
from collections import defaultdict
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset
import torchsig.signals as tssignals
from tqdm import tqdm


class RML2016GeneratorWithFreqBW:
    """生成带信号+纯噪声的RML2016数据集，并打上SNR、载频、带宽和信号存在性标签"""

    def __init__(self, output_dir="./rml2016_freq_bw_dataset", batch_size=512, noise_ratio=0.2):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.batch_size = batch_size
        self.noise_ratio = noise_ratio

        # 参数
        self.num_iq_samples = 4096
        self.samples_per_combo = 10
        self.snr_range = list(range(-30, 31, 2))
        # 定义你需要的信号类型
        self.selected_modulations = [
            # 'am-dsb',  # AM
            # 'am-usb',  # USB
            # 'lfm_data',  # LFM
            # # 'lfm_radar',  # NLFM
            'tone',  # 单音脉冲
            # 'ofdm-64',  # OFDM
            # # '2ask',  # 2ASK
            # '4ask',  # 4ASK
            # '8ask',  # 8ASK
            # '2fsk',  # 2FSK
            # '4fsk',  # 4FSK
            # '8fsk',  # 8FSK
            # 'bpsk',  # BPSK
            # # 'qpsk',  # UQPSK
            # # '2gmsk',  # GMSK
            # 'qpsk',  # QPSK
            # '16qam',  # 16QAM
            # '32qam',  # 32QAM
            # '64qam',  # 64QAM
            # # '128qam_cross',  # 128QAM
            # '256qam',  # 256QAM
            # '8psk',  # 8PSK
            # # '16psk'  # 16APSK
        ]

        # 只使用所选的信号类型
        self.all_modulations = self.selected_modulations
        # self.all_modulations = tssignals.TorchSigSignalLists.all_signals

        # 计算样本数量
        self.total_signal_samples = len(self.all_modulations) * len(self.snr_range) * self.samples_per_combo
        self.total_noise_samples = int(self.total_signal_samples * self.noise_ratio)
        self.total_samples = self.total_signal_samples + self.total_noise_samples

    def generate_and_save(self):
        print("🚀 正在生成数据集...")
        h5_file = os.path.join(self.output_dir, "dec_freq_bw_Limit_with_noise_test.h5")

        with h5py.File(h5_file, 'w') as f:
            iq_data_dset = f.create_dataset('X', (self.total_samples, 2, self.num_iq_samples), dtype='float32')
            modulation_dset = f.create_dataset('Y', (self.total_samples,), dtype='int32')
            snr_dset = f.create_dataset('Z', (self.total_samples,), dtype='float32')
            freq_dset = f.create_dataset('F', (self.total_samples,), dtype='float32')
            bw_dset = f.create_dataset('B', (self.total_samples,), dtype='float32')
            signal_presence_dset = f.create_dataset('S', (self.total_samples,), dtype='int32')

            f.attrs['total_samples'] = self.total_samples
            f.attrs['num_iq_samples'] = self.num_iq_samples
            f.attrs['modulation_types'] = [mod for mod in self.all_modulations]
            f.attrs['snr_range'] = self.snr_range
            f.attrs['data_format'] = '(samples, 2, num_iq_samples)'

            batch_iq, batch_mod, batch_snr, batch_freq, batch_bw, batch_signal_presence = [], [], [], [], [], []
            idx = 0
            pbar = tqdm(total=self.total_samples, unit="sample")

            # -----------------生成带信号样本-----------------
            print("📡 生成带信号样本...")
            signal_metadata = DatasetMetadata(
                num_iq_samples_dataset=self.num_iq_samples,
                fft_size=64,
                num_signals_max=1,  # 有信号样本：最少1个信号，最多1个信号
                num_signals_min=1,
                sample_rate=100e6,
                snr_db_min=0,
                snr_db_max=30,
                signal_duration_min=1.00 * self.num_iq_samples / 100e6,
                signal_duration_max=1.00 * self.num_iq_samples / 100e6,
                signal_bandwidth_min=100e6 / 5,#20MHz
                signal_bandwidth_max=100e6 / 5,#20MHz
                signal_center_freq_min=0e6, #-40MHz
                signal_center_freq_max=0e6, #40MHz
                class_list=self.all_modulations,
            )

            signal_dataset = TorchSigIterableDataset(dataset_metadata=signal_metadata)

            for signal in signal_dataset:
                if idx >= self.total_signal_samples:
                    break

                # 由于设置了num_signals_min=1，这里应该总是有信号
                if len(signal.component_signals) == 0:
                    print("警告：发现空信号样本，跳过")
                    continue

                comp = signal.component_signals[0]
                modulation = comp.metadata.class_name
                snr = comp.metadata.snr_db
                snr_rounded = max(-30, min(30, 2 * round(snr / 2)))
                freq = getattr(comp.metadata, 'center_freq', 0.0)
                bandwidth = getattr(comp.metadata, 'bandwidth', 0.0)

                iq_data = signal.data
                if isinstance(iq_data, np.ndarray):
                    iq_data = torch.from_numpy(iq_data)

                if len(iq_data) < self.num_iq_samples:
                    iq_data = torch.cat([iq_data, torch.zeros(self.num_iq_samples - len(iq_data), dtype=iq_data.dtype)])
                else:
                    iq_data = iq_data[:self.num_iq_samples]

                batch_iq.append(torch.stack([iq_data.real, iq_data.imag]).numpy())
                batch_mod.append(self.all_modulations.index(modulation))
                batch_snr.append(snr_rounded)
                batch_freq.append(freq)
                batch_bw.append(bandwidth)
                batch_signal_presence.append(1)

                idx += 1
                pbar.update(1)

                if len(batch_iq) >= self.batch_size:
                    start_idx = idx - len(batch_iq)
                    end_idx = idx
                    iq_data_dset[start_idx:end_idx] = np.stack(batch_iq, axis=0)
                    modulation_dset[start_idx:end_idx] = np.array(batch_mod, dtype=np.int32)
                    snr_dset[start_idx:end_idx] = np.array(batch_snr, dtype=np.float32)
                    freq_dset[start_idx:end_idx] = np.array(batch_freq, dtype=np.float32)
                    bw_dset[start_idx:end_idx] = np.array(batch_bw, dtype=np.float32)
                    signal_presence_dset[start_idx:end_idx] = np.array(batch_signal_presence, dtype=np.int32)

                    batch_iq.clear();
                    batch_mod.clear();
                    batch_snr.clear();
                    batch_freq.clear();
                    batch_bw.clear();
                    batch_signal_presence.clear()

            # -----------------生成纯噪声样本-----------------
            print("🔇 生成纯噪声样本...")

            noise_metadata = DatasetMetadata(
                num_iq_samples_dataset=self.num_iq_samples,
                fft_size=64,
                num_signals_max=0,  # 纯噪声样本
                num_signals_min=0,
                sample_rate=100e6,
                snr_db_min=-30,
                snr_db_max=30,
                class_list=self.all_modulations,
            )

            noise_dataset = TorchSigIterableDataset(dataset_metadata=noise_metadata)


            pbar = tqdm(total=self.total_noise_samples, desc="Generating noise samples", unit="sample")

            for signal in noise_dataset:
                if idx >= self.total_samples:  # 总样本数
                    break

                iq_data = signal.data
                if isinstance(iq_data, np.ndarray):
                    iq_data = torch.from_numpy(iq_data)

                # 调整长度
                if len(iq_data) < self.num_iq_samples:
                    iq_data = torch.cat([iq_data, torch.zeros(self.num_iq_samples - len(iq_data), dtype=iq_data.dtype)])
                else:
                    iq_data = iq_data[:self.num_iq_samples]

                # 随机分配 SNR 和调制标签（保持与信号样本一致的结构）
                snr = np.random.choice(self.snr_range)
                modulation_idx = np.random.randint(0, len(self.all_modulations))

                batch_iq.append(torch.stack([iq_data.real, iq_data.imag]).numpy())
                batch_mod.append(modulation_idx)
                batch_snr.append(snr)
                batch_freq.append(0.0)
                batch_bw.append(0.0)
                batch_signal_presence.append(0)  # 无信号标志

                idx += 1
                pbar.update(1)

                # 批次写入 HDF5
                if len(batch_iq) >= self.batch_size:
                    start_idx = idx - len(batch_iq)
                    end_idx = idx
                    iq_data_dset[start_idx:end_idx] = np.stack(batch_iq, axis=0)
                    modulation_dset[start_idx:end_idx] = np.array(batch_mod, dtype=np.int32)
                    snr_dset[start_idx:end_idx] = np.array(batch_snr, dtype=np.float32)
                    freq_dset[start_idx:end_idx] = np.array(batch_freq, dtype=np.float32)
                    bw_dset[start_idx:end_idx] = np.array(batch_bw, dtype=np.float32)
                    signal_presence_dset[start_idx:end_idx] = np.array(batch_signal_presence, dtype=np.int32)

                    batch_iq.clear()
                    batch_mod.clear()
                    batch_snr.clear()
                    batch_freq.clear()
                    batch_bw.clear()
                    batch_signal_presence.clear()

            # 写入剩余样本
            if batch_iq:
                start_idx = idx - len(batch_iq)
                end_idx = idx
                iq_data_dset[start_idx:end_idx] = np.stack(batch_iq, axis=0)
                modulation_dset[start_idx:end_idx] = np.array(batch_mod, dtype=np.int32)
                snr_dset[start_idx:end_idx] = np.array(batch_snr, dtype=np.float32)
                freq_dset[start_idx:end_idx] = np.array(batch_freq, dtype=np.float32)
                bw_dset[start_idx:end_idx] = np.array(batch_bw, dtype=np.float32)
                signal_presence_dset[start_idx:end_idx] = np.array(batch_signal_presence, dtype=np.int32)

            pbar.close()
            print(f"✅ 纯噪声生成完成，共 {idx}/{self.total_noise_samples} 样本")

            return h5_file


if __name__ == "__main__":
    generator = RML2016GeneratorWithFreqBW(output_dir="./rml2016_freq_bw", batch_size=1024)
    h5_file = generator.generate_and_save()
    print(f"数据集已保存至 {h5_file}")