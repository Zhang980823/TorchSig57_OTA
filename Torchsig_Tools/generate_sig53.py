import os
import torch
import numpy as np
import h5py
from collections import defaultdict
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset
import torchsig.signals as tssignals
from tqdm import tqdm

visible_ids = []
for i in range(torch.cuda.device_count()):
    if "4090" in torch.cuda.get_device_name(i):
        visible_ids.append(str(i))
if visible_ids:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_ids)
    print(f" GPU supported: {', '.join(visible_ids)} ")
else:
    print("CPU supported only")


class RML2016GeneratorFast:

    def __init__(self, output_dir="./sig53_dataset", batch_size=512):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.batch_size = batch_size

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"Using GPU: {torch.cuda.get_device_name(self.device.index)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.num_iq_samples   = 1024
        self.samples_per_combo = 5000
        self.step             = 2
        self.snr_range        = list(range(0, 50, self.step))
        self.all_modulations  = tssignals.TorchSigSignalLists.all_signals
        self.sample_rate      = 10e6   # ← 统一管理

        print("\n=== 数据集配置信息 ===")
        print(f" 调制类型数量: {len(self.all_modulations)}")
        print(f" SNR范围: {self.snr_range} (共{len(self.snr_range)}个SNR值)")
        print(f" 每个组合样本数: {self.samples_per_combo:,}")
        print(f" 每个样本IQ长度: {self.num_iq_samples}")

        self.total_samples = (len(self.all_modulations)
                              * len(self.snr_range)
                              * self.samples_per_combo)
        print(f" 预期总样本数: {self.total_samples:,}")
        print(f" 预期数据维度: X({self.total_samples}, 2, {self.num_iq_samples})")
        print(f"               Y({self.total_samples},)  调制标签")
        print(f"               Z({self.total_samples},)  SNR标签")
        print(f"               BW({self.total_samples},) 带宽标签 ← 新增")
        print(f"               FC({self.total_samples},) 中心频率标签 ← 新增")

        print(f" 调制类型列表:")
        for i, mod in enumerate(self.all_modulations):
            print(f"    {i:2d}. {mod}")

        self.metadata = DatasetMetadata(
            num_iq_samples_dataset=self.num_iq_samples,
            fft_size=64,
            num_signals_max=1,
            num_signals_min=1,
            sample_rate=self.sample_rate,
            snr_db_min=0,
            snr_db_max=50,
            signal_duration_min=1.00 * self.num_iq_samples / self.sample_rate,
            signal_duration_max=1.00 * self.num_iq_samples / self.sample_rate,
            signal_bandwidth_min=0.3e6,
            signal_bandwidth_max=0.3e6,
            signal_center_freq_min=1e6,
            signal_center_freq_max=1e6,
            class_list=self.all_modulations,
        )

    def generate_and_save(self):
        print(f"\n 开始生成数据集 [{self.device}] ...")
        h5_file = os.path.join(self.output_dir, "sig53_fast_4090.h5")

        with h5py.File(h5_file, 'w') as f:

            # ===== 创建数据集 =====
            iq_dset  = f.create_dataset('X',  (self.total_samples, 2, self.num_iq_samples), dtype='float32')
            mod_dset = f.create_dataset('Y',  (self.total_samples,), dtype='int32')
            snr_dset = f.create_dataset('Z',  (self.total_samples,), dtype='float32')
            bw_dset  = f.create_dataset('BW', (self.total_samples,), dtype='float32')  # 带宽 (Hz)
            fc_dset  = f.create_dataset('FC', (self.total_samples,), dtype='float32')  # 中心频率 (Hz)

            # ===== 全局属性 =====
            f.attrs['total_samples']    = self.total_samples
            f.attrs['num_iq_samples']   = self.num_iq_samples
            f.attrs['samples_per_combo'] = self.samples_per_combo
            f.attrs['sample_rate_hz']   = self.sample_rate
            f.attrs['modulation_types'] = [str(m) for m in self.all_modulations]
            f.attrs['snr_range']        = self.snr_range
            f.attrs['bw_range_hz']      = [0.3e6, 0.3e6]
            f.attrs['fc_range_hz']      = [1e6,   1e6]
            f.attrs['data_format']      = '(samples, 2, num_iq_samples)'
            f.attrs['bw_unit']          = 'Hz'
            f.attrs['fc_unit']          = 'Hz'

            # ===== 生成循环 =====
            dataset  = TorchSigIterableDataset(dataset_metadata=self.metadata)
            counters = defaultdict(lambda: defaultdict(int))

            idx      = 0
            batch_iq  = []
            batch_mod = []
            batch_snr = []
            batch_bw  = []   # ← 新增
            batch_fc  = []   # ← 新增

            pbar = tqdm(total=self.total_samples, unit="sample", desc="生成进度")

            for signal in dataset:
                if idx >= self.total_samples:
                    break
                if len(signal.component_signals) == 0:
                    continue

                comp       = signal.component_signals[0]
                modulation = comp.metadata.class_name
                snr        = comp.metadata.snr_db
                snr_rounded = max(0, min(max(self.snr_range),
                                         self.step * round(snr / self.step)))

                if counters[modulation][snr_rounded] >= self.samples_per_combo:
                    continue

                # ===== 提取带宽和中心频率 =====
                # TorchSig 里存的是归一化值（相对于 sample_rate）
                # 需要 × sample_rate 还原为 Hz
                bw_norm = float(comp.metadata.bandwidth)      # 归一化带宽
                fc_norm = float(comp.metadata.center_freq)    # 归一化中心频率

                bw_hz = bw_norm * self.sample_rate            # 实际带宽 (Hz)
                fc_hz = fc_norm * self.sample_rate            # 实际中心频率 (Hz)

                # ===== IQ数据处理 =====
                iq_data = signal.data
                if isinstance(iq_data, np.ndarray):
                    iq_data = torch.from_numpy(iq_data)

                iq_data = iq_data.to(self.device)
                if iq_data.size(0) < self.num_iq_samples:
                    pad = torch.zeros(self.num_iq_samples - iq_data.size(0),
                                      dtype=iq_data.dtype, device=self.device)
                    iq_data = torch.cat([iq_data, pad])
                else:
                    iq_data = iq_data[:self.num_iq_samples]

                iq_standard = torch.stack([iq_data.real, iq_data.imag], dim=0)

                batch_iq.append(iq_standard.cpu().numpy())
                batch_mod.append(self.all_modulations.index(modulation))
                batch_snr.append(snr_rounded)
                batch_bw.append(bw_hz)    # ← 新增
                batch_fc.append(fc_hz)    # ← 新增

                counters[modulation][snr_rounded] += 1
                idx += 1
                pbar.update(1)

                # ===== 批量写入 =====
                if len(batch_iq) >= self.batch_size:
                    self._flush_batch(
                        idx, batch_iq, batch_mod, batch_snr, batch_bw, batch_fc,
                        iq_dset, mod_dset, snr_dset, bw_dset, fc_dset
                    )
                    batch_iq.clear(); batch_mod.clear(); batch_snr.clear()
                    batch_bw.clear(); batch_fc.clear()

            # ===== 写入剩余 =====
            if batch_iq:
                self._flush_batch(
                    idx, batch_iq, batch_mod, batch_snr, batch_bw, batch_fc,
                    iq_dset, mod_dset, snr_dset, bw_dset, fc_dset
                )

            pbar.close()

            # ===== 统计信息 =====
            self._print_stats(idx, counters, h5_file)

        return h5_file

    # ------------------------------------------------------------------ #
    def _flush_batch(self, idx, batch_iq, batch_mod, batch_snr, batch_bw, batch_fc,
                     iq_dset, mod_dset, snr_dset, bw_dset, fc_dset):
        n         = len(batch_iq)
        start_idx = idx - n
        end_idx   = idx
        iq_dset [start_idx:end_idx] = np.stack(batch_iq,  axis=0)
        mod_dset[start_idx:end_idx] = np.array(batch_mod, dtype=np.int32)
        snr_dset[start_idx:end_idx] = np.array(batch_snr, dtype=np.float32)
        bw_dset [start_idx:end_idx] = np.array(batch_bw,  dtype=np.float32)
        fc_dset [start_idx:end_idx] = np.array(batch_fc,  dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _print_stats(self, idx, counters, h5_file):
        print(f"\n=== 生成完成统计 ===")
        print(f" 实际生成样本数: {idx:,}/{self.total_samples:,}")

        print(f"\n 各调制类型样本分布:")
        mod_stats = [(mod, sum(counters[mod].values()))
                     for mod in self.all_modulations]
        mod_stats.sort(key=lambda x: x[1], reverse=True)
        expected_per_mod = len(self.snr_range) * self.samples_per_combo
        for mod, count in mod_stats:
            status = "✓" if count == expected_per_mod else "⚠"
            print(f"  {status} {mod:<20}: {count:>6,} (预期 {expected_per_mod:,})")

        print(f"\n SNR分布统计:")
        expected_per_snr = len(self.all_modulations) * self.samples_per_combo
        for snr_val in self.snr_range:
            count  = sum(counters[mod][snr_val] for mod in self.all_modulations)
            status = "✓" if count == expected_per_snr else "⚠"
            print(f"  {status} SNR {snr_val:>2}dB: {count:>8,} (预期 {expected_per_snr:,})")

        file_size_mb = os.path.getsize(h5_file) / (1024 * 1024)
        print(f"\n 文件大小: {file_size_mb:.2f} MB")
        print(f" 数据集保存至: {h5_file}")


if __name__ == "__main__":
    generator = RML2016GeneratorFast(
        output_dir="./sig57_pretrain_1024LEN_5000NUM_0to50SNRS_SP2_FBFIX_LBL",
        batch_size=1024
    )
    h5_file = generator.generate_and_save()
    print(f"\n 数据集已保存至: {h5_file}")
