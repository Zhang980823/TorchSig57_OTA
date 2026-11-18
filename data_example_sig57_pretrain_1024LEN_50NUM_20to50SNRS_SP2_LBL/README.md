#  仿真数据示例（每类、每档 SNR 各 50 个样本）

这是 Sig57-OTA 项目使用的 **TorchSig 仿真数据子集**，供数据读取、USRP 发射流程和特征提取流程演示使用。本目录中的 IQ 数据尚未经过 USRP 空口传输。

## 数据规模

- 57 种调制/信号类型，类别名和顺序见 HDF5 属性 `modulation_types`。
- SNR：20、22、24、26、28、30、32、34、36、38、40、42、44、46、48 dB。
- 每个“类别 × SNR”组合从原来的 5,000 个样本中抽取 50 个。
- 共 `57 × 15 × 50 = 42,750` 个样本；每个样本包含 1,024 个复数 IQ 点。
- 原始文件约 32.68 GiB；子集单文件约 275.70 MiB，使用 gzip 无损压缩。

目录名沿用原始数据的 `20to50` 命名；实际 SNR 不含 50 dB，以文件中的标签和属性为准。（信噪比范围为软件换算值，与真实值不对应，真实值以接收端信噪比估计值为准）

## 文件和发布方式

- `sig53_fast_4090.h5`：包含所有 SNR 的单文件版本，保留原文件名，方便现有代码读取。
- `github_shards/sig57_snr_*dB.h5`：按 SNR 分成 15 个文件，每个文件包含 57 类、每类 50 个样本，共 2,850 个样本，约 18.09–18.82 MiB。
- `manifest.json`：原始分组数量、抽样参数、运行库版本、输出文件大小和 SHA-256 校验值。

合并文件与全部分片包含**同一批样本**，两种存储形式不应合并当成两份训练数据。


本目录的 `.gitignore` 默认忽略本地合并文件，提交该目录时保留分片、说明和清单。若改用 Git LFS 发布合并文件，请相应调整忽略规则。

## HDF5 格式

原有数据集名称、数据类型、IQ 数值和类别顺序保持不变：

- `X`：`float32`，形状 `(N, 2, 1024)`，第 2 维依次为 I、Q。
- `Y`：`int32`，形状 `(N,)`，类别标签为 0–56。
- `Z`：`float32`，形状 `(N,)`，SNR 标签，单位 dB。
- `BW`：`float32`，形状 `(N,)`，带宽，单位 Hz。
- `FC`：`float32`，形状 `(N,)`，中心频率，单位 Hz。
- `source_indices`：新增的 `int64` 数据集，形状 `(N,)`，记录对应原始 HDF5 的 **0 起始行号**。

原始属性中记录的采样率为 20 MHz，带宽范围为 0.1–2 MHz，中心频率范围为 3–6 MHz。子集保留原始物理参数和其他元信息，更新 `total_samples`、`samples_per_combo` 和各文件的 `snr_range`，并添加来源和抽样属性。

每个文件内的样本按原文件行号升序排列；类别和 SNR 并不连续分块，应通过 `Y`、`Z` 筛选。

## 读取示例

只需 `numpy` 和 `h5py`，读取和裁剪均无需安装 TorchSig、PyTorch 或 CUDA。

```python
from pathlib import Path
import h5py

data_dir = Path("data_example_sig57_pretrain_1024LEN_50NUM_20to50SNRS_SP2_LBL")
with h5py.File(data_dir / "github_shards/sig57_snr_20dB.h5", "r") as f:
    iq = f["X"][0]                          # (2, 1024), float32
    label = int(f["Y"][0])
    modulation = f.attrs["modulation_types"][label]
    snr = float(f["Z"][0])
    source_row = int(f["source_indices"][0])
    print(iq.shape, modulation, snr, source_row)

# 按需遍历完整子集，避免依赖本地合并文件：
for path in sorted((data_dir / "github_shards").glob("*.h5")):
    with h5py.File(path, "r") as f:
        print(path.name, f["X"].shape, f.attrs["snr_range"])
```

Python 中保持上述数组顺序。MATLAB `h5read` 的维度顺序与 Python 不同，现有 MATLAB 发射脚本已按其读取顺序处理，裁剪时无需转置。

现有 `read_sig53.py` 和 MATLAB 发射脚本可指向本目录的合并文件，也可以指向单个 SNR 分片。重新进行 OTA 实验时，应从所选子集重新生成发射 `.dat` 文件和配套 `tx_metadata.mat`；后续接收和特征提取使用这套新索引。

## 复现裁剪

在仓库根目录运行；输出目录必须尚不存在，脚本拒绝覆盖已有结果：

```bash
python scripts/subset_sig57.py \
  data_example_sig57_pretrain_1024LEN_5000NUM_20to50SNRS_SP2_LBL/sig53_fast_4090.h5 \
  sig57_subset_reproduced \
  --samples-per-combo 50 --seed 42 --split-by-snr
```

抽样使用 NumPy `PCG64`，随机种子为 `42`，对每个类别与 SNR 组合独立、均匀、无放回抽样。脚本只读取原文件，分批复制 IQ 数据，无需将约 35 GB 的原始 IQ 全部载入内存。本次运行环境为 NumPy 1.24.3、h5py 3.9.0；源索引保存在每个文件中，便于精确追溯抽样结果。

脚本对每个输出文件执行以下核验，全部通过后才写出清单：

1. 每个声明的类别与 SNR 组合恰好 50 个样本，源索引无重复。
2. `X/Y/Z/BW/FC` 的所有输出样本逐字节等于原文件对应样本。
3. 数据类型、原始元信息以及更新后的数量和 SNR 属性正确。
4. 原文件大小和修改时间未发生变化。

此命令复现的是对已有仿真文件的裁剪。当前仓库 `Torchsig_Tools/generate_sig53.py` 中的生成参数与原始文件属性不同，不能直接将其默认参数视为本数据的生成配置。
