import torch
from torch.utils.data import Dataset
import h5py


class RML2016TorchDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = h5py.File(h5_path, 'r')
        self.X = self.file['X']
        self.Y = self.file['Y']
        self.Z = self.file['Z']
        self.total_samples = self.file.attrs['total_samples']

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        iq = torch.tensor(self.X[idx], dtype=torch.float32)  # (2, num_iq_samples)
        label = torch.tensor(self.Y[idx], dtype=torch.long)
        snr = torch.tensor(self.Z[idx], dtype=torch.float32)
        return iq, label, snr


# 用法
from torch.utils.data import DataLoader

dataset = RML2016TorchDataset("./sig53_fast_4090/sig53_fast_4090.h5")

loader = DataLoader(dataset, batch_size=256, shuffle=True)

for iq, label, snr in loader:
    print(iq.shape, label.shape, snr.shape)
    break