"""
Utility functions for dealing with the Signal type
"""

from typing import List
import numpy as np
from torchsig.utils.dsp import low_pass_iterative_design


def check_signal_class(name: str, possible_names: List[str]) -> bool:
    is_type_signal = [n in name for n in possible_names]
    return any(is_type_signal)


def random_limiting_filter_design(
    bandwidth: float,
    sample_rate: float,
    rng: np.random.Generator = np.random.default_rng(seed=None),
    transition_bandwidth: float | None = None,
):
    """
    NOTE:
    TorchSig builders 依赖函数名 random_limiting_filter_design。
    这里保留签名，但去掉随机化：cutoff 固定为 bandwidth/2（bandwidth 为双边占用带宽）。
    rng 参数保留但不使用，仅用于兼容上层调用。
    """

    bandwidth = float(bandwidth)
    sample_rate = float(sample_rate)

    nyq = sample_rate / 2.0
    cutoff = bandwidth / 2.0  # 双边带宽 -> 半带宽

    if cutoff <= 0:
        raise ValueError(f"bandwidth must be >0, got {bandwidth}")
    if cutoff >= nyq:
        raise ValueError(f"bandwidth/2 must be < Nyquist ({nyq}), got cutoff={cutoff}")

    # 固定过渡带宽（确定性）
    if transition_bandwidth is None:
        transition_bandwidth = 0.10 * cutoff  # 10% half-bandwidth
        # 防止过渡带太窄导致滤波器超长/设计失败（按需调）
        transition_bandwidth = max(transition_bandwidth, 10e3)  # >=10 kHz

    transition_bandwidth = float(transition_bandwidth)
    if transition_bandwidth <= 0:
        raise ValueError("transition_bandwidth must be > 0")

    # 必须留出 stopband
    max_tb = nyq - cutoff
    if transition_bandwidth >= max_tb:
        transition_bandwidth = 0.9 * max_tb
        if transition_bandwidth <= 0:
            raise ValueError("No room for transition band: cutoff too close to Nyquist.")

    lpf = low_pass_iterative_design(
        cutoff=cutoff,
        transition_bandwidth=transition_bandwidth,
        sample_rate=sample_rate,
    )
    return lpf


# def random_limiting_filter_design(
#         bandwidth: float,
#         sample_rate: float,
#         rng: np.random.Generator = np.random.default_rng(seed=None)
# ):
#     """Design a coarse bandwidth limiting filter with randomized parameters
#     using TorchSig's iterative designer function.
#
#     Args:
#         bandwidth (float): Occupied bandwidth for signal (Hz).
#         sample_rate (float): Signal sampling rate (Hz).
#         rng (np.random.Generator, optional): Random number generator. Defaults to np.random.default_rng(seed=None).
#
#     Returns:
#         np.array: filter taps of designed low pass filter.
#     """
#
#     # randomize the cutoff
#     cutoff = rng.uniform(0.8*bandwidth/2,0.95*sample_rate/2)
#
#     # calculate maximum transition bandwidth
#     max_transition_bandwidth = sample_rate/2 - cutoff
#
#     # transition bandwidth is randomized value less than max transition bandwidth
#     transition_bandwidth = rng.uniform(0.5,1.5)*max_transition_bandwidth
#
#     # design bandwidth-limiting filter
#     lpf = low_pass_iterative_design(cutoff=cutoff,transition_bandwidth=transition_bandwidth,sample_rate=sample_rate)
#
#     return lpf