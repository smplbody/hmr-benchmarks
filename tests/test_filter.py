import numpy as np
import torch

from mmhuman3d.core.filter.builder import build_filter


#  test different data type
def test_data_type_torch():
    noisy_input = torch.randn((100, 17, 3))
    cfg = dict(type='OneEuroFilter', min_cutoff=0.004, beta=0.7)
    oneeuro = build_filter(cfg)
    out_g = oneeuro(noisy_input)
    cfg = dict(type='Gaus1dFilter', window_size=11, sigma=4)
    gaus1d = build_filter(cfg)
    out_s = gaus1d(noisy_input)
    cfg = dict(type='SGFilter', window_size=11, polyorder=2)
    savgol = build_filter(cfg)
    out_o = savgol(noisy_input)
    # verify the correctness
    accel_input = noisy_input[:-2] - 2 * noisy_input[1:-1] + noisy_input[2:]
    accel_out_g = out_g[:-2] - 2 * out_g[1:-1] + out_g[2:]
    accel_input_abs = torch.mean(torch.abs(accel_input))
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_g))
    accel_out_s = out_s[:-2] - 2 * out_s[1:-1] + out_s[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_s))
    accel_out_o = out_o[:-2] - 2 * out_o[1:-1] + out_o[2:]
    assert accel_input_abs >= torch.mean(torch.abs(accel_out_o))
    assert out_g.shape == noisy_input.shape == out_s.shape == out_o.shape


def test_data_type_torch_zero():
    noisy_input = torch.zeros((50, 20, 3))
    cfg = dict(type='OneEuroFilter', min_cutoff=0.004, beta=0.7)
    oneeuro = build_filter(cfg)
    out_g = oneeuro(noisy_input)
    cfg = dict(type='Gaus1dFilter', window_size=11, sigma=4)
    gaus1d = build_filter(cfg)
    out_s = gaus1d(noisy_input)
    cfg = dict(type='SGFilter', window_size=11, polyorder=2)
    savgol = build_filter(cfg)
    out_o = savgol(noisy_input)
    # verify the correctness
    accel_input = noisy_input[:-2] - 2 * noisy_input[1:-1] + noisy_input[2:]
    accel_out_g = out_g[:-2] - 2 * out_g[1:-1] + out_g[2:]
    assert torch.mean(accel_input) >= torch.mean(accel_out_g)
    accel_out_s = out_s[:-2] - 2 * out_s[1:-1] + out_s[2:]
    assert torch.mean(accel_input) >= torch.mean(accel_out_s)
    accel_out_o = out_o[:-2] - 2 * out_o[1:-1] + out_o[2:]
    assert torch.mean(accel_input) >= torch.mean(accel_out_o)
    assert out_g.shape == noisy_input.shape == out_s.shape == out_o.shape


def test_data_type_torch_cuda():
    noisy_input = torch.randn((3, 24, 4)).cuda()
    cfg = dict(type='OneEuroFilter', min_cutoff=0.0004, beta=0.7)
    oneeuro = build_filter(cfg)
    out_g = oneeuro(noisy_input)
    cfg = dict(type='Gaus1dFilter', window_size=6, sigma=1)
    gaus1d = build_filter(cfg)
    out_s = gaus1d(noisy_input)
    cfg = dict(type='SGFilter', window_size=7, polyorder=2)
    savgol = build_filter(cfg)
    out_o = savgol(noisy_input)
    assert out_g.shape == noisy_input.shape == out_s.shape == out_o.shape


def test_data_type_np():
    noisy_input = np.random.rand(100, 24, 6)
    cfg = dict(type='OneEuroFilter', min_cutoff=0.004, beta=0.1)
    oneeuro = build_filter(cfg)
    out_g = oneeuro(noisy_input)
    cfg = dict(type='Gaus1dFilter', window_size=5, sigma=2)
    gaus1d = build_filter(cfg)
    out_s = gaus1d(noisy_input)
    cfg = dict(type='SGFilter', window_size=5, polyorder=2)
    savgol = build_filter(cfg)
    out_o = savgol(noisy_input)
    assert out_g.shape == noisy_input.shape == out_s.shape == out_o.shape


def test_smooth():
    test_data_type_torch()
    test_data_type_torch_cuda()
    test_data_type_np()
    test_data_type_torch_zero()
