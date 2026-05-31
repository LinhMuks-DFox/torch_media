#!/usr/bin/env python3
"""Golden reference values for the audio functional unit tests, computed with torchaudio 2.5.1.

Run in the project venv:
    .venv/bin/python unit_test/audio/functional/gen_golden.py

Filterbanks and melspectrogram are deterministic and cross-language comparable, so the printed
constants are hard-coded into main.cpp. Randomized-input tests use libtorch self-reference instead
(e.g. normalized spectrogram == unnormalized / Sum(win^2)).
"""
import torch
import torchaudio.functional as F
import torchaudio.transforms as T


def show(name, value):
    print(f"{name} = {value}")


# bug#3 -- db_to_amplitude(x, ref, power) = ref * (10^(0.1*x))^power
show("db_to_amp(20,1,1)", float(F.DB_to_amplitude(torch.tensor(20.0), 1.0, 1.0)))    # 100
show("db_to_amp(20,1,0.5)", float(F.DB_to_amplitude(torch.tensor(20.0), 1.0, 0.5)))  # 10
show("db_to_amp(10,2,1)", float(F.DB_to_amplitude(torch.tensor(10.0), 2.0, 1.0)))    # 20

# bug#2 -- amplitude_to_DB(x, multiplier, amin, db_multiplier, top_db); no internal square
show("amp_to_DB power [1,.1,.01,.001] mult=10",
     F.amplitude_to_DB(torch.tensor([[1., .1, .01, .001]]), 10.0, 1e-10, 0.0, 80.0).tolist())  # [0,-10,-20,-30]
show("amp_to_DB mag [1,.5,.25] mult=20",
     F.amplitude_to_DB(torch.tensor([[1., .5, .25]]), 20.0, 1e-10, 0.0, 80.0).tolist())          # [0,-6.02,-12.04]

# bug#4 -- convolve mode lengths ('same' = first input length)
a = torch.tensor([1., 2., 3., 4., 5.]).view(1, -1)
b = torch.tensor([1., 1., 1.]).view(1, -1)
for m in ["full", "same", "valid"]:
    show(f"convolve {m} len", F.convolve(a, b, mode=m).shape[-1])  # 7 / 5 / 3

# bug#1 -- mel filterbank: slaney differs from htk; slaney sum (norm=None)
fb_s = F.melscale_fbanks(201, 0.0, 8000.0, 64, 16000, norm=None, mel_scale="slaney")
fb_h = F.melscale_fbanks(201, 0.0, 8000.0, 64, 16000, norm=None, mel_scale="htk")
show("fb_slaney != fb_htk", bool(not torch.allclose(fb_s, fb_h)))
show("fb_slaney.sum", round(float(fb_s.sum()), 3))  # 194.677

# bug#5 -- spectrogram normalized (power=2): unnormalized/normalized == Sum(win^2)
win = torch.hann_window(400)
show("Sum(win^2)", float(win.pow(2).sum()))  # 150

# melspectrogram (htk) end-to-end golden, deterministic sine input
sig = torch.sin(2 * torch.pi * 440.0 * torch.arange(16000.0) / 16000.0).unsqueeze(0)
mel = T.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512, hop_length=256, n_mels=64,
                       f_min=0.0, f_max=8000.0, power=2.0, norm=None, center=True,
                       pad_mode="reflect", mel_scale="htk")(sig)
show("melspec htk shape", tuple(mel.shape))          # (1, 64, 63)
show("melspec htk sum", round(float(mel.sum()), 3))  # 1548267.875

# Tier 1 -- create_dct (DCT-II matrix)
for norm in (None, "ortho"):
    d = F.create_dct(4, 8, norm)
    show(f"create_dct(4,8,{norm}) sum/[0,0]/[3,2]",
         [round(float(d.sum()), 6), round(float(d[0, 0]), 6), round(float(d[3, 2]), 6)])

# Tier 1 -- mfcc (DCT over dB / log mel)
mk = dict(n_fft=512, win_length=512, hop_length=256, n_mels=64, f_min=0.0, f_max=8000.0,
          power=2.0, norm=None, mel_scale="htk", center=True, pad_mode="reflect")
for log_mels in (False, True):
    mf = T.MFCC(sample_rate=16000, n_mfcc=13, norm="ortho", log_mels=log_mels, melkwargs=mk)(sig)
    show(f"mfcc(log_mels={log_mels}) shape/sum/[0,0,0]",
         [tuple(mf.shape), round(float(mf.sum()), 4), round(float(mf[0, 0, 0]), 4)])

# Tier 1 -- griffinlim (deterministic, rand_init=False)
wavg = torch.sin(2 * torch.pi * 5.0 * torch.arange(2000.0) / 2000.0).unsqueeze(0)
wing = torch.hann_window(256)
specg = F.spectrogram(wavg, pad=0, window=wing, n_fft=256, hop_length=64, win_length=256, power=2.0,
                      normalized=False, center=True, pad_mode="reflect", onesided=True)
recg = F.griffinlim(specg, window=wing, n_fft=256, hop_length=64, win_length=256, power=2.0,
                    n_iter=32, momentum=0.99, length=None, rand_init=False)
show("griffinlim shape/sum/[0,1000]", [tuple(recg.shape), round(float(recg.sum()), 4), round(float(recg[0, 1000]), 6)])

# Tier 1 -- resample
xr = torch.sin(2 * torch.pi * 3.0 * torch.arange(64.0) / 64.0).unsqueeze(0)
for (a, b) in [(64, 32), (64, 48)]:
    rr = F.resample(xr, a, b, lowpass_filter_width=6, rolloff=0.99)
    show(f"resample {a}->{b} shape/sum/[0,1]", [tuple(rr.shape), round(float(rr.sum()), 6), round(float(rr[0, 1]), 6)])
