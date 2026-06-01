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

# task01 -- lfilter / biquad / filtfilt (IIR core). Fixed deterministic signal.
fwav = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
show("lfilter a=[1,-.3,.05] b=[.5,.1,0]",
     [round(float(x), 8) for x in F.lfilter(fwav, torch.tensor([1.0, -0.3, 0.05]), torch.tensor([0.5, 0.1, 0.0]))])
show("biquad (.2,.2,.2 / 1,-.5,.1)",
     [round(float(x), 8) for x in F.biquad(fwav, 0.2, 0.2, 0.2, 1.0, -0.5, 0.1)])
show("filtfilt a=[1,-.3,.05] b=[.5,.1,0]",
     [round(float(x), 8) for x in F.filtfilt(fwav, torch.tensor([1.0, -0.3, 0.05]), torch.tensor([0.5, 0.1, 0.0]))])
fimp = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
show("lfilter one-pole a=[1,-.5] b=[1,0] (0.5^n)",
     [round(float(x), 8) for x in F.lfilter(fimp, torch.tensor([1.0, -0.5]), torch.tensor([1.0, 0.0]))])
_a2 = torch.tensor([[1.0, -0.3, 0.05], [1.0, -0.5, 0.1]])
_b2 = torch.tensor([[0.5, 0.1, 0.0], [0.2, 0.2, 0.2]])
_o2 = F.lfilter(torch.stack([fwav, fwav * 0.5]), _a2, _b2, batching=True)
show("lfilter batched row1", [round(float(x), 8) for x in _o2[1]])

# task02 -- biquad designers. filt_wav(), sr=16000, f=2000, Q=0.707, gain=6 (deemph/riaa: own rates).
_fw = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
def _bd(n, t): show(n, [round(float(x), 7) for x in t])
_bd("allpass", F.allpass_biquad(_fw, 16000, 2000.0, 0.707))
_bd("lowpass", F.lowpass_biquad(_fw, 16000, 2000.0, 0.707))
_bd("highpass", F.highpass_biquad(_fw, 16000, 2000.0, 0.707))
_bd("bandpass_F", F.bandpass_biquad(_fw, 16000, 2000.0, 0.707, const_skirt_gain=False))
_bd("bandpass_T", F.bandpass_biquad(_fw, 16000, 2000.0, 0.707, const_skirt_gain=True))
_bd("bandreject", F.bandreject_biquad(_fw, 16000, 2000.0, 0.707))
_bd("equalizer_g6", F.equalizer_biquad(_fw, 16000, 2000.0, 6.0, 0.707))
_bd("band_F", F.band_biquad(_fw, 16000, 2000.0, 0.707, noise=False))
_bd("band_T", F.band_biquad(_fw, 16000, 2000.0, 0.707, noise=True))
_bd("bass_g6", F.bass_biquad(_fw, 16000, 6.0, 100.0, 0.707))
_bd("treble_g6", F.treble_biquad(_fw, 16000, 6.0, 3000.0, 0.707))
for _r in (44100, 48000):
    _bd(f"deemph_{_r}", F.deemph_biquad(_fw, _r))
for _r in (44100, 48000, 88200, 96000):
    _bd(f"riaa_{_r}", F.riaa_biquad(_fw, _r))

# task03 -- simple effects contrast/dcshift/gain on filt_wav()
_fw = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
def _ef(n, t): show(n, [round(float(x), 7) for x in t])
_ef("contrast_75", F.contrast(_fw, 75.0)); _ef("contrast_0", F.contrast(_fw, 0.0))
_ef("gain_6", F.gain(_fw, 6.0))
_ef("dcshift_0.3", F.dcshift(_fw, 0.3)); _ef("dcshift_-0.3", F.dcshift(_fw, -0.3))
_ef("dcshift_lim_pos", F.dcshift(_fw.clone(), 0.5, limiter_gain=0.1))
_ef("dcshift_lim_neg", F.dcshift(_fw.clone(), -0.5, limiter_gain=0.1))

# task06 -- companding & emphasis on filt_wav()
_fw = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
show("mulaw_enc", [int(x) for x in F.mu_law_encoding(_fw, 256)])
show("mulaw_dec", [round(float(x), 7) for x in F.mu_law_decoding(F.mu_law_encoding(_fw, 256), 256)])
show("preemph_097", [round(float(x), 7) for x in F.preemphasis(_fw, 0.97)])
show("deemph_097", [round(float(x), 7) for x in F.deemphasis(_fw, 0.97)])

# task09 -- compute_deltas / linear_fbanks / spectral_centroid
_sg = torch.tensor([[1.,2.,3.,4.,5.,6.],[6.,5.,4.,3.,2.,1.]])
show("deltas_w5", F.compute_deltas(_sg, 5).tolist())
show("deltas_w3", F.compute_deltas(_sg, 3).tolist())
show("linfbank_6_3", F.linear_fbanks(6, 0.0, 8000.0, 3, 16000).tolist())

# task07 -- fftconvolve / add_noise / speed
_x = torch.tensor([[1.,2.,3.,4.]]); _y = torch.tensor([[1.,1.,1.]])
for _m in ("full","valid","same"):
    show(f"fftconv_{_m}", F.fftconvolve(_x, _y, mode=_m).tolist())
_w = torch.tensor([[1.,2.,3.,4.],[0.5,-0.5,0.5,-0.5]]); _n = torch.tensor([[0.1,0.1,0.1,0.1],[0.2,-0.2,0.2,-0.2]])
show("add_noise", F.add_noise(_w, _n, torch.tensor([10.0,3.0])).tolist())
_sp = torch.sin(2*torch.pi*2.0*torch.arange(16.0)/16.0).reshape(1,16)
_o,_l = F.speed(_sp, 16, 2.0, torch.tensor([16])); show("speed_out", [round(float(x),6) for x in _o.flatten()]); show("speed_len", _l.tolist())

# task13 -- frechet_distance (edit_distance is non-tensor, closed-form tested in C++)
_I = torch.eye(2)
show("frechet_shift", round(float(F.frechet_distance(torch.tensor([0.,0.]), _I, torch.tensor([1.,1.]), _I)), 6))
show("frechet_diag", round(float(F.frechet_distance(torch.tensor([1.,2.]), torch.tensor([[2.,0.],[0.,3.]]), torch.tensor([0.,0.]), _I)), 6))

# task18 -- dither (TPDF deterministic; RPDF/GPDF RNG-dependent)
_dw = torch.tensor([[0.1,-0.2,0.3,-0.4,0.5,-0.6],[0.05,-0.05,0.15,-0.15,0.25,-0.25]])
show("dither_tpdf", F.dither(_dw, "TPDF", noise_shaping=False).tolist())
show("dither_tpdf_ns", F.dither(_dw, "TPDF", noise_shaping=True).tolist())

# task08 -- phase_vocoder / inverse_spectrogram / pitch_shift (deterministic inputs)
import math as _m
_A = torch.arange(1,13,dtype=torch.float32).reshape(1,3,4)
_spec = torch.complex(_A, _A*0.5); _pa = torch.linspace(0, _m.pi, 3)[...,None]
_pv = F.phase_vocoder(_spec, 1.5, _pa)
show("pv_real", [round(float(x),6) for x in _pv.real.flatten()])
show("pv_imag", [round(float(x),6) for x in _pv.imag.flatten()])
_t = torch.arange(2000.0)/16000.0; _sine = torch.sin(2*_m.pi*440*_t).reshape(1,2000); _win = torch.hann_window(400)
_S = F.spectrogram(_sine, pad=0, window=_win, n_fft=400, hop_length=200, win_length=400, power=None, normalized=False)
show("inv_recon_err", round(float((F.inverse_spectrogram(_S,2000,0,_win,400,200,400,False)-_sine).abs().max()),6))
_ps = F.pitch_shift(_sine, 16000, 4, n_fft=512)
show("ps_sum", round(float(_ps.sum()),5)); show("ps_500", round(float(_ps[0,500]),6))

# task12 -- loudness (ITU-R BS.1770)
import math as _m
_t = torch.arange(16000.0)/16000.0; _tone = torch.sin(2*_m.pi*1000*_t)
show("loudness_2ch", round(float(F.loudness(torch.stack([_tone, 0.5*_tone]), 16000)),5))
show("loudness_mono", round(float(F.loudness(_tone.reshape(1,16000), 16000)),5))

# task14 -- beamforming (deterministic complex PD matrices)
_A = torch.arange(1,25,dtype=torch.float32).reshape(2,3,4); _spec = torch.complex(_A, _A*0.1)
show("psd_re", [round(float(x),5) for x in F.psd(_spec).real.flatten()])
_Ms = torch.complex(torch.arange(1,13,dtype=torch.float32).reshape(3,2,2), torch.arange(1,13,dtype=torch.float32).reshape(3,2,2)*0.2)
_ps = _Ms @ _Ms.conj().transpose(-1,-2) + 2*torch.eye(2)
_Mn = torch.complex(torch.arange(13,25,dtype=torch.float32).reshape(3,2,2)*0.1, torch.arange(1,13,dtype=torch.float32).reshape(3,2,2)*0.3)
_pn = _Mn @ _Mn.conj().transpose(-1,-2) + 2*torch.eye(2)
_ws = F.mvdr_weights_souden(_ps,_pn,0)
show("souden_re", [round(float(x),5) for x in _ws.real.flatten()]); show("souden_im", [round(float(x),5) for x in _ws.imag.flatten()])

# task04 -- overdrive / phaser / flanger
_w = torch.tensor([0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8]).reshape(1,8)
show("overdrive", [round(float(x),6) for x in F.overdrive(_w,20.0,20.0).flatten()])
show("phaser_sine", [round(float(x),6) for x in F.phaser(_w,8000,sinusoidal=True).flatten()])
_wf = torch.tensor([[0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8],[0.2,-0.1,0.4,-0.3,0.6,-0.5,0.8,-0.7]])
show("flanger_lin", [round(float(x),6) for x in F.flanger(_wf,8000,interpolation="linear").flatten()])

# task11 -- sliding_window_cmn / detect_pitch_frequency
import math as _m
_sg = torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.],[9.,10.]])
show("cmn_basic", F.sliding_window_cmn(_sg,3,2,False,False).tolist())
show("cmn_center", F.sliding_window_cmn(_sg,3,2,True,False).tolist())
show("cmn_normvars", F.sliding_window_cmn(_sg,4,2,False,True).tolist())
_sine = torch.sin(2*_m.pi*220*torch.arange(8000.0)/8000.0).reshape(1,8000)
show("pitch_median", round(float(F.detect_pitch_frequency(_sine,8000).median()),3))

# task05 -- vad (silence-then-tone trim length)
import math as _m
_sr=16000; _t=torch.arange(int(0.5*_sr))/_sr
_sig=torch.cat([torch.zeros(int(0.5*_sr)), 0.5*torch.sin(2*_m.pi*300*_t)]).reshape(1,-1)
show("vad_out_len", int(F.vad(_sig,_sr).size(-1)))
