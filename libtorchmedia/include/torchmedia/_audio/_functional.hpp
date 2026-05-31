#pragma once
#include <cmath>
#include <numeric>
#include <stdexcept>
#include "torch/csrc/autograd/generated/variable_factories.h"
#ifndef LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
#define LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
#include <ATen/core/TensorBody.h>
#include <torch/nn/functional/conv.h>
#include <torch/nn/options/conv.h>
#include "../globel_include.hpp"
#include "_functional_methods_options.hpp"

namespace torchmedia::audio::functional {
    enum convolve_mode { full, valid, same };

    inline auto _apply_convolve_mode(tensor_t conv_result, const int64_t x_length, const int64_t y_length,
                                     const convolve_mode mode) -> tensor_t {
        if (mode == valid) {
            const auto target_length = std::max(x_length, y_length) - std::min(x_length, y_length) + 1;
            const auto start_idx = (conv_result.size(-1) - target_length) / 2;
            return conv_result.slice(-1, start_idx, start_idx + target_length);
        }
        if (mode == same) {
            const auto start_idx = (conv_result.size(-1) - x_length) / 2;
            return conv_result.slice(-1, start_idx, start_idx + x_length);
        }
        if (mode != full) {
            handle_exceptions<torch::Tensor, std::invalid_argument>(
                    torch::empty({1}), "Unrecognized convolve mode. Use full, valid, or same.");
        }
        return conv_result; // full mode (and the no-exceptions fallthrough)
    }
    inline auto db_to_amplitude(tensor_t x, float ref, float power) -> tensor_t {
        // torchaudio: DB_to_amplitude(x, ref, power) = ref * (10^(0.1*x))^power
        return ref * torch::pow(torch::pow(10.0, 0.1 * x), power);
    }

    inline auto convolve(tensor_t x, tensor_t y, const convolve_mode mode) -> tensor_t {
        using namespace torch::indexing;

        if (x.dim() == 0 || y.dim() == 0) {
            handle_exceptions<class torch::Tensor, std::invalid_argument>(torch::empty({1}),
                                                                          "Inputs must be at least 1D.");
        }

        // 1. 维度对齐 (Align Dimensions)
        auto n_dims_x = x.dim();
        auto n_dims_y = y.dim();
        if (n_dims_x < n_dims_y) {
            for (int i = 0; i < n_dims_y - n_dims_x; i++) {
                x = x.unsqueeze(0);
            }
            n_dims_x = n_dims_y;
        } else if (n_dims_y < n_dims_x) {
            for (int i = 0; i < n_dims_x - n_dims_y; i++) {
                y = y.unsqueeze(0);
            }
            n_dims_y = n_dims_x;
        }

        // torchaudio's 'same'/'valid' crop is defined w.r.t. the ORIGINAL first input x, so record its
        // length BEFORE the swap. Convolution is commutative, so swapping to keep the longer signal first
        // is fine for the math; only the final crop length must use the original x length.
        const auto original_x_size = x.size(-1);
        const auto original_y_size = y.size(-1);
        if (x.size(-1) < y.size(-1)) {
            std::swap(x, y);
        }

        const auto x_size = x.size(-1);
        const auto y_size = y.size(-1);
        const auto leading_dims_count = n_dims_x - 1;
        const auto shape_x = x.sizes();
        const auto shape_y = y.sizes();

        // 2. 广播 (Broadcasting) 计算共同形状
        std::vector<int64_t> new_shape(leading_dims_count);
        for (int i = 0; i < leading_dims_count; i++) {
            new_shape[i] = std::max(shape_x[i], shape_y[i]);
        }

        auto broadcast_shape_x = new_shape;
        broadcast_shape_x.push_back(x_size);
        x = x.broadcast_to(broadcast_shape_x);

        auto broadcast_shape_y = new_shape;
        broadcast_shape_y.push_back(y_size);
        y = y.broadcast_to(broadcast_shape_y);

        // 3. 计算总信号数量 (Flatten Batch)
        auto num_signals = 1LL;
        for (int i = 0; i < leading_dims_count; i++) {
            num_signals *= new_shape[i];
        }

        // 4. Reshape 为分组卷积 (Grouped Convolution) 兼容的形状
        // 关键修复:
        // Input  => [1, num_signals, x_size]  (Batch=1, Channels=num_signals)
        // Weight => [num_signals, 1, y_size]  (Out=num_signals, In/Groups=1)
        const auto reshaped_x = x.reshape({1, num_signals, x_size});

        // 注意：Convolution 实际上是互相关，严格数学卷积需要 flip kernel。
        // 原版代码有 flip，这里保留。
        const auto reshaped_y = y.flip(-1).reshape({num_signals, 1, y_size});

        // 5. 执行卷积
        // padding(y_size - 1) 是为了实现 'full' 模式，之后再裁切
        const auto conv_out =
                torch::nn::functional::conv1d(reshaped_x, reshaped_y,
                                              torch::nn::functional::Conv1dFuncOptions()
                                                      .stride(1)
                                                      .groups(num_signals) // 分组数 = 信号数，实现 Depthwise
                                                      .padding(y_size - 1));

        // 6. 恢复形状
        // conv_out 目前是 [1, num_signals, out_len]
        const auto output_length = conv_out.size(-1);
        auto output_shape = new_shape;
        output_shape.push_back(output_length);

        // 先 reshape 去掉 batch=1 维度，再 reshape 回广播后的维度
        const auto result = conv_out.reshape(output_shape);

        // 7. 应用裁剪模式 (Full/Valid/Same) — crop length uses the ORIGINAL x/y lengths
        return _apply_convolve_mode(result, original_x_size, original_y_size, mode);
    }


    inline auto amplitude_to_DB(tensor_t signal, const amplitude_to_db_option option = {}) -> tensor_t {
        if (signal.is_complex()) {
            signal = signal.abs(); // sqrt(real^2 + imag^2)
        }
        // torchaudio: db = multiplier*log10(clamp(x, amin)) - multiplier*db_multiplier.
        // `signal` is already power (multiplier=10) or magnitude (multiplier=20) — do NOT square here.
        const float amin_val = std::max(option.amin, std::numeric_limits<float>::min());
        auto db = option.multiplier * torch::log10(torch::clamp_min(signal, amin_val));
        db = db - option.multiplier * option.db_multiplier;
        if (option.apply_top_db) {
            // per-sample max over the (freq, time) dims for batched input
            const auto max_db = db.amax({-2, -1}, /*keepdim=*/true);
            db = torch::maximum(db, max_db - option.top_db);
        }

        return db;
    }


    inline auto spectrogram(tensor_t signal, spectrogram_option option) -> tensor_t {
        if (const int pad_amount = option._pad; pad_amount > 0) {
            signal = constant_pad_nd(signal, {pad_amount, pad_amount}, 0);
        }

        const int n_fft = option._n_fft;
        int hop_length = option._hop_length;
        int win_length = option._win_length;
        tensor_t window =
                option._window.defined()
                        ? option._window
                        : torch::hann_window(win_length,
                                             tensor_options_t().dtype(signal.dtype()).device(signal.device()));
        // Always compute a complex STFT with normalized=false, then normalize ONCE on the complex
        // amplitude before applying power. This matches torchaudio and fixes: (a) double normalization
        // (normalized was passed to stft AND applied again), (b) the wrong normalization exponent
        // (dividing the power spectrum by sqrt(sum win^2) instead of by (sum win^2)^(power/2)), and
        // (c) abs() over stft's view_as_real output when return_complex was false.
        auto spec_f = torch::stft(signal, n_fft, hop_length, win_length, window, option._center,
                                  option._pad_mode, /*normalized=*/false, option._onesided,
                                  /*return_complex=*/true);
        if (option._normalized) {
            if (option._normalize_method == "window") {
                spec_f = spec_f / window.pow(2).sum().sqrt();
            } else if (option._normalize_method == "frame_length") {
                spec_f = spec_f / win_length;
            }
        }
        if (option._return_complex) {
            return spec_f;
        }
        return torch::pow(torch::abs(spec_f), option._power);
    }


    inline auto mel_filter_bank(int n_mels, const double f_min, double f_max, const int sample_rate,
                                int n_stft_bins, // 通常 = n_fft/2 + 1
                                const std::string &norm, // "slaney" 或 ""
                                const std::string &mel_scale // "htk" / "slaney"
                                ) -> tensor_t {
        using namespace torch::indexing;
        // 如果外部没指定 f_max 或给了 <=0，则默认设为 Nyquist 频率
        if (f_max <= 0.0) {
            f_max = sample_rate / 2.0;
        }

        // 一些辅助函数：赫兹转 mel，mel 转赫兹
        // Hz<->mel. HTK: mel = 2595*log10(1+f/700). Slaney: piecewise (linear < 1000Hz, log >= 1000Hz).
        auto hz_to_mel = [&](const double freq) -> double {
            if (mel_scale != "slaney") {
                return 2595.0 * std::log10(1.0 + freq / 700.0);
            }
            const double f_sp = 200.0 / 3.0;
            const double min_log_hz = 1000.0;
            const double min_log_mel = min_log_hz / f_sp;
            const double logstep = std::log(6.4) / 27.0;
            if (freq >= min_log_hz) {
                return min_log_mel + std::log(freq / min_log_hz) / logstep;
            }
            return freq / f_sp;
        };
        auto mel_to_hz = [&](const double mel) -> double {
            if (mel_scale != "slaney") {
                return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
            }
            const double f_sp = 200.0 / 3.0;
            const double min_log_hz = 1000.0;
            const double min_log_mel = min_log_hz / f_sp;
            const double logstep = std::log(6.4) / 27.0;
            if (mel >= min_log_mel) {
                return min_log_hz * std::exp(logstep * (mel - min_log_mel));
            }
            return f_sp * mel;
        };

        const double mel_min = hz_to_mel(f_min);
        const double mel_max = hz_to_mel(f_max);

        // 在 mel 坐标上等距取 n_mels+2 个点
        const auto mels = torch::linspace(mel_min, mel_max, n_mels + 2);

        // 逐个转换回赫兹频率
        const auto hz_points = mels.clone();
        for (int i = 0; i < hz_points.size(0); i++) {
            const auto mel_val = hz_points[i].item<double>();
            hz_points[i] = mel_to_hz(mel_val);
        }

        // 计算每个 STFT bin 对应的赫兹频率
        // freq_bin[i] = i * (sample_rate / 2) / (n_stft_bins - 1)
        const auto bin_frequencies = torch::linspace(0, sample_rate / 2.0, n_stft_bins);

        // 创建 [n_mels, n_stft_bins] 的滤波器
        auto fb = torch::zeros({n_mels, n_stft_bins}, torch::kFloat);

        for (int m = 1; m <= n_mels; m++) {
            const auto left = hz_points[m - 1].item<double>();
            const auto center = hz_points[m].item<double>();
            const auto right = hz_points[m + 1].item<double>();

            for (int f = 0; f < n_stft_bins; f++) {
                if (const auto freq = bin_frequencies[f].item<double>(); freq >= left && freq <= center) {
                    fb[m - 1][f] = static_cast<float>((freq - left) / (center - left));
                } else if (freq > center && freq <= right) {
                    fb[m - 1][f] = static_cast<float>((right - freq) / (right - center));
                } else {
                    fb[m - 1][f] = 0.0f;
                }
            }
        }

        // 如果 norm == "slaney"，需要对每个 mel 过滤器再做归一化 (按带宽)
        if (norm == "slaney") {
            for (int m = 0; m < n_mels; m++) {
                auto row = fb.select(0, m);
                const float enorm = 2.0f / (hz_points[m + 2].item<float>() - hz_points[m].item<float>());
                (void) row.mul_(enorm);
            }
        }

        return fb;
    }

    inline auto mel_scale(const_tensor_lref_t spec, const_tensor_lref_t fb) -> tensor_t {
        const auto sizes = spec.sizes();
        const auto ndim = sizes.size();
        int64_t freq = sizes[ndim - 2];
        int64_t time = sizes[ndim - 1];

        int64_t batch = 1;
        for (int i = 0; i < ndim - 2; i++) {
            batch *= sizes[i];
        }
        // reshape => [batch, freq, time]
        auto spec_3d = spec.reshape({batch, freq, time});
        // 转置 => [batch, time, freq]
        spec_3d = spec_3d.transpose(1, 2);

        // fb => [n_mels, freq], 做 transpose => [freq, n_mels]
        const auto fb_t = fb.transpose(0, 1); // [freq, n_mels]

        // [batch, time, freq] matmul [freq, n_mels] => [batch, time, n_mels]
        auto mel_3d = matmul(spec_3d, fb_t);

        // 再转置回 => [batch, n_mels, time]
        mel_3d = mel_3d.transpose(1, 2);

        // 最后 reshape => [..., n_mels, time]
        std::vector<int64_t> final_shape;
        final_shape.push_back(batch);
        final_shape.push_back(mel_3d.size(1)); // n_mels
        final_shape.push_back(mel_3d.size(2)); // time
        auto mel_out = mel_3d.reshape(final_shape);

        return mel_out;
    }

    inline auto melspectrogram(const_tensor_lref_t waveform, const mel_spectrogram_option &opt) -> tensor_t {
        // 1) 构建 spectrogram_option
        spectrogram_option sp_opt;
        sp_opt._pad = opt.pad;
        sp_opt._n_fft = opt.n_fft;
        sp_opt._win_length = opt.win_length;
        sp_opt._hop_length = opt.hop_length;
        sp_opt._center = opt.center;
        sp_opt._pad_mode = opt.pad_mode;
        sp_opt._onesided = opt.onesided;
        sp_opt._power = opt.power; // 1 => 幅度谱, 2 => 功率谱
        sp_opt._normalized = opt.normalized;
        sp_opt._return_complex = false; // 我们要实数谱，后面才能和 fb 做乘法

        // 2) 调用已有 spectrogram 函数，得到实数的 [batch, freq, time] 等形状
        const auto spec = spectrogram(waveform, sp_opt);

        // 3) 构建 mel filter bank
        const int n_stft_bins = opt.n_fft / 2 + 1;
        const auto fb = mel_filter_bank(opt.n_mels, opt.f_min, opt.f_max, opt.sample_rate, n_stft_bins, opt.norm,
                                        opt.mel_scale); // [n_mels, n_stft_bins]

        // 4) 做 mel-scale
        auto mel_spectrogram = mel_scale(spec, fb);
        // mel_spectrogram => [..., n_mels, time]

        return mel_spectrogram;
    }

    // DCT-II matrix of shape [n_mels, n_mfcc] (mirrors torchaudio.functional.create_dct).
    // dct[n,k] = cos(pi/n_mels * (n+0.5) * k); norm=="" -> *2; norm=="ortho" -> col0 *= 1/sqrt(2), then *sqrt(2/n_mels).
    inline auto create_dct(int n_mfcc, int n_mels, const std::string &norm = "ortho") -> tensor_t {
        const double pi = std::acos(-1.0);
        const auto n = torch::arange(n_mels, torch::kFloat64).unsqueeze(1); // [n_mels, 1]
        const auto k = torch::arange(n_mfcc, torch::kFloat64).unsqueeze(0); // [1, n_mfcc]
        auto dct = torch::cos((pi / n_mels) * (n + 0.5) * k);               // [n_mels, n_mfcc]
        if (norm.empty()) {
            dct = dct * 2.0;
        } else { // "ortho"
            dct.select(1, 0).mul_(1.0 / std::sqrt(2.0));
            dct = dct * std::sqrt(2.0 / n_mels);
        }
        return dct.to(torch::kFloat32);
    }

    // MFCC: DCT over the dB (or log) mel spectrogram (mirrors torchaudio.transforms.MFCC).
    inline auto mfcc(const_tensor_lref_t waveform, const mfcc_option &opt) -> tensor_t {
        const auto mel_spec = melspectrogram(waveform, opt.mel); // [..., n_mels, time]
        tensor_t feat;
        if (opt.log_mels) {
            feat = torch::log(mel_spec + 1e-6);
        } else {
            const auto db_opt = amplitude_to_db_option()
                                        .set_multiplier(10.0f)
                                        .set_amin(1e-10f)
                                        .set_db_multiplier(0.0f)
                                        .set_top_db(opt.top_db);
            feat = amplitude_to_DB(mel_spec, db_opt);
        }
        const auto dct_mat = create_dct(opt.n_mfcc, opt.mel.n_mels, opt.norm); // [n_mels, n_mfcc]
        // apply the DCT along the mel axis: [..., n_mels, time] -> [..., n_mfcc, time]
        return torch::matmul(feat.transpose(-2, -1), dct_mat).transpose(-2, -1);
    }

    // Griffin-Lim phase reconstruction from a (power) spectrogram (mirrors torchaudio.functional.griffinlim).
    // Uses torch::istft/stft (the same ATen ops torchaudio calls) so rand_init=false is reproducible.
    inline auto griffinlim(const_tensor_lref_t specgram, const griffinlim_option &opt) -> tensor_t {
        const auto window = opt.window.defined()
                                    ? opt.window
                                    : torch::hann_window(opt.win_length, tensor_options_t()
                                                                                 .dtype(torch::kFloat)
                                                                                 .device(specgram.device()));
        const double momentum = opt.momentum / (1.0 + opt.momentum);

        const auto shape = specgram.sizes().vec();
        const int64_t freq = shape[shape.size() - 2];
        const int64_t frames = shape[shape.size() - 1];
        auto spec = specgram.reshape({-1, freq, frames}).pow(1.0 / opt.power); // magnitude

        tensor_t angles;
        if (opt.rand_init) {
            const auto r = torch::rand(spec.sizes(), spec.options());
            angles = torch::polar(torch::ones_like(r), 2.0 * std::acos(-1.0) * r); // unit modulus, random phase
        } else {
            angles = torch::ones(spec.sizes(), spec.options().dtype(torch::kComplexFloat)); // 1 + 0j
        }

        tensor_t tprev = torch::zeros({}, spec.options());
        for (int i = 0; i < opt.n_iter; i++) {
            const auto inverse = torch::istft(spec * angles, opt.n_fft, opt.hop_length, opt.win_length, window,
                                              /*center=*/true, /*normalized=*/false, /*onesided=*/true,
                                              /*length=*/c10::nullopt, /*return_complex=*/false);
            const auto rebuilt = torch::stft(inverse, opt.n_fft, opt.hop_length, opt.win_length, window,
                                             /*center=*/true, /*pad_mode=*/"reflect", /*normalized=*/false,
                                             /*onesided=*/true, /*return_complex=*/true);
            angles = rebuilt;
            if (opt.momentum > 0.0) {
                angles = angles - tprev * momentum;
            }
            angles = angles / (angles.abs() + 1e-16);
            tprev = rebuilt;
        }

        const c10::optional<int64_t> length = opt.length > 0 ? c10::optional<int64_t>(opt.length) : c10::nullopt;
        auto waveform = torch::istft(spec * angles, opt.n_fft, opt.hop_length, opt.win_length, window,
                                     /*center=*/true, /*normalized=*/false, /*onesided=*/true, length,
                                     /*return_complex=*/false);

        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 2);
        out_shape.push_back(waveform.size(-1));
        return waveform.reshape(out_shape);
    }

    // Band-limited sinc resampling expressed as a strided conv1d (mirrors torchaudio.functional.resample,
    // resampling_method="sinc_interp_hann"). Uses the same ATen ops, so it matches torchaudio point-wise.
    inline auto resample(const_tensor_lref_t waveform, int orig_freq, int new_freq, const resample_option &opt = {})
            -> tensor_t {
        const int gcd = std::gcd(orig_freq, new_freq);
        const int of = orig_freq / gcd;
        const int nf = new_freq / gcd;
        const int lfw = opt.lowpass_filter_width;
        const double base_freq = std::min(of, nf) * opt.rolloff;
        const int width = static_cast<int>(std::ceil(lfw * of / base_freq));
        const double pi = std::acos(-1.0);
        const auto options = tensor_options_t().dtype(torch::kFloat).device(waveform.device());

        // Build the [nf, 1, kernel_width] sinc filterbank.
        const auto idx = torch::arange(-width, width + of, options).reshape({1, 1, -1}) / of;     // [1,1,K]
        const auto phase = torch::arange(0, -nf, -1, options).reshape({nf, 1, 1}) / nf;            // [nf,1,1]
        auto t = (phase + idx) * base_freq;
        t = t.clamp(-lfw, lfw);
        const auto window = torch::cos(t * pi / lfw / 2.0).pow(2);
        t = t * pi;
        const double scale = base_freq / of;
        auto kernels = torch::where(t == 0, torch::ones_like(t), torch::sin(t) / t) * window * scale;

        // Apply: pad, strided conv1d, crop to the target length.
        const auto shape = waveform.sizes().vec();
        const int64_t length = shape[shape.size() - 1];
        auto wav = waveform.reshape({-1, length});
        const int64_t num_wavs = wav.size(0);
        wav = torch::constant_pad_nd(wav, {width, width + of}, 0);
        auto resampled = torch::nn::functional::conv1d(wav.unsqueeze(1), kernels,
                                                       torch::nn::functional::Conv1dFuncOptions().stride(of));
        resampled = resampled.transpose(1, 2).reshape({num_wavs, -1});
        const int64_t target_length = static_cast<int64_t>(std::ceil(static_cast<double>(nf) * length / of));
        resampled = resampled.slice(-1, 0, target_length);

        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1);
        out_shape.push_back(resampled.size(-1));
        return resampled.reshape(out_shape);
    }
} // namespace torchmedia::audio::functional
#endif // LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
