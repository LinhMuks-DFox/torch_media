#pragma once
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

    auto inline _check_shape_compatible(const tensor_t &x, const tensor_t &y) -> bool {
        if (x.ndimension() != y.ndimension())
            return false;
        for (auto i = 0; i < x.ndimension() - 1; i++) {
            const auto xi = x.size(i);
            const auto yi = y.size(i);
            if (xi == yi || xi == 1 || yi == 1)
                continue;
            return false;
        }
        return true;
    }

    inline auto _apply_convolve_mode(tensor_t conv_result, const int64_t x_length, const int64_t y_length,
                                     const convolve_mode mode) -> tensor_t {
        switch (mode) {
            case full:
                return conv_result;
            case valid: {
                const auto target_length = std::max(x_length, y_length) - std::min(x_length, y_length) + 1;
                auto start_idx = (conv_result.size(-1) - target_length) / 2;
                return conv_result.slice(-1, start_idx, start_idx + target_length);
            }
            case same: {
                auto start_idx = (conv_result.size(-1) - x_length) / 2;
                return conv_result.slice(-1, start_idx, start_idx + x_length);
            }
            default:
                handle_exceptions<class torch::Tensor, std::invalid_argument>(
                        torch::empty({1}), " Unrecognized mode value.Please specify one of full, valid, same.");
        }
    }
    inline auto db_to_amplitude(tensor_t x, float ref, float power) -> tensor_t {
        return torch::pow(10.0, x / (20.0 * power)) * ref;
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

        // 确保 x 是较长的信号 (虽然数学上卷积可交换，但通常用短的做 kernel 比较直观)
        // 这里的 swap 逻辑保留原版，视需求可去
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

        // 7. 应用裁剪模式 (Full/Valid/Same)
        return _apply_convolve_mode(result, x_size, y_size, mode);
    }


    inline auto amplitude_to_DB(tensor_t signal, const amplitude_to_db_option option = {}) -> tensor_t {
        if (signal.is_complex()) {
            signal = signal.abs(); // 等价于 sqrt(real^2 + imag^2)
        }
        float amin_val = std::max(option.amin, std::numeric_limits<float>::min());
        const auto power = torch::pow(signal, 2.0);
        auto db = 10.0 * torch::log10(clamp(power, amin_val, std::numeric_limits<float>::max()));
        db = db * option.db_multiplier;
        if (option.apply_top_db) {
            const auto max_db = db.max().item<float>();
            db = max(db, torch::tensor(max_db - option.top_db, db.options()));
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
        auto spec_f = stft(signal, n_fft, hop_length, win_length, window, option._center, option._pad_mode,
                           option._normalized, option._onesided, option._return_complex);
        if (option._return_complex) {
            return spec_f;
        }
        auto spec = torch::pow(torch::abs(spec_f), option._power);
        if (option._normalized) {
            if (option._normalize_method == "window") {
                spec /= window.pow(2).sum().sqrt();
            } else if (option._normalize_method == "frame_length") {
                spec /= win_length;
            }
        }
        return spec;
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
        auto hz_to_mel = [&](const double freq) {
            if (mel_scale != "slaney") {
                // HTK 风格
                return 2595.0 * std::log10(1.0 + freq / 700.0);
            }
            // Slaney 风格公式 (approx)
            return 2595.0 * std::log10(1.0 + freq / 700.0);
        };
        auto mel_to_hz = [&](const double mel) {
            if (mel_scale != "slaney") {
                return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
            }
            return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
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
} // namespace torchmedia::audio::functional
#endif // LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
