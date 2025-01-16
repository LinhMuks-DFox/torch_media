#pragma once
#ifndef _LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
#define _LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
#include <ATen/core/TensorBody.h>
#include <c10/core/TensorImpl.h>
#include <sox.h>
#include <stdexcept>
#include <string>
#include <torch/nn/functional/conv.h>
#include <torch/nn/options/conv.h>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>
#include <vector>

namespace torchmedia::audio::functional {
    using tensor_t = torch::Tensor;

    enum convolve_mode { full, valid, same };

    auto inline _check_shape_compatible(tensor_t &x, tensor_t &y) -> bool {
        if (x.ndimension() != y.ndimension())
            return false;
        for (auto i = 0; i < x.ndimension() - 1; i++) {
            auto xi = x.size(i);
            auto yi = y.size(i);
            if (xi == yi || xi == 1 || yi == 1)
                continue;
            return false;
        }
        return true;
    }

    inline auto _apply_convolve_mode(torch::Tensor conv_result, int64_t x_length,
                                     int64_t y_length, convolve_mode mode)
        -> torch::Tensor {
        switch (mode) {
            case full:
                return conv_result;
            case valid: {
                auto target_length =
                        std::max(x_length, y_length) - std::min(x_length, y_length) + 1;
                auto start_idx = (conv_result.size(-1) - target_length) / 2;
                return conv_result.slice(-1, start_idx, start_idx + target_length);
            }
            case same: {
                auto start_idx = (conv_result.size(-1) - x_length) / 2;
                return conv_result.slice(-1, start_idx, start_idx + x_length);
            }
            default:
                throw std::invalid_argument(
                    "Unrecognized mode value. Please specify one of full, valid, same.");
        }
    }

    inline auto convolve(torch::Tensor x, torch::Tensor y, convolve_mode mode)
        -> torch::Tensor {
        using namespace torch::indexing;

        if (x.dim() == 0 || y.dim() == 0) {
            throw std::invalid_argument("convolve: x or y is zero-dimensional.");
        }
        auto ndims_x = x.dim();
        auto ndims_y = y.dim();
        if (ndims_x < ndims_y) {
            for (int i = 0; i < (ndims_y - ndims_x); i++) {
                x = x.unsqueeze(0);
            }
            ndims_x = ndims_y;
        } else if (ndims_y < ndims_x) {
            for (int i = 0; i < (ndims_x - ndims_y); i++) {
                y = y.unsqueeze(0);
            }
            ndims_y = ndims_x;
        }
        if (x.size(-1) < y.size(-1)) {
            std::swap(x, y);
        }
        auto x_size = x.size(-1);
        auto y_size = y.size(-1);
        auto leading_dims_count = ndims_x - 1;
        auto shape_x = x.sizes();
        auto shape_y = y.sizes();
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
        auto num_signals = 1LL;
        for (int i = 0; i < leading_dims_count; i++) {
            num_signals *= new_shape[i];
        }
        auto reshaped_x = x.reshape({num_signals, 1, x_size});
        auto reshaped_y = y.flip(-1).reshape({num_signals, 1, y_size});
        auto conv_out =
                torch::nn::functional::conv1d(reshaped_x, reshaped_y,
                                              torch::nn::functional::Conv1dFuncOptions()
                                              .stride(1)
                                              .groups(num_signals)
                                              .padding(y_size - 1));
        auto output_length = conv_out.size(-1);
        auto output_shape = new_shape;
        output_shape.push_back(output_length);
        auto result = conv_out.reshape(output_shape);
        return _apply_convolve_mode(result, x_size, y_size, mode);
    }

    // 1) 定义参数选项结构
    struct amplitude_to_db_option {
        float amin = 1e-10f;
        float top_db = 80.0f;
        float db_multiplier = 1.0f;
        bool apply_top_db = true;

        auto set_amin(float a) -> amplitude_to_db_option & {
            amin = a;
            return *this;
        }

        auto set_top_db(float t) -> amplitude_to_db_option & {
            top_db = t;
            return *this;
        }

        auto set_db_multiplier(float m) -> amplitude_to_db_option & {
            db_multiplier = m;
            return *this;
        }

        auto set_apply_top_db(bool b) -> amplitude_to_db_option & {
            apply_top_db = b;
            return *this;
        }
    };

    inline auto amplitude_to_DB(tensor_t signal, amplitude_to_db_option option = {})
        -> tensor_t {
        if (signal.is_complex()) {
            signal = signal.abs(); // 等价于 sqrt(real^2 + imag^2)
        }
        float amin_val = std::max(option.amin, std::numeric_limits<float>::min());
        auto power = torch::pow(signal, 2.0);
        auto db = 10.0 * torch::log10(torch::clamp(
                      power, amin_val, std::numeric_limits<float>::max()));
        db = db * option.db_multiplier;
        if (option.apply_top_db) {
            float max_db = db.max().item<float>();
            db = torch::max(db, torch::tensor(max_db - option.top_db, db.options()));
        }

        return db;
    }

    struct spectrogram_option {
        int _pad = 0;
        tensor_t _window = {};
        int _n_fft = 400;
        int _hop_length = 200;
        int _win_length = 400;
        float _power = 2.0;
        bool _normalized = false;
        std::string _normalize_method = "window"; // window, frame_length
        bool _center = true;
        std::string _pad_mode = "reflect";
        bool _onesided = true;
        bool _return_complex = true; // when true, power becomes optional;

        auto pad(int p) -> spectrogram_option & {
            _pad = p;
            return *this;
        }

        auto window(tensor_t w) -> spectrogram_option & {
            _window = w;
            return *this;
        }

        auto n_fft(int n) -> spectrogram_option & {
            _n_fft = n;
            return *this;
        }

        auto hop_length(int h) -> spectrogram_option & {
            _hop_length = h;
            return *this;
        }

        auto win_length(int w) -> spectrogram_option & {
            _win_length = w;
            return *this;
        }

        auto power(float p) -> spectrogram_option & {
            _power = p;
            return *this;
        }

        auto normalized(bool n) -> spectrogram_option & {
            _normalized = n;
            return *this;
        }

        auto normalize_method(const std::string &n) -> spectrogram_option & {
            _normalize_method = n;
            return *this;
        }

        auto center(bool c) -> spectrogram_option & {
            _center = c;
            return *this;
        }

        auto pad_mode(const std::string &p) -> spectrogram_option & {
            _pad_mode = p;
            return *this;
        }

        auto onesided(bool o) -> spectrogram_option & {
            _onesided = o;
            return *this;
        }

        auto return_complex(bool r) -> spectrogram_option & {
            _return_complex = r;
            return *this;
        }
    };

    inline auto spectrogram(tensor_t signal, spectrogram_option option)
        -> tensor_t {
        int pad_amount = option._pad;
        if (pad_amount > 0) {
            signal = torch::constant_pad_nd(signal, {pad_amount, pad_amount}, 0);
        }

        int n_fft = option._n_fft;
        int hop_length = option._hop_length;
        int win_length = option._win_length;
        tensor_t window =
                option._window.defined()
                    ? option._window
                    : torch::hann_window(win_length, torch::TensorOptions()
                                         .dtype(signal.dtype())
                                         .device(signal.device()));
        auto spec_f = torch::stft(signal, n_fft, hop_length, win_length, window,
                                  option._center, option._pad_mode,
                                  option._normalized, // 第八个参数
                                  option._onesided, // 第九个参数
                                  option._return_complex // 第十个参数
        );
        if (option._return_complex) {
            return spec_f;
        } else {
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
    }

    struct melspectrogram_option {
        int sample_rate = 16000;
        int n_fft = 400;
        int win_length = 400;
        int hop_length = 200;
        int pad = 0;

        double f_min = 0.0;
        double f_max = 0.0; // 若设为 0，后续可默认为 sample_rate/2
        int n_mels = 128;

        double power = 2.0;
        bool normalized = false;
        bool center = true;
        std::string pad_mode = "reflect";
        bool onesided = true;
        std::string norm = ""; // "slaney" 或空字符串等
        std::string mel_scale = "htk"; // "htk" 或 "slaney"
    };

    inline auto mel_filter_bank(int n_mels, double f_min, double f_max,
                                int sample_rate,
                                int n_stft_bins, // 通常 = n_fft/2 + 1
                                const std::string &norm, // "slaney" 或 ""
                                const std::string &mel_scale // "htk" / "slaney"
    ) -> torch::Tensor {
        using namespace torch::indexing;
        // 如果外部没指定 f_max 或给了 <=0，则默认设为 Nyquist 频率
        if (f_max <= 0.0) {
            f_max = sample_rate / 2.0;
        }

        // 一些辅助函数：赫兹转 mel，mel 转赫兹
        auto hz_to_mel = [&](double freq) {
            if (mel_scale == "slaney") {
                // Slaney 风格公式 (approx)
                return 2595.0 * std::log10(1.0 + freq / 700.0);
            } else {
                // HTK 风格
                return 2595.0 * std::log10(1.0 + freq / 700.0);
            }
        };
        auto mel_to_hz = [&](double mel) {
            if (mel_scale == "slaney") {
                return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
            } else {
                return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
            }
        };

        double mel_min = hz_to_mel(f_min);
        double mel_max = hz_to_mel(f_max);

        // 在 mel 坐标上等距取 n_mels+2 个点
        auto mels = torch::linspace(mel_min, mel_max, n_mels + 2);

        // 逐个转换回赫兹频率
        auto hz_points = mels.clone();
        for (int i = 0; i < hz_points.size(0); i++) {
            double mel_val = hz_points[i].item<double>();
            hz_points[i] = mel_to_hz(mel_val);
        }

        // 计算每个 STFT bin 对应的赫兹频率
        // freq_bin[i] = i * (sample_rate / 2) / (n_stft_bins - 1)
        auto bin_frequencies = torch::linspace(0, sample_rate / 2.0, n_stft_bins);

        // 创建 [n_mels, n_stft_bins] 的滤波器
        auto fb = torch::zeros({n_mels, n_stft_bins}, torch::kFloat);

        for (int m = 1; m <= n_mels; m++) {
            double left = hz_points[m - 1].item<double>();
            double center = hz_points[m].item<double>();
            double right = hz_points[m + 1].item<double>();

            for (int f = 0; f < n_stft_bins; f++) {
                double freq = bin_frequencies[f].item<double>();
                if (freq >= left && freq <= center) {
                    fb[m - 1][f] = float((freq - left) / (center - left));
                } else if (freq > center && freq <= right) {
                    fb[m - 1][f] = float((right - freq) / (right - center));
                } else {
                    fb[m - 1][f] = 0.0f;
                }
            }
        }

        // 如果 norm == "slaney"，需要对每个 mel 过滤器再做归一化 (按带宽)
        if (norm == "slaney") {
            for (int m = 0; m < n_mels; m++) {
                auto row = fb.index({m, Slice()});
                float enorm =
                        2.0f / (hz_points[m + 2].item<float>() - hz_points[m].item<float>());
                row.mul_(enorm);
            }
        }

        return fb;
    }

    inline auto mel_scale(torch::Tensor spec, torch::Tensor fb) -> torch::Tensor {
        auto sizes = spec.sizes();
        int64_t ndim = sizes.size();
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
        auto fb_t = fb.transpose(0, 1); // [freq, n_mels]

        // [batch, time, freq] matmul [freq, n_mels] => [batch, time, n_mels]
        auto mel_3d = torch::matmul(spec_3d, fb_t);

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

    inline auto melspectrogram(torch::Tensor waveform, melspectrogram_option opt)
        -> torch::Tensor {
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
        auto spec = spectrogram(waveform, sp_opt);

        // 3) 构建 mel filter bank
        int n_stft_bins = opt.n_fft / 2 + 1;
        auto fb = mel_filter_bank(opt.n_mels, opt.f_min, opt.f_max, opt.sample_rate,
                                  n_stft_bins, opt.norm,
                                  opt.mel_scale); // [n_mels, n_stft_bins]

        // 4) 做 mel-scale
        auto mel_specgram = mel_scale(spec, fb);
        // mel_specgram => [..., n_mels, time]

        return mel_specgram;
    }
} // namespace torchmedia::audio::functional
#endif // _LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
