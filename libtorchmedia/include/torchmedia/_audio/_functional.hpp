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
                            option._normalized,    // 第八个参数
                            option._onesided,      // 第九个参数
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
} // namespace torchmedia::audio::functional
#endif // _LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP