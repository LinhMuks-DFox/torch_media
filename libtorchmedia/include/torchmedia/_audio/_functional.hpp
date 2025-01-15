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

inline auto convolve(tensor_t x, tensor_t y, convolve_mode mode) -> tensor_t {
  /*
      x (tensor_t): First convolution operand, with shape `(..., N)`.
      y (tensor_t): Second convolution operand, with shape `(..., M)`
          (leading dimensions must be broadcast-able with those of ``x``).
      mode (str, optional): Must be one of ("full", "valid", "same").

          * "full": Returns the full convolution result, with shape `(..., N + M
     - 1)`. (Default)
          * "valid": Returns the segment of the full convolution result
     corresponding to where the two inputs overlap completely, with shape `(...,
     max(N, M) - min(N, M) + 1)`.
          * "same": Returns the center segment of the full convolution result,
     with shape `(..., N)`.
  */
  if (!_check_shape_compatible(x, y)) {
    throw std::invalid_argument("Shapes are not compatible for convolution.");
  }

  auto x_size = x.size(-1);
  auto y_size = y.size(-1);
  if (x.size(-1) < y.size(-1)) {
    std::swap(x, y);
  }

  if (x.sizes().slice(0, -1) != y.sizes().slice(0, -1)) {
    auto new_shape = x.sizes().slice(0, -1).vec();
    for (size_t i = 0; i < new_shape.size(); i++) {
      new_shape[i] = std::max(x.size(i), y.size(i));
    }
    new_shape.push_back(x.size(-1));
    x = x.broadcast_to(new_shape);
    new_shape.pop_back();
    new_shape.push_back(y.size(-1));
    y = y.broadcast_to(new_shape);
  }
  auto num_signals =
      torch::tensor(x.sizes().slice(0, -1)).prod().item<int64_t>();
  auto reshaped_x = x.reshape({num_signals, x.size(-1)});
  auto reshaped_y = y.reshape({num_signals, y.size(-1)});
  auto output = torch::nn::functional::conv1d(
      reshaped_x, reshaped_y.flip(-1).unsqueeze(1),
      torch::nn::functional::Conv1dFuncOptions()
          .stride(1)
          .groups(reshaped_x.size(0))
          .padding(reshaped_y.size(-1) - 1));
  auto output_shape = x.sizes().slice(0, -1).vec();
  output_shape.push_back(-1);
  auto result = output.reshape(output_shape);
  return _apply_convolve_mode(result, x_size, y_size, mode);
}

inline auto amplitude_to_DB(tensor_t signal, float amin, float db_multiplier,
                            float topdb, bool apply_topdb) -> tensor_t {
  // Ensure amin is greater than 0 to avoid log of zero
  amin = std::max(amin, std::numeric_limits<float>::min());

  // Convert amplitude to power
  tensor_t power = torch::pow(signal, 2.0);

  // Apply logarithm
  tensor_t db = 10.0 * torch::log10(torch::clamp(
                           power, amin, std::numeric_limits<float>::max()));

  // Apply multiplier
  db = db * db_multiplier;

  // Apply top_db limit if needed
  if (apply_topdb) {
    float max_db = db.max().item<float>();
    db = torch::max(db, torch::tensor(max_db - topdb));
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
  bool _return_complex = false; // when true, power becomes optional;

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
          : torch::hann_window(win_length,
                               torch::TensorOptions().dtype(signal.dtype()));

  auto spec_f =
      torch::stft(signal, n_fft, hop_length, win_length, window, option._center,
                  option._pad_mode, option._onesided, option._return_complex);

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