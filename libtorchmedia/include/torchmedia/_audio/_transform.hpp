#pragma once
#ifndef _LIB_TORCH_MEDIA_AUDIO_TRANSFOMR_HPP
#define _LIB_TORCH_MEDIA_AUDIO_TRANSFOMR_HPP
#include "_functional.hpp"
#include <ATen/core/TensorBody.h>
#include <c10/core/TensorImpl.h>
#include <sox.h>
#include <torch/nn/module.h>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>
namespace audio4torch::transform {

class Spectrogram : public torch::nn::Module {
private:
  torchmedia::audio::functional::spectrogram_option option;

public:
  auto forward(torch::Tensor signal) -> torch::Tensor {
    return torchmedia::audio::functional::spectrogram(signal, this->option);
  }
};

} // namespace audio4torch::transform
#endif // _LIB_TORCH_MEDIA_AUDIO_TRANSFOMR_HPP