#pragma once
#ifndef _LIB_TORCH_MEDIA_AUDIO_IO_HPP
#define _LIB_TORCH_MEDIA_AUDIO_IO_HPP
#include <ATen/core/TensorBody.h>
#include <c10/core/TensorImpl.h>
#include <sox.h>
#include <stdexcept>
#include <string>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>
#include <vector>

namespace torchmedia::audio::io {
struct load_audio_t {
  torch::Tensor data;
  int sample_rate;
};

auto inline load_audio(const std::string &path) -> load_audio_t {
  // Initialize SoX library
  if (sox_init() != SOX_SUCCESS) {
    throw std::runtime_error("Failed to initialize SoX");
  }

  sox_format_t *format = sox_open_read(path.c_str(), nullptr, nullptr, nullptr);
  if (!format) {
    sox_quit();
    throw std::runtime_error("Failed to open audio file");
  }

  int sample_rate = format->signal.rate;
  size_t num_samples = format->signal.length;
  int num_channels = format->signal.channels;

  // Calculate the total number of samples to read
  size_t total_samples = num_samples * num_channels;

  std::vector<sox_sample_t> samples(total_samples);
  size_t samples_read = sox_read(format, samples.data(), total_samples);
  if (samples_read != total_samples) {
    sox_close(format);
    sox_quit();
    throw std::runtime_error("Failed to read audio samples");
  }

  sox_close(format);
  sox_quit();

  // Convert samples to float
  // Create a torch tensor from the samples
  auto options = torch::TensorOptions().dtype(torch::kInt32);
  torch::Tensor tensor =
      torch::from_blob(samples.data(), {static_cast<int64_t>(samples_read)},
                       options)
          .clone();

  // Convert tensor to float and normalize
  tensor = tensor.to(torch::kFloat32) / static_cast<float>(SOX_SAMPLE_MAX);

  // Reshape the tensor to [num_channels, num_samples_per_channel] if necessary
  if (num_channels > 1) {
    tensor = tensor
                 .view({static_cast<int64_t>(num_channels),
                        static_cast<int64_t>(samples_read / num_channels)})
                 .contiguous();
  }

  return {tensor, sample_rate};
}

#define LOAD_AUDIO(path) load_audio(path).data

auto inline save_audio(torch::Tensor audio_tensor, const std::string &path,
                       int sample_rate, int sample_depth) -> bool {
  // Initialize SoX library
  if (sox_init() != SOX_SUCCESS) {
    throw std::runtime_error("Failed to initialize SoX");
  }

  // Set up SoX format parameters
  sox_signalinfo_t signal;
  signal.rate = sample_rate;
  signal.channels = audio_tensor.size(
      0); // Assuming audio_tensor is of shape [num_channels, num_samples]
  signal.precision = sample_depth;
  signal.length = 0;
  signal.mult = nullptr;

  sox_encodinginfo_t encoding;
  encoding.encoding = SOX_ENCODING_SIGN2;
  encoding.bits_per_sample = sample_depth;
  encoding.compression = 0;
  encoding.reverse_bytes = sox_option_default;
  encoding.reverse_nibbles = sox_option_default;
  encoding.reverse_bits = sox_option_default;
  encoding.opposite_endian = sox_false;

  // Open the output file
  sox_format_t *format = sox_open_write(path.c_str(), &signal, &encoding,
                                        nullptr, nullptr, nullptr);
  if (!format) {
    sox_quit();
    throw std::runtime_error("Failed to open output audio file");
  }

  // Convert tensor to sox_sample_t and denormalize
  auto float_tensor = audio_tensor * static_cast<float>(SOX_SAMPLE_MAX);
  auto int_tensor = float_tensor.to(torch::kInt32);
  std::vector<sox_sample_t> samples(int_tensor.numel());
  std::memcpy(samples.data(), int_tensor.data_ptr<int32_t>(),
              samples.size() * sizeof(int32_t));

  // Write samples to the output file
  size_t samples_written = sox_write(format, samples.data(), samples.size());
  if (samples_written != samples.size()) {
    sox_close(format);
    sox_quit();
    throw std::runtime_error("Failed to write audio samples");
  }

  sox_close(format);
  sox_quit();

  return true;
}
} // namespace torchmedia::audio::io
#endif // _LIB_TORCH_MEDIA_AUDIO_IO_HPP