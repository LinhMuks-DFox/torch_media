#pragma once
#include <filesystem>
#include <stdexcept>
#ifndef LIB_TORCH_MEDIA_AUDIO_IO_HPP
#define LIB_TORCH_MEDIA_AUDIO_IO_HPP
#include <sox.h>
#include "../globel_include.hpp"
namespace torchmedia::audio::io {
    struct load_audio_t {
        tensor_t data;
        int sample_rate;
    };

#ifdef SOX_H
    auto inline sox_backend_load_audio(const std::string &path) -> load_audio_t {

        if (std::filesystem::exists(path)) {
            handle_exceptions<load_audio_t, std::runtime_error>(std::string("File dose not exist: ") + path);
        }
        if (sox_init() != SOX_SUCCESS) {
            handle_exceptions<load_audio_t, std::runtime_error>(std::string("Fail to init sox: ") + path);
        }

        sox_format_t *format = sox_open_read(path.c_str(), nullptr, nullptr, nullptr);
        if (!format) {
            sox_quit();
            handle_exceptions<load_audio_t, std::runtime_error>(std::string("Can not open file: ") + path);
        }

        const int sample_rate = format->signal.rate;
        const size_t num_samples = format->signal.length;
        const int num_channels = format->signal.channels;
        const size_t total_samples = num_samples * num_channels;

        std::vector<sox_sample_t> samples(total_samples);
        size_t samples_read = sox_read(format, samples.data(), total_samples);
        if (samples_read != total_samples) {
            sox_close(format);
            sox_quit();
            handle_exceptions<load_audio_t, std::runtime_error>(std::string("Could not load audio file ") + path);
        }

        sox_close(format);
        sox_quit();
        const auto options = tensor_options_t().dtype(torch::kInt32);
        tensor_t tensor = torch::from_blob(samples.data(), {static_cast<int64_t>(samples_read)}, options).clone();
        tensor = tensor.to(torch::kFloat32) / static_cast<float>(SOX_SAMPLE_MAX);
        if (num_channels > 1) {
            tensor =
                    tensor.view({static_cast<int64_t>(num_channels), static_cast<int64_t>(samples_read / num_channels)})
                            .contiguous();
        }

        return {tensor.contiguous(), sample_rate};
    }

    auto inline sox_backend_save_audio(tensor_t audio_tensor, const std::string &path, const int sample_rate,
                                       const int sample_depth) -> bool {
        // Initialize SoX library
        if (sox_init() != SOX_SUCCESS) {
            return false;
        }

        // Set up SoX format parameters
        sox_signalinfo_t signal;
        signal.rate = sample_rate;
        signal.channels = audio_tensor.size(0); // Assuming audio_tensor is of shape [num_channels, num_samples]
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
        sox_format_t *format = sox_open_write(path.c_str(), &signal, &encoding, nullptr, nullptr, nullptr);
        if (!format) {
            sox_quit();
            return false;
        }

        // Convert tensor to sox_sample_t and denormalize
        const auto float_tensor = audio_tensor * static_cast<float>(SOX_SAMPLE_MAX);
        const auto int_tensor = float_tensor.to(torch::kInt32);
        std::vector<sox_sample_t> samples(int_tensor.numel());
        std::memcpy(samples.data(), int_tensor.data_ptr<int32_t>(), samples.size() * sizeof(int32_t));

        // Write samples to the output file
        size_t samples_written = sox_write(format, samples.data(), samples.size());
        if (samples_written != samples.size()) {
            sox_close(format);
            sox_quit();
            return false;
        }

        sox_close(format);
        sox_quit();

        return true;
    }

#define LOAD_AUDIO_BACK_END sox_backend_load_audio
#define SAVE_AUDIO_BACK_END sox_backend_save_audio
#endif

#ifndef LOAD_AUDIO_BACK_END
#error "No backend detected, libtorchmedia requires sox"
#endif
    auto inline load_audio(const std::string &path) -> load_audio_t { return LOAD_AUDIO_BACK_END(path); }

    auto inline load_audio(std::filesystem::path &path) -> load_audio_t { return LOAD_AUDIO_BACK_END(path.string()); }

    auto inline save_audio(tensor_t audio_tensor, const std::string &path, const int sample_rate,
                           const int sample_depth) -> bool {
        return SAVE_AUDIO_BACK_END(audio_tensor, path, sample_rate, sample_depth);
    }

    auto inline save_audio(tensor_t audio_tensor, const std::filesystem::path &path, const int sample_rate,
                           const int sample_depth) -> bool {
        return SAVE_AUDIO_BACK_END(audio_tensor, path.string(), sample_rate, sample_depth);
    }
} // namespace torchmedia::audio::io
#undef LOAD_AUDIO_BACK_END
#undef SAVE_AUDIO_BACK_END
#endif // LIB_TORCH_MEDIA_AUDIO_IO_HPP
