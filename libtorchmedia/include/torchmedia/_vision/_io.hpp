#pragma once
#ifndef LIB_TORCH_MEDIA_VISION_IO_HPP
#define LIB_TORCH_MEDIA_VISION_IO_HPP

#include <filesystem>
#include <stdexcept>
#include <string>
#include <torch/torch.h>

#include "../globel_include.hpp"

// stb_image / stb_image_write: header-only image codecs (public domain / MIT). The implementation is
// emitted in the one TU that defines TORCHMEDIA_IO_IMPLEMENTATION (same macro as dr_wav), keeping the
// library header-only while avoiding ODR violations. See develop_log/2026-05-31/progress04.
#ifdef TORCHMEDIA_IO_IMPLEMENTATION
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#endif
#include "_vendor/stb_image.h"
#include "_vendor/stb_image_write.h"

namespace torchmedia::vision::io {

    // Load an image into a [C, H, W] float32 tensor in [0, 1].
    inline auto load_image(const std::string &path) -> tensor_t {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("File does not exist: " + path);
        }
        int w = 0, h = 0, c = 0;
        unsigned char *data = stbi_load(path.c_str(), &w, &h, &c, 0);
        if (data == nullptr) {
            throw std::runtime_error("Could not load image: " + path);
        }
        // stb gives [H, W, C] uint8 row-major; -> float [0,1] -> [C, H, W]. permute+contiguous copies out
        // of the stb buffer, so it is safe to free immediately afterwards.
        const auto hwc = torch::from_blob(data, {h, w, c}, torch::kUInt8).to(torch::kFloat32) / 255.0;
        auto chw = hwc.permute({2, 0, 1}).contiguous();
        stbi_image_free(data);
        return chw;
    }

    // Save a [C, H, W] float tensor (values in [0,1]) as a PNG. Returns false on bad input / IO error.
    inline auto save_image(const tensor_t &img, const std::string &path) -> bool {
        if (img.dim() != 3) {
            return false;
        }
        const auto c = img.size(0);
        const auto h = img.size(1);
        const auto w = img.size(2);
        // [C,H,W] float [0,1] -> [H,W,C] uint8 (clamp, scale, round).
        const auto hwc = (img.detach().to(torch::kCPU).clamp(0.0, 1.0) * 255.0 + 0.5)
                                 .to(torch::kUInt8)
                                 .permute({1, 2, 0})
                                 .contiguous();
        const int stride = static_cast<int>(w * c);
        const int ok = stbi_write_png(path.c_str(), static_cast<int>(w), static_cast<int>(h),
                                      static_cast<int>(c), hwc.data_ptr<uint8_t>(), stride);
        return ok != 0;
    }

    inline auto load_image(const char *path) -> tensor_t { return load_image(std::string(path)); }
    inline auto load_image(const std::filesystem::path &path) -> tensor_t { return load_image(path.string()); }
    inline auto save_image(const tensor_t &img, const char *path) -> bool {
        return save_image(img, std::string(path));
    }
    inline auto save_image(const tensor_t &img, const std::filesystem::path &path) -> bool {
        return save_image(img, path.string());
    }
} // namespace torchmedia::vision::io

#endif // LIB_TORCH_MEDIA_VISION_IO_HPP
