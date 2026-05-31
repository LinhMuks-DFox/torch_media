#pragma once
#ifndef LIB_TORCH_MEDIA_VISION_FUNCTIONAL_HPP
#define LIB_TORCH_MEDIA_VISION_FUNCTIONAL_HPP
#include <vector>
#include <torch/torch.h>
#include "../globel_include.hpp"

// Image tensors follow torchvision's [..., C, H, W] layout, float in [0, 1].
namespace torchmedia::vision::functional {

    inline auto hflip(const tensor_t &img) -> tensor_t { return img.flip(-1); }

    inline auto vflip(const tensor_t &img) -> tensor_t { return img.flip(-2); }

    // RGB -> grayscale: L = 0.2989 R + 0.587 G + 0.114 B (ITU-R 601-2 luma, torchvision weights).
    inline auto rgb_to_grayscale(const tensor_t &img, int num_output_channels = 1) -> tensor_t {
        const auto r = img.select(-3, 0);
        const auto g = img.select(-3, 1);
        const auto b = img.select(-3, 2);
        auto l = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(-3); // [..., 1, H, W]
        if (num_output_channels == 3) {
            std::vector<int64_t> reps(l.dim(), 1);
            reps[l.dim() - 3] = 3;
            l = l.repeat(reps);
        }
        return l;
    }

    // Per-channel standardization: (img - mean) / std.
    inline auto normalize(const tensor_t &img, const std::vector<double> &mean, const std::vector<double> &stddev)
            -> tensor_t {
        const std::vector<float> mf(mean.begin(), mean.end());
        const std::vector<float> sf(stddev.begin(), stddev.end());
        const auto m = torch::tensor(mf, img.options()).reshape({-1, 1, 1});
        const auto s = torch::tensor(sf, img.options()).reshape({-1, 1, 1});
        return (img - m) / s;
    }

    // Center crop (assumes height/width <= image size; oversize padding is a Tier-2 follow-up).
    inline auto center_crop(const tensor_t &img, int64_t height, int64_t width) -> tensor_t {
        const auto h = img.size(-2);
        const auto w = img.size(-1);
        const auto top = (h - height) / 2;
        const auto left = (w - width) / 2;
        return img.slice(-2, top, top + height).slice(-1, left, left + width);
    }

    // Brightness: blend toward black by `factor`, then clamp to [0, 1].
    inline auto adjust_brightness(const tensor_t &img, double factor) -> tensor_t {
        return (img * factor).clamp(0.0, 1.0);
    }

    // Invert colors (float bound 1.0).
    inline auto invert(const tensor_t &img) -> tensor_t { return 1.0 - img; }

} // namespace torchmedia::vision::functional
#endif // LIB_TORCH_MEDIA_VISION_FUNCTIONAL_HPP
