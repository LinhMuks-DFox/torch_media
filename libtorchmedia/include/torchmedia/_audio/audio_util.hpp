//
// Created by Mux on 25-1-16.
//

#ifndef LIB_TORCH_MEDIA_AUDIO_UTIL_HPP
#define LIB_TORCH_MEDIA_AUDIO_UTIL_HPP
#include "../globel_include.hpp"

namespace torchmedia::audio {
    enum SignalWiseCompareMode {
        MSE, MAE, SNR
    };

    auto inline signal_wise_almost_equal(const torch::Tensor &left, const torch::Tensor &right, float threshold = 1e-7,
                                         SignalWiseCompareMode mode = SignalWiseCompareMode::MSE) -> bool {
        if (left.sizes() != right.sizes()) {
            return false; // 形状不匹配，直接返回 false
        }

        torch::Tensor compared;
        if (mode == SignalWiseCompareMode::MSE) {
            compared = torch::nn::functional::mse_loss(left, right,
                                                       torch::nn::functional::MSELossFuncOptions().reduction(
                                                           torch::kMean));
        } else if (mode == SignalWiseCompareMode::SNR) {
            auto signal_power = left.pow(2).sum();
            auto noise_power = (left - right).pow(2).sum();
            if (noise_power.item<float>() == 0.0f) {
                return true;
            }
            auto snr = 10 * torch::log10(signal_power / noise_power);
            return snr.item<float>() > threshold;
        } else {
            compared = torch::nn::functional::l1_loss(left, right,
                                                      torch::nn::functional::L1LossFuncOptions().
                                                      reduction(torch::kMean));
        }

        return compared.item<float>() < threshold;
    }
}
#endif //LIB_TORCH_MEDIA_AUDIO_UTIL_HPP
