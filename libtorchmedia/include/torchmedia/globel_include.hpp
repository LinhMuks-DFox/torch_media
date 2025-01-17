#ifndef LIB_TORCH_MEDIA_GLOBEL_INCLUDE_HPP
#define LIB_TORCH_MEDIA_GLOBEL_INCLUDE_HPP
#include <filesystem>
#include <string>
#include <vector>
#include <stdexcept>

#include <torch/torch.h>
#include <fmt/core.h>

#include "util.hpp"
#define LIB_TORCHMEDIA_CHECK_FILE_EXISTS(path) std::filesystem::exists(path)
namespace torchmedia {
    using tensor_t = torch::Tensor;
    using const_tensor_lref_t = const tensor_t&;
    using tensor_rref_t = tensor_t&&;
    using tensor_options_t = torch::TensorOptions;
    using str_t = std::string;
    using path_t = std::filesystem::path;

}
#endif //LIB_TORCH_MEDIA_GLOBEL_INCLUDE_HPP
