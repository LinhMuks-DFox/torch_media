#pragma once
#include "ATen/core/TensorBody.h"
#include "c10/util/ArrayRef.h"
#include "fmt/core.h"
#include <ostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#ifndef LIBTORCH_MEDIA_BASIC
#define LIBTORCH_MEDIA_BASIC
namespace torchmedia::basic {
auto inline to_device(torch::Tensor tensor, const std::string &device)
    -> torch::Tensor {
  return tensor.to(torch::TensorOptions().device(device));
};

auto inline to_string(const torch::Tensor &obj) -> std::string {
  std::stringstream ss;
  ss << obj << std::flush;
  return ss.str();
}
auto inline to_string(const torch::IntArrayRef &t) -> std::string {
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < static_cast<int>(t.size()); ++i) {
    ss << t[i];
    if (i + 1 < static_cast<int>(t.size())) {
      ss << ", ";
    }
  }
  ss << "]" << std::flush;
  return ss.str();
}

auto inline print_tensor(const torch::Tensor &t) -> void {
  fmt::print("{}", to_string(t));
}

} // namespace torchmedia::basic
#endif