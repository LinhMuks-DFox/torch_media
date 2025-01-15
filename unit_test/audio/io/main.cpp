#include <fmt/core.h>
#include <torch/torch.h>
#include <torchmedia/audio.hpp>
#include <torchmedia/basic.hpp>
int main() {
  fmt::print("hello torch!");
  auto t = torch::tensor({1, 2, 3});
  fmt::print("{}", torchmedia::basic::to_string(t));
  return 0;
}