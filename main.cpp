#include "fmt/base.h"
#include <audio.hpp>
#include <basic.hpp>
#include <fmt/core.h>
#include <torch/torch.h>
int main() {
  fmt::print("hello torch!");
  auto t = torch::tensor({1, 2, 3});
  fmt::print("{}", torchmedia::to_string(t));
  return 0;
}