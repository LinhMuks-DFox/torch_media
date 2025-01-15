#pragma once
#include <sstream>
#include <string>
#ifndef LIBTORCH_MEDIA_BASIC
#define LIBTORCH_MEDIA_BASIC
namespace torchmedia::basic {
template <class TorchData>
auto inline to_string(const TorchData &obj) -> std::string {
  std::stringstream ss;
  ss << obj << std::flush;
  return ss.str();
}
} // namespace torchmedia::basic
#endif