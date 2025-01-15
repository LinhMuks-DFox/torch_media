#include "torchmedia/basic.hpp"
#include <fmt/core.h>
#include <torch/torch.h>
#include <torchmedia.hpp>
int main() {
  auto audio = torchmedia::audio::io::load_audio("dummy_audio_440Hz.wav").data;
  audio = torchmedia::basic::to_device(audio, "mps");
  torchmedia::basic::print_tensor(audio);

  torch::save(audio, "audio.pt");
  return 0;
}