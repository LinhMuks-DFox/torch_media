#include "c10/core/TensorOptions.h"
#include "fmt/base.h"
#include "torch/types.h"
#include "torchmedia/_audio/_functional.hpp"
#include "torchmedia/basic.hpp"
#include <fmt/core.h>
#include <torch/torch.h>
#include <torchmedia.hpp>
void test_spectrogram() {
  auto audio = torchmedia::audio::io::load_audio("dummy_audio_440Hz.wav").data;
  audio = audio.to(torch::TensorOptions{}.device("mps"));

  using torchmedia::audio::functional::amplitude_to_DB;
  using torchmedia::audio::functional::spectrogram;
  using torchmedia::audio::functional::spectrogram_option;
  auto spe = spectrogram(audio, spectrogram_option());
  auto db = amplitude_to_DB(spe);
  torchmedia::basic::print_tensor(db);
}

void test_convolve() {
  auto audio = torchmedia::audio::io::load_audio("dummy_audio_440Hz.wav")
                   .data.unsqueeze(0)
                   .unsqueeze(0);
  audio = audio.to(torch::TensorOptions{}.device("mps"));
  using torchmedia::audio::functional::convolve;
  using torchmedia::audio::functional::convolve_mode;
  using torchmedia::basic::to_string;

  auto kernel =
      torch::tensor({0, 1, 0, 1},
                    torch::TensorOptions().dtype(torch::kFloat32).device("mps"))
          .unsqueeze(0)
          .unsqueeze(0);

  fmt::print("Audio shape: {}\n, kernel shape:{}\n", to_string(audio.sizes()),
             to_string(kernel.sizes()));

  auto convolved = convolve(audio, kernel, convolve_mode::full);
  torchmedia::basic::print_tensor(convolved);
}
int main() {
  test_spectrogram();
  return 0;
}