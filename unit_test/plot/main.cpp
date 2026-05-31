// Smoke test for torchmedia::plot (matplot++ / gnuplot). Verifies the calls succeed and produce files.
#include <cmath>
#include <filesystem>
#include <torch/torch.h>
#include <torchmedia/plot.hpp>
#include "test_util.hpp"

int main() {
    namespace plot = torchmedia::plot;
    const double pi = std::acos(-1.0);

    auto wav = torch::sin(2.0 * pi * 5.0 * torch::arange(500, torch::kFloat32) / 500.0).reshape({1, 500});
    plot::save_waveform(wav, "plot_waveform.png");
    TM_CHECK(std::filesystem::exists("plot_waveform.png"));

    auto spec = torch::rand({64, 40});
    plot::save_spectrogram(spec, "plot_spectrogram.png");
    TM_CHECK(std::filesystem::exists("plot_spectrogram.png"));

    return tm_test::summary("plot_test");
}
