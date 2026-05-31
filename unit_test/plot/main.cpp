// Smoke test for torchmedia::plot (matplot++ / gnuplot). Verifies the calls succeed and produce files.
// gnuplot writes asynchronously, so we issue all saves first, then poll for the output files.
#include <chrono>
#include <cmath>
#include <filesystem>
#include <thread>
#include <torch/torch.h>
#include <torchmedia/plot.hpp>
#include "test_util.hpp"

static bool wait_for_file(const std::string &p) {
    for (int i = 0; i < 200; ++i) { // up to ~10s
        if (std::filesystem::exists(p) && std::filesystem::file_size(p) > 0)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

int main() {
    namespace plot = torchmedia::plot;
    const double pi = std::acos(-1.0);
    auto wav = torch::sin(2.0 * pi * 5.0 * torch::arange(500, torch::kFloat32) / 500.0).reshape({1, 500});
    auto spec = torch::rand({64, 40});

    // convenience wrappers
    plot::save_waveform(wav, "plot_waveform.png");
    plot::save_spectrogram(spec, "plot_spectrogram.png");
    // Plotter chaining API
    plot::Plotter().heatmap(torch::rand({32, 20})).colorbar().title("chain").save("plot_chain_heatmap.png");
    plot::Plotter().waveform(wav).title("wave").xlabel("n").ylabel("x").save("plot_chain_wave.png");

    // gnuplot is async: poll for each output file after all saves are issued.
    TM_CHECK(wait_for_file("plot_waveform.png"));
    TM_CHECK(wait_for_file("plot_spectrogram.png"));
    TM_CHECK(wait_for_file("plot_chain_heatmap.png"));
    TM_CHECK(wait_for_file("plot_chain_wave.png"));

    return tm_test::summary("plot_test");
}
