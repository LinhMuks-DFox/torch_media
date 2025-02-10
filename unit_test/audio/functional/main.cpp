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
    torchmedia::util::print_tensor(db);

    torch::save(db.detach().cpu(), "db.pt");
}

void test_melspectrogram() {
    auto audio = torchmedia::audio::io::load_audio("dummy_audio_440Hz.wav").data;
    audio = audio.to(torch::TensorOptions{}.device("mps"));
    using torchmedia::audio::functional::amplitude_to_DB;
    using torchmedia::audio::functional::mel_spectrogram_option;
    using torchmedia::audio::functional::melspectrogram;
    using torchmedia::util::print_tensor;
    mel_spectrogram_option mel_opt;
    mel_opt.sample_rate = 16000;
    mel_opt.n_fft = 512;
    mel_opt.win_length = 512;
    mel_opt.hop_length = 256;
    mel_opt.n_mels = 128;
    mel_opt.f_min = 0.0;
    mel_opt.f_max = 8000.0;
    mel_opt.power = 2.0;

    auto mel_spec = melspectrogram(audio, mel_opt);
    auto mel_db = amplitude_to_DB(mel_spec);
    fmt::print("MelSpectrogram shape: {}\n", torchmedia::util::to_string(mel_db.sizes()));
    print_tensor(mel_db);
}

void test_convolve() {
    auto audio = torchmedia::audio::io::load_audio("dummy_audio_440Hz.wav").data.unsqueeze(0).unsqueeze(0);
    audio = audio.to(torch::TensorOptions{}.device("mps"));
    using torchmedia::audio::functional::convolve;
    using torchmedia::audio::functional::convolve_mode;
    using torchmedia::util::to_string;

    const auto kernel = torch::tensor({0, 1, 0, 1}, torch::TensorOptions().dtype(torch::kFloat32).device("mps"))
                                .unsqueeze(0)
                                .unsqueeze(0);

    fmt::print("Audio shape: {}\n, kernel shape:{}\n", to_string(audio.sizes()), to_string(kernel.sizes()));

    auto convolved = convolve(audio, kernel, convolve_mode::full);
    torchmedia::util::print_tensor(convolved);
}

int main() {
    test_spectrogram();
    return 0;
}
