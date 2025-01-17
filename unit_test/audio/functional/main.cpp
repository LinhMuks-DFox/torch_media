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
    // 1. 加载音频，移动到 MPS (若无 MPS 可改成 CPU / CUDA)
    auto audio = torchmedia::audio::io::load_audio("dummy_audio_440Hz.wav").data;
    audio = audio.to(torch::TensorOptions{}.device("mps"));

    // 2. 引入相关命名空间 (模拟你在 test_spectrogram 里做的方式)
    using torchmedia::audio::functional::amplitude_to_DB;
    using torchmedia::audio::functional::melspectrogram;
    using torchmedia::audio::functional::mel_spectrogram_option;
    using torchmedia::util::print_tensor;

    // 3. 创建 MelSpectrogram 配置
    mel_spectrogram_option mel_opt;
    mel_opt.sample_rate = 16000; // 或按实际采样率改
    mel_opt.n_fft = 512;
    mel_opt.win_length = 512;
    mel_opt.hop_length = 256;
    mel_opt.n_mels = 128; // Mel 滤波器数
    mel_opt.f_min = 0.0; // 最低频率
    mel_opt.f_max = 8000.0; // 若不指定则默认 Nyquist=sample_rate/2
    mel_opt.power = 2.0; // 做功率谱
    // 其它参数如 normalized/center/pad_mode 可按需修改

    // 4. 调用 melspectrogram 得到 mel 频谱
    auto mel_spec = melspectrogram(audio, mel_opt);

    // 5. （可选）再把 Mel 频谱转换到 dB
    auto mel_db = amplitude_to_DB(mel_spec);

    // 6. 打印结果形状或局部数值
    fmt::print("MelSpectrogram shape: {}\n",
               torchmedia::util::to_string(mel_db.sizes()));
    print_tensor(mel_db); // 如只想看简单概览，可改成 print_tensor_info()
}

void test_convolve() {
    auto audio = torchmedia::audio::io::load_audio("dummy_audio_440Hz.wav")
            .data.unsqueeze(0)
            .unsqueeze(0);
    audio = audio.to(torch::TensorOptions{}.device("mps"));
    using torchmedia::audio::functional::convolve;
    using torchmedia::audio::functional::convolve_mode;
    using torchmedia::util::to_string;

    const auto kernel =
            torch::tensor({0, 1, 0, 1},
                          torch::TensorOptions().dtype(torch::kFloat32).device("mps"))
            .unsqueeze(0)
            .unsqueeze(0);

    fmt::print("Audio shape: {}\n, kernel shape:{}\n", to_string(audio.sizes()),
               to_string(kernel.sizes()));

    auto convolved = convolve(audio, kernel, convolve_mode::full);
    torchmedia::util::print_tensor(convolved);
}

int main() {
    test_spectrogram();
    return 0;
}
