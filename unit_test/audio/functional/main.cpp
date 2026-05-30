// Assertion-based tests for the audio functional layer (target: 100% coverage of _functional*.hpp).
// Golden values verified against torchaudio 2.5.1 (see gen_golden.py). Bug regression tests are RED
// against the pre-fix _functional.hpp and GREEN after task02's fixes.
#include <cmath>
#include <stdexcept>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "test_util.hpp"

using namespace torchmedia::audio::functional;
static const double PI = std::acos(-1.0);

// ---------------- convolve ----------------
static void test_convolve_full_length() {
    auto x = torch::tensor({1.f, 2.f, 3.f, 4.f}).reshape({1, 4});
    auto y = torch::tensor({1.f, 1.f}).reshape({1, 2});
    TM_CHECK(convolve(x, y, full).size(-1) == 5); // N + M - 1
}

// bug#4: 'same' returns the FIRST input's length; 'valid' = max-min+1 (x shorter than y)
static void test_convolve_modes() {
    auto x = torch::tensor({1.f, 2.f, 3.f}).reshape({1, 3});
    auto y = torch::tensor({1.f, 1.f, 1.f, 1.f, 1.f}).reshape({1, 5});
    TM_CHECK(convolve(x, y, full).size(-1) == 7);
    TM_CHECK(convolve(x, y, same).size(-1) == 3);  // first input length
    TM_CHECK(convolve(x, y, valid).size(-1) == 3); // 5 - 3 + 1
}

static void test_convolve_broadcast_and_errors() {
    // differing ndim -> dimension-alignment (unsqueeze) branches
    TM_CHECK(convolve(torch::randn({1, 10}), torch::randn({4}), full).size(-1) == 13); // unsqueeze y
    TM_CHECK(convolve(torch::randn({5}), torch::randn({1, 3}), full).size(-1) == 7);   // unsqueeze x

    // multi-dim leading dims with broadcasting
    auto x = torch::randn({2, 1, 10});
    auto y = torch::randn({1, 3, 4});
    auto out = convolve(x, y, full);
    TM_CHECK(out.size(0) == 2 && out.size(1) == 3);

    bool threw_0d = false;
    try {
        convolve(torch::tensor(1.0f), torch::tensor(1.0f), full);
    } catch (const std::invalid_argument &) {
        threw_0d = true;
    }
    TM_CHECK(threw_0d);

    bool threw_0d_y = false; // x is >=1D, y is 0D -> exercises the y.dim()==0 side of the ||
    try {
        (void) convolve(torch::tensor({1.f, 2.f}).reshape({1, 2}), torch::tensor(1.0f), full);
    } catch (const std::invalid_argument &) {
        threw_0d_y = true;
    }
    TM_CHECK(threw_0d_y);

    bool threw_mode = false;
    try {
        auto a = torch::tensor({1.f, 2.f}).reshape({1, 2});
        auto b = torch::tensor({1.f}).reshape({1, 1});
        (void) convolve(a, b, static_cast<convolve_mode>(99));
    } catch (const std::invalid_argument &) {
        threw_mode = true;
    }
    TM_CHECK(threw_mode);
}

// ---------------- db_to_amplitude (bug#3) ----------------
static void test_db_to_amplitude() {
    TM_CHECK_CLOSE(db_to_amplitude(torch::tensor(20.0f), 1.0f, 1.0f).item<float>(), 100.0, 1e-2);
    TM_CHECK_CLOSE(db_to_amplitude(torch::tensor(20.0f), 1.0f, 0.5f).item<float>(), 10.0, 1e-2);
    TM_CHECK_CLOSE(db_to_amplitude(torch::tensor(10.0f), 2.0f, 1.0f).item<float>(), 20.0, 1e-2);
}

// ---------------- amplitude_to_DB (bug#2) ----------------
static void test_amplitude_to_DB_power() {
    auto spec = torch::tensor({1.0f, 0.1f, 0.01f, 0.001f}).reshape({1, 4});
    auto opt = amplitude_to_db_option().set_db_multiplier(0.0f).set_top_db(80.0f);
    auto db = amplitude_to_DB(spec, opt);
    auto expected = torch::tensor({0.0f, -10.0f, -20.0f, -30.0f}).reshape({1, 4});
    TM_CHECK_TENSOR_CLOSE(db, expected, 1e-3, 1e-4);
}

static void test_amplitude_to_DB_magnitude_complex_flags() {
    // magnitude path (multiplier = 20): golden [0, -6.0206, -12.0412]
    auto mag = torch::tensor({1.0f, 0.5f, 0.25f}).reshape({1, 3});
    auto db = amplitude_to_DB(mag, amplitude_to_db_option().set_multiplier(20.0f).set_db_multiplier(0.0f));
    auto expected = torch::tensor({0.0f, -6.0206f, -12.0412f}).reshape({1, 3});
    TM_CHECK_TENSOR_CLOSE(db, expected, 1e-3, 1e-4);

    // complex input -> internal abs; apply_top_db=false path
    auto c = torch::complex(torch::tensor({3.0f, 0.0f}), torch::tensor({4.0f, 0.0f})).reshape({1, 2});
    auto dbc = amplitude_to_DB(c, amplitude_to_db_option().set_multiplier(20.0f).set_db_multiplier(0.0f).set_apply_top_db(false));
    TM_CHECK_CLOSE(dbc[0][0].item<float>(), 20.0f * std::log10(5.0f), 1e-2);

    // top_db clamp path: power [1, 1e-5] -> [0, -50] dB, clamp to max-30 = -30
    auto big = torch::tensor({1.0f, 1e-5f}).reshape({1, 2});
    auto db2 = amplitude_to_DB(big, amplitude_to_db_option().set_multiplier(10.0f).set_db_multiplier(0.0f).set_top_db(30.0f));
    TM_CHECK_CLOSE(db2[0][1].item<float>(), -30.0f, 1e-3);
}

// ---------------- spectrogram (bug#5/#6) ----------------
static void test_spectrogram_shape() {
    auto spec = spectrogram(torch::randn({1, 16000}), spectrogram_option().return_complex(true));
    TM_CHECK(spec.is_complex());
    TM_CHECK(spec.size(-2) == 201);
}

static void test_spectrogram_power_abs() {
    auto sig = torch::randn({1, 4000});
    auto win = torch::hann_window(400);
    auto c = spectrogram(sig, spectrogram_option().window(win).return_complex(true));
    auto p = spectrogram(sig, spectrogram_option().window(win).power(2.0).return_complex(false));
    TM_CHECK_TENSOR_CLOSE(p, c.abs().pow(2.0), 1e-3, 1e-4);
}

static void test_spectrogram_normalized() {
    auto sig = torch::randn({1, 4000});
    auto win = torch::hann_window(400);
    auto sp_n = spectrogram(sig, spectrogram_option().window(win).power(2.0).return_complex(false).normalized(true));
    auto sp_un =
            spectrogram(sig, spectrogram_option().window(win).power(2.0).return_complex(false).normalized(false));
    TM_CHECK_TENSOR_CLOSE(sp_n, sp_un / win.pow(2).sum(), 1e-3, 1e-4);
}

static void test_spectrogram_branches() {
    auto sig = torch::randn({1, 4000});
    // pad > 0 branch
    auto sp_pad = spectrogram(sig, spectrogram_option().pad(100).return_complex(true));
    TM_CHECK(sp_pad.size(-2) == 201);
    // default window (none set -> hann) branch
    auto sp_defwin = spectrogram(sig, spectrogram_option().power(2.0).return_complex(false));
    TM_CHECK(sp_defwin.size(-2) == 201);
    // frame_length normalization branch
    auto sp_fl = spectrogram(sig, spectrogram_option()
                                          .window(torch::hann_window(400))
                                          .power(2.0)
                                          .return_complex(false)
                                          .normalized(true)
                                          .normalize_method("frame_length"));
    TM_CHECK(sp_fl.size(-2) == 201);
    // normalized=true but an unrecognized method -> neither normalize branch taken
    auto sp_none = spectrogram(sig, spectrogram_option()
                                            .window(torch::hann_window(400))
                                            .power(2.0)
                                            .return_complex(false)
                                            .normalized(true)
                                            .normalize_method("none"));
    TM_CHECK(sp_none.size(-2) == 201);
}

// ---------------- mel (bug#1) ----------------
static void test_mel_filter_bank_slaney() {
    auto fb_sla = mel_filter_bank(64, 0.0, 8000.0, 16000, 201, "", "slaney");
    auto fb_htk = mel_filter_bank(64, 0.0, 8000.0, 16000, 201, "", "htk");
    TM_CHECK(!torch::allclose(fb_sla, fb_htk));
    TM_CHECK_CLOSE(fb_sla.sum().item<double>(), 194.677, 0.5); // torchaudio golden (norm=None)
}

static void test_mel_filter_bank_branches() {
    // f_max <= 0 -> default to Nyquist; norm="slaney" branch
    auto fb = mel_filter_bank(40, 0.0, 0.0, 16000, 201, "slaney", "htk");
    TM_CHECK(fb.size(0) == 40 && fb.size(1) == 201);
}

static void test_melspectrogram() {
    auto sig = torch::sin(2.0 * PI * 440.0 * torch::arange(16000, torch::kFloat32) / 16000.0).reshape({1, 16000});
    mel_spectrogram_option opt;
    opt.sample_rate = 16000;
    opt.n_fft = 512;
    opt.win_length = 512;
    opt.hop_length = 256;
    opt.n_mels = 64;
    opt.f_min = 0.0;
    opt.f_max = 8000.0;
    opt.power = 2.0;
    opt.mel_scale = "htk";
    opt.norm = "";
    auto mel = melspectrogram(sig, opt);
    TM_CHECK(mel.size(-2) == 64 && mel.size(-1) == 63);                          // golden shape (1,64,63)
    TM_CHECK_CLOSE(mel.sum().item<double>(), 1548267.875, 1548267.875 * 0.03);   // golden sum, 3% tol
}

// ---------------- option setters ----------------
static void test_option_setters() {
    auto so = spectrogram_option()
                      .pad(0)
                      .window(torch::hann_window(400))
                      .n_fft(400)
                      .hop_length(200)
                      .win_length(400)
                      .power(2.0)
                      .normalized(false)
                      .normalize_method("window")
                      .center(true)
                      .pad_mode("reflect")
                      .onesided(true)
                      .return_complex(false);
    TM_CHECK(so._n_fft == 400);
    auto ao = amplitude_to_db_option()
                      .set_multiplier(10.0f)
                      .set_amin(1e-10f)
                      .set_top_db(80.0f)
                      .set_db_multiplier(0.0f)
                      .set_apply_top_db(true);
    TM_CHECK(ao.multiplier == 10.0f);
}

int main() {
    test_convolve_full_length();
    test_convolve_modes();
    test_convolve_broadcast_and_errors();
    test_db_to_amplitude();
    test_amplitude_to_DB_power();
    test_amplitude_to_DB_magnitude_complex_flags();
    test_spectrogram_shape();
    test_spectrogram_power_abs();
    test_spectrogram_normalized();
    test_spectrogram_branches();
    test_mel_filter_bank_slaney();
    test_mel_filter_bank_branches();
    test_melspectrogram();
    test_option_setters();
    return tm_test::summary("audio_test_functional");
}
