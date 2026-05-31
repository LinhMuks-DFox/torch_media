// Image I/O tests (stb_image): round-trip + error/overload branches.
#define TORCHMEDIA_IO_IMPLEMENTATION
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "test_util.hpp"

using namespace torchmedia::vision;

static void test_roundtrip() {
    auto img = torch::arange(3 * 8 * 8, torch::kFloat32).reshape({3, 8, 8}) / static_cast<double>(3 * 8 * 8);
    const std::string path = "roundtrip_img.png";
    TM_CHECK(io::save_image(img, path));
    auto loaded = io::load_image(path);
    TM_CHECK(loaded.size(0) == 3 && loaded.size(1) == 8 && loaded.size(2) == 8);
    const float max_err = (loaded - img).abs().max().item<float>();
    TM_CHECK(max_err < (1.0f / 255.0f) + 1e-4f); // uint8 quantization
}

static void test_io_branches() {
    // non-existent file -> throws
    bool threw_missing = false;
    try {
        (void) io::load_image("nonexistent_xyz.png");
    } catch (const std::runtime_error &) {
        threw_missing = true;
    }
    TM_CHECK(threw_missing);

    // non-image file -> stbi_load returns null -> throws
    { std::ofstream f("not_an_image.png"); f << "not a png"; }
    bool threw_bad = false;
    try {
        (void) io::load_image("not_an_image.png");
    } catch (const std::runtime_error &) {
        threw_bad = true;
    }
    TM_CHECK(threw_bad);

    // wrong dim -> false
    TM_CHECK(io::save_image(torch::zeros({8}), "bad.png") == false);

    // const char* and std::filesystem::path overloads
    auto img = torch::rand({3, 4, 4});
    TM_CHECK(io::save_image(img, std::filesystem::path("rt_path.png")));
    TM_CHECK(io::load_image(std::filesystem::path("rt_path.png")).size(0) == 3);
    TM_CHECK(io::save_image(img, "rt_cstr.png"));
    TM_CHECK(io::load_image("rt_cstr.png").size(0) == 3);
}

int main() {
    test_roundtrip();
    test_io_branches();
    return tm_test::summary("vision_test_io");
}
