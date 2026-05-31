// Assertion-based tests for the vision functional layer. Golden values from torchvision 0.20.1
// (see gen_golden.py). Image: [3,4,4] float in [0,1].
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "test_util.hpp"

using namespace torchmedia::vision::functional;

static torch::Tensor make_img() {
    return torch::arange(48, torch::kFloat32).reshape({3, 4, 4}) / 47.0; // [C,H,W] in [0,1]
}

static void test_flip() {
    auto img = make_img();
    auto h = hflip(img);
    TM_CHECK_CLOSE(h[0][0][0].item<double>(), 0.0638, 1e-3); // was [0,0,3]
    TM_CHECK_CLOSE(h[0][0][3].item<double>(), 0.0, 1e-6);    // was [0,0,0]
    TM_CHECK_CLOSE(vflip(img).sum().item<double>(), 24.0, 1e-3);
}

static void test_rgb_to_grayscale() {
    auto img = make_img();
    auto g = rgb_to_grayscale(img);
    TM_CHECK(g.size(-3) == 1);
    TM_CHECK_CLOSE(g[0][0][0].item<double>(), 0.277447, 1e-4);
    auto g3 = rgb_to_grayscale(img, 3); // num_output_channels=3 branch
    TM_CHECK(g3.size(-3) == 3);
    TM_CHECK_CLOSE(g3[2][0][0].item<double>(), 0.277447, 1e-4);
}

static void test_normalize() {
    auto img = make_img();
    auto n = normalize(img, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5});
    TM_CHECK_CLOSE(n[0][0][0].item<double>(), -1.0, 1e-5);
}

static void test_center_crop() {
    auto cc = center_crop(make_img(), 2, 2);
    TM_CHECK(cc.size(-3) == 3 && cc.size(-2) == 2 && cc.size(-1) == 2);
    TM_CHECK_CLOSE(cc.sum().item<double>(), 6.0, 1e-3);
}

static void test_adjust_brightness() {
    auto img = make_img();
    TM_CHECK_CLOSE(adjust_brightness(img, 1.5)[0][0][3].item<double>(), 0.095745, 1e-4);
    TM_CHECK_CLOSE(adjust_brightness(img, 100.0).max().item<double>(), 1.0, 1e-6); // clamp branch
}

static void test_invert() {
    TM_CHECK_CLOSE(invert(make_img())[0][0][0].item<double>(), 1.0, 1e-6);
}

int main() {
    test_flip();
    test_rgb_to_grayscale();
    test_normalize();
    test_center_crop();
    test_adjust_brightness();
    test_invert();
    return tm_test::summary("vision_test_functional");
}
