//
// Created by Mux on 25-1-17.
//

#ifndef LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_METHODS_OPTIONS_HPP
#define LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_METHODS_OPTIONS_HPP
#include "../globel_include.hpp"

namespace torchmedia::audio::functional {
    typedef struct spectrogram_option {
        int _pad = 0;
        tensor_t _window = {};
        int _n_fft = 400;
        int _hop_length = 200;
        int _win_length = 400;
        double _power = 2.0;
        bool _normalized = false;
        std::string _normalize_method = "window"; // window, frame_length
        bool _center = true;
        std::string _pad_mode = "reflect";
        bool _onesided = true;
        bool _return_complex = true; // when true, power becomes optional;

        auto pad(const int p) -> spectrogram_option & {
            _pad = p;
            return *this;
        }

        auto window(const tensor_t &w) -> spectrogram_option & {
            _window = w;
            return *this;
        }

        auto n_fft(const int n) -> spectrogram_option & {
            _n_fft = n;
            return *this;
        }

        auto hop_length(const int h) -> spectrogram_option & {
            _hop_length = h;
            return *this;
        }

        auto win_length(const int w) -> spectrogram_option & {
            _win_length = w;
            return *this;
        }

        auto power(const double p) -> spectrogram_option & {
            _power = p;
            return *this;
        }

        auto normalized(const bool n) -> spectrogram_option & {
            _normalized = n;
            return *this;
        }

        auto normalize_method(const std::string &n) -> spectrogram_option & {
            _normalize_method = n;
            return *this;
        }

        auto center(const bool c) -> spectrogram_option & {
            _center = c;
            return *this;
        }

        auto pad_mode(const std::string &p) -> spectrogram_option & {
            _pad_mode = p;
            return *this;
        }

        auto onesided(const bool o) -> spectrogram_option & {
            _onesided = o;
            return *this;
        }

        auto return_complex(const bool r) -> spectrogram_option & {
            _return_complex = r;
            return *this;
        }
    } spectrogram_option_t;

    typedef struct amplitude_to_db_option {
        float amin = 1e-10f;
        float top_db = 80.0f;
        float db_multiplier = 1.0f;
        bool apply_top_db = true;

        auto set_amin(const float a) -> amplitude_to_db_option & {
            amin = a;
            return *this;
        }

        auto set_top_db(const float t) -> amplitude_to_db_option & {
            top_db = t;
            return *this;
        }

        auto set_db_multiplier(const float m) -> amplitude_to_db_option & {
            db_multiplier = m;
            return *this;
        }

        auto set_apply_top_db(const bool b) -> amplitude_to_db_option & {
            apply_top_db = b;
            return *this;
        }
    } amplitude_to_db_option_t;

    typedef struct mel_spectrogram_option {
        int sample_rate = 16000;
        int n_fft = 400;
        int win_length = 400;
        int hop_length = 200;
        int pad = 0;

        double f_min = 0.0;
        double f_max = 0.0; // 若设为 0，后续可默认为 sample_rate/2
        int n_mels = 128;

        double power = 2.0;
        bool normalized = false;
        bool center = true;
        std::string pad_mode = "reflect";
        bool onesided = true;
        std::string norm; // "slaney" 或空字符串等
        std::string mel_scale = "htk"; // "htk" 或 "slaney"
    } mel_spectrogram_option_t;
}
#endif //LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_METHODS_OPTIONS_HPP
