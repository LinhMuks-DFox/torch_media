# TorchMedia

### Quick start

```C++
#include <libtrochmedia.hpp>

int main() {
    using io = torchmedia::audio::io;
    using functional = torchmedia::audio::functional;
    auto my_audio = io::load_audio("path/to/audio.wav").data;
    auto spectrogram = functional::spectorgram(my_audio, functional::spectrogram_options_t{}.windows(400).n_fft(200));
    fmt::print("{}", torchmedia::basic::to_string(spectrogram));
    return 0;
}
```
### Dependence
* fmt (any version)
* libtorch(any version)
* sox(any version)