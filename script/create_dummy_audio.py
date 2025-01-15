import numpy as np
import wave
import struct

# 参数设置
sample_rate = 44100  # 采样率 (Hz)
duration = 0.01  # 音频时长 (秒)
frequency = 440.0  # 正弦波频率 (Hz)
amplitude = 32767  # 振幅 (最大值为 32767)

# 生成时间序列
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# 生成正弦波
waveform = amplitude * np.sin(2 * np.pi * frequency * t)

# 创建 WAV 文件
output_file = "dummy_audio_440Hz.wav"
with wave.open(output_file, "w") as wav_file:
    # 设置 WAV 文件参数
    n_channels = 1  # 单声道
    sampwidth = 2  # 2 bytes (16-bit)
    n_frames = len(waveform)
    wav_file.setnchannels(n_channels)
    wav_file.setsampwidth(sampwidth)
    wav_file.setframerate(sample_rate)

    # 写入数据
    for value in waveform:
        wav_file.writeframes(struct.pack("<h", int(value)))
