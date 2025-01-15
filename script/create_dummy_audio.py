import numpy as np
import wave
import struct
import random

# 参数设置
sample_rate = 44100  # 采样率 (Hz)
duration = 10  # 音频时长 (秒)
num_waves = 10  # 叠加的sin波数量
max_amplitude = 32767  # 最大振幅 (16-bit最大值)

# 生成时间序列
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# 初始化waveform
waveform = np.zeros_like(t)

# 叠加若干随机sin波
for _ in range(num_waves):
    # 在一定范围内随机生成频率、相位、幅度
    # 你可以根据需求自行调整这些范围
    freq = random.uniform(100.0, 8000.0)          # 随机频率范围 (Hz)
    phase = random.uniform(0, 2 * np.pi)          # 随机相位
    amplitude = random.uniform(0.3, 1.0) * max_amplitude  # 随机幅度比例
    
    # 生成一个正弦波并叠加
    waveform += amplitude * np.sin(2 * np.pi * freq * t + phase)

# 防止幅度溢出(Clipping)
waveform = np.clip(waveform, -max_amplitude, max_amplitude)

# 创建并写入WAV文件
output_file = "dummy_audio_440Hz.wav"
with wave.open(output_file, "w") as wav_file:
    n_channels = 1       # 单声道
    sampwidth = 2        # 2 bytes (16-bit)
    wav_file.setnchannels(n_channels)
    wav_file.setsampwidth(sampwidth)
    wav_file.setframerate(sample_rate)

    # 写入数据
    for value in waveform:
        wav_file.writeframes(struct.pack("<h", int(value)))