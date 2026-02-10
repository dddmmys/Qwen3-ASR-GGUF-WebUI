# coding=utf-8
import numpy as np

def load_audio(audio_path, sample_rate=16000, start_second=None, duration=None):
    """加载音频文件并转换为 16kHz PCM，支持按需加载指定片段"""
    from pydub import AudioSegment
    
    # 使用 pydub 的参数来减少解码量（如果可能）
    load_kwargs = {
        "frame_rate": sample_rate, 
        "channels": 1
    }
    if start_second is not None: load_kwargs['start_second'] = start_second
    if duration: load_kwargs['duration'] = duration

    audio_segment = AudioSegment.from_file(audio_path, **load_kwargs)

    bit_depth = audio_segment.sample_width * 8
    max_val = float(1 << (bit_depth - 1))
    
    audio = np.array(
        audio_segment
        .set_channels(1)
        .set_frame_rate(sample_rate)
        .get_array_of_samples(),
    ) / max_val

    return audio.astype(np.float32)
