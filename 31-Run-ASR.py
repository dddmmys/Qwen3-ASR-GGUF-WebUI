# coding=utf-8
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr_gguf.inference import QwenASREngine, itn, load_audio

def main():
    # 路径配置
    model_dir = Path("./model")
    encoder_onnx = model_dir / "qwen3_asr_encoder.int8.onnx" # 或 .int8.onnx / .fp32.onnx
    llm_gguf = model_dir / "qwen3_asr_llm.q8_0.gguf"
    mel_filters = model_dir / "mel_filters.npy"
    
    audio_path = "睡前消息.m4a"
    
    # 1. 初始化引擎 (即插即用)
    engine = QwenASREngine(
        encoder_onnx=str(encoder_onnx),
        llm_gguf=str(llm_gguf),
        mel_filters=str(mel_filters),
        use_dml=False
    )
    
    # 2. 加载音频 (使用 pydub 片段加载，比 librosa 快且省内存)
    print(f"\n加载音频: {audio_path}\n")
    audio = load_audio(audio_path, start_second=0, duration=120)
    
    # 3. 执行转录
    result = engine.transcribe(
        audio=audio,
        context="这是1004期睡前消息，主持人叫督工，助理叫静静。",
        language="Chinese",
        chunk_size_sec=40.0
    )
    
    # 4. ITN 后处理
    print("\n" + "="*20 + " ITN 处理后 " + "="*20)
    print(itn(result))
    print("="*52)
    
    # 5. 优雅退出
    engine.shutdown()

if __name__ == "__main__":
    main()
