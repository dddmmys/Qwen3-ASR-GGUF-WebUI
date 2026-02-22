import os
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr_gguf.inference.encoder import QwenAudioEncoder
from qwen_asr_gguf.inference.utils import load_audio

def calculate_cosine_similarity(v1, v2):
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()
    return np.dot(v1_flat, v2_flat) / (np.linalg.norm(v1_flat) * np.linalg.norm(v2_flat))

def main():
    audio_file = "test.mp3"
    if not os.path.exists(audio_file):
        print(f"Error: 找不到 {audio_file}")
        sys.exit(1)

    model_dir = os.path.join(Path(__file__).parent.absolute(), "model")
    mel_filters = os.path.join(model_dir, "mel_filters.npy")

    print("[1/4] 载入音频文件...")
    # 只取前30秒，足够反映出量化对大特征的影响
    audio = load_audio(audio_file, start_second=0, duration=0.0)
    print(f"  音频长度: {len(audio)/16000:.2f} 秒")

    # ----- FP16 -----
    print("\n[2/4] 载入 FP16 Encoder 并推理...")
    fp16_encoder = QwenAudioEncoder(
        frontend_path=os.path.join(model_dir, "qwen3_asr_encoder_frontend.fp16.onnx"),
        backend_path=os.path.join(model_dir, "qwen3_asr_encoder_backend.fp16.onnx"),
        mel_filters_path=mel_filters,
        use_dml=True,
        warmup_sec=5.0,
        verbose=False
    )
    fp16_embd, fp16_time = fp16_encoder.encode(audio)
    print(f"  FP16 推理完成，耗时: {fp16_time:.2f}s, 输出形状: {fp16_embd.shape}")

    # ----- INT4 -----
    print("\n[3/4] 载入 INT4 Encoder 并推理...")
    int4_encoder = QwenAudioEncoder(
        frontend_path=os.path.join(model_dir, "qwen3_asr_encoder_frontend.int4.onnx"),
        backend_path=os.path.join(model_dir, "qwen3_asr_encoder_backend.int4.onnx"),
        mel_filters_path=mel_filters,
        use_dml=True,
        warmup_sec=5.0,
        verbose=False
    )
    int4_embd, int4_time = int4_encoder.encode(audio)
    print(f"  INT4 推理完成，耗时: {int4_time:.2f}s, 输出形状: {int4_embd.shape}")

    # 释放显存等资源
    del fp16_encoder
    del int4_encoder

    print("\n[4/4] 计算余弦相似度...")
    if fp16_embd.shape != int4_embd.shape:
        print("  ⚠️ 形状不完全一致，将对齐最小长度")
        min_len = min(fp16_embd.shape[0], int4_embd.shape[0])
        fp16_embd = fp16_embd[:min_len]
        int4_embd = int4_embd[:min_len]

    sim = calculate_cosine_similarity(fp16_embd, int4_embd)
    print(f"  🎯 余弦相似度 (Cosine Similarity): {sim:.5f}")

    mae = np.mean(np.abs(fp16_embd - int4_embd))
    print(f"  🎯 平均绝对误差 (MAE): {mae:.5f}")

if __name__ == '__main__':
    main()
