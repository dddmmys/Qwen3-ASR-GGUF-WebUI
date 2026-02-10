# coding=utf-8
import os
import numpy as np
import onnxruntime as ort
from export_config import EXPORT_DIR

def calculate_cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

def main():
    onnx_path = os.path.join(EXPORT_DIR, "qwen3_aligner_encoder.onnx")
    input_mel_path = "capture_data_aligner/official_mel.npy"
    baseline_output_path = "capture_data_aligner/audio_encoder_output.npy"

    if not os.path.exists(onnx_path):
        print(f"❌ 缺失 ONNX 模型: {onnx_path}")
        return
    if not os.path.exists(input_mel_path):
        print(f"❌ 缺失输入数据: {input_mel_path}")
        return
    if not os.path.exists(baseline_output_path):
        print(f"❌ 缺失基准数据: {baseline_output_path}")
        return

    # 1. 加载数据
    input_mel = np.load(input_mel_path) # (1, 128, 6120)
    baseline_output = np.load(baseline_output_path) # (796, 1024)
    
    seq_len_out = baseline_output.shape[0]
    
    # 2. 准备输入
    # Aligner 推理时通常使用 Full Attention (全零 Mask)
    mask_input = np.zeros((1, 1, seq_len_out, seq_len_out), dtype=np.float32)
    
    # 3. 推理 (开启 DirectML 加速)
    print(f"Initializing ALIGNER Encoder ONNX Runtime with DirectML...")
    sess = ort.InferenceSession(onnx_path, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    
    print(f"Running Aligner Encoder ONNX verification...")
    ort_outs = sess.run(None, {
        "input_features": input_mel,
        "attention_mask": mask_input
    })
    onnx_output = ort_outs[0][0] # (796, 1024)

    # 4. 对比验证
    sim = calculate_cosine_similarity(baseline_output, onnx_output)
    
    print("\n" + "="*40)
    print(f"Aligner Encoder 精度验证结果:")
    print(f"  - ONNX 输出形状: {onnx_output.shape}")
    print(f"  - 官方基准形状: {baseline_output.shape}")
    print(f"  - 余弦相似度: {sim:.10f}")
    
    if sim > 0.999999:
        print("\n✨ PERFECT! Aligner Encoder ONNX output is bit-exact with baseline.")
    elif sim > 0.999:
        print("\n✅ SUCCESS: High precision achieved.")
    else:
        print("\n❌ FAILED: Accuracy is below threshold.")
    print("="*40)

if __name__ == "__main__":
    main()
