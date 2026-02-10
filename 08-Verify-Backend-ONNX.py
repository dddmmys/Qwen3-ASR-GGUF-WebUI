# coding=utf-8
import os
import numpy as np
import onnxruntime as ort
from export_config import EXPORT_DIR

def calculate_cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

def create_block_diagonal_mask(seq_len, cu_seqlens):
    """根据官方 cu_seqlens 创建块对角注意力掩码"""
    mask = np.full((seq_len, seq_len), -1e9, dtype=np.float32)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        mask[start:end, start:end] = 0.0
    return mask

def main():
    onnx_path = os.path.join(EXPORT_DIR, "qwen3_asr_backend.onnx")
    input_path = "capture_data/encoder_backend_input.npy"  # (371, 896)
    output_path = "capture_data/encoder_backend_output.npy"  # (371, 1024)
    cu_seqlens_path = "capture_data/encoder_cu_seqlens.npy" # [0, 104, 208, 312, 371]

    if not os.path.exists(onnx_path) or not os.path.exists(input_path):
        print("❌ 缺失文件，请确保执行过导出和捕获数据。")
        return

    # 1. 加载数据
    backend_input = np.load(input_path)
    baseline_output = np.load(output_path)
    cu_seqlens = np.load(cu_seqlens_path)
    
    seq_len = backend_input.shape[0]
    
    # 2. 构造 Mask
    # ONNX 期望 (B, 1, T, T)
    mask = create_block_diagonal_mask(seq_len, cu_seqlens)
    mask_input = mask[np.newaxis, np.newaxis, :, :] # (1, 1, 371, 371)
    
    # 对输入增加 Batch 维
    backend_input_batch = backend_input[np.newaxis, :, :]
    
    # 3. 推理
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    print(f"Running Backend ONNX verification (SeqLen: {seq_len})...")
    ort_outs = sess.run(None, {
        "hidden_states": backend_input_batch,
        "attention_mask": mask_input
    })
    onnx_output = ort_outs[0][0] # (371, 1024)

    # 4. 对比
    sim = calculate_cosine_similarity(baseline_output, onnx_output)
    
    print("\n" + "="*40)
    print(f"Backend 验证结果 (Backend Verification):")
    print(f"  - 形状: ONNX {onnx_output.shape} vs 官方 {baseline_output.shape}")
    print(f"  - 余弦相似度: {sim:.8f}")
    
    if sim > 0.9999:
        print("\n✨ PERFECT! The Backend ONNX is consistent with PyTorch implementation.")
    elif sim > 0.99:
        print("\n✅ SUCCESS: High similarity.")
    else:
        print("\n❌ MISMATCH: Check attention implementation or mask logic.")
    print("="*40)

if __name__ == "__main__":
    main()
