import torch
import numpy as np
import onnxruntime as ort
import sys
import os
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).parent.absolute()
CUSTOM_MODEL_DIR = PROJECT_ROOT / "model" / "qwen_asr_custom"
if str(CUSTOM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_MODEL_DIR))

from modeling_qwen3_asr_onnx import StatefulAudioEncoderWrapper, DiscreteAudioEncoder
from modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from export_config import MODEL_DIR, EXPORT_DIR

def verify_encoder():
    print("--- 正在验证 Audio Encoder ONNX 模型 ---")
    
    # 1. 加载原始 PyTorch 模型
    full_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    encoder = full_model.thinker.audio_tower.eval()
    
    # 指定 ONNX 路径
    onnx_dir = Path(EXPORT_DIR) / "onnx"
    discrete_path = onnx_dir / "qwen3_asr_encoder_discrete.onnx"
    stateful_path = onnx_dir / "qwen3_asr_encoder_stateful.onnx"
    
    # 2. 准备测试数据 (B=1, T=128, F=128) - 使用 8 的倍化以对齐下采样
    dummy_mel = torch.randn(1, 128, 128)
    
    # 3. 运行 PyTorch 原生推理
    py_discrete = DiscreteAudioEncoder(encoder).eval()
    with torch.no_grad():
        expected_out = py_discrete(dummy_mel).numpy()
    
    # 4. 验证 Discrete ONNX
    print("\n[Test 1] 验证 Discrete ONNX...")
    sess_discrete = ort.InferenceSession(str(discrete_path))
    onnx_discrete_out = sess_discrete.run(None, {"mel": dummy_mel.numpy()})[0]
    
    mse_discrete = np.mean((expected_out - onnx_discrete_out) ** 2)
    print(f"Discrete ONNX MSE: {mse_discrete:.2e}")
    if mse_discrete < 1e-5:
        print("✅ Discrete 验证通过!")
    else:
        print("❌ Discrete 验证失败: 误差过大")

    # 5. 验证 Stateful ONNX (分块推理)
    print("\n[Test 2] 验证 Stateful ONNX (流式分块)...")
    sess_stateful = ort.InferenceSession(str(stateful_path))
    
    # 将 128 帧分为两块: 64 + 64
    chunk1 = dummy_mel[:, :64, :].numpy()
    chunk2 = dummy_mel[:, 64:, :].numpy()
    conv_state = np.zeros((1, 8, 128), dtype=np.float32)
    
    # 运行第一块
    out1, next_conv_state = sess_stateful.run(None, {
        "mel": chunk1,
        "conv_state": conv_state
    })
    
    # 运行第二块
    out2, _ = sess_stateful.run(None, {
        "mel": chunk2,
        "conv_state": next_conv_state
    })
    
    # 拼接分块输出
    # 注意：下采样倍数是 8，理论上 50 帧 -> 6.25 -> 取整?
    # 实际上 conv_out 是根据 3 层 s=2 得出的
    # 我们检查拼接后的总长度
    onnx_stateful_out = np.concatenate([out1, out2], axis=1)
    
    # 这里的对比比较微妙，因为卷积边界会有少许不同
    # 理想情况下，由于我们传递了 8 帧状态，结果应与全局推理高度一致
    # 但由于下采样对齐问题，可能需要检查中心部分
    
    print(f"原生输出 Shape: {expected_out.shape}")
    print(f"分块拼接 Shape: {onnx_stateful_out.shape}")
    
    # 如果形状一致，计算 MSE
    if expected_out.shape == onnx_stateful_out.shape:
        mse_stateful = np.mean((expected_out - onnx_stateful_out) ** 2)
        print(f"Stateful ONNX MSE (Full): {mse_stateful:.2e}")
        if mse_stateful < 1e-3: # 流式允许稍大的累积误差
             print("✅ Stateful 验证通过!")
        else:
             print("⚠️ Stateful 存在一定误差，请人工检查连续性")
    else:
        print("❌ Stateful 形状不匹配，流式下采样逻辑可能存在对齐偏移")

if __name__ == "__main__":
    verify_encoder()
