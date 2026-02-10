# coding=utf-8
import os
import sys
import torch
from pathlib import Path

# 1. 导入配置和自定义模型定义
# 确保项目路径在 sys.path 中以便正常导入 qwen_asr
sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent / "qwen_asr_gguf" / "export"))

from export_config import MODEL_DIR, EXPORT_DIR
from qwen_asr import Qwen3ASRModel
from qwen3_asr_custom.modeling_qwen3_asr_onnx import Qwen3ASRFrontendOnnx

def export_frontend():
    # 2. 准备路径
    model_path = str(MODEL_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    onnx_path = os.path.join(EXPORT_DIR, "qwen3_asr_frontend.onnx")
    
    print(f"Loading official model from: {model_path}")
    
    # 3. 加载官方模型并提取 AudioTower
    # 加载到 CPU 即可，方便导出
    asr_model = Qwen3ASRModel.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    audio_tower = asr_model.model.thinker.audio_tower
    
    # 4. 初始化 Wrapper
    frontend_model = Qwen3ASRFrontendOnnx(audio_tower)
    frontend_model.eval()
    
    # 5. 准备 dummy 输入 (Batch, Mel, Time)
    # Qwen3-ASR 标准：Mel=128
    # 我们取一个典型的 Time 长度，比如 3000 (约 30s)
    dummy_input = torch.randn(1, 128, 3000)
    
    # 6. 执行导出
    print(f"Exporting frontend to ONNX: {onnx_path}...")
    
    torch.onnx.export(
        frontend_model,
        (dummy_input,),
        onnx_path,
        input_names=["input_features"],
        output_names=["frontend_output"],
        dynamic_axes={
            "input_features": {0: "batch", 2: "time"},
            "frontend_output": {0: "batch", 1: "time"},
        },
        opset_version=18,
        do_constant_folding=True, 
        dynamo=True
    )
    
    print(f"✅ Frontend ONNX export complete: {onnx_path}")

if __name__ == "__main__":
    export_frontend()
