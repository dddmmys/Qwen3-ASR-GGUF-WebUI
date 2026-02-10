# coding=utf-8
import torch
import os
import sys
import numpy as np
import librosa
from pathlib import Path

# 环境路径准备
sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr import Qwen3ForcedAligner
from export_config import ALIGNER_MODEL_DIR

def save_tensor(name, tensor, folder):
    """将 Tensor 转换为 Numpy 并保存"""
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = tensor
    
    path = os.path.join(folder, f"{name}.npy")
    np.save(path, data)
    print(f"✅ 已保存 {name}: {data.shape} -> {path}")

def main():
    # 1. 目录准备
    output_folder = "capture_data_aligner"
    os.makedirs(output_folder, exist_ok=True)
    
    model_path = str(ALIGNER_MODEL_DIR)
    audio_path = "test.mp3"
    text_path = "test.txt"
    
    if not os.path.exists(text_path):
        print(f"❌ 找不到文本文件: {text_path}")
        return
        
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # 2. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 正在初始化 Aligner 模型 (Device: {device}) ---")
    
    # 强制 float32 模式以获取最纯净的数值参考
    aligner = Qwen3ForcedAligner.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map=device
    )
    
    captured_data = {}

    # 3. 挂载钩子 (Hooks)
    # Aligner 结构: model -> thinker (Qwen2VL) -> audio_tower + llm
    thinker = aligner.model.thinker
    audio_tower = thinker.audio_tower

    # --- A. Audio Encoder 内部钩子 ---
    
    # 拦截进入 Transformer 之前的卷积前端输出
    def hook_conv_out(module, input, output):
        captured_data["conv_frontend_output"] = output

    # 拦截 Transformer 层的第一层输入 (检查隐向量和 mask)
    def hook_first_layer(module, input, kwargs):
        captured_data["encoder_layer0_input"] = input[0]
        # Aligner 可能会在这里传入特殊的 cu_seqlens

    # 拦截音频编码器的最终输出
    def hook_audio_tower_final(module, input, output):
        # 官方返回的是 BaseModelOutput，取其 last_hidden_state
        captured_data["audio_encoder_output"] = output.last_hidden_state

    # --- B. Thinker (LLM) 与 Logits 钩子 ---

    def hook_thinker_input(module, input, kwargs):
        # 截获最终拼装后的 Token IDs 和注意力掩码
        if "input_ids" in kwargs:
            captured_data["input_ids"] = kwargs["input_ids"]
        elif len(input) > 0:
            captured_data["input_ids"] = input[0]

    def hook_lm_head(module, input, output):
        # 截获最终的 Logits 概率分布
        captured_data["raw_logits"] = output

    # 挂载
    handles = []
    # 语音前端
    handles.append(audio_tower.conv_out.register_forward_hook(hook_conv_out))
    # 语音第一层 (含 mask)
    handles.append(audio_tower.layers[0].register_forward_pre_hook(hook_first_layer, with_kwargs=True))
    # 语音编码器总输出
    handles.append(audio_tower.register_forward_hook(hook_audio_tower_final))
    
    # 文本/多模态输入
    handles.append(thinker.register_forward_pre_hook(hook_thinker_input, with_kwargs=True))
    # 最终分类头
    handles.append(thinker.lm_head.register_forward_hook(hook_lm_head))

    print(f"已成功挂载 {len(handles)} 个拦截点，准备打桩。")

    # 4. 执行推理并捕获
    print("\n--- 启动对齐推理并执行拦截 ---")
    try:
        # 手动执行一部分预处理以捕捉原始音频和 Mel
        audio_array, _ = librosa.load(audio_path, sr=16000)
        captured_data["input_waveform"] = audio_array
        
        # 捕捉经过处理后的 Mel
        word_list, aligner_input_text = aligner.aligner_processor.encode_timestamp(text, "Chinese")
        inputs = aligner.processor(
            text=[aligner_input_text],
            audio=audio_array,
            return_tensors="pt",
            padding=True,
        )
        captured_data["official_mel"] = inputs["input_features"]

        # 执行正式 align 推理，触发所有挂载好的 Hook
        results = aligner.align(
            audio=audio_path,
            text=text,
            language="Chinese"
        )
    except Exception as e:
        print(f"❌ 捕获过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        # 卸载钩子，释放内存
        for h in handles:
            h.remove()

    # 5. 保存结果
    print("\n--- 正在将截获的 Tensor 持久化至磁盘 ---")
    for name, tensor in captured_data.items():
        save_tensor(name, tensor, output_folder)

    print(f"\n✨ 打桩完成！数据存放在: {output_folder}/")
    print("你可以使用这些 .npy 文件作为 ONNX 验证的 Golden Baseline。")

if __name__ == "__main__":
    main()
