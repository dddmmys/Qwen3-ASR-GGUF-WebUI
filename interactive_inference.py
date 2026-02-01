import torch
from qwen_asr import Qwen3ASRModel
import soundfile as sf
import numpy as np
import time
import os

MODEL_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"
FORCED_ALIGNER_PATH = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ForcedAligner-0.6B"

def main():
    # Load the model
    print(f"正在从以下路径加载模型: {MODEL_DIR}...")
    try:
        asr = Qwen3ASRModel.from_pretrained(
            MODEL_DIR,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    print("\n" + "="*50)
    print("交互式推理模式")
    print("输入 'q'、'quit' 或 'exit' 退出程序。")
    print("请输入音频文件的完整路径（例如：input.mp3）")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("请输入音频路径 >>> ").strip()
            
            # Check for exit commands
            if user_input.lower() in ('q', 'quit', 'exit'):
                print("正在退出...")
                break
                
            if not user_input:
                continue
                
            # Handle quoted paths (common when copying from Windows context menu)
            if user_input.startswith('"') and user_input.endswith('"'):
                user_input = user_input[1:-1]
            if user_input.startswith("'") and user_input.endswith("'"):
                user_input = user_input[1:-1]
                
            audio_path = user_input
            
            if not os.path.exists(audio_path):
                print(f"错误：找不到文件：{audio_path}")
                continue
                
            print(f"正在处理：{audio_path}")
            
            start_time = time.time()
            
            try:
                results = asr.transcribe(
                    audio=audio_path,
                    language=None, # Auto-detect
                    return_time_stamps=False,
                )
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print("\n----- 转录结果 -----")
                if isinstance(results, list):
                    for i, r in enumerate(results):
                        print(f"[样本 {i}] 语言: {r.language}")
                        print(f"[样本 {i}] 文本: {r.text}")
                else:
                    print(f"语言: {results.language}")
                    print(f"文本: {results.text}")
                
                print(f"\n[耗时]: {elapsed_time:.4f} 秒")
                print("-" * 33 + "\n")
                
            except Exception as e:
                print(f"转录出错: {e}")
                print("-" * 33 + "\n")

        except KeyboardInterrupt:
            print("\n正在退出...")
            break
        except Exception as e:
            print(f"发生未知错误: {e}")

if __name__ == "__main__":
    main()
