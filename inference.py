import torch
from qwen_asr import Qwen3ASRModel
import soundfile as sf
import numpy as np

MODEL_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"
FORCED_ALIGNER_PATH = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ForcedAligner-0.6B"

def main():
    # Load the model
    print(f"Loading model from {MODEL_DIR}...")
    asr = Qwen3ASRModel.from_pretrained(
        MODEL_DIR,
        # forced_aligner=FORCED_ALIGNER_PATH,
        # forced_aligner_kwargs=dict(
        #     dtype=torch.bfloat16,
        #     device_map="cuda:0",
        #     # attn_implementation="flash_attention_2",
        # ),
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )


    audio_path = "input.mp3"
    
    results = asr.transcribe(
        audio=audio_path,
        language=None, # Auto-detect
        return_time_stamps=False,
    )
    
    print("\n===== Result =====")
    for i, r in enumerate(results):
        print(f"[sample {i}] language={r.language}")
        print(f"[sample {i}] text={r.text}")
    

if __name__ == "__main__":
    main()
