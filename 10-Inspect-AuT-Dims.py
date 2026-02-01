import torch
from pathlib import Path
import sys

# Add custom path
PROJECT_ROOT = Path(__file__).parent.absolute()
CUSTOM_MODEL_DIR = PROJECT_ROOT / "model" / "qwen_asr_custom"
if str(CUSTOM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_MODEL_DIR))

from modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from export_config import MODEL_DIR

def inspect_dimensions():
    print(f"Loading model from: {MODEL_DIR}")
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    encoder = model.thinker.audio_tower
    print("\n--- Audio Encoder Dimensions ---")
    print(f"d_model: {encoder.config.d_model}")
    print(f"num_heads: {encoder.config.encoder_attention_heads}")
    
    # Inspect weights
    for name, param in encoder.named_parameters():
        if "self_attn.q_proj.weight" in name:
            print(f"{name} shape: {list(param.shape)}")
            # weight shape is [out_features, in_features]
            # out_features = num_heads * head_dim
            head_dim = param.shape[0] // encoder.config.encoder_attention_heads
            print(f"Inferred head_dim: {head_dim}")
        
        if "conv" in name and "weight" in name:
            print(f"{name} shape: {list(param.shape)}")

if __name__ == "__main__":
    inspect_dimensions()
