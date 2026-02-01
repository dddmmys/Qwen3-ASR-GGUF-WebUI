import os
import sys
import torch
import torch.nn as nn
import json
from pathlib import Path
from transformers import AutoTokenizer

# 将项目根目录添加到 sys.path 以便导入 qwen_asr
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from export_config import EXPORT_DIR
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRTextConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRThinkerTextModel, Qwen3ASRThinkerTextPreTrainedModel

class Qwen3ASRStandaloneLLM(Qwen3ASRThinkerTextPreTrainedModel):
    """
    原生 ASR LLM 类，用于加载验证。
    匹配导出的 HF 结构：
    - model.* -> self.model (骨干网络)
    - lm_head.* -> self.lm_head (输出头)
    """
    def __init__(self, config: Qwen3ASRTextConfig):
        super().__init__(config)
        # 在 Qwen3ASRThinkerTextPreTrainedModel 中，base_model_prefix = "model"
        # 因此 from_pretrained 会自动将 "model.*" 加载到 self.model 中
        self.model = Qwen3ASRThinkerTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return logits

def verify_exported_model():
    """
    验证导出的 HF 格式模型是否可以被原生代码正确加载并运行。
    """
    model_path = Path(EXPORT_DIR) / "llm_hf"
    print(f"--- 正在进行原生标准加载验证 ---")
    print(f"模型路径: {model_path}")
    
    if not model_path.exists():
        print(f"❌ 错误: 模型路径 {model_path} 不存在。")
        return

    # 1. 加载配置
    print("\n[1/4] 正在加载配置...")
    try:
        # 首先加载为字典，以处理自定义的 model_type 或 architectures
        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        # 实例化原生配置
        config = Qwen3ASRTextConfig(**config_dict)
        print(f"✅ 配置加载成功。词表大小: {config.vocab_size}")
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return

    # 2. 加载分词器
    print("\n[2/4] 正在加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ 分词器加载成功。词表大小: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ 加载分词器失败: {e}")

    # 3. 使用 from_pretrained 加载模型
    print("\n[3/4] 正在通过 from_pretrained() 加载模型...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 注意：我们使用本地定义的类进行加载
        # 它会查找 model.safetensors 并正确映射 "model.*" 和 "lm_head.*"
        model = Qwen3ASRStandaloneLLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float32, # 或 bfloat16
            device_map=device
        )
        print("✅ 模型通过 from_pretrained() 加载成功！")
        
    except Exception as e:
        print(f"❌ from_pretrained() 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 虚拟推理测试
    print("\n[4/4] 正在运行虚推理测试...")
    try:
        model.eval()
        # 准备随机输入 [batch_size=1, seq_len=10]
        dummy_input = torch.randint(0, config.vocab_size, (1, 10)).to(device)
        
        with torch.no_grad():
            logits = model(dummy_input)
            
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出 Logits 形状: {logits.shape}")
        
        if logits.shape == (1, 10, config.vocab_size):
            print(f"✅ Logits 形状符合预期: [1, 10, {config.vocab_size}]")
            if torch.isnan(logits).any():
                print("❌ 警告: Logits 中包含 NaN！")
            else:
                print("✅ 数值正常 (无 NaN)。")
        else:
            print(f"❌ Logits 形状不匹配！得到的是 {logits.shape}")

    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ 原生验证流程结束。")

if __name__ == "__main__":
    verify_exported_model()

if __name__ == "__main__":
    verify_exported_model()
