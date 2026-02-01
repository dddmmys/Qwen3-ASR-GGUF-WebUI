# Qwen3-ASR LLM (Thinker) 导出经验总结

## 1. 核心发现：Qwen3-ASR 的真实身份

在尝试将 Qwen3-ASR 的 LLM 部分导出为 GGUF 时，我们发现它并不是标准的 `Qwen2` 模型，而是具备以下特征：

*   **Q-Norm / K-Norm**: Attention 层包含 `q_norm` 和 `k_norm`。
*   **MRoPE (Multimodal RoPE)**: 使用了 3D 位置编码逻辑（虽然音频是 1D 的，但代码沿用了 Omni/VL 的结构）。

这些特征表明它属于 **Qwen3-VL** 架构系列。这是相对较新的架构，虽然在算子层面（如 IMRoPE, RMSNorm）与 Qwen2-VL 有继承关系，但它是独立的模型类型。


## 2. 导出策略 (Export Strategy)

### 2.1 架构配置
在导出 `config.json` 时，明确指定新架构：
*   `architectures`: 设置为 `["Qwen3VLForConditionalGeneration"]`。
*   `model_type`: 设置为 `qwen3_vl`。

### 2.2 权重映射
原始模型将 LLM 包装在 `thinker` 下，导出时需要“剥壳”：
*   `thinker.model.*` -> `model.*`
*   `thinker.lm_head.*` -> `lm_head.*`

### 2.3 坑点：SafeTensors 共享内存报错
**问题**：`Qwen3` 的 `lm_head` 和 `embed_tokens` 权重通常是绑定的（共享内存）。`safetensors` 不支持保存共享内存的张量，会报 `RuntimeError: Some tensors share memory`。
**解决**：在导出时，对 `lm_head.weight` 进行 `.clone()`，强制分配独立内存。
```python
# 21-Export-ASR-LLM.py
if key.startswith("thinker.lm_head."):
    new_key = key.replace("thinker.lm_head.", "lm_head.")
    # Clone to separate memory from embed_tokens if they are tied
    new_state_dict[new_key] = as_state_dict[key].clone()
```

## 3. 验证策略 (Verification Strategy)

### 3.1 坑点：Transformers 注册冲突
**问题**：在验证脚本中，如果你尝试用 `AutoConfig.register("qwen3_vl", ...)` 来注册一个自定义类，而 `transformers` 内部已经保留了这个名字（或检测到冲突），会报错 `ValueError: 'qwen3_vl' is already used`。

**尝试失败的方案**：
```python
# ❌ 这种写法会和 transformers 内置 registry 打架
AutoConfig.register("qwen3_vl", MyConfig)
AutoModel.from_pretrained(path)
```

### 3.2 解决方案：定义原生 Standalone 类并使用 `from_pretrained`
**成功方案**：在项目中定义一个专用的 `Qwen3ASRStandaloneLLM` 类，通过继承 `Qwen3ASRThinkerTextPreTrainedModel` (基类) 并组合 `Qwen3ASRThinkerTextModel` (Backbone) 与 `lm_head`，可以直接利用官方的 `from_pretrained` 方法进行一键加载。

```python
# ✅ 这种写法最标准，权重自动匹配
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRThinkerTextModel, 
    Qwen3ASRThinkerTextPreTrainedModel
)

class Qwen3ASRStandaloneLLM(Qwen3ASRThinkerTextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3ASRThinkerTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

# 直接加载全套权重（包含 model.* 和 lm_head.*）
model = Qwen3ASRStandaloneLLM.from_pretrained(model_path, config=config)
```

## 4. GGUF 转换阶段 (GGUF Conversion)

### 4.1 核心挑战：配置加载“张冠李戴”
在初次转换时，日志显示 `embedding length = 4096`，而我们实际是 **2048**。
**原因**：转换器调用的 `AutoConfig.from_pretrained` 识别到 `model_type="qwen3_vl"` 后，由于本地库或远程配置的干扰，加载了一个默认的大模型配置，导致参数完全失配，且丢失了 `mrope_section`。

### 4.2 解决方案：猴子补丁 (Monkey Patching)
在转换脚本中注入以下补丁，强制转换器认准本地文件：

```python
# 23-Convert-LLM-GGUF.py 中的核心补丁
def patched_load_hparams(dir_model: Path, is_mistral_format: bool):
    print(f"💉 [补丁] 强制从本地 config.json 加载参数")
    with open(dir_model / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)

# 应用补丁
ModelBase.load_hparams = staticmethod(patched_load_hparams)
TextModel.get_vocab_base_pre = lambda self, tok: "qwen2" # 强制识别分词器
```

### 4.3 验证结果 (llama-bench)
转换后的模型在 `llama.cpp` 中成功加载，推理性能（Vulkan 后端）：
- **PP512** (Prompt Processing): ~6927 t/s
- **TG128** (Text Generation): ~82 t/s
- **参数规模**: 2.03 B (F16)

## 5. 总结与产物
*   **流程**: 权重重命名 (21) -> 原生代码验证 (22) -> 补丁辅助转换 (23)
*   **脚本**: 
    - [21-Export-ASR-LLM.py](file:///d:/qwen3-asr/21-Export-ASR-LLM.py)
    - [22-Verify-ASR-LLM.py](file:///d:/qwen3-asr/22-Verify-ASR-LLM.py)
    - [23-Convert-LLM-GGUF.py](file:///d:/qwen3-asr/23-Convert-LLM-GGUF.py)
*   **最后一步**: 确保 GGUF 转换过程中出现了 `gguf: mrope sections: [24, 20, 20, 0]`，这是 ASR 正常工作的基石。
