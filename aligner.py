# coding=utf-8
import os
import sys
import time
import gc
import numpy as np
import onnxruntime as ort
import librosa
from pathlib import Path
from typing import List, Dict, Any

import unicodedata

# ==========================================
# 0. 环境配置
# ==========================================
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# 导入 GGUF 绑定
from qwen_asr_gguf.inference import llama

# ==========================================
# 1. 对齐处理器：NativeAlignerProcessor (去除外部依赖)
# ==========================================
class NativeAlignerProcessor:
    """复刻官方逻辑，处理文本分词、时间戳编码与 LIS 修正"""
    def is_kept_char(self, ch: str) -> bool:
        if ch == "'": return True
        cat = unicodedata.category(ch)
        return cat.startswith("L") or cat.startswith("N")

    def clean_token(self, token: str) -> str:
        return "".join(ch for ch in token if self.is_kept_char(ch))

    def is_cjk_char(self, ch: str) -> bool:
        code = ord(ch)
        return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or
                0x20000 <= code <= 0x2A6DF or 0x2A700 <= code <= 0x2B73F or
                0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF or
                0xF900 <= code <= 0xFAFF)

    def split_segment_with_chinese(self, seg: str) -> List[str]:
        tokens, buf = [], []
        def flush_buf():
            nonlocal buf
            if buf: tokens.append("".join(buf)); buf = []
        for ch in seg:
            if self.is_cjk_char(ch):
                flush_buf(); tokens.append(ch)
            else: buf.append(ch)
        flush_buf()
        return tokens

    def tokenize_general(self, text: str) -> List[str]:
        tokens = []
        for seg in text.split():
            cleaned = self.clean_token(seg)
            if cleaned: tokens.extend(self.split_segment_with_chinese(cleaned))
        return tokens

    def encode_timestamp(self, text: str, language: str) -> (List[str], str):
        # 默认使用通用分词逻辑 (支持中英混排)
        word_list = self.tokenize_general(text)
        aligner_text = "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
        aligner_text = "<|audio_start|><|audio_pad|><|audio_end|>" + aligner_text
        return word_list, aligner_text

    def fix_timestamp(self, data: List[int]) -> List[int]:
        n = len(data)
        if n == 0: return []
        # LIS (最长递增子序列) 算法实现
        dp, parent = [1] * n, [-1] * n
        for i in range(1, n):
            for j in range(i):
                if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1; parent[i] = j
        max_idx = dp.index(max(dp))
        lis_indices, idx = [], max_idx
        while idx != -1: lis_indices.append(idx); idx = parent[idx]
        lis_indices.reverse()
        
        is_normal = [False] * n
        for idx in lis_indices: is_normal[idx] = True
        
        result = [int(x) for x in data]
        i = 0
        while i < n:
            if not is_normal[i]:
                j = i
                while j < n and not is_normal[j]: j += 1
                anomaly_count = j - i
                left = next((result[k] for k in range(i-1, -1, -1) if is_normal[k]), None)
                right = next((result[k] for k in range(j, n) if is_normal[k]), None)
                
                if anomaly_count <= 2:
                    for k in range(i, j):
                        if left is None: result[k] = right
                        elif right is None: result[k] = left
                        else: result[k] = left if (k - (i-1)) <= (j - k) else right
                else:
                    if left is not None and right is not None:
                        step = (right - left) / (anomaly_count + 1)
                        for k in range(i, j): result[k] = int(left + step * (k - i + 1))
                    elif left is not None:
                        for k in range(i, j): result[k] = left
                    elif right is not None:
                        for k in range(i, j): result[k] = right
                i = j
            else: i += 1
        return result

    def parse_timestamp(self, word_list: List[str], timestamp: np.ndarray) -> List[Dict]:
        fixed = self.fix_timestamp(timestamp.tolist())
        return [{"text": w, "start_time": fixed[i*2], "end_time": fixed[i*2+1]} 
                for i, w in enumerate(word_list)]

# ==========================================
# 2. 音频前端：FastWhisperMel
# ==========================================
class FastWhisperMel:
    def __init__(self, filter_path: str):
        self.filters = np.load(filter_path)
        
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """输入: PCM 采样 (1D float32), 输出: Mel 频谱 (128, T)"""
        # 1. 核心 STFT
        stft = librosa.stft(audio, n_fft=400, hop_length=160, window='hann', center=True)
        magnitudes = np.abs(stft) ** 2
        
        # 2. 映射到 Mel 域
        mel_spec = np.dot(self.filters.T, magnitudes)
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        
        # 3. 标准化
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        
        # 4. 帧对齐：官方逻辑是丢弃 stft(center=True) 产生的多余帧
        n_frames = audio.shape[-1] // 160
        log_spec = log_spec[:, :n_frames]
        
        return log_spec.astype(np.float32)

def _get_feat_lengths(t):
    """计算下采样后的长度 (用于生成注意掩码)"""
    t_leave = t % 100
    feat_len = (t_leave - 1) // 2 + 1
    out_len = ((feat_len - 1) // 2 + 1 - 1) // 2 + 1 + (t // 100) * 13
    return int(out_len)

# ==========================================
# 2. 核心全链路对齐器 (GGUF + ONNX 版)
# ==========================================
class Qwen3GGUFForcedAligner:
    def __init__(
        self, 
        model_dir: str = "model",
        llm_name: str = "qwen3_aligner_llm.q8_0.gguf"
    ):
        model_path = Path(model_dir)
        print(f"--- 正在初始化 Qwen3-GGUF 强制对齐器 ---")
        
        # A. 加载音频编码器 (ONNX 合并版)
        print("正在加载音频编码器 (Merged ONNX)...")
        opt = ort.SessionOptions()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.encoder_sess = ort.InferenceSession(
            str(model_path / "qwen3_aligner_encoder.onnx"),
            sess_options=opt, providers=providers
        )
        print(f"编码器 EP: {self.encoder_sess.get_providers()[0]}")
        self.mel_extractor = FastWhisperMel(str(model_path / "mel_filters.npy"))

        # B. 加载 Thinker (GGUF)
        llm_path = model_path / llm_name
        print(f"正在加载语言模型 ({llm_name})...")
        self.model = llama.LlamaModel(str(llm_path))
        # 对齐任务 Batch 较大，ctx 也要设够
        self.ctx = llama.LlamaContext(self.model, n_ctx=4096, n_batch=4096)
        self.embedding_table = llama.get_token_embeddings_gguf(str(llm_path))

        # C. 初始化本地对齐处理器
        self.align_processor = NativeAlignerProcessor()

        # D. 获取关键 Token ID
        self.ID_AUDIO_PAD = 151676 # <|audio_pad|>
        self.ID_TIMESTAMP = self.model.token_to_id("<timestamp>")
        self.TIMESTAMP_STEP_MS = 80.0 # 核心步长
        
        print(f"初始化完成。设备: CPU (llama.cpp)")

    def _prepare_prompt(self, text: str, language: str, n_audio_tokens: int) -> (List[str], np.ndarray):
        """构造符合官方标准的原生 Prompt (无 Chat Template)"""
        word_list, aligner_text = self.align_processor.encode_timestamp(text, language)
        
        # 1. 展开音频占位符
        audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
        # 官方逻辑直接拼接：audio_start + N个pad + audio_end
        expanded_audio = f"<|audio_start|>{'<|audio_pad|>' * n_audio_tokens}<|audio_end|>"
        full_text = aligner_text.replace(audio_placeholder, expanded_audio)
        
        # 2. 直接进行 Tokenize (原生模式)
        input_ids = self.model.tokenize(full_text, add_special=False, parse_special=True)
        return word_list, np.array(input_ids, dtype=np.int32)

    def align(self, audio_file: str, text: str, language: str = "Chinese"):
        t_start = time.time()
        
        # 1. 提取音频特征
        audio, _ = librosa.load(audio_file, sr=16000)
        mel = self.mel_extractor(audio)
        mel_input = mel[np.newaxis, ...] # (1, 128, T)
        
        # 计算掩码
        t_mel = mel.shape[1]
        t_out = _get_feat_lengths(t_mel)
        mask_input = np.zeros((1, 1, t_out, t_out), dtype=np.float32)
        
        audio_features = self.encoder_sess.run(None, {
            "input_features": mel_input,
            "attention_mask": mask_input
        })[0]
        
        if audio_features.ndim == 3: audio_features = audio_features[0]
        n_audio_tokens = audio_features.shape[0]
        
        # 2. 构造 Prompt (传入音频 Token 数量进行展开)
        word_list, input_ids = self._prepare_prompt(text, language, n_audio_tokens)
        n_tokens = len(input_ids)
        print(f"Prompt 构造完毕，总 Token 数: {n_tokens} (音频占位符: {n_audio_tokens})")
        
        # 3. 构造 Embedding 矩阵并填入音频
        full_embd = self.embedding_table[input_ids].copy()
        audio_indices = np.where(input_ids == self.ID_AUDIO_PAD)[0]
        
        if len(audio_indices) != n_audio_tokens:
            print(f"⚠️ 警告: 占位符数量({len(audio_indices)})与特征数量({n_audio_tokens})不匹配，尝试强制同步...")
            match_len = min(len(audio_indices), n_audio_tokens)
            for i in range(match_len):
                full_embd[audio_indices[i]] = audio_features[i]
        else:
            for i, idx in enumerate(audio_indices):
                full_embd[idx] = audio_features[i]
        
        # 4. GGUF NAR 推理
        # 设置位置编码：Qwen3 是 4D-RoPE，[pos, pos, pos, 0]
        pos_base = np.arange(0, n_tokens, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(n_tokens, dtype=np.int32)])
        
        # 注意：4D-RoPE 必须申请 4 倍 Token 的 Batch 容量
        batch = llama.LlamaBatch(n_tokens * 4, embd_dim=1024)
        batch.set_embd(full_embd, pos=pos_arr)
        for i in range(n_tokens): batch.logits[i] = 1 # 我们需要一次性拿到全量 logits
        
        self.ctx.clear_kv_cache()
        self.ctx.decode(batch)
        
        # 5. 提取并解析 Logits
        ts_pos_indices = np.where(input_ids == self.ID_TIMESTAMP)[0]
        timestamp_indices = []
        for idx in ts_pos_indices:
            ptr = self.ctx.get_logits_ith(idx)
            # 对齐模型前 5000 个 Logits 对应 0-400s 的索引
            logits_arr = np.ctypeslib.as_array(ptr, shape=(152064,))
            timestamp_indices.append(np.argmax(logits_arr[:5000]))
            
        # 6. 后处理逻辑 (LIS 修正 + 毫秒转换)
        timestamp_ms = np.array(timestamp_indices) * self.TIMESTAMP_STEP_MS
        final_results = self.align_processor.parse_timestamp(word_list, timestamp_ms)
        
        t_total = time.time() - t_start
        print(f"--- [对齐完成] 耗时: {t_total:.3f}s ---")
        
        return final_results

# ==========================================
# 3. 测试入口
# ==========================================
def main():
    # 文件路径
    audio_path = "test.mp3"
    text_path = "test.txt"
    
    if not os.path.exists(text_path):
        print(f"❌ 未找到文本文件: {text_path}")
        return
        
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # 1. 初始化对齐器
    aligner = Qwen3GGUFForcedAligner()
    
    # 2. 执行对齐
    results = aligner.align(audio_path, text)
    
    # 3. 输出结果
    print("\n" + "="*50)
    print(f"{'Text':<20} | {'Start':<8} | {'End':<8}")
    print("-" * 50)
    for it in results:
        s = it['start_time'] / 1000.0
        e = it['end_time'] / 1000.0
        print(f"{it['text']:<20} | {s:7.3f}s | {e:7.3f}s")
    print("="*50)

if __name__ == "__main__":
    main()
