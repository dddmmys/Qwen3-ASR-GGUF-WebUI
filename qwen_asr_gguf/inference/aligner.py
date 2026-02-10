# coding=utf-8
import time
import unicodedata
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Optional
from pathlib import Path

from .schema import AlignerResult
from .encoder import FastWhisperMel, get_feat_lengths
from . import llama

class AlignerProcessor:
    """文本预处理与时间戳修正逻辑"""
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

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for seg in text.split():
            cleaned = self.clean_token(seg)
            if not cleaned: continue
            # 处理中文字符切分
            buf = []
            for ch in cleaned:
                if self.is_cjk_char(ch):
                    if buf: tokens.append("".join(buf)); buf = []
                    tokens.append(ch)
                else: buf.append(ch)
            if buf: tokens.append("".join(buf))
        return tokens

    def fix_timestamps(self, data: np.ndarray) -> np.ndarray:
        """LIS (最长递增子序列) 修正算法"""
        n = len(data)
        if n == 0: return data
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
        
        result = data.copy()
        i = 0
        while i < n:
            if not is_normal[i]:
                j = i
                while j < n and not is_normal[j]: j += 1
                anomaly_count = j - i
                left = next((result[k] for k in range(i-1, -1, -1) if is_normal[k]), None)
                right = next((result[k] for k in range(j, n) if is_normal[k]), None)
                if anomaly_count <= 2:
                    for k in range(i, j): result[k] = right if left is None else (left if right is None else (left if (k-i+1) <= (j-k) else right))
                else:
                    if left is not None and right is not None:
                        step = (right - left) / (anomaly_count + 1)
                        for k in range(i, j): result[k] = int(left + step * (k - i + 1))
                    elif left is not None: result[i:j] = left
                    elif right is not None: result[i:j] = right
                i = j
            else: i += 1
        return result

class QwenForcedAligner:
    """Qwen3 强制对齐器 (GGUF + ONNX 版)"""
    def __init__(self, encoder_onnx: str, llm_gguf: str, mel_filters: str, verbose: bool = True, use_dml: bool = True):
        self.verbose = verbose
        if verbose: print("--- [Aligner] 初始化对齐器 ---")
        
        # 1. 编码器 (同步加载，非流式)
        opt = ort.SessionOptions()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        if use_dml and  'DmlExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'DmlExecutionProvider') 
            
        self.encoder = ort.InferenceSession(encoder_onnx, sess_options=opt, providers=providers)
        self.mel_extractor = FastWhisperMel(mel_filters)
        
        # 识别精度
        try:
            fe_type = self.encoder.get_inputs()[0].type
            self.input_dtype = np.float16 if 'float16' in fe_type else np.float32
        except: self.input_dtype = np.float32

        # 2. LLM GGUF
        self.model = llama.LlamaModel(llm_gguf)
        self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf)
        self.ctx = llama.LlamaContext(self.model, n_ctx=4096, n_batch=4096)
        
        self.processor = AlignerProcessor()
        self.ID_AUDIO_PAD = 151676
        self.ID_TIMESTAMP = self.model.token_to_id("<timestamp>")
        self.STEP_MS = 80.0

    def align(self, audio: np.ndarray, text: str, language: str = "Chinese") -> List[AlignerResult]:
        t0 = time.time()
        
        # 1. Encoder 推理
        mel = self.mel_extractor(audio, dtype=self.input_dtype)
        mel_input = mel[np.newaxis, ...]
        t_out = get_feat_lengths(mel.shape[1])
        mask_input = np.zeros((1, 1, t_out, t_out), dtype=self.input_dtype)
        
        audio_embd = self.encoder.run(None, {"input_features": mel_input, "attention_mask": mask_input})[0]
        if audio_embd.ndim == 3: audio_embd = audio_embd[0]
        n_aud = audio_embd.shape[0]
        
        # 2. 构造 Prompt
        words = self.processor.tokenize(text)
        align_text = "<|audio_start|><|audio_pad|><|audio_end|>" + "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
        # 展开占位符
        expanded_audio = f"<|audio_start|>{'<|audio_pad|>' * n_aud}<|audio_end|>"
        full_text = align_text.replace("<|audio_start|><|audio_pad|><|audio_end|>", expanded_audio)
        
        input_ids = self.model.tokenize(full_text, add_special=False, parse_special=True)
        n_tok = len(input_ids)
        
        # 3. 构造 Embedding
        full_embd = self.embedding_table[input_ids].copy()
        audio_indices = np.where(np.array(input_ids) == self.ID_AUDIO_PAD)[0]
        for i, idx in enumerate(audio_indices):
            if i < n_aud: full_embd[idx] = audio_embd[i]
            
        # 4. GGUF NAR 推理 (4D-RoPE)
        pos_base = np.arange(0, n_tok, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(n_tok, dtype=np.int32)])
        batch = llama.LlamaBatch(n_tok * 4, embd_dim=1024)
        batch.set_embd(full_embd, pos=pos_arr)
        for i in range(n_tok): batch.logits[i] = 1
        
        self.ctx.clear_kv_cache()
        self.ctx.decode(batch)
        
        # 5. 解析 Logits
        ts_indices = np.where(np.array(input_ids) == self.ID_TIMESTAMP)[0]
        raw_ts = []
        for idx in ts_indices:
            logits_ptr = self.ctx.get_logits_ith(idx)
            logits = np.ctypeslib.as_array(logits_ptr, shape=(152064,))
            raw_ts.append(np.argmax(logits[:5000])) # 0-400s
            
        # 6. 后处理
        fixed_ts = self.processor.fix_timestamps(np.array(raw_ts))
        ms = fixed_ts * self.STEP_MS
        
        results = [
            AlignerResult(text=w, start_time=ms[i*2]/1000.0, end_time=ms[i*2+1]/1000.0)
            for i, w in enumerate(words)
        ]
        
        if self.verbose:
            print(f"--- [Aligner] 对齐完成，耗时: {time.time()-t0:.2f}s ---")
        return results
