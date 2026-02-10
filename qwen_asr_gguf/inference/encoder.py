# coding=utf-8
import os
import time
import numpy as np
import onnxruntime as ort
import librosa
from .schema import MsgType, StreamingMessage

class FastWhisperMel:
    """基于 NumPy 和 Librosa 的 Mel 提取器"""
    def __init__(self, filter_path: str):
        self.filters = np.load(filter_path) # (201, 128)
        
    def __call__(self, audio: np.ndarray, dtype=np.float32) -> np.ndarray:
        # 1. STFT
        stft = librosa.stft(audio, n_fft=400, hop_length=160, window='hann', center=True)
        # 2. 能量谱
        magnitudes = np.abs(stft) ** 2
        # 3. Mel 映射
        mel_spec = np.dot(self.filters.T, magnitudes)
        # 4. 取对数
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        # 5. 归一化
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        # 6. 帧对齐：丢弃 stft(center=True) 产生的多余帧
        n_frames = audio.shape[-1] // 160
        log_spec = log_spec[:, :n_frames]
        return log_spec.astype(dtype)

def get_feat_lengths(t_mel: int) -> int:
    """计算下采样后的特征长度"""
    t_leave = t_mel % 100
    feat_len = (t_leave - 1) // 2 + 1
    out_len = ((feat_len - 1) // 2 + 1 - 1) // 2 + 1 + (t_mel // 100) * 13
    return int(out_len)

def encoder_worker_proc(to_enc_q, from_enc_q, encoder_path: str, mel_filters_path: str, warmup_sec: float = 0, use_dml: bool = True):
    """音频编码器后台进程"""
    # 初始化 ONNX Session
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['CPUExecutionProvider']
    if use_dml and  'DmlExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'DmlExecutionProvider') 
        
    try:
        encoder_sess = ort.InferenceSession(encoder_path, sess_options=sess_opts, providers=providers)
    except Exception as e:
        print(f"[Encoder] Initialization failed: {e}")
        return
    
    mel_extractor = FastWhisperMel(mel_filters_path)
    
    # 检测输入精度
    try:
        fe_input_type = encoder_sess.get_inputs()[0].type
        input_dtype = np.float16 if 'float16' in fe_input_type else np.float32
    except:
        input_dtype = np.float32

    # 预热选项
    if warmup_sec > 0:
        dummy_wav = np.random.randn(int(16000 * warmup_sec)).astype(np.float32)
        dummy_mel = mel_extractor(dummy_wav, dtype=input_dtype)
        dummy_mel_input = dummy_mel[np.newaxis, ...]
        t_out = get_feat_lengths(dummy_mel.shape[1])
        dummy_mask = np.zeros((1, 1, t_out, t_out), dtype=input_dtype)
        _ = encoder_sess.run(None, {"input_features": dummy_mel_input, "attention_mask": dummy_mask})
    
    # 宣告就绪
    from_enc_q.put(StreamingMessage(MsgType.MSG_READY))
    
    while True:
        msg: StreamingMessage = to_enc_q.get()
        
        if msg.msg_type == MsgType.CMD_STOP:
            from_enc_q.put(StreamingMessage(MsgType.MSG_DONE))
            break
            
        if msg.msg_type == MsgType.CMD_ENCODE:
            audio_chunk = msg.data
            t0 = time.time()
            
            mel = mel_extractor(audio_chunk, dtype=input_dtype) 
            mel_input = mel[np.newaxis, ...]
            
            t_mel = mel.shape[1]
            t_out = get_feat_lengths(t_mel)
            mask_input = np.zeros((1, 1, t_out, t_out), dtype=input_dtype)
            
            audio_embd = encoder_sess.run(None, {
                "input_features": mel_input,
                "attention_mask": mask_input
            })[0]
            
            if audio_embd.ndim == 3: audio_embd = audio_embd[0]
            
            encode_time = time.time() - t0
            from_enc_q.put(StreamingMessage(MsgType.MSG_EMBD, data=audio_embd, is_last=msg.is_last, encode_time=encode_time))
