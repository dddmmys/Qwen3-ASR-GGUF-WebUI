# coding=utf-8
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional
import numpy as np

class MsgType(Enum):
    CMD_ENCODE = auto()   # 主进程 -> Encoder: 编码请求
    CMD_STOP = auto()     # 主进程 -> Encoder: 停止请求
    MSG_EMBD = auto()     # Encoder -> 主进程: 返回特征
    MSG_READY = auto()    # Encoder -> 主进程: 就绪信号
    MSG_DONE = auto()     # Encoder -> 主进程: 已退出信号

@dataclass
class StreamingMessage:
    """音频编码进程通信协议"""
    msg_type: MsgType
    data: Any = None         # 存放音频 chunk 或 embedding 结果
    is_last: bool = False    # 标记是否为最后一段音频
    encode_time: float = 0.0 # 编码器实际耗时

@dataclass
class DecodeResult:
    """LLM 解码内核输出标准化"""
    text: str = ""           # 包含前缀的完整文本
    new_text: str = ""       # 本次增量生成的文本
    stable_tokens: List[int] = field(default_factory=list)
    t_prefill: float = 0.0   # 预填充耗时 (ms)
    t_generate: float = 0.0  # 生成耗时 (ms)
    n_prefill: int = 0       # 预填充 token 数
    n_generate: int = 0      # 生成 token 数
    is_aborted: bool = False # 是否因重复或其他原因熔断中断

@dataclass
class AlignerResult:
    """对齐结果标准化"""
    text: str
    start_time: float        # 单位：秒
    end_time: float          # 单位：秒
