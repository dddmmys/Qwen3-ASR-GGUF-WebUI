# -*- coding: utf-8 -*-
"""
波形生成模块
负责从音频文件生成波形数据，用于前端绘制波形图。
依赖 pydub 和 numpy 库，需要安装 ffmpeg 以支持多种音频格式。
"""

import os
import numpy as np
from pydub import AudioSegment


def generate_waveform(audio_filepath, num_points=500):
    """
    从音频文件生成波形数据（归一化振幅序列）和音频时长。

    该函数使用 pydub 加载音频文件，将其转换为单声道，然后提取原始样本值。
    如果样本点数超过指定的 num_points，则进行等间隔下采样；
    否则直接使用所有样本点。最后将振幅归一化到 [-1, 1] 范围内。

    参数:
        audio_filepath (str): 音频文件的完整路径（支持常见格式如 wav, mp3, flac 等）
        num_points (int, optional): 期望生成的波形点数，默认为 500。
            如果实际样本点数少于该值，则返回所有点。

    返回:
        tuple: 包含两个元素的元组
            - waveform (list): 归一化后的振幅列表，每个元素为 float，范围约为 [-1, 1]
            - duration (float): 音频时长，单位为秒

    抛出:
        FileNotFoundError: 如果音频文件不存在
        pydub.exceptions.CouldntDecodeError: 如果音频格式无法解码
        Exception: 其他 pydub 或 numpy 相关错误
    """
    # ==================== 加载音频文件 ====================
    # 使用 pydub 的 AudioSegment 加载音频文件
    # 该函数会自动根据文件扩展名选择合适的解码器
    audio_segment = AudioSegment.from_file(audio_filepath)

    # 获取音频时长（毫秒）
    duration_milliseconds = len(audio_segment)

    # 转换为秒
    duration_seconds = duration_milliseconds / 1000.0

    # ==================== 转换为单声道 ====================
    # 如果音频是多声道的，转换为单声道以避免后续处理复杂化
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)

    # ==================== 获取原始样本数组 ====================
    # 从 AudioSegment 对象中获取样本值数组（int16 范围）
    raw_samples = np.array(audio_segment.get_array_of_samples())

    # 样本总数
    total_samples = len(raw_samples)

    # ==================== 下采样处理 ====================
    # 如果样本数小于或等于目标点数，则直接使用所有样本
    if total_samples <= num_points:
        # 直接使用原始样本，转换为 float32 并归一化
        # 原始样本是 int16，范围 -32768 ~ 32767，除以 2^15 归一化到 -1 ~ 1
        waveform_normalized = raw_samples.astype(np.float32) / (2**15)

    else:
        # 需要下采样：等间隔选取 num_points 个点
        # 生成均匀分布的索引，包含首尾
        indices = np.linspace(
            start=0, stop=total_samples - 1, num=num_points, dtype=int
        )

        # 根据索引选取样本
        sampled_samples = raw_samples[indices]

        # 归一化
        waveform_normalized = sampled_samples.astype(np.float32) / (2**15)

    # ==================== 返回结果 ====================
    # 将 numpy 数组转换为 Python 列表，以便 JSON 序列化
    waveform_list = waveform_normalized.tolist()

    return waveform_list, duration_seconds
