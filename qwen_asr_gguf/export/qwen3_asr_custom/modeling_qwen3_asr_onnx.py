# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3ASRFrontendOnnx(nn.Module):
    """
    Qwen3-ASR 音频前端卷积部分 (CNN Wrapper)
    负责将 Mel 频谱 (B, 128, T) 进行 8 倍下采样并投影到隐藏层维度。
    """
    def __init__(self, audio_tower):
        super().__init__()
        # 直接引用官方模型的子模块
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out
        
        # 保存相关参数用于 view 变换
        self.config = audio_tower.config
        
    def forward(self, input_features: torch.Tensor):
        """
        Args:
            input_features: (Batch, Mel_Bins, Time) -> (B, 128, T)
        Returns:
            projected_embeds: (Batch, T_downsampled, Hidden_Size)
        """
        # 1. 准备输入：添加通道维度 (B, 1, 128, T)
        # 注意：导出的 ONNX 应该直接接受这个 4D 输入或者在内部处理 3D -> 4D
        x = input_features.unsqueeze(1)
        
        # 2. 三层卷积 + GELU (每次下采样 2 倍，共 8 倍)
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        
        # 3. 维度变换：B, C, F, T -> B, T, C*F
        b, c, f, t = x.shape
        # 这里使用 permute + reshape 以符合 Transformer 的输入格式
        # 官方逻辑是 permute(0, 3, 1, 2) 即 B, T, C, F 然后 view 成 B, T, C*F
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, t, c * f)
        
        # 4. 线性投影到 d_model 维度
        x = self.conv_out(x)
        
        return x
