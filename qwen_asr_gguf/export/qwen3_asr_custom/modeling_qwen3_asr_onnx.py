# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Qwen3ASRFrontendFullOnnx(nn.Module):
    """
    Qwen3-ASR 完整前端 (Precision Wrapper)
    """
    def __init__(self, audio_tower):
        super().__init__()
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out
        self.pos_embed_table = audio_tower.positional_embedding.positional_embedding
        
    def _get_feat_extract_output_lengths(self, input_lengths):
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    def forward(self, input_features: torch.Tensor):
        b, mel, t = input_features.shape
        chunk_size = 100
        expected_len = self._get_feat_extract_output_lengths(t)
        pad_len = (chunk_size - (t % chunk_size)) % chunk_size
        if pad_len > 0:
            x = F.pad(input_features, (0, pad_len))
        else:
            x = input_features
        num_chunks = x.shape[2] // chunk_size
        x = x.transpose(1, 2).contiguous()
        x = x.view(num_chunks, chunk_size, mel).transpose(1, 2).unsqueeze(1)
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        b_c, c, f, t_c = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(num_chunks, t_c, c * f)
        x = self.conv_out(x)
        pos_embed = self.pos_embed_table[:t_c, :].unsqueeze(0)
        x = x + pos_embed
        x = x.view(1, -1, x.shape[-1])
        x = x[:, :expected_len, :]
        return x

class Qwen3ASRAudioAttentionOnnx(nn.Module):
    def __init__(self, raw_attn):
        super().__init__()
        self.num_heads = raw_attn.num_heads
        self.head_dim = raw_attn.head_dim
        self.scaling = raw_attn.scaling
        self.q_proj = raw_attn.q_proj
        self.k_proj = raw_attn.k_proj
        self.v_proj = raw_attn.v_proj
        self.out_proj = raw_attn.out_proj

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output

class Qwen3ASRBackendOnnx(nn.Module):
    def __init__(self, audio_tower):
        super().__init__()
        self.layers = nn.ModuleList()
        for raw_layer in audio_tower.layers:
            # Monkey batch
            raw_layer.self_attn = Qwen3ASRAudioAttentionOnnx(raw_layer.self_attn)
            self.layers.append(raw_layer)
        self.ln_post = audio_tower.ln_post
        self.proj1 = audio_tower.proj1
        self.act = audio_tower.act
        self.proj2 = audio_tower.proj2
        
    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            residual = hidden_states
            hidden_states = layer.self_attn_layer_norm(hidden_states)
            hidden_states = layer.self_attn(hidden_states, attention_mask=attention_mask)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.fc1(hidden_states)
            hidden_states = layer.activation_fn(hidden_states)
            hidden_states = layer.fc2(hidden_states)
            hidden_states = residual + hidden_states
        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states
