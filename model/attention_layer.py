import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from timm.layers import trunc_normal_
from transformers.cache_utils import Cache

from typing import Dict, Iterable, Optional, Tuple

try:
    from torch.nn.functional import scaled_dot_product_attention
    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

try:
    from flash_attn import flash_attn_func
except (ImportError, RuntimeError, OSError):
    flash_attn_func = None


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, RMSNorm))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class RoPE(nn.Module):
    def __init__(self, dim, scale=40):
        super(RoPE,self).__init__()
        assert dim % 2 == 0
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim//2, 2).float() / (dim//2)))
        self.register_buffer('inv_freq',inv_freq)
        self.scale = scale

    def forward(self,x):
        t = torch.arange(x, device=self.inv_freq.device).type_as(self.inv_freq) / self.scale
        freqs = torch.einsum('i,j->ij',t, self.inv_freq)
        return torch.cat([freqs,freqs],dim=-1)
    
def rotate_half(x):
    x1, x2 = x.chunk(2,dim=-1)
    return torch.cat([-x2,x1],dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )
def apply_rotary(x,cos,sin):
    x_rot, x_base = x.split(cos.shape[-1], dim=-1)
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat([x_rot,x_base],dim=-1)

def set_lambda(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


# class  MHSA(nn.Module):
#     def __init__(self, 
#         embed_dim,
#         num_heads,
#         dropout_p=0.1,
#         cross_attn=False
#     ):
#         super(MHSA, self).__init__()
#         self.cross_attn = cross_attn
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dims = embed_dim // num_heads
        
#         self.num_kv_heads = num_heads
#         self.n_rep = self.num_heads // self.num_kv_heads
#         self.dropout_p = dropout_p
#         self.scale = self.head_dims ** -0.5

#         self.q = nn.Linear(embed_dim,embed_dim)
#         self.k = nn.Linear(embed_dim,embed_dim,bias=False)
#         self.v = nn.Linear(embed_dim,embed_dim)

#         self.out_proj = nn.Linear(embed_dim,embed_dim)
#         self.norm = nn.LayerNorm(embed_dim,eps=1e-6,elementwise_affine=True)

        
#     def forward(
#         self,
#         x,
#         xa=None,
#         mask=None,
#         kv_cache=None,
#     ):
        
#         b, tgt_len, dim = x.size()
#         q = self.q(x)

            
#         if kv_cache is None or xa is None or self.k not in kv_cache:
#             k = self.k(x if xa is None else xa)
#             v = self.v(x if xa is None else xa)
#         else:
#             k = kv_cache[self.k]
#             v = kv_cache[self.v]
    
#         q = rearrange(q,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims) 
#         k = rearrange(k,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)
#         v = rearrange(v,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)

#         if scaled_dot_product_attention:
#             attn = scaled_dot_product_attention(q,k,v,is_causal=mask is not None and tgt_len > 1)
#             attn = rearrange(attn,'b h l d -> b l (h d)')
#         else:
#             attn = self.attn(q,k,v,dropout_p=self.dropout_p,mask=mask)
        
#         out = self.out_proj(attn)
#         return out
    
#     def attn(self,q,k,v,dropout_p,mask):
#         b, h, l, d = q.size()
        
#         qk = torch.einsum('bhld,bhkd->bhlk',q,k) * self.scale
#         if not self.cross_attn:
#             qk += mask[:l,:l]
#         qk = qk.softmax(dim=-1)
#         qk = F.dropout(qk,p=dropout_p)
#         attn = torch.einsum('bhlk,bhkd->bhld',qk,v)
#         output = rearrange(attn,'b h l d -> b l (h d)')
#         return output

class  MLA(nn.Module):
    def __init__(self, 
        hidden_states,
        kv_comp,
        d_rope,
        layer_idx,
        num_heads=8,
        dropout_p=0.0,
        cross_attn=False,
        share_compression=False,
        max_position_embeddings=2048
    ):
        super(MLA, self).__init__()
        self.cross_attn = cross_attn
        self.hidden_states = hidden_states
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.head_dims = hidden_states // num_heads
        self.split_dim = self.head_dims - d_rope
        self.d_rope = d_rope
        self.max_position_embeddings = max_position_embeddings
        self.dropout_p = dropout_p
        self.share_compression = share_compression
        self.scale = math.sqrt(self.head_dims)

        self.w_dq = nn.Linear(hidden_states,kv_comp)
        self.w_uq = nn.Linear(kv_comp,self.num_heads * self.split_dim)
        self.w_qr = nn.Linear(kv_comp,self.num_heads * d_rope)

        self.w_dkv = nn.Linear(hidden_states,kv_comp)
        self.w_uk = nn.Linear(kv_comp,self.num_heads * self.split_dim)
        self.w_uv = nn.Linear(kv_comp,self.num_heads * self.head_dims)

        self.w_kr = nn.Linear(hidden_states,self.num_heads * d_rope)

        self.rotary_emb = RotaryEmbedding(
            self.num_heads,
            max_position_embeddings=self.max_position_embeddings,
        )

        self.out_proj = nn.Linear(hidden_states,hidden_states)
        self.norm = nn.LayerNorm(hidden_states,eps=1e-6,elementwise_affine=True)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None
    ):
        b, tgt_len, dim = hidden_states.size()
        
        # Compression Q
        c_q = self.w_dq(hidden_states)
        q_c = rearrange(self.w_uq(c_q), 'b l (h k) -> b h l k', h=self.num_heads, k=self.split_dim)
        q_r = rearrange(self.w_qr(c_q), 'b l (h r) -> b h l r', h=self.num_heads, r=self.d_rope)

        # Compression K, V
        if past_key_value is not None:
            c_kv = past_key_value[self.w_dkv]
            k_r = rearrange(past_key_value[self.w_kr],'b l (h r) -> b h l r', h=self.num_heads, r=self.d_rope)
        else:
            c_kv = self.w_dkv(hidden_states)
            k_r = rearrange(self.w_kr(hidden_states), 'b l (h r) -> b h l r', h=self.num_heads, r=self.d_rope)

        k = rearrange(self.w_uk(c_kv), 'b l (h k) -> b h l k', h=self.num_heads, k=self.split_dim)
        v = rearrange(self.w_uv(c_kv), 'b l (h d) -> b h l d', h=self.num_heads, d=self.head_dims)
        

        kv_seq_len = k.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        cos, sin = self.rotary_emb(k_r, seq_len=kv_seq_len)

        # cos, sin = self.rotary_emb(seq_len=kv_seq_len, device=hidden_states.device)
        q_r, k_r = apply_rotary_pos_emb(q_r, k_r, cos, sin, position_ids)

        q = torch.cat([q_c, q_r], dim=-1)
        k = torch.cat([k, k_r], dim=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        if scaled_dot_product_attention and not flash_attn_func:
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.dropout_p if self.training else 0)
        elif flash_attn_func:
            attn_output = flash_attn_func(q, k, v, causal=True)
        else:
            attn_output = self.attn(q,k,v,self.dropout_p,attention_mask)

        attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')
        out = self.out_proj(attn_output)
        return out
    
    def attn(self,q,k,v,dropout_p,mask):
        b, h, l, d = q.size()
        
        qk = torch.einsum('bhld,bhkd->bhlk',q,k) * self.scale
        if not self.cross_attn:
            qk += mask[:l,:l]
        qk = qk.softmax(dim=-1)
        qk = F.dropout(qk,p=dropout_p)
        attn = torch.einsum('bhlk,bhkd->bhld',qk,v)
        output = rearrange(attn,'b h l d -> b l (h d)')
        return output

class Expert(nn.Module):
    def __init__(self,embed_dim,ffn_dim):
        super(Expert,self).__init__()
        self.fc1 = nn.Linear(embed_dim,ffn_dim)
        self.fc2 = nn.Linear(ffn_dim,embed_dim)

    def forward(self,x):
        return self.fc2(F.gelu(self.fc1(x)))


class SwishGLU(nn.Module):
    def __init__(self,embed_dim,expand):
        super(SwishGLU,self).__init__()
        dim_inner = embed_dim * expand
        self.fc1 = nn.Linear(embed_dim,dim_inner)
        self.fc2 = nn.Linear(embed_dim,dim_inner)
        self.fc3 = nn.Linear(dim_inner,embed_dim)

    def forward(self,x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)
    
class MoE(nn.Module):
    def __init__(self,hidden_states,ffn_dim,top_k,n_shared,n_experts):
        super(MoE,self).__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.shared_experts = nn.ModuleList([
            Expert(hidden_states,ffn_dim)

            for _ in range(n_shared)
        ])
        self.router_experts = nn.ModuleList([
            Expert(hidden_states,ffn_dim)
            for _ in range(n_experts)
        ])
        self.gate = nn.Linear(hidden_states,n_experts)
        self.aux_loss = 0

    def forward(self,x):
        shared_out = sum(expert(x) for expert in self.shared_experts)

        routed_logits = self.gate(x)
        probs = F.softmax(routed_logits, dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)

        expert_counts = torch.zeros(self.n_experts,device=x.device)
        expert_counts.scatter_add(
            0, 
            top_idx.view(-1), 
            torch.ones_like(top_idx.view(-1), 
            dtype=torch.float)
        )
        self.aux_loss += expert_counts.float().var() * 0.003

        routed_out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_mask = top_idx[...,k]
            expert_contrib = torch.zeros_like(x).to(torch.bfloat16)

            for expert_idx in range(self.n_experts):
                mask = (expert_mask == expert_idx)
                if mask.any():
                    expert_out = self.router_experts[expert_idx](x[mask])
                    expert_contrib[mask] = expert_out * top_probs[...,k][mask].unsqueeze(-1)
            routed_out += expert_contrib
        return shared_out + routed_out
    

# class QFormer(nn.Module):
#     def __init__(self, 
#                  embed_dim,
#                  num_heads,
#                  cross_attn,
#                  expand=2
#                  ):
#         super(QFormer, self).__init__()

#         self.attn = MHSA(
#             embed_dim=embed_dim,
#             num_heads=embed_dim//64,
#             cross_attn=False,
#         )

#         self.norm1 = RMSNorm(embed_dim,eps=1e-6)

#         self.cross_attn = (
#             MHSA(
#             embed_dim=embed_dim,
#             num_heads=embed_dim//64,
#             cross_attn=True,
#         ) if cross_attn else None)
        
        
#         self.norm2 = (RMSNorm(embed_dim,eps=1e-6) if cross_attn else None)

#         self.SwiGLU = SwishGLU(embed_dim,expand=expand)
        
#         self.norm3 = RMSNorm(embed_dim,eps=1e-6)


#     def forward(self, 
#         x: Tensor,
#         xa: Optional[Tensor] = None,
#         mask: Optional[Tensor] = None,
#     ):
#         x = x + self.attn(x=self.norm1(x),mask=mask)
        
#         if self.cross_attn:
#             x = x + self.cross_attn(x=self.norm2(x),xa=xa)

#         x = x + self.SwiGLU(self.norm3(x))
#         return x
    
class DSConv(nn.Module):
    def __init__(
        self,
        embed_dim,
        kernel_size,
        pooling_size=2,
    ):
        super().__init__()

        self.dconv = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size // 2), groups=embed_dim, bias=False)
        self.pconv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.norm = RMSNorm(embed_dim, eps=1e-5)
        self.act_fn = nn.SiLU()
        self.pool = nn.AvgPool1d(kernel_size, stride=pooling_size,padding=(kernel_size//2))

    def forward(self, x):
        residual = x
        x_permuted = x.permute(0, 2, 1).contiguous()
        x_permuted = self.dconv(x_permuted)
        x_permuted = self.pconv(x_permuted)
        x = x_permuted.permute(0, 2, 1).contiguous()
        x = self.act_fn(self.norm(x))
        x = x + residual
        x = x.permute(0, 2, 1).contiguous()
        x = self.pool(x)
        x = x.permute(0, 2, 1).contiguous()
        return x

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# class Resampler(nn.Module):
#     def __init__(
#             self, 
#             embed_dim,
#             num_heads,
#             num_query,
#             n_ctx,
#         ):
#         super(Resampler, self).__init__()

#         self.num_heads = num_heads
#         self.head_dims = embed_dim // num_heads
#         self.n_ctx = n_ctx
        
#         self.query = nn.Parameter(torch.zeros(1,num_query,embed_dim))
#         nn.init.normal_(self.query, mean=0.0, std=0.02)

#         self.q_ln = nn.LayerNorm(embed_dim,eps=1e-5)
#         self.kv_ln = nn.LayerNorm(embed_dim,eps=1e-5)
#         self.kv_proj = nn.Identity()
#         self.attn = nn.MultiheadAttention(embed_dim, self.num_heads,batch_first=True)

#         self.register_buffer("q_pos_embeds", sinusoids(num_query, embed_dim))
#         self.register_buffer("kv_pos_embeds", sinusoids(n_ctx, embed_dim))

#     def forward(self, 
#         x: Tensor,
#         mask: Optional[Tensor] = None,
#     ):
#         q = self.q_ln(self.query.to(x.device))
#         x = self.kv_ln(self.kv_proj(x))
#         out, attn = self.attn(
#             q+self.q_pos_embeds,
#             x+self.kv_pos_embeds,
#             x
#         )
#         return out

class Resampler(nn.Module):
    def __init__(
            self, 
            embed_dim,
            num_heads,
            num_query,
            n_ctx,
        ):
        super(Resampler, self).__init__()

        self.num_heads = num_heads
        self.head_dims = embed_dim // num_heads
        self.n_ctx = n_ctx
        
        self.query = nn.Parameter(torch.randn(1,num_query,embed_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        
        self.q_ln = nn.LayerNorm(embed_dim,eps=1e-5)
        self.kv_ln = nn.LayerNorm(embed_dim,eps=1e-5)
        
        self.kv_proj = nn.Identity()
        self.out_proj = nn.Linear(embed_dim,embed_dim)

        self.register_buffer("q_pos_embeds", self.sinusoids(num_query, embed_dim))
        self.register_buffer("kv_pos_embeds", self.sinusoids(n_ctx, embed_dim))
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.constant_(self.q_ln.bias, 0)
        nn.init.constant_(self.q_ln.weight, 1.0)
        nn.init.constant_(self.kv_ln.bias, 0)
        nn.init.constant_(self.kv_ln.weight, 1.0)
    
    def sinusoids(self, length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


    def forward(self, 
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.q_ln(self.query.to(x.device))
        x = self.kv_ln(self.kv_proj(x))

        q = rearrange(q + self.q_pos_embeds,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims) 
        k = rearrange(x + self.kv_pos_embeds,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)
        v = rearrange(x,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)

        attn = scaled_dot_product_attention(q,k,v)
        attn = rearrange(attn,'b h l d -> b l (h d)')
        x = self.out_proj(attn)
        return x


class  MHSA(nn.Module):
    def __init__(self, 
        embed_dim,
        num_heads,
    ):
        super(MHSA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dims = embed_dim // num_heads

        self.q = nn.Linear(embed_dim,embed_dim,bias=True)
        self.k = nn.Linear(embed_dim,embed_dim,bias=False)
        self.v = nn.Linear(embed_dim,embed_dim,bias=True)

        self.out_proj = nn.Linear(embed_dim,embed_dim,bias=True)
    
    def forward(
        self,
        x,
        xa=None,
        mask=None,
    ):
        
        b, tgt_len, dim = x.size()
        q = self.q(x)
        k = self.k(x if xa is None else xa)
        v = self.v(x if xa is None else xa)
       
        q = rearrange(q,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims) 
        k = rearrange(k,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)
        v = rearrange(v,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)

        attn = scaled_dot_product_attention(q,k,v,is_causal=mask is not None)
        attn = rearrange(attn,'b h l d -> b l (h d)')
       
        out = self.out_proj(attn)
        return out

class Attention(nn.Module):
    def __init__(
        self, 
        embed_dim,
        num_heads,
    ):
    
        super(Attention, self).__init__()
        self.attn = MHSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.cross_attn = MHSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.norm1 = nn.LayerNorm(embed_dim,eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim,eps=1e-5)

    def forward(self, 
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(x=self.norm2(x),xa=xa)            
        return x

class SwishGLU(nn.Module):
    def __init__(self,embed_dim,expand=4):
        super(SwishGLU,self).__init__()
        dim_inner = embed_dim * expand
        self.fc1 = nn.Linear(embed_dim,dim_inner)
        self.fc2 = nn.Linear(embed_dim,dim_inner)
        self.fc3 = nn.Linear(dim_inner,embed_dim)

    def forward(self,x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)

# class  MHSA(nn.Module):
#     def __init__(self, 
#         embed_dim,
#         num_heads,
#         cross_attn=True,
#     ):
#         super(MHSA, self).__init__()
#         self.cross_attn = cross_attn
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dims = embed_dim // num_heads
        
#         self.num_kv_heads = num_heads
#         self.scale = self.head_dims ** -0.5

#         self.q = nn.Linear(embed_dim,embed_dim)
#         self.k = nn.Linear(embed_dim,embed_dim,bias=False)
#         self.v = nn.Linear(embed_dim,embed_dim)

#         self.out_proj = nn.Linear(embed_dim,embed_dim)

#     def attn(self,q,k,v,mask=None):
#         b, h, l, d = q.size()
        
#         qk = torch.einsum('bhld,bhkd->bhlk',q,k) * self.scale
#         if not self.cross_attn:
#             qk += mask[:l,:l]
#         attn_score = qk.softmax(dim=-1)
#         # attn_score = F.dropout(attn_score,p=dropout_p)
#         attn = torch.einsum('bhlk,bhkd->bhld',attn_score,v)
#         return attn, attn_score

#     def forward(
#         self,
#         x,
#         xa=None,
#         mask=None,
#         attn_weight=False
#     ):
        
#         b, tgt_len, dim = x.size()
#         q = self.q(x)
#         k = self.k(x if xa is None else xa)
#         v = self.v(x if xa is None else xa)
       
#         q = rearrange(q,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims) 
#         k = rearrange(k,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)
#         v = rearrange(v,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)

#         if not attn_weight:
#             attn = scaled_dot_product_attention(q,k,v,is_causal=mask is not None)
#             attn_weights = None
#         else:
#             attn, attn_weights = self.attn(q,k,v)
#         attn = rearrange(attn,'b h l d -> b l (h d)')
       
#         out = self.out_proj(attn)
#         return out, attn_weights

# class Attention(nn.Module):
#     def __init__(self, 
#                  embed_dim,
#                  num_heads,
#                  cross_attn,
#                  num_query,
#                  n_ctx,
#                  ):
#         super(Attention, self).__init__()
#         self.attn = Resampler(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             num_query=num_query,
#             n_ctx=n_ctx
#         )

#         self.norm = RMSNorm(embed_dim,eps=1e-5)

#         self.cross_attn = (
#             MHSA(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             cross_attn=True
#         ) if cross_attn else None)

#     def forward(self, 
#         x: Tensor,
#         xa: Optional[Tensor] = None,
#         mask: Optional[Tensor] = None,
#         attn_weight: Optional[bool] = False
#     ):
#         x = self.attn(x)
        
#         if self.cross_attn:
#             residual = x
#             x, attn_weights = self.cross_attn(x=self.norm(x),xa=xa,attn_weight=attn_weight)
#             x = x + residual
#         return x, attn_weights




class QAttention(nn.Module):
    def __init__(self, 
                 embed_dim,
                 num_heads,
                 cross_attn,
                 num_query,
                 n_ctx,
                 expand=4
                 ):
        super(QAttention, self).__init__()

        self.attn = Resampler_v1(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_query=num_query,
            n_ctx=n_ctx,
        )

        self.norm1 = RMSNorm(embed_dim,eps=1e-5)

        self.cross_attn = (
            MHSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            cross_attn=True,
        ) if cross_attn else None)
        
        
        self.norm2 = (RMSNorm(embed_dim,eps=1e-5) if cross_attn else None)

        self.SwiGLU = SwishGLU(embed_dim,expand=expand)

        self.norm3 = RMSNorm(embed_dim,eps=1e-5)

    def forward(self, 
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        x = self.attn(x)
        
        if self.cross_attn:
            x = x + self.cross_attn(x=self.norm1(x),xa=xa)

        x = x + self.SwiGLU(self.norm2(x))
        return x