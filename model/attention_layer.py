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
    
    
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

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
