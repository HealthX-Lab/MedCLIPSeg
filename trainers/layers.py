import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ProbCrossAttention(nn.Module):
    """
    Simple and stable probabilistic cross-attention:
      - keys/values each have mean + variance
      - softplus to keep variance positive
      - attention scores adjusted by key uncertainty
    """
    def __init__(self, dim, beta: float = 2.35, gate_init: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim * 2)  # mean + logvar
        self.v_proj = nn.Linear(dim, dim * 2)  # mean + logvar
        self.out_proj = nn.Linear(dim, dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.eps = 1e-6
        self.gate = nn.Parameter(torch.tensor(gate_init))  # Initial residual mix
        self.beta = beta

    def forward(self, query, context, sample=True, num_samples=1):
        B, Tq, C = query.shape
        _, Tk, _ = context.shape

        Q = self.q_proj(query)  # [B, Tq, C]

        # Keys: mean + variance
        K_out = self.k_proj(context)
        K_mu, K_logvar = K_out[..., :C], K_out[..., C:]
        K_mu = self.norm_k(K_mu)
        K_var = F.softplus(K_logvar) + self.eps  # positive

        # Values: mean + variance
        V_out = self.v_proj(context)
        V_mu, V_logvar = V_out[..., :C], V_out[..., C:]
        V_mu = self.norm_v(V_mu)
        V_var = F.softplus(V_logvar) + self.eps  # positive

        # Attention scores
        scale = math.sqrt(C)
        mean_scores = torch.matmul(Q, K_mu.transpose(1, 2)) / scale

        var_penalty = torch.matmul(Q.pow(2), K_var.transpose(1, 2)) / C
        scores = mean_scores - self.beta * torch.sqrt(var_penalty)
    
        attn_weights = F.softmax(scores, dim=-1)

        eps = torch.randn_like(V_var)
        V_sample = V_mu + torch.sqrt(V_var) * eps
        out = torch.matmul(attn_weights, V_sample) # average over samples

        gate = torch.sigmoid(self.gate)
        proj_out = self.out_proj(out)
        fused = gate * proj_out + (1 - gate) * query

        return fused

class TwoWayTransformerLayer(nn.Module):
    def __init__(self, embed_dim, beta=2.35, gate_init=0.0):
        super().__init__()
        self.cross_attn_img_to_txt = ProbCrossAttention(embed_dim, beta, gate_init)
        self.cross_attn_txt_to_img = ProbCrossAttention(embed_dim, beta, gate_init)

    def forward(self, img_tokens, txt_tokens):
        img_tokens = self.cross_attn_img_to_txt(img_tokens, txt_tokens)
        txt_tokens = self.cross_attn_txt_to_img(txt_tokens, img_tokens)
        return img_tokens, txt_tokens

class PVL_Adapter(nn.Module):
    def __init__(self,
                 in_channels_vis: int,
                 in_channels_txt: int,
                 adapter_channels: int,
                 beta: float,
                 gate_init: int):
        
        super().__init__()

        # Down projection
        self.proj_vis_down = nn.Sequential(nn.Linear(in_channels_vis, adapter_channels, bias=False))
        self.proj_txt_down = nn.Linear(in_channels_txt, adapter_channels, bias=False)

        # Up projection
        self.proj_vis_up = nn.Linear(adapter_channels, in_channels_vis, bias=False)
        self.proj_txt_up = nn.Linear(adapter_channels, in_channels_txt, bias=False)

        # Cross-modal interaction
        self.two_way = TwoWayTransformerLayer(adapter_channels, beta, gate_init)

    def forward(self, vis, text):

        v = self.proj_vis_down(vis)
        t = self.proj_txt_down(text)

        v_fused, t_fused = self.two_way(v, t)

        vis_out = self.proj_vis_up(v_fused)
        txt_out = self.proj_txt_up(t_fused)

        return vis_out, txt_out