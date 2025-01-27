# Copyright (c) 2025 Yang Li, Microsoft. All rights reserved.
# MIT License.
# 
# --------------------------------------------------------
# MoE Modules
#   Follow MoE in Mixtral(https://huggingface.co/docs/trans
#   formers/v4.48.0/en/model_doc/mixtral#overview)
# --------------------------------------------------------
# 
# Created on Mon Jan 27 2025.


import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from functools import partial
from jaxtyping import Float, Int
from typing import Tuple

from einops import rearrange, repeat


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Top2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = 4 * config.n_embd
        self.hidden_dim = config.config.n_embd

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = nn.GELU()

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class Router(nn.Module):
    def __init__(self,
                embed_dim: int,
                n_exp: int = 8,
                ):
        # Set up router net
        super().__init__()
        self.gate_net = nn.Linear(embed_dim, n_exp, bias=False)
        
    def forward(self,
                embeds: Float[Tensor, "*batch token embed"])->Float[Tensor, "*batch token exp"]:
     
        logits_exp = self.gate_net(embeds)
        return logits_exp
    
class MoELayer(nn.Module):
    def __init__(self,
            config,
            ):
        super().__init__()
        
        # Configs
        self.n_exp_per_token = config.n_exp_per_token
        self.jitter_noise = config.router_jitter_noise 
        self.embed_dim = config.n_embd
        self.n_exp = config.n_exp
        
        # Set up experts
        self.exps = nn.ModuleList([MLP(config) for _ in range(config.n_exp)])
        
        # Set up router
        self.router = Router(embed_dim = self.embed_dim, n_exp = self.n_exp)
        
    def forward(self, 
                x: Float[Tensor, "*batch token embed"]):
        
        batch, n_token, hidden_dim = x.size(0), x.size(-2), x.size(-1)
        
        # Add random noise for capacity balance 
        if self.training and self.jitter_noise > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        
        # routing
        x = rearrange(x, "batch token embed -> (batch token) embed")
        logits_exps = self.router(x)
        weights_exps = F.softmax(logits_exps, dim=-1, dtype=torch.float)
        weights_selected_exps, idx_selected_exps = torch.topk(weights_exps, self.n_exp_per_token, dim=-1) 
        weights_selected_exps = weights_selected_exps.to(x.dtype)
        
        # Exps forwarding 
        final_x = torch.zeros_like(x)
        
        mask_exps = rearrange(F.one_hot(idx_selected_exps, num_classes=self.n_exp), "batch_token exp_per_token n_exp -> n_exp exp_per_token batch_token")
        
        for id, exp in enumerate(self.exps):
            idx_exp, top_x = torch.where(mask_exps[id])
            
            # Indexing token for current expert
            x_curr = x[None, top_x].reshape(-1, hidden_dim)
            x_curr_hidden = exp(x_curr) * weights_exps[top_x, idx_exp, None]
            
            final_x.index_add_(0, top_x, x_curr_hidden.to(x.dtype))
            
        final_x = final_x.reshape(batch, n_token, hidden_dim)
        
        return final_x, logits_exps 
    
def balancing_loss_func(
    all_router_logits: Tuple[Float[Tensor, "*batch token exps"]],
    n_exp: int,
    n_exp_per_token=int,
) -> Tensor:

    all_router_logits = torch.cat([logits_exp for logits_exp in all_router_logits], dim=0)
    
    # Same as routing in MoELayer
    weights_exps = F.softmax(all_router_logits, dim=-1)

    _, idx_selected_exps = torch.topk(weights_exps, n_exp_per_token, dim=-1)

    mask_exps = F.one_hot(idx_selected_exps, n_exp)
 
    n_tokens_per_exp = torch.mean(mask_exps.float(), dim=0)

    prob_per_exp = torch.mean(weights_exps, dim=0)

    overall_loss = torch.sum(n_tokens_per_exp * prob_per_exp.unsqueeze(0))
    return overall_loss * n_exp
            