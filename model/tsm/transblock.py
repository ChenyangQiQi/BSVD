import torch.nn as nn
import torch
import math
from torch.nn import LayerNorm

class Mlp(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout = 0.):
        super(Attention, self).__init__()
        self.heads = heads
        context_dim = context_dim or query_dim
        hidden_dim = max(query_dim, context_dim)
        # self.dim_head = int(hidden_dim / self.heads)
        self.dim_head = dim_head
        self.all_head_dim = self.heads * self.dim_head

        ## All linear layers (including query, key, and value layers and dense block layers) 
        ## preserve the dimensionality of their inputs and are tiled over input index dimensions # 
        # (i.e. applied as a 1 × 1 convolution).
        self.query = nn.Linear(query_dim, self.all_head_dim) # (b n d_q) -> (b n hd)
        self.key = nn.Linear(context_dim, self.all_head_dim) # (b m d_c) -> (b m hd)
        self.value = nn.Linear(context_dim, self.all_head_dim) # (b m d_c) -> (b m hd)
        self.out = nn.Linear(self.all_head_dim, query_dim) # (b n d) -> (b n d)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.heads, self.dim_head)
        x = x.view(*new_x_shape) # (b n hd) -> (b n h d)
        return x.permute(0, 2, 1, 3) # (b n h d) -> (b h n d)

    def forward(self, query, context=None):
        if context is None:
            context = query
        mixed_query_layer = self.query(query) # (b n d_q) -> (b n hd)
        mixed_key_layer = self.key(context) # (b m d_c) -> (b m hd)
        mixed_value_layer = self.value(context) # (b m d_c) -> (b m hd)

        query_layer = self.transpose_for_scores(mixed_query_layer) # (b h n d)
        key_layer = self.transpose_for_scores(mixed_key_layer) # (b h m d)
        value_layer = self.transpose_for_scores(mixed_value_layer) # (b h m d)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (b h n m)
        attention_scores = attention_scores / math.sqrt(self.dim_head) # (b h n m)
        attention_probs = self.softmax(attention_scores) # (b h n m)
        attention_probs = self.attn_dropout(attention_probs) # (b h n m)

        context_layer = torch.matmul(attention_probs, value_layer) # (b h n m) , (b h m d) -> (b h n d)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (b h n d)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class TransBlock(nn.Module):
    def __init__(self, hidden_size, droppath=0., heads=8):
        super(TransBlock, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size)
        self.attn = Attention(hidden_size, heads=heads)
        # self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.drop_path =  nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.attn(self.attention_norm(x))) + x
        x = self.drop_path(self.ffn(self.ffn_norm(x))) + x
        return x


