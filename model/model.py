import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from base import BaseModel, init_weights
import torch
import random
import numpy as np
import numbers

#transformer
import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


nl = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "leaky_relu": nn.LeakyReLU,
    "none": lambda x: x,
}

rnn = {
    "lstm": nn.LSTM,
    "gru": nn.GRU
}

rnn_default_initial_states = {
    "lstm": lambda h_size, bs, dev: (torch.zeros((1, bs, h_size)).to(dev), torch.zeros((1, bs, h_size)).to(dev)),
    "gru": lambda h_size, bs, dev: torch.zeros((1, bs, h_size)).to(dev)
}


# --------------- AUXILIAR CLASSES ---------------

class Embedding(BaseModel):
    def __init__(self, input_dim, embedding_dims=[], dropout=0.5, non_linearities='relu'):
        super(Embedding, self).__init__()
        self.non_linearities = non_linearities

        self.dropout = nn.Dropout(dropout)
        self.nl = nl[non_linearities]()

        self.denses = None
        if len(embedding_dims) > 0:
            seqs = []
            for i in range(len(embedding_dims)):
                linear = nn.Linear(input_dim if i==0 else embedding_dims[i-1], embedding_dims[i])
                init_weights(linear)
                seqs.append(nn.Sequential(self.dropout, linear, self.nl))
            self.denses = nn.Sequential(*seqs)

    def forward(self, x):
        return self.denses(x) if self.denses is not None else x


# --------------- TEST ---------------

class BasicMLP(BaseModel):
    # MLP with that predicts all traits at once.
    def __init__(self, input_dim, seq_length, output_dim=5, embedding_dims=[32, 32], dropout=0.5, non_linearities='relu'):
        super(BasicMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.nl = nl[non_linearities]()

        self.backbone = Embedding(input_dim*seq_length, embedding_dims=embedding_dims, dropout=dropout, non_linearities=non_linearities)
        
        input_head_dim = input_dim if len(embedding_dims) == 0 else embedding_dims[-1]
        self.head = nn.Sequential(nn.Linear(input_head_dim, output_dim), 
                                    nn.Softmax(1))

    def forward(self, x):
        x = rearrange(x, 'b t f -> b (t f)')
        x = self.backbone(x)
        return self.head(x)

class BasicMLPIndiv(BaseModel):
    # MLP with an exclusive head (MLP) for each OCEAN trait
    def __init__(self, input_dim, seq_length, output_dim=5, embedding_dims=[32, 32], dropout=0.5, non_linearities='relu'):
        super(BasicMLPIndiv, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.nl = nl[non_linearities]()

        self.backbone = Embedding(input_dim*seq_length, embedding_dims=embedding_dims, dropout=dropout, non_linearities=non_linearities)
        
        input_head_dim = input_dim if len(embedding_dims) == 0 else embedding_dims[-1]
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(input_head_dim, 1), nn.Sigmoid()) for i in range(output_dim)])

    def forward(self, x):
        #x = torch.zeros_like(x).to(x.device)
        x = rearrange(x, 'b t f -> b (t f)')
        x = self.backbone(x)
        pred = torch.hstack([h(x) for h in self.heads])
        return pred

# --------------- RNN ---------------

class EncoderRNN(BaseModel):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.5, rnn_type='lstm'):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.dropout = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        self.rnn = rnn[rnn_type](input_dim, hidden_dim, n_layers, batch_first=True)
        
        init_weights(self.rnn)

    def get_initial_state(self, batch_size, device):
        return rnn_default_initial_states[self.rnn_type](self.hidden_dim, batch_size, device)

    def forward(self, x, state=None):
        x = self.dropout(x)
        if state is None:
            outputs, state = self.rnn(x)#, (h0, c0))
        else:
            outputs, state = self.rnn(x, state)
        return outputs, state


class RecurrentNN(BaseModel):
    # Participants are split and each goes through the same encoder. They share NO information between them.
    def __init__(self, input_dim, output_dim, seq_length=100, rnn_type='lstm', hidden_dim=10, n_layers=(1,1), embedding_dims=[512, 512], linear_output_dims=[512, 512], dropout=0.5, non_linearities='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # batch_first => (batch_size, seq_len, features)
        self.embedding = Embedding(self.input_dim, embedding_dims, dropout=dropout, non_linearities=non_linearities)
        self.encoder = EncoderRNN(embedding_dims[-1], hidden_dim, n_layers[0], rnn_type=rnn_type, dropout=dropout) # same hidden dim for encoder and decoder
        self.head = nn.Sequential(Embedding(hidden_dim, linear_output_dims + [output_dim, ], dropout=dropout, non_linearities=non_linearities), 
                                nn.Softmax(1))

    def forward(self, x):
        x = self.embedding(x) # -> (batch_size, seq_length, embed_dim)
        output, state = self.encoder(x) # -> output is (batch_size, seq_length, hidden_size)
        output = output[:,-1] # we only take the output from the last step -> (batch_size, hidden_size)
        output = self.head(output) # -> (batch_size, 5)
        return output

# --------------- Transformer ---------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def get_attention_maps(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #print(f"Attention shape: {attn.shape}")
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def get_attention_maps(self, x):
        x, attn_map = self.attn.get_attention_maps(self.norm1(x))
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_map


class PoseTransformerTemporal(BaseModel):
    def __init__(self, seq_length=100, input_dim=15, output_dim=5, embed_dim_ratio=8, depth=4,
                 num_heads=8, mlp_ratio=1.5, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frames (int, tuple): input frames number
            input_features (int): number of features
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * input_dim   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        self.out_dim = embed_dim

        ### temporal patch embedding
        self.Temporal_patch_to_embedding = nn.Linear(input_dim, embed_dim)
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, seq_length, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=seq_length, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(nn.Linear(embed_dim, 5), 
                                    nn.Softmax(1))

    def forward(self, x):
        # INPUT: [batch_size, sequence, features]
        b, f, p = x.shape

        x = self.Temporal_patch_to_embedding(x)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x) # -> (b, f, embed_dim)
        x = self.weighted_mean(x) # -> (b, 1, embed_dim)
        x = x.view(b, -1) # -> (b, embed_dim)
        x = self.head(x) # -> (b, 5)
        return x
    