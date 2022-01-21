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
    def __init__(self, fv_version, input_dim, embedding_dims=[], orders=[0, ], dropout=0.5, non_linearities='relu', contiguous=False):
        super(Embedding, self).__init__()
        self.fv_version = fv_version
        self.orders = orders
        self.non_linearities = non_linearities
        self.contiguous = contiguous

        self.dropout = nn.Dropout(dropout)
        self.nl = nl[non_linearities]()

        self.denses = None
        if len(embedding_dims) > 0:
            seqs = []
            for i in range(len(embedding_dims)):
                linear = nn.Linear(input_dim * len(orders) if i==0 else embedding_dims[i-1], embedding_dims[i])
                init_weights(linear)
                seqs.append(nn.Sequential(self.dropout, linear, self.nl))
            self.denses = nn.Sequential(*seqs)

    def forward(self, x):
        x = transform_input_order(x, self.orders, fv_version=self.fv_version, contiguous=self.contiguous)
        return self.denses(x) if self.denses is not None else x


# --------------- RNN ---------------

class EncoderRNN(BaseModel):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.5, rnn_type='lstm'):
        super(Encoder, self).__init__()
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
        return state


# --------------- TCN ---------------

class TemporalBlock(BaseModel):
    """
        Original TCN. Chom1d replaced by padding to the left only. This way, we can choose not to keep the same output dimension.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, non_linearities="relu", dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.padding = padding
        self.dilation = dilation
        
        conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation))
        conv1.weight.data.normal_(0, 0.01)
        relu1 = nl[non_linearities]()
        dropout1 = nn.Dropout(dropout)
        self.block1 = nn.Sequential(conv1, relu1, dropout1)

        conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size=1, stride=stride, padding=0, dilation=1)) # 1D1
        conv2.weight.data.normal_(0, 0.01)
        relu2 = nl[non_linearities]()
        dropout2 = nn.Dropout(dropout)
        self.block2 = nn.Sequential(conv2, relu2, dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nl[non_linearities]()
        self.init_weights()

    def init_weights(self):
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x, log=False):
        y = F.pad(x, (self.padding, 0))
        y = self.block1(y)
        #y = F.pad(y, (0, 0)) # second has kernel size 1 and dilation 1
        y = self.block2(y)
            
        res = x if self.downsample is None else self.downsample(x) # downsample if needed
        
        # slice needed if temporal dimensions do not match (due to not padding)
        # this means we discard the left-most outputs
        if y.shape[2] < res.shape[2]:
            diff = res.shape[2] - y.shape[2]
            res = res[:,:,diff:]
        return self.relu(y + res) # residual connection

class TemporalConvNet(BaseModel):
    def __init__(self, num_inputs, num_channels, orders=[0, ], dilations=None, kernel_sizes=2, keep_length=True, dropout=0.2, non_linearities="relu"):
        super(TemporalConvNet, self).__init__()
        
        self.orders = orders

        self.layers = []
        num_levels = len(num_channels)
        assert type(kernel_sizes) is int or num_levels == len(kernel_sizes)
        assert dilations is None or num_levels == len(dilations)
        for i in range(num_levels):
            dilation_size = 2 ** i if dilations is None else dilations[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            kernel_size = kernel_sizes if type(kernel_sizes) is int else kernel_sizes[i]
            self.layers += [TemporalBlock(in_channels, out_channels, 
                                     kernel_size, 
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size if keep_length else 0, 
                                     dropout=dropout, non_linearities=non_linearities)]

        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)


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

class PoseTransformer(BaseModel):
    def __init__(self, orders=[0,], num_frame=100, num_joints=80, embed_dim_ratio=8, depth=4,
                 num_heads=8, mlp_ratio=1.5, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
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
        self.orders = orders
        in_chans = len(self.orders) * 3 # in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        self.out_dim = embed_dim     #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        #self.head = nn.Sequential(
        #    nn.LayerNorm(embed_dim),
        #    nn.Linear(embed_dim , out_dim),
        #)


    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        b  = x.shape[0]
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    # forward for 1 participant of the session
    def forward(self, x):
        # INPUT: [batch_size, sequence, joint_num, channels]
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)
        x = self.forward_features(x)
        #x = self.head(x)

        #x = x.view(b, 1, p, -1)
        return x

class PoseTransformerTemporal(BaseModel):
    def __init__(self, orders=[0,], num_frame=100, num_joints=80, embed_dim_ratio=8, depth=4,
                 num_heads=8, mlp_ratio=1.5, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
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
        self.orders = orders
        in_chans = len(self.orders) * 3 # in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        self.out_dim = embed_dim

        ### temporal patch embedding
        self.Temporal_patch_to_embedding = nn.Linear(in_chans * num_joints, embed_dim)
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)


    def temporal_forward(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> b f (p c)', )

        x = self.Temporal_patch_to_embedding(x)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    # forward for 1 participant of the session
    def forward(self, x):
        # INPUT: [batch_size, sequence, joint_num, channels]
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.temporal_forward(x)
        return x
    
    def get_attention_maps(self, x):
        # INPUT: [batch_size, sequence, joint_num, channels]
        b, f, p, _ = x.shape
        x = rearrange(x, 'b f p c  -> b f (p c)', )

        x = self.Temporal_patch_to_embedding(x)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        att_maps = []
        for blk in self.blocks:
            x, att_map = blk.get_attention_maps(x)
            att_maps.append(att_map)

        return torch.stack(att_maps, axis=1)
    
class PoseTransformerTemporal_hierarchical(BaseModel):
    def __init__(self, orders=[0,], num_frame=100, num_joints=80, embed_dim_ratio=8, depth=4,
                 num_heads=8, mlp_ratio=1.5, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None,
                 reduction=2, pooling='average'):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
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
        self.orders = orders
        in_chans = len(self.orders) * 3 # in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        self.out_dim = embed_dim#int(np.sum([embed_dim // (reduction**i) for i in range(depth)]))

        ### temporal patch embedding
        self.Temporal_patch_to_embedding = nn.Linear(in_chans * num_joints, embed_dim)
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if pooling == 'average':
            self.pooling = torch.nn.AvgPool1d(reduction, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
        elif pooling == 'max':
            self.pooling = torch.nn.MaxPool1d(reduction, stride=None, padding=0, ceil_mode=False)
        else:
            raise Exception(f"'{pooling}' not implemented as pooling method.")
            
        self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Temporal_norm = norm_layer(self.out_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame // (reduction**(depth-1)), out_channels=1, kernel_size=1)


    def temporal_forward(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> b f (p c)', )

        x = self.Temporal_patch_to_embedding(x)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i != len(self.blocks) - 1:
                x = self.pooling(x.transpose(1,2)).transpose(1,2)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    # forward for 1 participant of the session
    def forward(self, x):
        # INPUT: [batch_size, sequence, joint_num, channels]
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.temporal_forward(x)
        return x
    
    