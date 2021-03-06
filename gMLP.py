import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import Dataset,  DataLoader
import numpy as np
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self,  survival_prob,  fn):
        super().__init__()
        self.prob = torch.rand(1)

        self.survival_prob = survival_prob
        self.fn = fn

    def forward(self,  x):
        if self.prob <= self.survival_prob:
            return self.fn(x) + x
        else:
            return self.fn(x)


class PreNorm(nn.Module):
    def __init__(self,  dim,  fn,  **kwargs):
        super().__init__(**kwargs)
        self.norm = nn.LayerNorm(normalized_shape=dim,  eps=1e-6)
        self.fn = fn

    def forward(self,  x,  **kwargs):
        return self.fn(self.norm(x))


class SpatialGatingUnit(nn.Module):
    def __init__(self,  dim_seq,  dim_ff):
        super().__init__()

        self.proj = nn.Linear(dim_seq,  dim_seq)
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)
        self.norm = nn.LayerNorm(normalized_shape=dim_ff//2,  eps=1e-6)

        self.dim_ff = dim_ff

        self.activation = nn.GELU()


    def forward(self,  x):

        # x: shape = (batch,  dim_seq,  channel)
        res,  gate = torch.split(tensor=x,  split_size_or_sections=self.dim_ff//2,  dim=2) #ch
        # res, gate: shape = (batch,  dim_seq,  channel//2)
        gate = self.norm(gate)
        # gate: shape = (batch,  dim_seq,  channel//2)
        gate = torch.transpose(gate,  1,  2)
        # gate: shape = (batch,  channel//2,  dim_seq)
        gate = self.proj(gate)
        # gate: shape = (batch,  channel//2,  dim_seq)
        # gate = self.activation(gate)
        gate = torch.transpose(gate,  1,  2)
        # gate: shape = (batch,  dim_seq,  channel//2)
        return gate * res

class gMLPBlock(nn.Module):
    def __init__(self,  dim,  dim_ff,  seq_len):
        super().__init__()


        self.proj_in = nn.Linear(dim,  dim_ff)
        self.activation = nn.GELU()
        self.sgu = SpatialGatingUnit(seq_len,  dim_ff)
        self.proj_out = nn.Linear(dim_ff//2,  dim) #ch

    def forward(self,  x):
        # shape=(B,  seq,  dim) --> (B,  seq,  dim_ff) --> (B,  seq,  dim_ff/2) --> (B,  seq,  dim)
        x = self.proj_in(x)
        x = self.activation(x)
        x = self.sgu(x)
        x = self.proj_out(x)

        return x

class gMLPFeatures(nn.Module):
    def __init__(self,  survival_prob=0.5,  image_size=256,  patch_size=16,  dim=128,  depth=30,  ff_mult=2,  num_classes=196):
        super().__init__()

        self.image_size = image_size

        self.patch_size = patch_size

        self.patch_rearrange = Rearrange('b c (h p) (w q) -> b (h w) (c p q)',  p=self.patch_size,  q=self.patch_size) #(b,  3 ,  256,  256) -> (b,  16*16,  3*16*16)


        dim_ff = dim * ff_mult

        initial_dim = 3 * (patch_size ** 2)

        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Linear(initial_dim , dim) #shape=(B,  seq,  dim)
        self.query = torch.nn.Parameter(torch.randn(1,1,dim))
        self.query.requires_grad = True
        self.dim = dim

        module_list = [Residual(survival_prob,
                        PreNorm(dim,
                            gMLPBlock(dim=dim,
                                      dim_ff=dim_ff,
                                      seq_len=num_patches+1,
                                      ))) for i in range(depth)]

        self.glayers = nn.Sequential(*module_list)

        self.norm = nn.LayerNorm(normalized_shape=dim,  eps=1e-5)

        self.classification_rearrange = Rearrange('b s d -> b (s d)')

        self.clssification_head = nn.Sequential(
                                                 nn.Linear(dim,  dim),
                                                 nn.ReLU(),
                                                 nn.Linear(dim,  num_classes)
                                                 )


    def extract_patches(self,  images):

        batch_size = images.size(0)
        patches = self.patch_rearrange(images)

        return patches

    def forward(self,  x):


        x = self.extract_patches(x) #shape=(B,  num_patches,  patch_size**2 * C)
        x = self.patch_embed(x) #shape=(B,  num_patches,  dim)
        B, seq, dim = x.shape
        query_1 = self.query.repeat((B, 1, 1))
        x = torch.cat([query_1, x], dim=1)
        x = self.glayers(x) #shape=(B,  num_patches,  dim)
        x = self.norm(x) #shape=(B,  num_patches,  dim)
        x = x[:,0:1,:]

        x = self.classification_rearrange(x)

        return x
