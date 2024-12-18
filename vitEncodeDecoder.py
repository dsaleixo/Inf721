
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from sklearn.model_selection import train_test_split
from dataSetFeatureExtractor import DataSetFeatureExtractor, ReadDataFeatureExtractor
from torch.utils.data import DataLoader


class AttentionInf21(nn.Module):
    #https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/vision_transformer/custom.py
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        #print("x ",x.shape)
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches , 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches , 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches , head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches )
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches, n_patches )
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches , n_patches )
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches , head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches , dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches , dim)
        x = self.proj_drop(x)  # (n_samples, n_patches , dim)

        return x



class Transformer(nn.Module):
    def __init__(self, emb_dim, n_heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        #self.mha1 = nn.MultiheadAttention(self.mha1, n_heads, batch_first=True)
        self.mha1 = AttentionInf21(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 10),
            nn.GELU(),
            nn.Linear(emb_dim * 10, emb_dim)
        )       

    def forward(self, x):
        x_norm = self.ln1(x)
        #print("att",x_norm.shape)
        attn_output= self.mha1(x_norm)
        #print("att",attn_output.shape)
        x = x + attn_output

        x_norm = self.ln2(x)
        ff_output = self.ff(x_norm)
        x = x + ff_output
        return x

class EncoderViT(nn.Module):
    def __init__(self, in_channels=3, p=4, dim=128, depth=1, latent_dim = 64):
        super(EncoderViT, self).__init__()
        patch_dim = in_channels * p * p
        self.num_patches = (dim // p) * (dim // p)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p),
          
            nn.Linear(patch_dim, dim),

        )
        self.att = nn.ModuleList([Transformer(dim,n_heads=8) for _ in range(depth)])
        self.pos_emb = nn.Embedding(self.num_patches, dim)
        self.proj = nn.Linear(dim, latent_dim)
       

    def forward(self, x:torch.Tensor):
        #print("ss " ,x.shape)
        x = self.to_patch_embedding(x)
        #print("ss " ,x.shape)
        pos = torch.arange(0, self.num_patches, dtype=torch.long, device=x.device)
        
        pos = self.pos_emb(pos)
       
        #print("pos " ,pos.shape)
        x += pos
        #print("ss " ,x.shape)
        for layer in self.att:
            x = layer(x)

        x = self.proj(x)
        #print("ss " ,x.shape)
        return x
    
class DecoderViT(nn.Module):
    def __init__(self, in_channels=15, p=4, dim=16, depth=1, latent_dim = 20):
        super(DecoderViT, self).__init__()
        patch_dim = in_channels * p * p
        self.num_patches = (16 // p) * (16 // p)
      
        self.to_image = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.GELU(),
            nn.Linear(patch_dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = 4, w = 4,p1=p, p2=p, c=in_channels),
        )
     
        self.l1=nn.LayerNorm(dim)
        self.l2=nn.Linear(dim, patch_dim)
        self.l3=nn.GELU()
        self.l4=nn.Linear(patch_dim, patch_dim)
        self.l5=nn.LayerNorm(patch_dim)
        self.l6=Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = 4, w = 4,p1=p, p2=p, c=in_channels)
       
        self.att = nn.ModuleList([Transformer(dim) for _ in range(depth)])
        self.pos_emb = nn.Embedding(self.num_patches, dim)
        self.proj = nn.Linear(latent_dim,dim)

    def forward(self, x):
  
        x = self.proj(x)

        pos = torch.arange(0, self.num_patches, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)
        x += pos
        for layer in self.att:
            x = layer(x)
        x = self.to_image(x)
     
     
        return x    

class TestViTEncoderDecoder():
    
    @staticmethod
    def test0():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("device =",device)

        datas = ReadDataFeatureExtractor.read(device)
        DataSetFeatureExtractor(datas)
        print(len(datas))
        X, Y  = train_test_split(datas, test_size=0.2, random_state=42)
        print(len(X),len(Y))
        
    @staticmethod
    def test1():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("device =",device)

        datas = ReadDataFeatureExtractor.read(device)
        datas = datas[:3]
        X = DataSetFeatureExtractor(datas)
        dataloader = DataLoader(X, batch_size=3, shuffle=True)
        
        num_channels=15
        dim=16
        p =4
        encoderViT=EncoderViT(in_channels=num_channels,p=p,dim=dim,depth=1,latent_dim=20)
        for batch_idx, batch_dados in enumerate(dataloader):
            y = encoderViT(batch_dados)
            print(y.shape)
        
        