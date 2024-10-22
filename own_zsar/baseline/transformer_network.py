import torch
import torch.nn as nn
import clip
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from clip.model import QuickGELU
from collections import OrderedDict


class CLIPVideo(nn.Module):
    def __init__(self, vit_backbone="ViT-B/16"):
        super(CLIPVideo, self).__init__()
        device = 'cuda'
        self.model, _ = clip.load(vit_backbone, device=device)

    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.permute(0,1,3,2,4,5) # (bs, nc, l, ch, h, w)
        x = x.reshape(bs*nc*l, ch, h, w) # (bs*nc*l, ch, h, w)
        video_features = self.model.encode_image(x) # Input requirement h=w=224
        # (bs*nc*l, 512)
        video_features = video_features.reshape(bs, nc*l, -1) # (bs, nc*l, 512)
        video_features = torch.mean(video_features, 1) # (bs, 512)
        return video_features
    
class CLIPClassifier(nn.Module):
    def __init__(self, out_features, vit_backbone="ViT-B/16"):
        super(CLIPClassifier, self).__init__()
        self.regressor = nn.Linear(512, out_features)
        self.vit_backbone = vit_backbone

    def forward(self, x):
        device = 'cuda'
        model, preprocess = clip.load(self.vit_backbone, device=device)
        bs, nc, ch, l, h, w = x.shape
        x = x.permute(0,1,3,2,4,5) # (bs, nc, l, ch, h, w)
        x = x.reshape(bs*nc*l, ch, h, w) # (bs*nc*l, ch, h, w)
        video_features = model.encode_image(x) # Input requirement h=w=224
        video_features = video_features.float()
        video_features = video_features.reshape(bs, nc*l, -1) # (bs, nc*l, 512)
        video_features = torch.mean(video_features, 1) # (bs, 512)

        video_features = self.regressor(video_features)
        return video_features
    
    def output(self, x):
        device = 'cuda'
        model, preprocess = clip.load(self.vit_backbone, device=device)
        bs, nc, ch, l, h, w = x.shape
        x = x.permute(0,1,3,2,4,5) # (bs, nc, l, ch, h, w)
        x = x.reshape(bs*nc*l, ch, h, w) # (bs*nc*l, ch, h, w)
        video_features = model.encode_image(x) # Input requirement h=w=224
        video_features = video_features.float()
        video_features = video_features.reshape(bs, nc*l, -1) # (bs, nc*l, 512)
        video_features = torch.mean(video_features, 1) # (bs, 512)
        return video_features
    

## Temporal context transformer

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class TemporalPool(nn.Module):
    def __init__(self):
        super(TemporalPool, self).__init__()

    def forward(self, x):
        x = self.temporal_pool(x)
        return x

    @staticmethod
    def temporal_pool(x):
        bs, t, n_feat = x.shape
        c = 2
        h = w = 16
        x = x.reshape(bs, t, c, h, w) # bs, t, c, h, w
        x = x.transpose(1, 2)  # bs, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(bs, t // 2, c*h*w)
        return x

class LCA(nn.Module):
    def __init__(self, LCA_drops, embed_dim=512):
        super(LCA, self).__init__()
        self.embed_dim = embed_dim

        if embed_dim == 512:        
            self.branch1 = nn.Sequential(
                nn.Conv2d(2, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 64, kernel_size=(3,3), dilation=1),
                QuickGELU(),
                CBAM(64),
                nn.Conv2d(64, 256, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[0]),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(16, 128, kernel_size=(3,3), dilation=2),
                QuickGELU(),
                CBAM(128),
                nn.Conv2d(128, 192, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[1]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.branch3 = nn.Sequential(
                nn.Conv2d(2, 4, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(4, 16, kernel_size=(3,3), dilation=4),
                QuickGELU(),
                CBAM(16),
                nn.Conv2d(16, 64, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[2]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif embed_dim == 768:
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 64, kernel_size=(3,3), dilation=1),
                QuickGELU(),
                CBAM(64),
                nn.Conv2d(64, 512, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[0]),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 32, kernel_size=(3,3), dilation=2),
                QuickGELU(),
                CBAM(32),
                nn.Conv2d(32, 128, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[1]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.branch3 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 32, kernel_size=(3,3), dilation=4),
                QuickGELU(),
                CBAM(32),
                nn.Conv2d(32, 128, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[2]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif embed_dim == 1024:
            self.branch1 = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 64, kernel_size=(3,3), dilation=1),
                QuickGELU(),
                CBAM(64),
                nn.Conv2d(64, 512, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[0]),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(16, 128, kernel_size=(3,3), dilation=2),
                QuickGELU(),
                CBAM(128),
                nn.Conv2d(128, 384, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[1]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.branch3 = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 16, kernel_size=(3,3), dilation=4),
                QuickGELU(),
                CBAM(16),
                nn.Conv2d(16, 128, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[2]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        else:
            exit("No LCA for embedding dimension = " + str(embed_dim))


    def forward(self, x):
        bs, t, _ = x.shape
        if self.embed_dim == 512:
            x = x.reshape(bs, t, 2, 16, 16)
            x = x.reshape(bs*t, 2, 16, 16)
        elif self.embed_dim == 768:
            x = x.reshape(bs, t, 3, 16, 16)
            x = x.reshape(bs*t, 3, 16, 16)
        elif self.embed_dim == 1024:
            x = x.reshape(bs, t, 4, 16, 16)
            x = x.reshape(bs*t, 4, 16, 16)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x1 = torch.squeeze(x1)
        x2 = torch.squeeze(x2)
        x3 = torch.squeeze(x3)
        out = torch.cat((x1, x2, x3), dim=1)
        out = out.reshape(bs, t, self.embed_dim)
        return out

# 2 branches
class LCA_ablation_1(nn.Module):
    def __init__(self, LCA_drops, embed_dim=512):
        super(LCA_ablation_1, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(1,1)),
            QuickGELU(),
            nn.Conv2d(8, 64, kernel_size=(3,3), dilation=2),
            QuickGELU(),
            CBAM(64),
            nn.Conv2d(64, 256, kernel_size=(1,1)),
            QuickGELU(),
            nn.Dropout(LCA_drops[1]),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(1,1)),
            QuickGELU(),
            nn.Conv2d(8, 64, kernel_size=(3,3), dilation=4),
            QuickGELU(),
            CBAM(64),
            nn.Conv2d(64, 256, kernel_size=(1,1)),
            QuickGELU(),
            nn.Dropout(LCA_drops[2]),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        bs, t, _ = x.shape
        x = x.reshape(bs, t, 2, 16, 16)
        x = x.reshape(bs*t, 2, 16, 16)

        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x1 = torch.squeeze(x1)
        x2 = torch.squeeze(x2)
        out = torch.cat((x1, x2), dim=1)
        out = out.reshape(bs, t, 512)
        return out

# 1 branch   
class LCA_ablation_2(nn.Module):
    def __init__(self, LCA_drops, embed_dim=512):
        super(LCA_ablation_2, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(1,1)),
            QuickGELU(),
            nn.Conv2d(8, 64, kernel_size=(3,3), dilation=4),
            QuickGELU(),
            CBAM(64),
            nn.Conv2d(64, 512, kernel_size=(1,1)),
            QuickGELU(),
            nn.Dropout(LCA_drops[2]),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        bs, t, _ = x.shape
        x = x.reshape(bs, t, 2, 16, 16)
        x = x.reshape(bs*t, 2, 16, 16)

        x1 = self.branch1(x)
        x1 = torch.squeeze(x1)

        x1 = x1.reshape(bs, t, 512)
        return x1


class TransformerBlock(nn.Module):
    def __init__(self, LCA_drops, d_model, n_head, drop_attn=0.0, droppath=0.0, lca_branch=3):
        super(TransformerBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=drop_attn)
        if lca_branch >= 3:
            self.lca = LCA(LCA_drops, embed_dim=d_model)
        elif lca_branch == 2:
            self.lca = LCA_ablation_1(LCA_drops, embed_dim=d_model)
        else:
            self.lca = LCA_ablation_2(LCA_drops, embed_dim=d_model)
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        #can keep droppath rate = 0.2 for ViTB/16. Source: 
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))


    def attention(self, x):
        return self.mha(x, x, x, need_weights=False)[0]
        
    def forward(self, x):
        x = x + self.drop_path(self.attention(self.ln_1(x)))

        # For LCA
        x = x + self.drop_path(self.lca(self.ln_2(x)))

        # For simple MLP
        # x = x + self.drop_path(self.mlp(self.ln_2(x)))

        return x
    

class CLIPTransformer(nn.Module):
    def __init__(self, T, LCA_drops, num_blocks=2, embed_dim=512, drop_attn=0.0, droppath=0.0, vit_backbone="ViT-B/16", lca_branch=3):
        super(CLIPTransformer, self).__init__()
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.vit_backbone = vit_backbone
        self.embed_dim = embed_dim

        n_head = embed_dim // 64
        self.transformer_block1 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath, lca_branch=lca_branch)
        # self.transformer_block2 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath)
        # self.transformer_block3 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath)

    def forward(self, x):
        device = 'cuda'
        model, preprocess = clip.load(self.vit_backbone, device=device)
        bs, nc, ch, l, h, w = x.shape
        x = x.permute(0,1,3,2,4,5) # (bs, nc, l, ch, h, w)
        x = x.reshape(bs*nc*l, ch, h, w) # (bs*nc*l, ch, h, w)

        video_features = model.encode_image(x) # Input requirement h=w=224
        video_features = video_features.float() # (bs*nc*l, 512)
        video_features = video_features.reshape(bs*nc, l, -1) # (bs*nc, l, 512)

        # Positional embedding
        video_features = video_features + self.positional_embedding
        video_features = self.transformer_block1(video_features)
        # video_features = self.transformer_block2(video_features)
        # video_features = self.transformer_block3(video_features)

        # (bs*nc, t, 512)
        video_features = video_features.reshape(bs, nc*l, self.embed_dim)
        video_features = torch.mean(video_features, 1) # (bs, 512)
        return video_features
    
class CLIPTransformerClassifier(nn.Module):
    def __init__(self, out_features, T, LCA_drops, num_blocks=2, embed_dim=512, drop_attn=0.0, droppath=0.0, vit_backbone="ViT-B/16", lca_branch=3):
        super(CLIPTransformerClassifier, self).__init__()
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.vit_backbone = vit_backbone
        self.embed_dim = embed_dim

        n_head = embed_dim // 64
        self.transformer_block1 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath, lca_branch=lca_branch)
        # self.transformer_block2 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath)
        # self.transformer_block3 = TransformerBlock(LCA_drops, d_model=embed_dim, n_head=n_head, drop_attn=drop_attn, droppath=droppath)

        self.regressor = nn.Linear(512, out_features)

    def output(self, x):
        device = 'cuda'
        model, preprocess = clip.load(self.vit_backbone, device=device)
        bs, nc, ch, l, h, w = x.shape
        x = x.permute(0,1,3,2,4,5) # (bs, nc, l, ch, h, w)
        x = x.reshape(bs*nc*l, ch, h, w) # (bs*nc*l, ch, h, w)
        video_features = model.encode_image(x) # Input requirement h=w=224
        video_features = video_features.float() # (bs*nc*l, 512)
        video_features = video_features.reshape(bs*nc, l, -1) # (bs*nc, l, 512)

        # Positional embedding
        video_features = video_features + self.positional_embedding
        video_features = self.transformer_block1(video_features)
        # video_features = self.transformer_block2(video_features)
        # video_features = self.transformer_block3(video_features)

        # (bs*nc, t, 512)
        video_features = video_features.reshape(bs, nc*l, self.embed_dim)
        video_features = torch.mean(video_features, 1) # (bs, 512)
        return video_features
    
    def forward(self, x):
        video_features = self.output(x)
        video_features = self.regressor(video_features)
        return video_features

