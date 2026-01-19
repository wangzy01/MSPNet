from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
import sys
sys.path.append("../")
from clip.model import LayerNorm, QuickGELU, DropPath
from ipdb import set_trace as st

class TemporalFusionAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath = 0., T=0, ):
        super().__init__()
        self.T = T

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head,)
           
        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x):
        l, bt, d = x.size()
        b = bt // self.T
        x = x.view(l, b, self.T, d) 

        msg_token = self.message_fc(x[0,:,:,:]) 
        msg_token = msg_token.view(b, self.T, 1, d) 
        
        msg_token = msg_token.permute(1,2,0,3).view(self.T, b, d) 
        msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token),self.message_ln(msg_token),self.message_ln(msg_token),need_weights=False)[0])
        msg_token = msg_token.view(self.T, 1, b, d).permute(1,2,0,3)
        
        x = torch.cat([x, msg_token], dim=0)
        
        x = x.view(l+1, -1, d)
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x[:l,:,:]
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None, use_checkpoint=False, T=8):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)] 
        self.width = width
        self.layers = layers
        
        self.resblocks = nn.Sequential(*[TemporalFusionAttentionBlock(width, heads, attn_mask, droppath[i], T) for i in range(layers)])
       
    def forward(self, x: torch.Tensor):
        if not self.use_checkpoint:
            return self.resblocks(x)
        else:
            return checkpoint_sequential(self.resblocks, 3, x)

class InterFrameInteractionTransforme(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 droppath = None, T = 8, use_checkpoint = False,CNT=2,CCC=3,use_PRF=False,use_PTM=False,):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.use_PRF = use_PRF
        self.use_PTM = use_PTM 

        if use_PRF:
            self.rsb_block = RSB_BLOCK(in_planes=CNT*CCC,efficient=use_checkpoint)

        self.conv0 = nn.Conv2d(in_channels=CNT*CCC, out_channels=3, kernel_size=1, stride=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        ############     

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        ## Attention Blocks
        self.transformer = Transformer(width, layers, heads, droppath=droppath, use_checkpoint=use_checkpoint, T=T,)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):

        # print("aaa:",x.shape)   #  torch.Size([16, 6, 224, 224]
        if self.use_PRF:
            x = self.rsb_block(x)
        # x = self.rsb_block(x)
        # print("bbb:",x.shape)
        x = self.conv0(x)
        x = self.conv1(x)  # shape = [b*t, width, grid, grid]
        # print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        
        x = x.permute(1, 0, 2)

        cls_x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            cls_x = cls_x @ self.proj
        
        return cls_x, x[:,1:,:]



class RSB_BLOCK(nn.Module):
    """
    from https://github.com/caiyuanhao1998/RSN/blob/master/exps/4XRSN18.coco/network.py
        class Bottleneck(nn.Module)
    """
    expansion = 1

    def __init__(self, in_planes, stride=1, groups=1, downsample=None, efficient=False):
        super(RSB_BLOCK, self).__init__()
        self.branch_ch = max(in_planes // 4, 1)
        self.conv_bn_relu1 = conv_bn_relu(in_planes, 4 * self.branch_ch, kernel_size=1,
                                          stride=stride, padding=0, groups=groups,
                                          has_bn=True, has_relu=True, efficient=efficient)

        self.conv_bn_relu2_1_1 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_2_1 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_2_2 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_3_1 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_3_2 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_3_3 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_4_1 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_4_2 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_4_3 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)
        self.conv_bn_relu2_4_4 = conv_bn_relu(self.branch_ch, self.branch_ch, kernel_size=3,
                                              stride=1, padding=1, groups=groups,
                                              has_bn=True, has_relu=True, efficient=efficient)

        self.conv_bn_relu3 = conv_bn_relu(4 * self.branch_ch, in_planes,
                                          kernel_size=1, stride=1, padding=0, groups=groups,
                                          has_bn=True, has_relu=False, efficient=efficient)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        spx = torch.split(out, max(out.size(1) // 4, 1), 1)
        out_1_1 = self.conv_bn_relu2_1_1(spx[0])

        out_2_1 = self.conv_bn_relu2_2_1(spx[1] + out_1_1)
        out_2_2 = self.conv_bn_relu2_2_2(out_2_1)

        out_3_1 = self.conv_bn_relu2_3_1(spx[2] + out_2_1)
        out_3_2 = self.conv_bn_relu2_3_2(out_3_1 + out_2_2)
        out_3_3 = self.conv_bn_relu2_3_3(out_3_2)

        out_4_1 = self.conv_bn_relu2_4_1(spx[3] + out_3_1)
        out_4_2 = self.conv_bn_relu2_4_2(out_4_1 + out_3_2)
        out_4_3 = self.conv_bn_relu2_4_3(out_4_2 + out_3_3)
        out_4_4 = self.conv_bn_relu2_4_4(out_4_3)

        out = torch.cat((out_1_1, out_2_2, out_3_3, out_4_4), 1)
        out = self.conv_bn_relu3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out

class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,
                 has_bn=True, has_relu=True, efficient=False, groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient

        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x

            return func

        func = _func_factory(self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x