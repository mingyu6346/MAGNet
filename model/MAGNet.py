import math
from model.smt import smt_t
from model.MobileNetV2 import mobilenet_v2
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

TRAIN_SIZE = 384


class MAGNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.rgb_backbone = smt_t(pretrained)
        self.d_backbone = mobilenet_v2(pretrained)

        # Fuse
        self.gfm2 = GFM(inc=512, expend_ratio=2)
        self.gfm1 = GFM(inc=256, expend_ratio=3)
        self.mafm2 = MAFM(inc=128)
        self.mafm1 = MAFM(inc=64)

        # Pred
        self.mcm3 = MCM(inc=512, outc=256)
        self.mcm2 = MCM(inc=256, outc=128)
        self.mcm1 = MCM(inc=128, outc=64)

        self.d_trans_4 = Trans(320, 512)
        self.d_trans_3 = Trans(96, 256)
        self.d_trans_2 = Trans(32, 128)
        self.d_trans_1 = Trans(24, 64)

        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        )

    def forward(self, x_rgb, x_d):
        # rgb
        _, (rgb_1, rgb_2, rgb_3, rgb_4) = self.rgb_backbone(x_rgb)

        # d
        _, d_1, d_2, d_3, d_4 = self.d_backbone(x_d)

        d_4 = self.d_trans_4(d_4)
        d_3 = self.d_trans_3(d_3)
        d_2 = self.d_trans_2(d_2)
        d_1 = self.d_trans_1(d_1)

        # Fuse
        fuse_4 = self.gfm2(rgb_4, d_4)  # [B, 512, 12, 12]
        fuse_3 = self.gfm1(rgb_3, d_3)  # [B, 256, 24, 24]
        fuse_2 = self.mafm2(rgb_2, d_2)  # [B, 128, 48, 48]
        fuse_1 = self.mafm1(rgb_1, d_1)  # [B, 64, 96, 96]

        # Pred
        pred_4 = F.interpolate(self.predtrans(fuse_4), TRAIN_SIZE, mode="bilinear", align_corners=True)
        pred_3, xf_3 = self.mcm3(fuse_3, fuse_4)
        pred_2, xf_2 = self.mcm2(fuse_2, xf_3)
        pred_1, xf_1 = self.mcm1(fuse_1, xf_2)

        return pred_1, pred_2, pred_3, pred_4


class Trans(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.apply(self._init_weights)

    def forward(self, d):
        return self.trans(d)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


# Conv_One_Identity
class COI(nn.Module):
    def __init__(self, inc, k=3, p=1):
        super().__init__()
        self.outc = inc
        self.dw = nn.Conv2d(inc, self.outc, kernel_size=k, padding=p, groups=inc)
        self.conv1_1 = nn.Conv2d(inc, self.outc, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(self.outc)
        self.bn2 = nn.BatchNorm2d(self.outc)
        self.bn3 = nn.BatchNorm2d(self.outc)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def forward(self, x):
        shortcut = self.bn1(x)

        x_dw = self.bn2(self.dw(x))
        x_conv1_1 = self.bn3(self.conv1_1(x))
        return self.act(shortcut + x_dw + x_conv1_1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MHMC(nn.Module):
    def __init__(self, dim, ca_num_heads=4, qkv_bias=True, proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = self.dim // ca_num_heads

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)
        for i in range(self.ca_num_heads):
            local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                   padding=(1 + i), stride=1,
                                   groups=dim // self.ca_num_heads)  # kernel_size 3,5,7,9 大核dw卷积，padding 1,2,3,4
            setattr(self, f"local_conv_{i + 1}", local_conv)
        self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                               groups=self.split_groups)
        self.bn = nn.BatchNorm2d(dim * expand_ratio)
        self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        v = self.v(x)
        s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1,
                                                                                          2)  # num_heads,B,C,H,W
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]  # B,C,H,W
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 2)
        s_out = s_out.reshape(B, C, H, W)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
        self.modulator = s_out
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
        x = s_out * v

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SAttention(nn.Module):
    def __init__(self, dim, sa_num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.sa_num_heads = sa_num_heads

        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        head_dim = dim // sa_num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
            self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
                                                                                                      N).transpose(1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x.permute(0, 2, 1).reshape(B, C, H, W)


# Multi-scale Awareness Fusion Module
class MAFM(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.outc = inc
        self.attention = MHMC(dim=inc)
        self.coi = COI(inc)
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1, stride=1),
            nn.BatchNorm2d(inc),
            nn.GELU()
        )
        self.pre_att = nn.Sequential(
            nn.Conv2d(inc * 2, inc * 2, kernel_size=3, padding=1, groups=inc * 2),
            nn.BatchNorm2d(inc * 2),
            nn.GELU(),
            nn.Conv2d(inc * 2, inc, kernel_size=1),
            nn.BatchNorm2d(inc),
            nn.GELU()
        )

        self.apply(self._init_weights)

    def forward(self, x, d):
        # multi = x * d
        # B, C, H, W = x.shape
        # x_cat = torch.cat((x, d, multi), dim=1)

        B, C, H, W = x.shape
        x_cat = torch.cat((x, d), dim=1)
        x_pre = self.pre_att(x_cat)
        # Attention
        x_reshape = x_pre.flatten(2).permute(0, 2, 1)  # B,C,H,W to B,N,C
        attention = self.attention(x_reshape, H, W)  # attention
        attention = attention.permute(0, 2, 1).reshape(B, C, H, W)  # B,N,C to B,C,H,W

        # COI
        x_conv = self.coi(attention)  # dw3*3,1*1,identity
        x_conv = self.pw(x_conv)  # pw

        return x_conv

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


# Decoder
class MCM(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, groups=outc),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(in_channels=outc, out_channels=1, kernel_size=1)
        )

        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=outc * 2, out_channels=outc * 2, kernel_size=3, padding=1, groups=outc * 2),
            nn.BatchNorm2d(outc * 2),
            nn.GELU(),
            nn.Conv2d(in_channels=outc * 2, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

        self.apply(self._init_weights)

    def forward(self, x1, x2):
        x2_upsample = self.upsample2(x2)  # 上采样
        x2_rc = self.rc(x2_upsample)  # 减少通道数
        shortcut = x2_rc

        x_cat = torch.cat((x1, x2_rc), dim=1)  # 拼接
        x_forward = self.rc2(x_cat)  # 减少通道数2
        x_forward = x_forward + shortcut
        pred = F.interpolate(self.predtrans(x_forward), TRAIN_SIZE, mode="bilinear", align_corners=True)  # 预测图

        return pred, x_forward

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


# Global Fusion Module
class GFM(nn.Module):
    def __init__(self, inc, expend_ratio=2):
        super().__init__()
        self.expend_ratio = expend_ratio
        assert expend_ratio in [2, 3], f"expend_ratio {expend_ratio} mismatch"

        self.sa = SAttention(dim=inc)
        self.dw_pw = DWPWConv(inc * expend_ratio, inc)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def forward(self, x, d):
        B, C, H, W = x.shape
        if self.expend_ratio == 2:
            cat = torch.cat((x, d), dim=1)
        else:
            multi = x * d
            cat = torch.cat((x, d, multi), dim=1)
        x_rc = self.dw_pw(cat).flatten(2).permute(0, 2, 1)
        x_ = self.sa(x_rc, H, W)
        x_ = x_ + x
        return self.act(x_)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class DWPWConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    x = torch.randn(1, 120, 14, 14)
    d = torch.randn(1, 3, 224, 224)

    model = MAGNet()
    #

    output = model(d, d)
    print(output.shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M")
