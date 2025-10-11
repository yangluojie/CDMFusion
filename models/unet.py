import math
import torch
import torch.nn as nn
import torch.fft as fft

# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm
# https://github.com/IGITUGraz/WeatherDiffusion
# https://github.com/ChenyangSi/FreeU

def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()
    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask
    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def get_calculation(stage, h1, h2, h3):
    calculations = {
        1: h1 + h2,
        2: h1 + h3,
        3: h1 + h2 + h3,
        4: h1 + h2 + h3,
        5: h1 + h2,
        6: h1 + h3,
        7: h1 + h2 + h3,
        8: h1 + h2 + h3
    }
    return calculations.get(stage, 0)  

def nonlinearity(x):
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x+h_

class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = 3
        in_channels3 = 3
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.in_channels3 = in_channels3

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in1 = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.conv_in2 = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.conv_in3 = torch.nn.Conv2d(in_channels3,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down1 = nn.ModuleList()
        self.down2 = nn.ModuleList()
        self.down3 = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down1.append(down)
            self.down2.append(down)
            self.down3.append(down)

        # middle
        self.mid1 = nn.Module()
        self.mid1.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid1.attn_1 = AttnBlock(block_in)
        self.mid1.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        

        self.mid2 = nn.Module()
        self.mid2.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid2.attn_1 = AttnBlock(block_in)
        self.mid2.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        

        self.mid3 = nn.Module()
        self.mid3.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid3.attn_1 = AttnBlock(block_in)
        self.mid3.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.b1 = 1.3 
        self.b2 = 1.4
        self.s1 = 0.9
        self.s2 = 0.1 

    def forward(self, x, t, stage=4, canshu=[0, 1, 1, 1, 1, 0]):
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        # downsampling
        hs1 = [self.conv_in1(x[:, 6:, :, :])]
        hs2 = [self.conv_in2(x[:, :3, :, :])]
        hs3 = [self.conv_in3(x[:, 3:6, :, :])]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h1 = self.down1[i_level].block[i_block](hs1[-1], temb)
                h2 = self.down2[i_level].block[i_block](hs2[-1], temb)
                h3 = self.down3[i_level].block[i_block](hs3[-1], temb)
                if len(self.down1[i_level].attn) > 0:
                    h1 = self.down1[i_level].attn[i_block](h1)
                    h2 = self.down2[i_level].attn[i_block](h2)
                    h3 = self.down3[i_level].attn[i_block](h3)
                hs1.append(get_calculation(stage, h1, h2, h3))
                hs2.append(h2)
                hs3.append(h3)
            if i_level != self.num_resolutions-1:
                hh1 = self.down1[i_level].downsample(hs1[-1])
                hh2 = self.down2[i_level].downsample(hs2[-1])
                hh3 = self.down3[i_level].downsample(hs3[-1])
                hs1.append(get_calculation(stage, hh1, hh2, hh3))
                hs2.append(hh2)
                hs3.append(hh3)
        # middle
        h1 = hs1[-1]
        h1 = self.mid1.block_1(h1, temb)
        h1 = self.mid1.attn_1(h1)
        h1 = self.mid1.block_2(h1, temb)
        h2 = hs2[-1]
        h2 = self.mid2.block_1(h2, temb)
        h2 = self.mid2.attn_1(h2)
        h2 = self.mid2.block_2(h2, temb)
        h3 = hs3[-1]
        h3 = self.mid3.block_1(h3, temb)
        h3 = self.mid3.attn_1(h3)
        h3 = self.mid3.block_2(h3, temb)
        h = get_calculation(stage, h1, h2, h3)
        self.b1 = canshu[1] 
        self.b2 = canshu[2] 
        self.s1 = canshu[3] 
        self.s2 = canshu[4]
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                
                stage_handlers = {
                    1: lambda: hs1.pop() + hs2.pop(),
                    2: lambda: hs1.pop() + hs3.pop(),
                    3: lambda: hs1.pop() + hs2.pop() + hs3.pop(),
                    4: lambda: hs1.pop() + hs2.pop() + hs3.pop(),
                    5: lambda: hs1.pop(), 
                    6: lambda: hs1.pop(),
                    7: lambda: hs1.pop(), 
                    8: lambda: hs1.pop() 
                }
                if canshu[5] == 0:
                    hs_sum = hs1.pop()
                else:
                    try:
                        hs_sum = stage_handlers[stage]()
                    except KeyError:
                        print("_________________________Error!")
                if stage == 4 or stage == 8:
                    if h.shape[1] == 512:
                        hidden_mean = h.mean(1).unsqueeze(1)
                        B = hidden_mean.shape[0]
                        hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
                        hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                        hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
                        h[:,:256] = h[:,:256] * ((self.b1 - 1 ) * hidden_mean + 1)
                        hs_sum = Fourier_filter(hs_sum, threshold=1, scale=self.s1)
                    if h.shape[1] == 256:
                        hidden_mean = h.mean(1).unsqueeze(1)
                        B = hidden_mean.shape[0]
                        hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
                        hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                        hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
                        h[:,:128] = h[:,:128] * ((self.b2 - 1 ) * hidden_mean + 1)
                        hs_sum = Fourier_filter(hs_sum, threshold=1, scale=self.s2)

                if canshu[5] == 0:
                    hs_sum =  hs_sum + hs2.pop() + hs3.pop()

                h = self.up[i_level].block[i_block](
                    torch.cat([ h,  hs_sum], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h
