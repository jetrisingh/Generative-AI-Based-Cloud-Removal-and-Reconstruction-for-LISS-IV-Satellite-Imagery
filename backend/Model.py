import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# Normalisation helper 
 
#Instance Norm — preserves per-image spectral relationships.
def IN(ch):
    return nn.InstanceNorm2d(ch, affine=True)
 
 
# CBAM: Channel + Spatial Attention 
# focus on cloud-affected regions 
 
class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg(x)) + self.fc(self.max(x)))
 
 
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max (dim=1, keepdim=True).values
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
 
 
class CBAM(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention()
 
    def forward(self, x):
        return self.sa(self.ca(x))
 
 
#Residual block (bottleneck) 
 
class ResBlock(nn.Module):
    def __init__(self, ch, use_cbam=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            IN(ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            IN(ch),
        )
        self.cbam = CBAM(ch) if use_cbam else nn.Identity()
 
    def forward(self, x):
        return x + self.cbam(self.block(x))
 
 
# Encoder block 
 
class EncoderBlock(nn.Module):
    """Conv(3×3) → IN → LeakyReLU  — no pooling, returns feature + pooled."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, bias=False),
            IN(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, bias=False),
            IN(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
 
    def forward(self, x):
        feat   = self.conv(x)   # skip connection
        pooled = self.pool(feat)
        return feat, pooled
 
 
#Decoder block
 
class DecoderBlock(nn.Module):
    """
    Bilinear upsample → concat skip → Conv → IN → ReLU.
    Bilinear avoids the checkerboard artifacts of ConvTranspose2d.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, bias=False),
            IN(out_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, bias=False),
            IN(out_ch),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd spatial dims from maxpool
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))
 
 
# U-Net
 
class CloudRemovalUNet(nn.Module):
    """
    U-Net for cloud removal.
 
    Args:
        in_ch   : 4  (3 cloudy bands + 1 cloud mask)
        out_ch  : 3  (3 reconstructed clear bands)
        base_ch : 64 (feature width; use 48 for 4 GB VRAM)
    """
    def __init__(self, in_ch=4, out_ch=3, base_ch=64):
        super().__init__()
        f = base_ch
 
        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(in_ch, f)       # skip: (B,  f, H,   W  )
        self.enc2 = EncoderBlock(f,     f*2)     # skip: (B, 2f, H/2, W/2)
        self.enc3 = EncoderBlock(f*2,   f*4)     # skip: (B, 4f, H/4, W/4)
        self.enc4 = EncoderBlock(f*4,   f*8)     # skip: (B, 8f, H/8, W/8)
 
        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ResBlock(f*8, use_cbam=True),
            ResBlock(f*8, use_cbam=True),
            ResBlock(f*8, use_cbam=False),
        )
 
        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(f*8, f*8, f*4)
        self.dec3 = DecoderBlock(f*4, f*4, f*2)
        self.dec2 = DecoderBlock(f*2, f*2, f)
        self.dec1 = DecoderBlock(f,   f,   f)
 
        # ── Output head ───────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(f, out_ch, 1),
            nn.Sigmoid(),     # output in [0,1] — matches your .npy data range
        )
 
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.ones_ (m.weight)
                nn.init.zeros_(m.bias)
 
    def forward(self, x):
        # Encode
        s1, p1 = self.enc1(x)    # s1 = skip, p1 = pooled input to next
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
 
        # Bottleneck
        b = self.bottleneck(p4)
 
        # Decode
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
 
        return self.head(d1)      # (B, 3, H, W) in [0,1]

if __name__ == "__main__":
    model = CloudRemovalUNet()

    x = torch.randn(1,4,256,256)

    y = model(x)

    print(y.shape) 