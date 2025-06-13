import torch
import torch.nn as nn

# 깊이별 분리 합성곱 (효율적인 연산을 위해)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 논문 그림 32 (b): 다운 샘플링 블록
class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock, self).__init__()
        self.main_path = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.main_path(x)
        x = x + res # 잔차 연결 (Add)
        x = self.pool(x)
        return x

# 논문 그림 32 (c)와 U-Net Skip Connection을 결합한 업 샘플링 블록
class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpSamplingBlock, self).__init__()
        # 크기를 2배 키우는 Transposed Convolution
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Skip Connection과 합쳐진 후의 Convolution
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(out_channels + skip_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        # U-Net의 핵심: Skip Connection을 채널 방향으로 합치기
        x = torch.cat([x, skip_connection], dim=1) 
        x = self.conv(x)
        return x
    
    
class SingleUNet(nn.Module):
    """분할 또는 복원에 사용될 단일 U-Net 구조"""
    def __init__(self, in_channels, out_channels_final):
        super(SingleUNet, self).__init__()
        self.enc1 = DownSamplingBlock(in_channels, 64)
        self.enc2 = DownSamplingBlock(64, 128)
        self.enc3 = DownSamplingBlock(128, 256)
        self.enc4 = DownSamplingBlock(256, 512)

        self.bottleneck = DepthwiseSeparableConv(512, 1024)

        self.dec4 = UpSamplingBlock(1024, 512, 512)
        self.dec3 = UpSamplingBlock(512, 256, 256)
        self.dec2 = UpSamplingBlock(256, 128, 128)
        self.dec1 = UpSamplingBlock(128, 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels_final, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        # Bottleneck
        b = self.bottleneck(e4)
        # Decoder
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        out = self.final_conv(d1)
        return out

class DualResUNet(nn.Module):
    """논문의 전체 세그멘테이션 모듈: 두 개의 U-Net을 직렬 연결"""
    def __init__(self, in_channels=3, num_classes=4):
        super(DualResUNet, self).__init__()
        # 네트워크 1: 분할 (Image -> Mask Logits)
        self.segmentation_net = SingleUNet(in_channels=in_channels, out_channels_final=num_classes)
        # 네트워크 2: 복원 (Mask Probs -> Reconstructed Image)
        self.reconstruction_net = SingleUNet(in_channels=num_classes, out_channels_final=in_channels)
        
        # 분할 결과에 사용할 Softmax
        self.softmax = nn.Softmax(dim=1)
        # 복원 결과에 사용할 Tanh
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 1. 분할 네트워크 통과
        # CrossEntropyLoss는 Logits를 입력으로 받으므로, 활성화 함수 전의 값을 반환
        segmentation_logits = self.segmentation_net(x)

        # 2. 복원 네트워크 입력 준비 (Softmax를 통과시켜 확률 맵으로 변환)
        mask_probs = self.softmax(segmentation_logits)

        # 3. 복원 네트워크 통과
        reconstruction = self.reconstruction_net(mask_probs)
        reconstruction = self.tanh(reconstruction) # 픽셀 값을 -1~1 사이로
        
        return segmentation_logits, reconstruction