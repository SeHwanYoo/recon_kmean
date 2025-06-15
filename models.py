# # import torch
# # import torch.nn as nn

# # # 깊이별 분리 합성곱 (효율적인 연산을 위해)
# # class DepthwiseSeparableConv(nn.Module):
# #     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
# #         super(DepthwiseSeparableConv, self).__init__()
# #         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
# #         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
# #         self.bn = nn.BatchNorm2d(out_channels)
# #         self.relu = nn.ReLU(inplace=True)

# #     def forward(self, x):
# #         x = self.depthwise(x)
# #         x = self.pointwise(x)
# #         x = self.bn(x)
# #         x = self.relu(x)
# #         return x

# # # 논문 그림 32 (b): 다운 샘플링 블록
# # class DownSamplingBlock(nn.Module):
# #     def __init__(self, in_channels, out_channels):
# #         super(DownSamplingBlock, self).__init__()
# #         self.main_path = nn.Sequential(
# #             DepthwiseSeparableConv(in_channels, out_channels),
# #             DepthwiseSeparableConv(out_channels, out_channels)
# #         )
# #         self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
# #         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

# #     def forward(self, x):
# #         res = self.shortcut(x)
# #         x = self.main_path(x)
# #         x = x + res # 잔차 연결 (Add)
# #         x = self.pool(x)
# #         return x

# # # 논문 그림 32 (c)와 U-Net Skip Connection을 결합한 업 샘플링 블록
# # class UpSamplingBlock(nn.Module):
# #     def __init__(self, in_channels, skip_channels, out_channels):
# #         super(UpSamplingBlock, self).__init__()
# #         # 크기를 2배 키우는 Transposed Convolution
# #         self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
# #         # Skip Connection과 합쳐진 후의 Convolution
# #         self.conv = nn.Sequential(
# #             DepthwiseSeparableConv(out_channels + skip_channels, out_channels),
# #             DepthwiseSeparableConv(out_channels, out_channels)
# #         )

# #     def forward(self, x, skip_connection):
# #         x = self.upsample(x)
# #         # U-Net의 핵심: Skip Connection을 채널 방향으로 합치기
# #         x = torch.cat([x, skip_connection], dim=1) 
# #         x = self.conv(x)
# #         return x
    
    
# # class SingleUNet(nn.Module):
# #     def __init__(self, in_channels, out_channels_final):
# #         super(SingleUNet, self).__init__()
# #         self.enc1 = DownSamplingBlock(in_channels, 64)   # 입력: 256x256 -> 출력: 128x128
# #         self.enc2 = DownSamplingBlock(64, 128)      # 입력: 128x128 -> 출력: 64x64
# #         self.enc3 = DownSamplingBlock(128, 256)     # 입력: 64x64 -> 출력: 32x32
# #         self.enc4 = DownSamplingBlock(256, 512)     # 입력: 32x32 -> 출력: 16x16

# #         self.bottleneck = DepthwiseSeparableConv(512, 1024)

# #         self.dec4 = UpSamplingBlock(1024, 512, 512) # 입력: 16x16 -> 출력: 32x32
# #         self.dec3 = UpSamplingBlock(512, 256, 256)  # 입력: 32x32 -> 출력: 64x64
# #         self.dec2 = UpSamplingBlock(256, 128, 128)  # 입력: 64x64 -> 출력: 128x128
# #         self.dec1 = UpSamplingBlock(128, 64, 64)    # 입력: 128x128 -> 출력: 256x256

# #         self.final_conv = nn.Conv2d(64, out_channels_final, kernel_size=1)

# #     def forward(self, x):
# #         # Encoder
# #         e1 = self.enc1(x)    # 크기: 128x128
# #         e2 = self.enc2(e1)   # 크기: 64x64
# #         e3 = self.enc3(e2)   # 크기: 32x32
# #         e4 = self.enc4(e3)   # 크기: 16x16
        
# #         # Bottleneck
# #         b = self.bottleneck(e4) # 크기: 16x16

# #         # Decoder (⭐ skip connection 연결 수정)
# #         # dec4는 16x16 -> 32x32로 업샘플링 후, 32x32 크기인 e3와 연결
# #         d4 = self.dec4(b, e3) 
# #         # dec3는 32x32 -> 64x64로 업샘플링 후, 64x64 크기인 e2와 연결
# #         d3 = self.dec3(d4, e2)
# #         # dec2는 64x64 -> 128x128로 업샘플링 후, 128x128 크기인 e1과 연결
# #         d2 = self.dec2(d3, e1)
# #         # dec1은 128x128 -> 256x256으로 업샘플링 후, 256x256 크기인 원본 x와 연결...
# #         # ... 하려니 채널 수가 안 맞음. 가장 바깥쪽 skip connection은 보통 첫 conv 레이어의 출력을 사용.
# #         # 이 구조에서는 enc1의 입력 x가 아니라, 채널 수가 맞는 e1을 사용하는 것이 맞음.
# #         # d2 -> 128채널, e1 -> 64채널. UpSamplingBlock(128, 64, 64)가 알아서 처리함.
# #         d1 = self.dec1(d2, e1)

# #         out = self.final_conv(d1)
# #         return out

# # class DualResUNet(nn.Module):
# #     """논문의 전체 세그멘테이션 모듈: 두 개의 U-Net을 직렬 연결"""
# #     def __init__(self, in_channels=3, num_classes=4):
# #         super(DualResUNet, self).__init__()
# #         # 네트워크 1: 분할 (Image -> Mask Logits)
# #         self.segmentation_net = SingleUNet(in_channels=in_channels, out_channels_final=num_classes)
# #         # 네트워크 2: 복원 (Mask Probs -> Reconstructed Image)
# #         self.reconstruction_net = SingleUNet(in_channels=num_classes, out_channels_final=in_channels)
        
# #         # 분할 결과에 사용할 Softmax
# #         self.softmax = nn.Softmax(dim=1)
# #         # 복원 결과에 사용할 Tanh
# #         self.tanh = nn.Tanh()

# #     def forward(self, x):
# #         # 1. 분할 네트워크 통과
# #         # CrossEntropyLoss는 Logits를 입력으로 받으므로, 활성화 함수 전의 값을 반환
# #         segmentation_logits = self.segmentation_net(x)

# #         # 2. 복원 네트워크 입력 준비 (Softmax를 통과시켜 확률 맵으로 변환)
# #         mask_probs = self.softmax(segmentation_logits)

# #         # 3. 복원 네트워크 통과
# #         reconstruction = self.reconstruction_net(mask_probs)
# #         reconstruction = self.tanh(reconstruction) # 픽셀 값을 -1~1 사이로
        
# #         return segmentation_logits, reconstruction

# import torch
# import torch.nn as nn

# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self, x):
#         return self.relu(self.bn(self.pointwise(self.depthwise(x))))

# class DownSamplingBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DownSamplingBlock, self).__init__()
#         self.main_path = nn.Sequential(
#             DepthwiseSeparableConv(in_channels, out_channels),
#             DepthwiseSeparableConv(out_channels, out_channels)
#         )
#         self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#     def forward(self, x):
#         # ⭐ 수정: 잔차 연결(res)을 먼저 계산하고 나중에 더함 (표준 ResNet 방식)
#         res = self.shortcut(x)
#         x = self.main_path(x)
#         return self.pool(x + res)

# class UpSamplingBlock(nn.Module):
#     def __init__(self, in_channels, skip_channels, out_channels):
#         super(UpSamplingBlock, self).__init__()
#         self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         # ⭐ 수정: 입력 채널 수를 정확히 계산 (업샘플링된 채널 + 스킵 연결 채널)
#         self.conv = nn.Sequential(
#             DepthwiseSeparableConv(out_channels + skip_channels, out_channels),
#             DepthwiseSeparableConv(out_channels, out_channels)
#         )
#     def forward(self, x, skip_connection):
#         x = self.upsample(x)
#         x = torch.cat([x, skip_connection], dim=1)
#         return self.conv(x)

# class SingleUNet(nn.Module):
#     def __init__(self, in_channels, out_channels_final):
#         super(SingleUNet, self).__init__()
        
#         # U-Net의 가장 바깥쪽 Skip Connection을 위한 초기 컨볼루션 레이어
#         self.in_conv = DepthwiseSeparableConv(in_channels, 64)

#         # Encoder
#         self.enc1 = DownSamplingBlock(64, 128)      # 256 -> 128
#         self.enc2 = DownSamplingBlock(128, 256)     # 128 -> 64
#         self.enc3 = DownSamplingBlock(256, 512)     # 64 -> 32
#         self.enc4 = DownSamplingBlock(512, 1024)    # 32 -> 16

#         self.bottleneck = DepthwiseSeparableConv(1024, 2048)

#         # Decoder (skip_channels 인자를 대칭되는 인코더의 출력 채널 수에 맞춤)
#         self.dec4 = UpSamplingBlock(in_channels=2048, skip_channels=1024, out_channels=1024) # 16 -> 32
#         self.dec3 = UpSamplingBlock(in_channels=1024, skip_channels=512, out_channels=512)   # 32 -> 64
#         self.dec2 = UpSamplingBlock(in_channels=512, skip_channels=256, out_channels=256)   # 64 -> 128
#         self.dec1 = UpSamplingBlock(in_channels=256, skip_channels=128, out_channels=128)   # 128 -> 256

#         self.final_conv = nn.Conv2d(128, out_channels_final, kernel_size=1)

#     def forward(self, x):
#         # Encoder
#         e1 = self.in_conv(x)  # 256x256, 64 channels
#         e2 = self.enc1(e1)    # 128x128, 128 channels
#         e3 = self.enc2(e2)    # 64x64, 256 channels
#         e4 = self.enc3(e3)    # 32x32, 512 channels
#         e5 = self.enc4(e4)    # 16x16, 1024 channels
        
#         # Bottleneck
#         b = self.bottleneck(e5) # 16x16, 2048 channels

#         # Decoder with CORRECTED skip connections
#         d4 = self.dec4(b, e5)   # Up to 32x32, connects with e4(512ch) -> No, e5(1024ch)
#         d3 = self.dec3(d4, e4)  # Up to 64x64, connects with e3(256ch) -> No, e4(512ch)
#         d2 = self.dec2(d3, e3)  # Up to 128x128, connects with e2(128ch) -> No, e3(256ch)
#         d1 = self.dec1(d2, e2)  # Up to 256x256, connects with e1(64ch) -> No, e2(128ch)
        
#         out = self.final_conv(d1)
#         return out

# # DualResUNet 클래스는 수정할 필요 없음
# class DualResUNet(nn.Module):
#     def __init__(self, in_channels=3, num_classes=4):
#         super(DualResUNet, self).__init__()
#         self.segmentation_net = SingleUNet(in_channels=in_channels, out_channels_final=num_classes)
#         self.reconstruction_net = SingleUNet(in_channels=num_classes, out_channels_final=in_channels)
#         self.softmax, self.tanh = nn.Softmax(dim=1), nn.Tanh()
#     def forward(self, x):
#         seg_logits = self.segmentation_net(x)
#         recon = self.reconstruction_net(self.softmax(seg_logits))
#         return seg_logits, self.tanh(recon)


import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 표준 U-Net 구성 요소 ---

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ConvTranspose2d를 사용하거나, Upsample + Conv를 사용할 수 있습니다.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 입력 텐서의 크기가 다를 경우를 대비한 패딩/크롭
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# --- 2. 새로 설계된 안정적인 SingleUNet ---

class StandardUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StandardUNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# --- 3. DualResUNet 래퍼는 그대로 사용, 내부 모델만 교체 ---

class DualResUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(DualResUNet, self).__init__()
        # 내부 모델을 새로 만든 StandardUNet으로 교체
        self.segmentation_net = StandardUNet(in_channels=in_channels, out_channels=num_classes)
        self.reconstruction_net = StandardUNet(in_channels=num_classes, out_channels=in_channels)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh() # 복원 이미지의 픽셀 값을 -1~1 또는 0~1로 맞춰주기 위함

    def forward(self, x):
        segmentation_logits = self.segmentation_net(x)
        mask_probs = self.softmax(segmentation_logits)
        reconstruction = self.reconstruction_net(mask_probs)
        # Tanh를 사용하려면 입력 이미지도 -1~1로 정규화하는 것이 좋음
        # Sigmoid를 사용하려면 입력 이미지를 0~1로 정규화
        return segmentation_logits, torch.sigmoid(reconstruction) # Tanh -> Sigmoid로 변경하여 안정성 확보