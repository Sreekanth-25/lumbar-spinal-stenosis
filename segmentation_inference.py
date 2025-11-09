import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = conv_block(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = nn.functional.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet50UNet(nn.Module):
    def __init__(self, pretrained=False, n_classes=1):
        super().__init__()
        weights = None
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=weights)
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if weights is not None:
            self.conv1.weight.data = resnet.conv1.weight.mean(dim=1, keepdim=True)
        self.bn1, self.relu, self.maxpool = resnet.bn1, resnet.relu, resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.center = conv_block(2048, 512)
        self.dec4 = DecoderBlock(512, 1024, 256)
        self.dec3 = DecoderBlock(256, 512, 128)
        self.dec2 = DecoderBlock(128, 256, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1),
        )

    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))
        p1 = self.maxpool(c1)
        e1, e2, e3, e4 = self.layer1(p1), self.layer2(self.layer1(p1)), self.layer3(self.layer2(self.layer1(p1))), self.layer4(self.layer3(self.layer2(self.layer1(p1))))
        center = self.center(e4)
        d4, d3, d2, d1 = self.dec4(center, e3), None, None, None
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, c1)
        out = self.final_conv(d1)
        out = nn.functional.interpolate(out, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return out

def preprocess(pil_img, size=IMG_SIZE):
    g = pil_img.convert("L").resize((size, size))
    arr = np.asarray(g).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def load_unet(weights="best_unet_resnet50.pth"):
    model = ResNet50UNet(pretrained=False, n_classes=1).to(DEVICE)
    state = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

@torch.inference_mode()
def segment_image(model, pil_img):
    x = preprocess(pil_img).to(DEVICE)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8)
    return mask, prob

def make_overlay(pil_img, mask):
    img = pil_img.convert("RGB").resize((mask.shape[1], mask.shape[0]))
    overlay = np.array(img).copy()
    overlay[mask > 0] = [255, 0, 0]
    out = cv2.addWeighted(np.array(img), 1.0, overlay, 0.4, 0)
    return Image.fromarray(out)