"""Submission Studio 2 - Klasifikasi Sampah (EfficientNet-B3).

Urutan kelas (alfabetis, sama dengan torchvision ImageFolder):
0 botol_plastik | 1 gelas_plastik | 2 kertas_kardus | 3 saset_kemasan | 4 styrofoam
"""
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

IMG_SIZE = 300
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 5
CLASS_NAMES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]


def build_model(pretrained=False):
    """Bangun EfficientNet-B3 + classifier kustom.

    pretrained=True hanya dipakai saat pelatihan (mengunduh bobot ImageNet).
    """
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features  # 1536
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.2),
        nn.Linear(1024, NUM_CLASSES),
    )
    return model


def get_model():
    """Bangun arsitektur saja - JANGAN mengunduh bobot atau memuat model.pth di sini."""
    return build_model(pretrained=False)  # output: logits [batch, 5]


def get_transform():
    """Transform validasi: PIL.Image (RGB) -> Tensor [3, H, W]."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
