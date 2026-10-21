import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Urutan kelas (alfabetis, sama dengan torchvision ImageFolder):
# 0 botol_plastik . 1 gelas_plastik . 2 kertas_kardus . 3 saset_kemasan . 4 styrofoam
CLASS_NAMES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
NUM_CLASSES = 5
IMG_SIZE = 300                       # ukuran input (native ConvNeXt-Tiny 224, tapi fully-convolutional sehingga 300 tetap bisa)
MEAN = [0.485, 0.456, 0.406]         # statistik ImageNet
STD = [0.229, 0.224, 0.225]


def get_model(pretrained=False):
    """Bangun arsitektur saja. Default pretrained=False: TIDAK mengunduh bobot
    dan TIDAK memuat model.pth (bobot dimuat oleh evaluate.py).
    pretrained=True hanya untuk keperluan latihan di notebook."""
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    model.classifier = nn.Sequential(
        model.classifier[0],         # LayerNorm2d bawaan ConvNeXt
        nn.Flatten(1),
        nn.Dropout(p=0.3),
        nn.Linear(768, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.2),
        nn.Linear(1024, NUM_CLASSES),
    )
    return model                     # output: logits [batch, 5]


def get_transform():
    """Transform validasi: PIL.Image (RGB) -> Tensor [3, H, W]."""
    return Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])
