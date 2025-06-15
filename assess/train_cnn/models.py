"""CNN model architectures for regression tasks."""

import torch
import torch.nn as nn
import torchvision.models as models


class RegressionHead(nn.Module):
    """Simple regression head for CNN backbones."""

    def __init__(self, in_features: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()


def get_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Get a CNN backbone with optional pretrained weights.

    Args:
        name: Name of the backbone architecture
        pretrained: Whether to use pretrained weights

    Returns:
        tuple of (backbone model, output features)
    """
    weights = "DEFAULT" if pretrained else None

    if name == "resnet18":
        model = models.resnet18(weights=weights)
        out_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-1])
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        out_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-1])
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        out_features = model.classifier[1].in_features
        model = nn.Sequential(*list(model.children())[:-1])
    elif name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights)
        out_features = model.classifier[1].in_features
        model = nn.Sequential(*list(model.children())[:-1])
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        out_features = model.classifier[3].in_features
        model = nn.Sequential(*list(model.children())[:-1])
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        out_features = model.classifier[2].in_features
        model = nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError(f"Unknown backbone: {name}")

    return model, out_features


class CNNRegressor(nn.Module):
    """CNN model for regression tasks."""

    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone, out_features = get_backbone(backbone_name, pretrained)
        self.regressor = RegressionHead(out_features)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)


MODELS = {
    "resnet18": lambda: CNNRegressor("resnet18"),
    "resnet50": lambda: CNNRegressor("resnet50"),
    "efficientnet_b0": lambda: CNNRegressor("efficientnet_b0"),
    "efficientnet_b3": lambda: CNNRegressor("efficientnet_b3"),
    "mobilenet_v3_small": lambda: CNNRegressor("mobilenet_v3_small"),
    "convnext_tiny": lambda: CNNRegressor("convnext_tiny"),
}
