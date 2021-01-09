from torchvision.models import resnet50
from torch.nn import Conv2d, Linear


def get_model(out_features=11):
    """Modifies the Resnet architecture for a greyscale image and specified number of output features."""
    model = resnet50()
    model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = Linear(in_features=2048, out_features=out_features, bias=True)
    return model

