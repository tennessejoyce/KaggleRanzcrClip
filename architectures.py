from torch.nn import Conv2d, Linear, Sequential


def Resnet18(out_features=11):
    """Modifies the Resnet architecture for a greyscale image and specified number of output features."""
    from torchvision.models import resnet18
    model = resnet18()
    model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = Linear(in_features=512, out_features=out_features, bias=True)
    return model


def EfficientNetB0(out_features=11):
    """Modifies the Resnet architecture for a greyscale image and specified number of output features."""
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_name('efficientnet-b0')
    model._conv_stem = Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    model._fc = Linear(in_features=1280, out_features=out_features, bias=True)
    # Remove the final swish layer (return logits)
    # model = Sequential(*(list(model.children())[:-1]))
    return model

def Resnet50(out_features=11):
    """Modifies the Resnet architecture for a greyscale image and specified number of output features."""
    from torchvision.models import resnet50
    model = resnet50()
    model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = Linear(in_features=2048, out_features=out_features, bias=True)
    return model
