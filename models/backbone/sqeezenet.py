import torch.nn as nn
from torchvision.models import squeezenet1_1


class DilationLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same_padding', dilation=1, bn=False):
        super(DilationLayer, self).__init__()
        if padding == 'same_padding':
            padding = int((kernel_size - 1) / 2 * dilation)
        self.Dconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, dilation=dilation)
        self.Drelu = nn.ReLU(inplace=True)
        self.Dbn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.Dconv(x)
        if self.Dbn is not None:
            x = self.Dbn(x)
        x = self.Drelu(x)
        return x


class FeatExtractorSqueezeNetx16(nn.Module):
    n_feats = [64, 128, 256, 512]

    def __init__(self, pretrained=True):
        super(FeatExtractorSqueezeNetx16, self).__init__()
        squeeze = squeezenet1_1(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            squeeze.features[0],
            squeeze.features[1],
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            squeeze.features[3],
            squeeze.features[4],
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            squeeze.features[6],
            squeeze.features[7],
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            squeeze.features[9],
            squeeze.features[10],
            squeeze.features[11],
            squeeze.features[12],
        )

        self.conv1[0].padding = (1, 1)

    def forward(self, x):
        x2 = self.conv1(x)
        x4 = self.conv2(x2)
        x8 = self.conv3(x4)
        x16 = self.conv4(x8)

        return x2, x4, x8, x16
