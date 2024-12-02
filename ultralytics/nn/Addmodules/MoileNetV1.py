import torch
from torch import nn

__all__ = ['MobileNetV1_n', 'MobileNetV1_s', 'MobileNetV1_m']


class DepthwiseSepConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_relu6: bool = True,
    ):
        """Constructs Depthwise seperable with pointwise convolution with relu and batchnorm respectively.

        Args:
            in_channels (int): input channels for depthwise convolution
            out_channels (int): output channels for pointwise convolution
            stride (int, optional): stride paramemeter for depthwise convolution. Defaults to 1.
            use_relu6 (bool, optional): whether to use standard ReLU or ReLU6 for depthwise separable convolution block. Defaults to True.
        """

        super().__init__()

        # Depthwise conv
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            (3, 3),
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.relu1 = nn.ReLU6() if use_relu6 else nn.ReLU()

        # Pointwise conv
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU6() if use_relu6 else nn.ReLU()

    def forward(self, x):
        """Perform forward pass."""

        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class MobileNetV1(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        depth_multiplier: float = 0.25,
        use_relu6: bool = True,
    ):
        """Constructs MobileNetV1 architecture

        Args:
            n_classes (int, optional): count of output neuron in last layer. Defaults to 1000.
            input_channel (int, optional): input channels in first conv layer. Defaults to 3.
            depth_multiplier (float, optional): network width multiplier ( width scaling ). Suggested Values - 0.25, 0.5, 0.75, 1.. Defaults to 1.0.
            use_relu6 (bool, optional): whether to use standard ReLU or ReLU6 for depthwise separable convolution block. Defaults to True.
        """

        super().__init__()

        # The configuration of MobileNetV1
        # input channels, output channels, stride
        config = (
            (32, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 512, 2),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 1024, 2),
            (1024, 1024, 1),
        )

        # Adding depthwise block in the model from the config
        layers = [nn.Conv2d(input_channel, int(32 * depth_multiplier), (3, 3), stride=2, padding=1)]
        for in_channels, out_channels, stride in config:
            layers.append(
                DepthwiseSepConvBlock(
                    int(in_channels * depth_multiplier),  # 输入通道
                    int(out_channels * depth_multiplier),  # 输出通道
                    stride,
                    use_relu6=use_relu6,
                )
            )

        # 将列表转换为 Sequential
        self.model = nn.Sequential(*layers)
        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        """Perform forward pass."""
        unique_tensors = {}
        for model in self.model:
            x = model(x)
            width, height = x.shape[2], x.shape[3]
            unique_tensors[(width, height)] = x
        result_list = list(unique_tensors.values())[-4:]
        return result_list


def MobileNetV1_n(width_mult=0.5):
    model = MobileNetV1(depth_multiplier=0.25)
    return model

def MobileNetV1_s(width_mult=1.0):
    model = MobileNetV1(depth_multiplier=0.5)
    return model

def MobileNetV1_m(width_mult=1.5):
    model = MobileNetV1(depth_multiplier=1)
    return model

if __name__ == "__main__":

    # Generating Sample image
    image_size = (1, 3, 224, 224)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v1 = MobileNetV1_m()

    out = mobilenet_v1(image)
    for i in range(len(out)):
        print(out[i].size())