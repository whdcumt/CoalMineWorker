import torch
import torch.nn as nn


__all__ = ['C3k2_HetConv1', 'C3k2_HetConv2']



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class HetConv(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1, p=4):
        """
        Initialize the HetConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param p: the value of P used in HetConv
        """
        super(HetConv, self).__init__()
        self.p = p
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filters = nn.ModuleList()
        self.convolution_1x1_index = []
        # Compute the indices of input channels fed to 1x1 convolutional kernels in all filters.
        # These indices of input channels are also the indices of the 1x1 convolutional kernels in the filters.
        # This is only executed when the HetConv class is created,
        # and the execution time is not included during inference.
        for i in range(self.p):
            self.convolution_1x1_index.append(self.compute_convolution_1x1_index(i))
        # Build HetConv filters.
        for i in range(self.p):
            self.filters.append(self.build_HetConv_filters(stride, p))

    def compute_convolution_1x1_index(self, i):
        """
        Compute the indices of input channels fed to 1x1 convolutional kernels in the i-th branch of filters (i=0, 1, 2,…, P-1). The i-th branch of filters consists of the {i, i+P, i+2P,…, i+N-P}-th filters.
        :param i: the i-th branch of filters in HetConv
        :return: return the required indices of input channels
        """
        index = [j for j in range(0, self.input_channels)]
        # Remove the indices of input channels fed to 3x3 convolutional kernels in the i-th branch of filters.
        while i < self.input_channels:
            index.remove(i)
            i += self.p
        return index

    def build_HetConv_filters(self, stride, p):
        """
        Build N/P filters in HetConv.
        :param stride: convolution stride
        :param p: the value of P used in HetConv
        :return: return N/P HetConv filters
        """
        temp_filters = nn.ModuleList()
        # nn.Conv2d arguments: nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        temp_filters.append(nn.Conv2d(self.input_channels//p, self.output_channels//p, 3, stride, 1, bias=False))
        temp_filters.append(nn.Conv2d(self.input_channels-self.input_channels//p, self.output_channels//p, 1, stride, 0, bias=False))
        return temp_filters

    def forward(self, input_data):
        """
        Define how HetConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        output_feature_maps = []
        # Loop P times to get output feature maps. The number of output feature maps = the batch size.
        for i in range(0, self.p):

            # M/P HetConv filter kernels perform the 3x3 convolution and output to N/P output channels.
            output_feature_3x3 = self.filters[i][0](input_data[:, i::self.p, :, :])
            # (M-M/P) HetConv filter kernels perform the 1x1 convolution and output to N/P output channels.
            output_feature_1x1 = self.filters[i][1](input_data[:, self.convolution_1x1_index[i], :, :])

            # Obtain N/P output feature map channels.
            output_feature_map = output_feature_1x1 + output_feature_3x3

            # Append N/P output feature map channels.
            output_feature_maps.append(output_feature_map)

        # Get the batch size, number of output channels (N/P), height and width of output feature map.
        N, C, H, W = output_feature_maps[0].size()
        # Change the value of C to the number of output feature map channels (N).
        C = self.p * C
        # Arrange the output feature map channels to make them fit into the shifted manner.
        return torch.cat(output_feature_maps, 1).view(N, self.p, C//self.p, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)



class CSPHet_Bottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.DualPConv = nn.Sequential(HetConv(dim, dim), HetConv(dim, dim))

    def forward(self, x):
        return self.DualPConv(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3kPConv(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(CSPHet_Bottleneck(c_) for _ in range(n)))


class C3k2_HetConv1(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else CSPHet_Bottleneck(self.c)for _ in range(n)
        )
        # 解析利用MLLABlock替换Bottneck


class C3k2_HetConv2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3kPConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in
            range(n)
        )
        # 解析利用MLLABlock替换C3k中的Bottneck


if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = C3k2_HetConv1(64, 128)

    out = model(image)
    print(out.size())
