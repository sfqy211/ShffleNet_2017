import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.weight = np.random.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for b in range(batch_size):
            for g in range(self.groups):
                for c in range(self.out_channels // self.groups):
                    for h in range(out_height):
                        for w in range(out_width):
                            h_start = h * self.stride
                            h_end = h_start + self.kernel_size
                            w_start = w * self.stride
                            w_end = w_start + self.kernel_size
                            output[b, g * (self.out_channels // self.groups) + c, h, w] = \
                                np.sum(x_padded[b, g * (in_channels // self.groups):(g + 1) * (in_channels // self.groups), h_start:h_end, w_start:w_end] * self.weight[g * (self.out_channels // self.groups) + c]) + self.bias[g * (self.out_channels // self.groups) + c]

        return output

class BatchNorm2D:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, channels)
        mean = np.mean(x_reshaped, axis=0)
        var = np.var(x_reshaped, axis=0)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        x_normalized = (x_reshaped - mean) / np.sqrt(var + self.eps)
        x_normalized = x_normalized.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)
        return self.gamma[None, :, None, None] * x_normalized + self.beta[None, :, None, None]

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

class ChannelShuffle:
    def __init__(self, groups):
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        channels_per_group = num_channels // self.groups
        x = x.reshape(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(0, 2, 1, 3, 4).reshape(batch_size, -1, height, width)
        return x

class MaxPool2D:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        output = np.zeros((batch_size, channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        output[b, c, h, w] = np.max(x_padded[b, c, h_start:h_end, w_start:w_end])

        return output

class AdaptiveAvgPool2D:
    def __init__(self, output_size):
        self.output_size = output_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out_height, out_width = self.output_size
        pool_height = height // out_height
        pool_width = width // out_width
        output = np.zeros((batch_size, channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * pool_height
                        h_end = h_start + pool_height
                        w_start = w * pool_width
                        w_end = w_start + pool_width
                        output[b, c, h, w] = np.mean(x[b, c, h_start:h_end, w_start:w_end])

        return output

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(out_features, in_features)
        self.bias = np.zeros(out_features)

    def forward(self, x):
        return np.dot(x, self.weight.T) + self.bias

class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

class ShuffleUnit:
    def __init__(self, in_channels, out_channels, stride, groups):
        self.stride = stride
        self.groups = groups
        mid_channels = out_channels // 4

        if self.stride == 2:
            self.residual = Sequential(
                Conv2D(in_channels, in_channels, 1, 1, 0, groups=groups),
                BatchNorm2D(in_channels),
                ReLU(),
                Conv2D(in_channels, in_channels, 3, stride, 1, groups=in_channels),
                BatchNorm2D(in_channels),
                Conv2D(in_channels, out_channels, 1, 1, 0, groups=1),
                BatchNorm2D(out_channels)
            )
            self.shortcut = Sequential(
                Conv2D(in_channels, in_channels, 3, stride, 1, groups=in_channels),
                BatchNorm2D(in_channels),
                Conv2D(in_channels, out_channels, 1, 1, 0, groups=1),
                BatchNorm2D(out_channels)
            )
        else:
            self.residual = Sequential(
                Conv2D(in_channels, mid_channels, 1, 1, 0, groups=1),
                BatchNorm2D(mid_channels),
                ReLU(),
                Conv2D(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels),
                BatchNorm2D(mid_channels),
                Conv2D(mid_channels, out_channels, 1, 1, 0, groups=1),
                BatchNorm2D(out_channels)
            )
            self.shortcut = Sequential()

        self.shuffle = ChannelShuffle(groups)
        self.relu = ReLU()

    def forward(self, x):
        residual = self.residual.forward(x)
        shortcut = self.shortcut.forward(x)
        output = self.relu.forward(residual + shortcut)
        return self.shuffle.forward(output)

class ShuffleNet:
    def __init__(self, num_classes=1000, groups=3):
        self.groups = groups
        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 240, 480, 960]

        self.conv1 = Conv2D(3, self.stage_out_channels[1], 3, 2, 1)
        self.bn1 = BatchNorm2D(self.stage_out_channels[1])
        self.relu = ReLU()
        self.maxpool = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.features = []
        input_channels = self.stage_out_channels[1]
        for i in range(len(self.stage_repeats)):
            output_channels = self.stage_out_channels[i + 2]
            for j in range(self.stage_repeats[i]):
                stride = 2 if j == 0 else 1
                self.features.append(ShuffleUnit(input_channels, output_channels, stride, groups=self.groups))
                input_channels = output_channels

        self.features = Sequential(*self.features)
        self.conv5 = Conv2D(input_channels, 1024, 1, 1, 0)
        self.bn5 = BatchNorm2D(1024)
        self.avgpool = AdaptiveAvgPool2D((1, 1))
        self.fc = Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu.forward(x)
        x = self.maxpool.forward(x)
        x = self.features.forward(x)
        x = self.conv5.forward(x)
        x = self.bn5.forward(x)
        x = self.relu.forward(x)
        x = self.avgpool.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc.forward(x)
        return x

# 创建 ShuffleNet 模型实例
model = ShuffleNet(num_classes=1000, groups=3)

# 创建一个随机输入张量
input_tensor = np.random.randn(1, 3, 224, 224)

# 前向传播
output = model.forward(input_tensor)

print(output.shape)  # 输出: (1, 1000)