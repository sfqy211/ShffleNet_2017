# test_shufflenet.py

import numpy as np
import time
from ShuffleNet import Conv2D, BatchNorm2D, ReLU, ChannelShuffle, MaxPool2D, AdaptiveAvgPool2D, Linear, Sequential, ShuffleUnit, ShuffleNet

# 单元测试

def test_conv2d():
    conv2d = Conv2D(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
    input_tensor = np.random.randn(1, 3, 10, 10)
    output = conv2d.forward(input_tensor)
    assert output.shape == (1, 6, 10, 10), f"Expected output shape (1, 6, 10, 10), got {output.shape}"
    print("Conv2D layer test passed.")

def test_batchnorm2d():
    batch_norm = BatchNorm2D(num_features=6)
    input_tensor = np.random.randn(1, 6, 10, 10)
    output = batch_norm.forward(input_tensor)
    assert output.shape == (1, 6, 10, 10), f"Expected output shape (1, 6, 10, 10), got {output.shape}"
    print("BatchNorm2D layer test passed.")

def test_relu():
    relu = ReLU()
    input_tensor = np.random.randn(1, 6, 10, 10)
    output = relu.forward(input_tensor)
    assert output.shape == (1, 6, 10, 10), f"Expected output shape (1, 6, 10, 10), got {output.shape}"
    print("ReLU layer test passed.")

def test_channelshuffle():
    channel_shuffle = ChannelShuffle(groups=2)
    input_tensor = np.random.randn(1, 6, 10, 10)
    output = channel_shuffle.forward(input_tensor)
    assert output.shape == (1, 6, 10, 10), f"Expected output shape (1, 6, 10, 10), got {output.shape}"
    print("ChannelShuffle layer test passed.")

def test_maxpool2d():
    max_pool = MaxPool2D(kernel_size=2, stride=2)
    input_tensor = np.random.randn(1, 6, 10, 10)
    output = max_pool.forward(input_tensor)
    assert output.shape == (1, 6, 5, 5), f"Expected output shape (1, 6, 5, 5), got {output.shape}"
    print("MaxPool2D layer test passed.")

def test_adaptiveavgpool2d():
    adaptive_avg_pool = AdaptiveAvgPool2D(output_size=(1, 1))
    input_tensor = np.random.randn(1, 6, 10, 10)
    output = adaptive_avg_pool.forward(input_tensor)
    assert output.shape == (1, 6, 1, 1), f"Expected output shape (1, 6, 1, 1), got {output.shape}"
    print("AdaptiveAvgPool2D layer test passed.")

def test_linear():
    linear = Linear(in_features=6, out_features=10)
    input_tensor = np.random.randn(1, 6)
    output = linear.forward(input_tensor)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    print("Linear layer test passed.")


def test_shufflenet():
    model = ShuffleNet(num_classes=1000, groups=3)
    input_tensor = np.random.randn(1, 3, 224, 224)
    output = model.forward(input_tensor)
    assert output.shape == (1, 1000), f"Expected output shape (1, 1000), got {output.shape}"
    print("ShuffleNet test passed.")

# 性能测试

def test_forward_pass_time():
    model = ShuffleNet(num_classes=1000, groups=3)
    input_tensor = np.random.randn(1, 3, 224, 224)
    start_time = time.time()
    output = model.forward(input_tensor)
    end_time = time.time()
    forward_time = end_time - start_time
    print(f"Forward pass time: {forward_time:.4f} seconds")

def test_shuffleunit():
    shuffle_unit = ShuffleUnit(in_channels=3, out_channels=6, stride=1, groups=3)
    input_tensor = np.random.randn(1, 3, 10, 10)
    output = shuffle_unit.forward(input_tensor)
    assert output.shape == (1, 6, 10, 10), f"Expected output shape (1, 6, 10, 10), got {output.shape}"
    print("ShuffleUnit test passed.")

def test_average_forward_pass_time():
    model = ShuffleNet(num_classes=1000, groups=3)
    input_tensor = np.random.randn(1, 3, 224, 224)
    num_runs = 1 #跑一轮就要1分钟
    total_time = 0

    for _ in range(num_runs):
        start_time = time.time()
        output = model.forward(input_tensor)
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = total_time / num_runs
    print(f"Average forward pass time over {num_runs} runs: {average_time:.4f} seconds")

# 运行所有测试

if __name__ == "__main__":
    test_conv2d()
    test_batchnorm2d()
    test_relu()
    test_channelshuffle()
    test_maxpool2d()
    test_adaptiveavgpool2d()
    test_linear()
    test_shufflenet()
    test_forward_pass_time()
    test_shuffleunit()
    test_average_forward_pass_time()