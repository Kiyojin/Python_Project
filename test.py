import torch
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: list,
                 image_size: tuple,
                 num_classes: int,
                 kernel_size: list,
                 use_batchnormalization: bool = False,
                 activation_function: torch.nn.Module = torch.nn.ReLU()
                 ):
        super().__init__()

        hidden_layers = []
        for out_channels, kernel in zip(hidden_channels, kernel_size):
            layer = torch.nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=kernel,
                                    padding=kernel // 2, bias=False)
            hidden_layers.append(layer)
            if use_batchnormalization:
                hidden_layers.append(torch.nn.BatchNorm2d(num_features=out_channels))
            hidden_layers.append(activation_function)
            input_channels = out_channels

        if len(hidden_channels) != len(kernel_size):
            raise ValueError('hidden_channels and kernel_size must have the same length')

        self.hidden_layers = torch.nn.Sequential(*hidden_layers)

        self.linear_layer0 = torch.nn.Linear(in_features=hidden_channels[-1] * image_size[0] * image_size[1],
                                             out_features=128)
        self.linear_layer1 = torch.nn.Linear(in_features=128,
                                             out_features=64)
        self.linear_layer2 = torch.nn.Linear(in_features=64,
                                             out_features=32)
        self.output_layer = torch.nn.Linear(in_features=32,
                                            out_features=num_classes)

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        input_images = self.hidden_layers(input_images)

        input_images = input_images.view(input_images.size(0), -1)

        input_images = self.linear_layer0(input_images)
        input_images = self.linear_layer1(input_images)
        input_images = self.linear_layer2(input_images)
        return self.output_layer(input_images)


model = MyCNN(
        input_channels=1,
        hidden_channels=[32, 64, 128],
        image_size=(100, 100),
        use_batchnormalization=True,
        num_classes=20,
        kernel_size=[3, 5, 7],
        activation_function=torch.nn.ELU())
