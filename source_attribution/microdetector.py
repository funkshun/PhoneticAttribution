import torch
from torch.nn import Softmax
from torchvision.models import resnet18


class PhonemicMicrodetector(torch.nn.Module):

    def __init__(self, phoneme, input_size, num_labels, device=None):
        super().__init__()

        self.phoneme = phoneme
        self.input_size = input_size

        backbone_layers = list(resnet18().children())[:-1]
        backbone_layers.append(torch.nn.Flatten())
        self.backbone = torch.nn.Sequential(*backbone_layers)

        test_data = torch.zeros((1, 3, input_size[0], input_size[1]))
        frontend_size = self.forward(test_data, setup=True)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(frontend_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_labels),
            # torch.nn.Softmax(dim=-1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x, setup=False):

        x = self.backbone(x)

        if setup:
            return x.size()[-1]

        x = self.fc(x)
        return x
