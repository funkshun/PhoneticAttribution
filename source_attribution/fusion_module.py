import torch
from torch.nn import Sigmoid

from .microdetector import PhonemicMicrodetector


class FusionModule(torch.nn.Module):

    def __init__(self, submodel_dict, args, device):

        self.submodels = {}
        self.one_hot = {phn: i for i, phn in enumerate(submodel_dict.keys())}

        for k, v in submodel_dict.items():

            self.submodels[k] = PhonemicMicrodetector(
                k,
                (128, args.target_spec_length),
                len(args.labels),
                device=device,
            )
            self.submodels[k].to(device)
            self.submodels[k].load_state_dict(torch.load(v, map_location=device))

        lstm_input_size = len(args.labels) + len(submodel_dict)
        self.lstm = torch.nn.LSTM(
            input_size=lstm_input_size, hidden_size=256, bidirectional=True
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=len(args.labels)),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):

        submodel_predictions = torch.stack(
            [
                torch.concat([self.onehot[phn], self.submodels[phn](spec)])
                for phn, spec in x
            ]
        )

        intermediate = self.lstm(submodel_predictions)

        return self.classifier(intermediate)
