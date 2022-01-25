import torch
import torch.nn as nn


class ANN(nn.Module):

    def __init__(self, args, input_size):
        super(ANN, self).__init__()

        assert args.ANN_num_layers >= 3  # input + hidden + output = 3

        self.loss = nn.MSELoss(reduction='mean')
        layers = []
        for i in range(args.ANN_num_layers - 2):
            layers.append(nn.Linear(args.ANN_hidden_size, args.ANN_hidden_size))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.backbone = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=args.ANN_hidden_size), nn.ReLU(),
            *layers,
        )

        self.output = nn.Linear(in_features=args.ANN_hidden_size, out_features=1)

    def forward(self, x, y):

        info = {'WRITER_KEYS': []}

        x = self.backbone(x)
        prediction = self.output(x)
        prediction = torch.squeeze(prediction, -1)
        loss = self.loss(prediction, y)

        # for calculating MSE
        info['training Loss'] = loss.item()
        info['WRITER_KEYS'].append('training Loss')

        # for calculating DC
        info['predictions'] = prediction.detach().cpu().numpy().tolist()
        info['targets'] = y.detach().cpu().numpy().tolist()

        return loss, info

    def inference(self, x):
        x = self.backbone(x)
        x = self.output(x)
        x = torch.squeeze(x, -1)

        return x

