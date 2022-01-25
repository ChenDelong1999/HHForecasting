import torch
import torch.nn as nn


class RNNs(nn.Module):

    def __init__(self, args, cell, input_size):
        super(RNNs, self).__init__()

        self.loss = nn.MSELoss()

        if cell == 'RNN':
            self.backbone = nn.RNN(input_size=input_size, batch_first=True,
                                   hidden_size=args.RNN_hidden_size, num_layers=args.RNN_num_layers, dropout=args.RNN_dropout)
        elif cell == 'LSTM':
            self.backbone = nn.LSTM(input_size=input_size, batch_first=True,
                                    hidden_size=args.RNN_hidden_size, num_layers=args.RNN_num_layers, dropout=args.RNN_dropout)
        elif cell == 'GRU':
            self.backbone = nn.GRU(input_size=input_size, batch_first=True,
                                   hidden_size=args.RNN_hidden_size, num_layers=args.RNN_num_layers, dropout=args.RNN_dropout)

        self.output = nn.Sequential(
            # nn.Linear(in_features=args.RNN_hidden_size, out_features=args.RNN_hidden_size), nn.ReLU(),
            nn.Linear(in_features=args.RNN_hidden_size, out_features=args.RNN_hidden_size), nn.ReLU(),
            nn.Linear(in_features=args.RNN_hidden_size, out_features=1))

    def forward(self, x, y):

        info = {'WRITER_KEYS': []}

        x, _ = self.backbone(x, None)
        prediction = self.output(x)
        prediction = torch.squeeze(prediction, -1)
        loss = self.loss(prediction, y)

        info['training Loss'] = loss.item()
        info['WRITER_KEYS'].append('training Loss')

        info['predictions'] = prediction.detach().cpu().numpy().tolist()
        info['targets'] = y.detach().cpu().numpy().tolist()

        return loss, info

    def inference(self, x):
        x, _ = self.backbone(x, None)
        x = self.output(x)
        x = torch.squeeze(x, -1)

        return x


