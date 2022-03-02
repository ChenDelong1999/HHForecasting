import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from model.ST_GCN.ST_GCN import ST_GCN


class Raindrop_encoder(nn.Module):
    def __init__(self, backbone_model, backbone_hidden_size, backbone_num_layers, dropout, input_size,
                 head_model='linear', head_hidden_size=36, head_num_layers=3, residual=True):
        super(Raindrop_encoder, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.backbone_model = backbone_model
        self.head_model = head_model
        self.residual = residual

        # Backbone
        if self.backbone_model == 'TCN':
            self.backbone = TemporalConvNet(num_inputs=input_size,
                                            num_channels=[backbone_hidden_size] * backbone_num_layers,
                                            dropout=dropout)
        elif self.backbone_model == 'RNN':
            self.backbone = nn.RNN(input_size=input_size, batch_first=True,
                                   hidden_size=backbone_hidden_size, num_layers=backbone_num_layers,
                                   dropout=dropout)
        elif self.backbone_model == 'LSTM':
            self.backbone = nn.LSTM(input_size=input_size, batch_first=True,
                                    hidden_size=backbone_hidden_size, num_layers=backbone_num_layers,
                                    dropout=dropout)
        elif self.backbone_model == 'GRU':
            self.backbone = nn.GRU(input_size=input_size, batch_first=True,
                                   hidden_size=backbone_hidden_size, num_layers=backbone_num_layers,
                                   dropout=dropout)
        elif self.backbone_model == 'ANN':
            assert backbone_num_layers >= 1  # input = 1

            layers = []
            for i in range(backbone_num_layers - 1):
                layers.append(nn.Conv1d(in_channels=backbone_hidden_size,
                                        out_channels=backbone_hidden_size, kernel_size=8, padding=7))
                layers.append(Chomp1d(7))
                layers.append(nn.ReLU())

            self.layers = nn.Sequential(*layers)
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=backbone_hidden_size, kernel_size=8, padding=7),
                Chomp1d(7),
                nn.ReLU(),
                *layers,
            )
        elif self.backbone_model == 'STGCN':
            self.graph_args = {}
            self.backbone = ST_GCN(in_channels=input_size,
                                   out_channels=backbone_hidden_size,
                                   num_layers=backbone_num_layers,
                                   graph_args=self.graph_args,
                                   edge_importance_weighting=True,
                                   mode='HHF')
        else:
            raise RuntimeError('model {} not defined!'.format(backbone_model))

        # Prediction head
        assert head_num_layers >= 2

        if self.head_model == 'conv1d':
            head_layers = []
            for i in range(head_num_layers - 2):
                head_layers.append(nn.Conv1d(in_channels=head_hidden_size, out_channels=head_hidden_size, kernel_size=2, padding=1))
                head_layers.append(Chomp1d(1))
                head_layers.append(nn.ReLU())
            self.head_layers = nn.Sequential(*head_layers)
            self.prediction_head = nn.Sequential(
                nn.Conv1d(in_channels=backbone_hidden_size+1, out_channels=head_hidden_size, kernel_size=2, padding=1), Chomp1d(1), nn.ReLU(),
                *head_layers,
                nn.Conv1d(in_channels=head_hidden_size, out_channels=1, kernel_size=3, padding=1)
            )
        elif self.head_model == 'linear':
            head_layers = []
            for i in range(head_num_layers - 2):
                head_layers.append(nn.Linear(in_features=head_hidden_size, out_features=head_hidden_size))
                head_layers.append(nn.ReLU())
            self.head_layers = nn.Sequential(*head_layers)
            self.prediction_head = nn.Sequential(
                nn.Linear(in_features=backbone_hidden_size+1, out_features=head_hidden_size), nn.ReLU(),
                *head_layers,
                nn.Linear(in_features=head_hidden_size, out_features=1)
            )
        else:
            raise RuntimeError('model {} not defined!'.format(head_model))

    def forward(self, raindrop, runoff_history, runoff):

        info = {'WRITER_KEYS': []}
        if self.backbone_model == 'TCN' or self.backbone_model == 'ANN':
            feature = self.backbone(raindrop.transpose(1, 2)).transpose(1, 2)
        elif self.backbone_model == 'STGCN':
            feature = self.backbone(raindrop.unsqueeze(2).transpose(1, 2).transpose(1, 3).unsqueeze(4)) \
                .transpose(1, 2).squeeze(-1)
        else:
            feature, a = self.backbone(raindrop, None)

        if self.head_model == 'linear':
            prediction = self.prediction_head(torch.cat([feature, torch.unsqueeze(runoff_history, -1)], dim=2))
        else:
            prediction = self.prediction_head(
                torch.cat([feature.transpose(1,2), torch.unsqueeze(runoff_history, 1)], dim=1)).transpose(1,2)

        prediction = torch.squeeze(prediction, -1)
        if self.residual is True:
            prediction += runoff_history
        loss = self.loss(prediction, runoff)

        # for calculating MSE
        info['training Loss'] = loss.item()
        info['WRITER_KEYS'].append('training Loss')

        # for calculating DC
        info['predictions'] = prediction.detach().cpu().numpy().tolist()
        info['targets'] = runoff.detach().cpu().numpy().tolist()

        return loss, info

    def inference(self, raindrop, runoff_history):
        if self.backbone_model == 'TCN' or self.backbone_model == 'ANN':
            feature = self.backbone(raindrop.transpose(1, 2)).transpose(1, 2)
        elif self.backbone_model == 'STGCN':
            feature = self.backbone(raindrop.unsqueeze(2).transpose(1, 2).transpose(1, 3).unsqueeze(4)) \
                .transpose(1, 2).squeeze(-1)
        else:
            feature, _ = self.backbone(raindrop, None)

        if self.head_model == 'linear':
            prediction = self.prediction_head(torch.cat([feature, torch.unsqueeze(runoff_history, -1)], dim=2))
        else:
            prediction = self.prediction_head(
                torch.cat([feature.transpose(1,2), torch.unsqueeze(runoff_history, 1)], dim=1)).transpose(1,2)

        prediction = torch.squeeze(prediction, -1)
        if self.residual is True:
            prediction += runoff_history
        return prediction

    def extract_feature(self, raindrop):
        if self.backbone_model == 'TCN' or self.backbone_model == 'ANN':
            feature = self.backbone(raindrop.transpose(1, 2)).transpose(1, 2)
        elif self.backbone_model == 'STGCN':
            feature = self.backbone(raindrop.unsqueeze(2).transpose(1, 2).transpose(1, 3).unsqueeze(4))\
                .transpose(1, 2).squeeze(-1)
        else:
            feature, _ = self.backbone(raindrop, None)

        feature = feature.reshape(-1, 36)
        return feature


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
