import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from openTSNE import TSNE
from PIL import Image
from torch import autograd


def save_backbone_features(model, data_loader):
    print('--- extract_backbone_features...')
    model.eval()
    features = []
    for index, (raindrop, runoff_history, runoff) in enumerate(data_loader):
        feature = model.extract_feature(raindrop.cuda().float())
        features.append(feature.detach().cpu().numpy())
    features = np.array(features)
    features = np.reshape(features, [-1, 36])
    print(features.shape)
    np.save('TunXi_features.npy', features)
    print('--- feature saved to TunXi_features.npy')


class GANDataset(torch.utils.data.Dataset):
    def __init__(self, forecast_range, mode, train_test_split_ratio, sample_length, dataset='ChangHua', training_set_scale=1):

        assert mode in ['train', 'test']
        assert forecast_range >= 1
        self.mode = mode
        self.sample_length = sample_length
        self.forecast_range = forecast_range
        self.training_set_scale = training_set_scale
        self.overlapping_split = 20

        # 加载屯溪编码器输出的特征
        self.TunXi_feature = np.load(r'TunXi_features.npy')
        self.TunXi_feature_cursor = 0

        csv_file = f'dataset/{dataset}/data.csv'
        print('--- reading', csv_file)
        df = pd.read_csv(csv_file)
        df = df.dropna()
        data = np.array(df)

        raindrop = data[:-self.forecast_range, 1:]
        raindrop = (raindrop - np.min(raindrop, axis=0)) / (np.max(raindrop, axis=0) - np.min(raindrop, axis=0))
        runoff_history = data[:-self.forecast_range, 0]
        runoff = data[self.forecast_range:, 0]

        train_test_spilt = int(train_test_split_ratio * len(raindrop))
        training_subset = int(train_test_spilt * self.training_set_scale)
        if self.mode == 'train':
            self.raindrop = raindrop[:training_subset, :]
            self.runoff_history = runoff_history[:training_subset]
            self.runoff = runoff[:training_subset]
        elif self.mode == 'test':
            self.raindrop = raindrop[train_test_spilt:, :]
            self.runoff_history = runoff_history[train_test_spilt:]
            self.runoff = runoff[train_test_spilt:]

        print(f'--- loaded [{dataset}] [{self.mode}] set: '
              f'input shape: {self.raindrop.shape}, '
              f'target shape: {self.runoff.shape}, len(dataset)={len(self)}')

    def __getitem__(self, index):
        start = int(index / self.overlapping_split * self.sample_length)  # 20-overlapping samples

        if self.TunXi_feature_cursor >= self.TunXi_feature.shape[0]:
            self.TunXi_feature_cursor = 0
        feature_start = self.TunXi_feature_cursor
        feature = self.TunXi_feature[feature_start:feature_start + self.sample_length]
        self.TunXi_feature_cursor = self.TunXi_feature_cursor + self.sample_length

        return torch.from_numpy(self.raindrop[start:start + self.sample_length]), \
               torch.from_numpy(self.runoff_history[start:start + self.sample_length]), \
               torch.from_numpy(self.runoff[start:start + self.sample_length]), \
               torch.from_numpy(feature)

    def __len__(self):
        return int(len(self.raindrop) / self.sample_length * self.overlapping_split) - self.overlapping_split

    def get_input_size(self):
        return self.raindrop.shape[1]

    def get_TunXi_feature(self):
        return self.TunXi_feature


class Discriminator_1DCNN(nn.Module):
    def __init__(self):
        super(Discriminator_1DCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(36, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),

            nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),

            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.encoder(x)
        x = self.fc(x.transpose(1, 2))
        x = torch.mean(x, dim=1)
        return x


def calc_gradient_penalty_ST(D, real_data, fake_data, term=None):
    if term is None:
        term = ['real', 'fake', 'real_fake', 'real_motion', 'fake_motion']
    loss = 0
    center = 0
    if 'real' in term:
        output = D(real_data.requires_grad_(True))
        gradients = autograd.grad(outputs=output, inputs=real_data, grad_outputs=torch.ones(output.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_real = gradients.norm(2, dim=1)
        GP_real = ((norm_real - center) ** 2).mean()
        loss += GP_real

    if 'fake' in term:
        output = D(fake_data.requires_grad_(True))
        gradients = autograd.grad(outputs=output, inputs=fake_data, grad_outputs=torch.ones(output.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_fake = gradients.norm(2, dim=1)
        GP_fake = ((norm_fake - center) ** 2).mean()
        loss += GP_fake

    if 'real_motion' in term:
        real_motion = real_data - real_data.mean(dim=1).unsqueeze(1)
        real_structure = real_data.mean(dim=1).unsqueeze(1)
        fake_structure = fake_data.mean(dim=1).unsqueeze(1)

        alpha = torch.rand(1).cuda()
        input = (alpha * real_motion + alpha * fake_structure + (1 - alpha) * real_structure).requires_grad_(True)
        output = D(input)
        gradients = autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones(output.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_real_motion = gradients.norm(2, dim=1)
        GP_real_motion = ((norm_real_motion - center) ** 2).mean()
        loss += GP_real_motion

    if 'fake_motion' in term:
        fake_motion = fake_data - fake_data.mean(dim=1).unsqueeze(1)
        real_structure = real_data.mean(dim=1).unsqueeze(1)
        fake_structure = fake_data.mean(dim=1).unsqueeze(1)

        alpha = torch.rand(1).cuda()
        input = (alpha * fake_motion + alpha * fake_structure + (1 - alpha) * real_structure).requires_grad_(True)
        output = D(input)
        gradients = autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones(output.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_fake_motion = gradients.norm(2, dim=1)
        GP_fake_motion = ((norm_fake_motion - center) ** 2).mean()
        loss += GP_fake_motion

    if 'real_fake' in term:
        alpha = torch.Tensor(np.random.random((real_data.size(0), 1, 1))).cuda()
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
        disc_interpolates = D(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_real_fake = gradients.norm()
        gradient_penalty = ((norm_real_fake - center) ** 2).mean()
        loss += gradient_penalty

    return loss


def plot_feature_tsne(prediction, feature):
    print('fitting tsne')
    tsne = TSNE()  # TSNE降维，降到2
    # 降维后的数据
    data = np.concatenate([prediction, feature], axis=0)
    data = tsne.fit(data)
    x_min, x_max = np.min(data), np.max(data)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理

    plt.scatter(data[:prediction.shape[0], 0], data[:prediction.shape[0], 1], color='r', s=1, label='ChangHua',
                alpha=0.5)
    plt.scatter(data[prediction.shape[0]:, 0], data[prediction.shape[0]:, 1], color='b', s=1, label='TunXi', alpha=0.5)
    plt.legend(loc="best", fontsize=6)

    image_path = 'temp_tsne.png'
    plt.savefig(image_path)
    plt.close()
    image_PIL = Image.open(image_path)
    img = np.array(image_PIL)
    plt.close()
    print('tsne ok')

    return img


def plot_backbone_features(prediction, feature):
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))
    for i, plt_image in enumerate(
            [prediction[0], feature[0], prediction[32], feature[32], prediction[63], feature[63]]):
        axes[i].matshow(plt_image, aspect='auto')
        if i % 2 == 0:
            axes[i].set_title("ChangHua feature")
        else:
            axes[i].set_title("TunXi feature")
        axes[i].grid(b=False)
    image_path = 'temp_matshow.png'
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
    image_PIL = Image.open(image_path)
    img = np.array(image_PIL)
    plt.close()
    return img
