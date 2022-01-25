import argparse
import os
import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid")

import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import CSVDataset
from utils import plot_result, get_deterministic_coefficient, get_mean_squared_error, print_results
from model.RNNs import RNNs
from model.ANNs import ANN
from model.CNNs import TCN

torch.manual_seed(19990319)
label_max = 1940.0
label_min = 0.69


'''
    prediction, target = main(args)

# Scatter plot
    data = pd.DataFrame(np.array([target, prediction]).T, columns=['target', 'prediction'])
    sns.lmplot(x='target', y='prediction', data=data, height=8)
    plt.xlim([0, max(target)])
    plt.ylim([0, max(target)])
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')

    plt.savefig('figures/{}_scatter.png'.format(args.run_dir.replace('/', '-')), bbox_inches='tight')

# Line plot
    plt.figure(figsize=(20, 5))
    plt.plot(prediction, label='prediction', linewidth=1, c='darkred', alpha=0.8)
    plt.plot(target, label='ground truth', linewidth=1, c='darkblue', alpha=0.8)
    plt.legend()
    plt.xlim([0, args.sample_length])
    plt.ylim([0, max(target)])
    plt.xlabel('Time (hour)')
    plt.ylabel('Flow')
    plt.savefig('figures/{}_line.png'.format(args.run_dir.replace('/', '-')), bbox_inches='tight')
'''
