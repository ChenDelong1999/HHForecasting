import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sns.set_theme(style="whitegrid")


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def get_deterministic_coefficient(predictions, targets):
    upper = np.sum((predictions - targets) ** 2)
    lower = np.sum((targets - np.average(targets)) ** 2)
    DC = 1 - upper / (lower + 1e-9)

    return DC * 100


def get_mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def plot_result_(y_predict, y, index):  # origin version
    y_predict = y_predict.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    fig = plt.figure()
    plt.plot(y[index, :], color='gray', label='Ground Truth')
    plt.plot(y_predict[index, :], color='r', label='Prediction')

    plt.legend()
    plt.xticks([])
    plt.yticks([])

    '''image_path = '.temp({}).png'.format(index)
    plt.savefig(image_path)
    plt.close()
    image_PIL = Image.open(image_path)
    img = np.array(image_PIL)'''

    fig.canvas.draw()
    plt.close()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return img


def plot_result(target, prediction):
    data = pd.DataFrame(np.array([target, prediction]).T, columns=['target', 'prediction'])

    # --- Scatter plot ---
    # fig_scatter = plt.figure()
    fig_scatter = sns.lmplot(x='target', y='prediction', data=data, height=8,
                             line_kws={'color': 'darkred'}).figure
    plt.xlim([0, 1500])
    plt.ylim([0, 1500])
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    # plt.savefig('figures/{}_scatter.png'.format(args.run_dir.replace('/', '-')), bbox_inches='tight')

    # --- Line plot ---
    fig_line = plt.figure(figsize=(30, 6))
    plt.plot(prediction, label='prediction', linewidth=1, c='darkred', alpha=0.5)
    plt.plot(target, label='ground truth', linewidth=1, c='darkblue', alpha=0.5)
    plt.legend(loc='upper left')
    plt.xlim([0, 10000])
    plt.xlabel('Time (hour)')
    plt.ylabel('Flow')

    # plt.savefig('new_1200.png')

    return fig_scatter, fig_line


def init_results(args, stage):
    results = {}
    arg_dict = vars(args)
    for key in arg_dict.keys():
        results[key] = []

    if stage == 1:
        metrics = ['TRAIN_MSE', 'TEST_MSE', 'TRAIN_DC', 'TEST_DC']
    if stage == 2:
        metrics = ['TEST_MSE', 'TEST_DC']

    for metric in metrics:
        results[metric] = []
    arg_dict = vars(args)
    for key in arg_dict.keys():
        results[key].append(arg_dict[key])
    return results


def print_results(results):
    file = 'results.csv'
    with open(file, 'a+') as f:
        print('\n--- printing results to file {}'.format(file))
        print('time', file=f, end=',')
        for key in results.keys():
            print(key, end=',', file=f)
        print(file=f)
        print(datetime.datetime.now(), file=f, end=',')

        for i in range(len(results[key])):
            for key in results.keys():
                print(results[key][i], end=',', file=f)
        print(file=f)


def print_args(args):
    options = vars(args)
    print('Args:')
    print('-' * 64)
    for key in options.keys():
        print(f'\t{key}:\t{options[key]}')

    print('-' * 64)


