import matplotlib.pyplot as plt
import datetime
import numpy as np
from PIL import Image
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


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
    fig_line = plt.figure(figsize=(20, 5))
    plt.plot(prediction, label='prediction', linewidth=1, c='darkred', alpha=0.8)
    plt.plot(target, label='ground truth', linewidth=1, c='darkblue', alpha=0.8)
    plt.legend()
    plt.ylim([0, 2000])
    plt.xlim([0, len(prediction)])
    plt.xlabel('Time (hour)')
    plt.ylabel('Flow')

    # plt.savefig('figures/{}_line.png'.format(args.run_dir.replace('/', '-')), bbox_inches='tight')

    return fig_scatter, fig_line


def print_results(args, results):
    file = 'results.csv'
    with open(file, 'a+') as f:
        print('\n--- printing results to file {}'.format(file))
        print(file=f)
        print(datetime.datetime.now())
        print(datetime.datetime.now(), file=f)
        for key in results.keys():
            print(key, end=',')
            print(key, end=',', file=f)
        print()
        print(file=f)

        for i in range(len(results[key])):
            for key in results.keys():
                print(results[key][i], end=',')
                print(results[key][i], end=',', file=f)
            print()
            print(file=f)
