import argparse
import os

import tqdm
import numpy as np
import torch
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CSVDataset
from utils import plot_result, get_deterministic_coefficient, get_mean_squared_error, print_results
from model.RNNs import RNNs
from model.ANNs import ANN
from model.CNNs import TCN

torch.manual_seed(19990319)


def test(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for index, (input_data, target) in enumerate(test_loader):
            input_data, target = input_data.cuda().float(), target.cuda().float()
            prediction = model.inference(input_data)

            predictions.extend(prediction.flatten().detach().cpu().numpy().tolist())
            targets.extend(target.flatten().detach().cpu().tolist())
    predictions, targets = np.array(predictions), np.array(targets)
    return predictions, targets


def train(model, train_loader, test_loader, writer, save_path):

    options = vars(args)
    print('Args:')
    print('-' * 64)
    for key in options.keys():
        print(f'\t{key}:\t{options[key]}')

    print('-' * 64)
    print('\tsaving checkpoints to:', save_path)
    log_file = save_path + '/TRAIN_LOG_{}.csv'.format(args.exp_description)
    print('\tsaving training log to:', save_path + log_file)

    print('\n--- starting training...')

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=5e-4, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(args.N_EPOCH):
        model.train()
        predictions = []
        targets = []

        for step, (input_data, target) in enumerate(train_loader):
            input_data, target = input_data.cuda().float(), target.cuda().float()

            optimizer.zero_grad()
            train_loss, info = model(input_data, target)
            train_loss.backward()
            optimizer.step()

            predictions.extend(info['predictions'])
            targets.extend(info['targets'])

            for key in info['WRITER_KEYS']:
                writer.add_scalar('step_log/' + key, info[key], global_step)

            global_step += 1

        predictions, targets = np.array(predictions), np.array(targets)
        TRAIN_MSE = get_mean_squared_error(predictions, targets)
        TRAIN_DC = get_deterministic_coefficient(predictions, targets)

        predictions, targets = test(model, test_loader)
        TEST_MSE = get_mean_squared_error(predictions, targets)
        TEST_DC = get_deterministic_coefficient(predictions, targets)

        tqdm.tqdm.write('Epoch: [{}/{}], TRAIN_MSE: {:.2f}, TEST_MSE: {:.2f}, TRAIN_DC: {:.2f}%, TEST_DC: {:.2f}%, '
                        .format(epoch + 1, args.N_EPOCH, TRAIN_MSE, TEST_MSE, TRAIN_DC, TEST_DC))
        writer.add_scalars('epoch_log/mean_squared_error', {'train': TRAIN_MSE, 'test': TEST_MSE}, epoch)
        writer.add_scalars('epoch_log/deterministic_coefficient', {'train': TRAIN_DC, 'test': TEST_DC}, epoch)

        with open(log_file, 'a+') as f:
            if epoch == 0:
                print('exp_description,', args.exp_description, file=f)
                print('Epoch,TRAIN_MSE,TEST_MSE,TRAIN_DC,TEST_DC', file=f)
            print('{},{},{},{},{}'.format(epoch,
                                          round(TRAIN_MSE, 2), round(TEST_MSE, 2),
                                          round(TRAIN_DC, 2), round(TEST_DC, 2)), file=f)

        if epoch% 50 == 0 or epoch == args.N_EPOCH:
            torch.save(model.state_dict(), save_path + '/epoch_{}.pt'.format(epoch))
            torch.save(model.state_dict(), save_path + '/last.pt'.format(epoch))
            img_scatter, img_line = plot_result(targets, predictions)
            writer.add_figure("prediction/scatter", img_scatter, epoch)
            writer.add_figure("prediction/line", img_line, epoch)

    return round(TRAIN_MSE, 2), round(TEST_MSE, 2), round(TRAIN_DC, 2), round(TEST_DC, 2)


def main(args):
    writer = SummaryWriter(comment=args.exp_description)
    save_path = writer.get_logdir()
    with open(save_path + '/parameters.txt', 'w') as f:
        for key in vars(args).keys():
            f.write('{} = {}\n'.format(key, vars(args)[key]))

    train_set = CSVDataset(forecast_range=args.forecast_range, dataset=args.dataset, mode='train',
                           train_test_split_ratio=args.train_test_split_ratio, sample_length=args.sample_length)
    test_set = CSVDataset(forecast_range=args.forecast_range, dataset=args.dataset, mode='test',
                          train_test_split_ratio=args.train_test_split_ratio, sample_length=args.sample_length)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    input_size = train_set.get_input_size()

    if args.model == 'RNN':
        model = RNNs(args, cell='RNN', input_size=input_size).cuda()
    elif args.model == 'LSTM':
        model = RNNs(args, cell='LSTM', input_size=input_size).cuda()
    elif args.model == 'GRU':
        model = RNNs(args, cell='GRU', input_size=input_size).cuda()
    elif args.model == 'ANN':
        model = ANN(args, input_size=input_size).cuda()
    elif args.model == 'TCN':
        model = TCN(args, input_size=input_size).cuda()
    else:
        raise RuntimeError('model {} not defined!'.format(args.model))

    for name, param in model.named_parameters():
        if 'weight' in name:
            init.xavier_uniform_(param)

    return train(model, train_loader, test_loader, writer, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flood Forecasting')
    parser.add_argument('--exp_description', default='exp', type=str)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--N_EPOCH', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--train_test_split_ratio', default=0.7, type=float)
    parser.add_argument('--sample_length', default=72, type=int)
    parser.add_argument('--forecast_range', default=6, type=int)  # 预见期
    parser.add_argument('--weight_decay', default=0.008, type=bool)

    parser.add_argument('--model', default='TCN', type=str)  # RNN, LSTM, GRU, ANN, TCN

    # RNN models
    parser.add_argument('--RNN_hidden_size', default=36, type=int)
    parser.add_argument('--RNN_num_layers', default=2, type=int)
    parser.add_argument('--RNN_dropout', default=0.2, type=float)
    # ANN models
    parser.add_argument('--ANN_hidden_size', default=36, type=int)
    parser.add_argument('--ANN_num_layers', default=3, type=int)
    # TCN models
    parser.add_argument('--CNN_hidden_size', default=36, type=int)
    parser.add_argument('--CNN_num_layers', default=3, type=int)

    args = parser.parse_args()

    results = {}
    arg_dict = vars(args)
    for key in arg_dict.keys():
        results[key] = []
    metrics = ['TRAIN_MSE', 'TEST_MSE', 'TRAIN_DC', 'TEST_DC']
    for metric in metrics:
        results[metric] = []

    args.exp_description = '__[EXP]__[{}]'.format(args.model)

    arg_dict = vars(args)
    for key in arg_dict.keys():
        results[key].append(arg_dict[key])

    TRAIN_MSE, TEST_MSE, TRAIN_DC, TEST_DC = main(args)

    results['TRAIN_MSE'].append(TRAIN_MSE)
    results['TEST_MSE'].append(TEST_MSE)
    results['TRAIN_DC'].append(TRAIN_DC)
    results['TEST_DC'].append(TEST_DC)

    print_results(args, results)
