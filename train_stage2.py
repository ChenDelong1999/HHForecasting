import argparse
import random
import tqdm
import numpy as np
import torch
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from utils import get_deterministic_coefficient, get_mean_squared_error, print_results, \
    freeze, plot_result, init_results, print_args
from adversarial_domain_adaptation_utils import GANDataset, Discriminator_1DCNN, calc_gradient_penalty_ST, plot_feature_tsne, plot_backbone_features
from model.Raindrop_encoder import Raindrop_encoder

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)


def evaluate(G, D, test_model, test_loader, TX_features, writer, epoch, total_step, tsne=False):
    G.eval()
    D.eval()
    test_model.eval()

    ChangHua_features = None
    predictions = []
    targets = []
    W_dis_all = []

    for step, (raindrop, runoff_history, runoff, feature) in enumerate(test_loader):
        raindrop, runoff_history, runoff, feature = \
            raindrop.cuda().float(), runoff_history.cuda().float(), runoff.cuda().float(), feature.cuda().float()

        ChangHua_feature = G(raindrop.transpose(1,2)).transpose(1,2)

        real_output_D = D(feature)
        fake_output_D = D(ChangHua_feature.detach())

        prediction = test_model.inference(raindrop, runoff_history)
        predictions.extend(prediction.flatten().detach().cpu().numpy().tolist())
        targets.extend(runoff.flatten().detach().cpu().tolist())
        W_dis_all.append((real_output_D - fake_output_D).detach().cpu().numpy().mean())

        if ChangHua_features is None:
            ChangHua_features = ChangHua_feature.detach().cpu().numpy()
        else:
            ChangHua_features = np.concatenate([ChangHua_features, ChangHua_feature.detach().cpu().numpy()], axis=0)

    # feature visualization
    fig = plot_backbone_features(ChangHua_feature.detach().cpu().numpy(), feature.cpu().numpy())
    writer.add_image("feature visualization", fig, epoch, dataformats='HWC')

    # tsne visualization，选取5w个屯溪特征和所有的昌化特征可视化
    if tsne is True:
        start = random.randint(0, int(TX_features.shape[0] / 10))
        ChangHua_features = ChangHua_features.reshape((ChangHua_features.shape[0] * ChangHua_features.shape[1], ChangHua_features.shape[2]))
        for i in range(5):
            if i == 0:
                features = TX_features[start:start + 10000]
            else:
                features = np.concatenate([features, TX_features[start:start + 10000]], axis=0)
            start += 100000
        fig = plot_feature_tsne(ChangHua_features, features)
        writer.add_image("feature TSNE", fig, epoch, dataformats='HWC')

    # test result visualization
    predictions, targets = np.array(predictions), np.array(targets)
    ideal_w = np.average(targets) / np.average(predictions)
    MSE = get_mean_squared_error(predictions, targets)
    DC = get_deterministic_coefficient(predictions, targets)

    ideal_MSE = get_mean_squared_error(predictions*ideal_w, targets)
    ideal_DC = get_deterministic_coefficient(predictions*ideal_w, targets)
    writer.add_scalars('epoch_log/mean_squared_error', {'direct_test': MSE}, epoch)
    writer.add_scalars('epoch_log/deterministic_coefficient', {'direct_test': DC}, epoch)
    writer.add_scalars('epoch_log/mean_squared_error', {'ideal': ideal_MSE}, epoch)
    writer.add_scalars('epoch_log/deterministic_coefficient', {'ideal': ideal_DC}, epoch)
    writer.add_scalar('epoch_log/ideal w', ideal_w, epoch)

    img_scatter, img_line = plot_result(targets, predictions)
    writer.add_figure("prediction/scatter", img_scatter, epoch)
    writer.add_figure("prediction/line", img_line, epoch)

    writer.add_scalars('step_log/W_distance',{'test': np.array(W_dis_all).mean()}, total_step)

    return MSE, DC, ideal_MSE, ideal_DC


def train(G, test_model, train_loader, test_loader, TunXi_features, writer, save_path):
    print_args(args)

    print('\tsaving checkpoints to:', save_path)
    log_file = save_path + '/TRAIN_LOG_{}.csv'.format(args.exp_description)
    print('\tsaving training log to:', save_path + log_file)

    print('\n--- starting training...')

    D = Discriminator_1DCNN().cuda()
    optimizer_D = torch.optim.RMSprop(D.parameters(), lr=args.lr)
    optimizer_G = torch.optim.RMSprop(G.parameters(), lr=args.lr)

    scheduler_D = lr_scheduler.CosineAnnealingLR(optimizer_D, args.N_EPOCH)
    scheduler_G = lr_scheduler.CosineAnnealingLR(optimizer_G, args.N_EPOCH)

    total_step = 0
    for epoch in range(args.N_EPOCH):
        G.train()
        D.train()

        for step, (raindrop, runoff_history, runoff, TunXi_feature) in enumerate(train_loader):
            raindrop, runoff, TunXi_feature = raindrop.cuda().float(), runoff.cuda().float(), TunXi_feature.cuda().float()

            optimizer_G.zero_grad()
            prediction = G(raindrop.transpose(1, 2)).transpose(1, 2)

            # ------------------------ #
            #    train Discriminator   #
            # ------------------------ #
            for critic_i in range(args.CRITIC_ITERS):
                optimizer_D.zero_grad()
                real_output_D = D(TunXi_feature)
                fake_output_D = D(prediction.detach())

                Loss_D_real = -torch.mean(real_output_D)
                Loss_D_fake = torch.mean(fake_output_D)
                gradient_penalty_Dr = calc_gradient_penalty_ST(D, TunXi_feature.data, prediction.data, term=['real_fake'])

                Loss_D = Loss_D_real + Loss_D_fake + args.w_gp * gradient_penalty_Dr
                Loss_D.backward()
                optimizer_D.step()

            # ----------------------- #
            #     train Generator     #
            # ----------------------- #
            optimizer_G.zero_grad()
            Loss_adv = -torch.mean(D(prediction))

            Loss_G = Loss_adv
            Loss_G.backward()
            optimizer_G.step()

            ###############################################
            #                    Logging                  #
            ###############################################

            W_dis = torch.mean(real_output_D).item() - torch.mean(fake_output_D).item()
            writer.add_scalars('step_log/W_distance', {'train': W_dis}, total_step)
            total_step += 1
        scheduler_D.step()
        scheduler_G.step()

        # Test
        test_model.backbone.load_state_dict(G.state_dict())
        TEST_MSE, TEST_DC, ideal_MSE, ideal_DC = evaluate(G, D, test_model, test_loader, TunXi_features, writer, epoch, total_step, tsne=False)
        tqdm.tqdm.write('Epoch: [{}/{}],  TEST_MSE: {:.5f}, TEST_DC: {:.2f}%, '.format(epoch + 1, args.N_EPOCH, TEST_MSE, TEST_DC))

        writer.add_scalar('epoch_log/learning rate', scheduler_D.get_last_lr()[-1], epoch)

        with open(log_file, 'a+') as f:
            if epoch == 0:
                print('exp_description,', args.exp_description, file=f)
                print('Epoch,TEST_MSE,TEST_DC', file=f)
            print('{},{},{}'.format(epoch, round(TEST_MSE, 2), round(TEST_DC, 2)), file=f)

        if epoch % 50 == 0 or epoch == args.N_EPOCH - 1:
            torch.save(G.state_dict(), save_path + '/epoch_{}.pt'.format(epoch))
            torch.save(G.state_dict(), save_path + '/last.pt'.format(epoch))

    torch.save(G.state_dict(), save_path + '/last.pt'.format(epoch))

    return round(TEST_MSE, 2), round(TEST_DC, 2)


def main(args):
    writer = SummaryWriter(comment=args.exp_description)
    save_path = writer.get_logdir()

    train_set = GANDataset(forecast_range=args.forecast_range, mode='train',
                           train_test_split_ratio=args.train_test_split_ratio,
                           sample_length=args.sample_length, training_set_scale=args.training_set_scale)
    test_set = GANDataset(forecast_range=args.forecast_range, mode='test',
                          train_test_split_ratio=args.train_test_split_ratio,
                          sample_length=args.sample_length, training_set_scale=args.training_set_scale)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    input_size = train_set.get_input_size()
    TunXI_features = test_set.get_TunXi_feature()

    backbone = Raindrop_encoder(args.backbone, args.backbone_hidden_size, args.backbone_num_layers,
                                args.dropout, input_size).backbone.cuda()

    for name, param in backbone.named_parameters():
        if 'weight' in name:
            init.xavier_uniform_(param)

    # test_model是将昌化编码器和屯溪prediction head结合的TCN模型
    if args.pre_structure == 'residual':
        test_model = Raindrop_encoder(args.backbone, args.backbone_hidden_size, args.backbone_num_layers,
                                      args.dropout, input_size, args.pre_head, args.pre_head_hidden_size,
                                      args.pre_head_num_layers, residual=True).cuda()
    else:
        test_model = Raindrop_encoder(args.backbone, args.backbone_hidden_size, args.backbone_num_layers,
                                      args.dropout, input_size, args.pre_head, args.pre_head_hidden_size,
                                      args.pre_head_num_layers, residual=False).cuda()

    TunXi_input_size = 11
    pretrained_model = Raindrop_encoder(args.pre_backbone, args.pre_backbone_hidden_size, args.pre_backbone_num_layers,
                                        args.pre_dropout, TunXi_input_size,
                                        args.pre_head, args.pre_head_hidden_size, args.pre_head_num_layers).cuda()
    pretrained_model.load_state_dict(torch.load(args.pretrained_weights))
    test_model.prediction_head.load_state_dict(pretrained_model.prediction_head.state_dict())
    freeze(test_model.prediction_head)
    return train(backbone, test_model, train_loader, test_loader, TunXI_features, writer, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flood Forecasting')
    parser.add_argument('--exp_description', default='stage2', type=str)
    parser.add_argument('--N_EPOCH', default=100, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--training_set_scale', default=1.0, type=float)
    parser.add_argument('--train_test_split_ratio', default=0.7, type=float)
    parser.add_argument('--sample_length', default=72, type=int)
    parser.add_argument('--CRITIC_ITERS', default=5, type=int)
    parser.add_argument('--forecast_range', default=6, type=int)
    parser.add_argument('--w_gp', default=10, help='weight for gradient penalty')

    # target model argument
    parser.add_argument('--backbone', default='TCN', type=str)  # RNN, LSTM, GRU, ANN, STGCN, TCN
    parser.add_argument('--backbone_hidden_size', default=36, type=int)
    parser.add_argument('--backbone_num_layers', default=3, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)

    # source model argument
    parser.add_argument('--pre_structure', default='residual', type=str)  # direct, residual
    parser.add_argument('--pretrained_weights', type=str, required=True)
    parser.add_argument('--pre_backbone', default='TCN', type=str)  # RNN, LSTM, GRU, ANN, STGCN, TCN
    parser.add_argument('--pre_backbone_hidden_size', default=36, type=int)
    parser.add_argument('--pre_backbone_num_layers', default=3, type=int)
    parser.add_argument('--pre_dropout', default=0.2, type=float)
    parser.add_argument('--pre_head', default='conv1d', type=str)  # conv1d, linear
    parser.add_argument('--pre_head_hidden_size', default=36, type=int)
    parser.add_argument('--pre_head_num_layers', default=3, type=int)

    args = parser.parse_args()
    results = init_results(args,stage=2)

    TEST_MSE, TEST_DC = main(args)

    results['TEST_MSE'].append(TEST_MSE)
    results['TEST_DC'].append(TEST_DC)

    print_results(results)
