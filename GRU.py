import datetime
from nbformat import write

import numpy as np
from tensorboard import summary
from zmq import device
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

from model.BiGRU import BiGRU as GRU
import os
import json
from utils import get_Nash_efficiency_coefficient, get_Kling_Gupta_efficiency, print_results

import glob
import gzip
import pickle
import pandas as pd
import torch.utils.data as uitilsData
from sklearn.preprocessing import MinMaxScaler
import numpy as np

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

class BenchMarkDataset(uitilsData.Dataset):
    """
    Prepare the dataset or a gauge for the models
    
    Parameters:
        path, str:
            Path to dataset which contains train and test set for each gauge as csv files
        sensorID, int:
            id of prepared gauge
        split, str:
            identify whether train or test set will be prepared
        gpu, int:
            id of used gpu
        scaler, obj:
            minmax scaler which is created based on train set
    
    """

    def __init__(self, path, sensorID, split="train", gpu=0, scaler=(None, None)):

        self.path = path
        self.split = split
        self.gpu = gpu
        self.sensorID = sensorID

        self.scalerX = scaler[0]
        self.scalerY = scaler[1]
        
        self.read()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        X = self.X[idx,:,:].to(device)   #.cuda()
        y = self.y[idx,:].to(device)
        
        return X, y
    
    def get_values(self):
        
        return (self.scalerX, self.scalerY)

    def read_file(self, file_path):
        X_path = file_path + "_x.csv"
        y_path = file_path + "_y.csv"
        X = pd.read_csv(X_path, index_col='datetime')
        y = pd.read_csv(y_path, index_col='datetime')

        X = X.values[:,:-7]
        y = y.values
        x_train_history = X[:,:72*3].reshape(-1, 72, 3)
        x_train_future = X[:,72*3:].reshape(-1, 120, 2)
        x_train_future = x_train_future[:,:,[0,1,1]]
        x_train_future[:,:,2] = 0
        ds_X = np.concatenate([x_train_history,x_train_future],axis=1)

        return ds_X, y

    def read(self):

        ds_X = np.random.rand(1, 192, 3)
        ds_y = np.random.rand(1, 120)

        
        file_path = self.path + str(self.sensorID) + "_" + self.split
        X, y = self.read_file(file_path)
        ds_X = np.concatenate((ds_X, X), 0) 
        ds_y = np.concatenate((ds_y, y), 0)
        
        ds_X = np.delete(ds_X, 0, 0)
        ds_y = np.delete(ds_y, 0, 0)

        shapeX = ds_X.shape
        
        
        if self.split == "train":

            self.scalerX = MinMaxScaler()
            self.scalerY = MinMaxScaler()
            
            self.scalerX.fit(ds_X.reshape((shapeX[0], shapeX[1] * shapeX[2])))
            self.scalerY.fit(ds_y)

        
        ds_X = self.scalerX.transform(ds_X.reshape((shapeX[0], shapeX[1] * shapeX[2])))
        ds_X = ds_X.reshape(shapeX)
        ds_y = self.scalerY.transform(ds_y)

        ds_X = torch.Tensor(ds_X)
        ds_y = torch.Tensor(ds_y)
        self.X = ds_X#.transpose(1,2)
        self.y = ds_y.unsqueeze(2)
        


def createSummaryFile(filename):
    """
    Create a summary file for the results of each gauge
        
    Parameters:
        filename, str:
            filename(path) of the summary file
    
    """
    
    Results = {}
    Results["Train"] = {}
    Results["Test"] = {}
    Results["Train"]["NSE"] = {}
    Results["Train"]["KGE"] = {}
    Results["Test"]["NSE"] = {}
    Results["Test"]["KGE"] = {}
    Results["Train"]["NSE"]["max"] = {}
    Results["Train"]["NSE"]["min"] = {}
    Results["Train"]["NSE"]["median"] = {}
    Results["Train"]["NSE"]["mean"] = {}
    Results["Train"]["KGE"]["max"] = {}
    Results["Train"]["KGE"]["min"] = {}
    Results["Train"]["KGE"]["median"] = {}
    Results["Train"]["KGE"]["mean"] = {}
    Results["Test"]["NSE"]["max"] = {}
    Results["Test"]["NSE"]["min"] = {}
    Results["Test"]["NSE"]["median"] = {}
    Results["Test"]["NSE"]["mean"] = {}
    Results["Test"]["KGE"]["max"] = {}
    Results["Test"]["KGE"]["min"] = {}
    Results["Test"]["KGE"]["median"] = {}
    Results["Test"]["KGE"]["mean"] = {}
    
    with open(filename, "w") as outfile:
        json.dump(Results, outfile)

def updateJSON(station_id, NSE_train, NSE_test, KGE_train, KGE_test, jsonFilePath):
    """
    Updates the summary file with results of a gauge
    
    Parameters:
        station_id, int:
            id of the gauge which results will be added to summary file
        NSE_train, list:
            NSE scores of train set
        NSE_test, list:
            NSE scores of test set
        KGE_train, list:
            KGE scores of train set
        KGE_test, list:
            KGE scores of test set
        jsonFilePath, str:
            path of the updated summary file
    
    """

    nse_train_max = np.max(NSE_train)
    nse_train_min = np.min(NSE_train)
    nse_train_median = np.median(NSE_train)
    nse_train_mean = np.mean(NSE_train)
    
    nse_test_max = np.max(NSE_test)
    nse_test_min = np.min(NSE_test)
    nse_test_median = np.median(NSE_test)
    nse_test_mean = np.mean(NSE_test)
    
    kge_train_max = np.max(KGE_train)
    kge_train_min = np.min(KGE_train)
    kge_train_median = np.median(KGE_train)
    kge_train_mean = np.mean(KGE_train)
    
    kge_test_max = np.max(KGE_test)
    kge_test_min = np.min(KGE_test)
    kge_test_median = np.median(KGE_test)
    kge_test_mean = np.mean(KGE_test)
    
    with open(jsonFilePath, "r") as jsonFile:
        Results = json.load(jsonFile)
    
    Results["Train"]["NSE"]["max"][station_id] = nse_train_max
    Results["Train"]["NSE"]["min"][station_id] = nse_train_min
    Results["Train"]["NSE"]["median"][station_id] = nse_train_median
    Results["Train"]["NSE"]["mean"][station_id] = nse_train_mean
    
    Results["Train"]["KGE"]["max"][station_id] = kge_train_max
    Results["Train"]["KGE"]["min"][station_id] = kge_train_min
    Results["Train"]["KGE"]["median"][station_id] = kge_train_median
    Results["Train"]["KGE"]["mean"][station_id] = kge_train_mean
    
    Results["Test"]["NSE"]["max"][station_id] = nse_test_max
    Results["Test"]["NSE"]["min"][station_id] = nse_test_min
    Results["Test"]["NSE"]["median"][station_id] = nse_test_median
    Results["Test"]["NSE"]["mean"][station_id] = nse_test_mean
    
    Results["Test"]["KGE"]["max"][station_id] = kge_test_max
    Results["Test"]["KGE"]["min"][station_id] = kge_test_min
    Results["Test"]["KGE"]["median"][station_id] = kge_test_median
    Results["Test"]["KGE"]["mean"][station_id] = kge_test_mean
    
    with open(jsonFilePath, "w") as jsonFile:
        json.dump(Results, jsonFile)
    
    

def GRU_main(sensorID, datasetPath, resultsPath, summaryFilePath, gpu=2):
    """
    Main function to get results of GRU model. It should be noted that
    batch size, learning rate or number of epochs can be change with updating
    corresponding values.
    
    
    Parameters:
        summaryFilePath, str:
            filename(path) of the summary file
        datasetPath, str:
            Path to dataset which contains train and test set for each gauge as csv files
        resultsPath, str:
            Path to save results of each gauge
        gpu, int:
            id of used gpu
    
    """
    
    sensorDirectory = resultsPath + "%d" % (sensorID)
    os.mkdir(sensorDirectory)
    summaryFile = summaryFilePath
    
    DATASET_PATH = datasetPath
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    LR = 1e-4
    EPOCHS = 50
    
    
    
    train_ = BenchMarkDataset(DATASET_PATH, sensorID, split='train', gpu=gpu)
    trainloader = torch.utils.data.DataLoader(train_, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_ = BenchMarkDataset(DATASET_PATH, sensorID, split='test', gpu=gpu, scaler=train_.get_values())
    testloader = torch.utils.data.DataLoader(test_, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    INPUT_DIM = 3
    HIDDEN_DIM = 64
    OUTPUT_DIM = 1
    NUM_LAYERS = 2
    DROPOUT = 0.2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    netGRU = GRU(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT)
    netGRU.to(device)
    
    optimizer = optim.Adam(netGRU.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    L1Loss = torch.nn.MSELoss()
    
    
    valid_model_path = sensorDirectory + "/valid.pth"
    
    val_loss_best = np.inf


    hist_loss = np.zeros(EPOCHS)
    hist_loss_val = np.zeros(EPOCHS)
    
    print('\n--- starting training...')
    for idx_epoch in range(EPOCHS):
        print('Epoch: [{}/{}]'.format(idx_epoch + 1, EPOCHS))
        running_loss = 0
        for idx_batch, (x, y) in enumerate(trainloader, 1):
            optimizer.zero_grad()
            y_hat = netGRU(x)
            minibatch_loss = L1Loss(y, y_hat[:, -120:])
            minibatch_loss.backward()
            optimizer.step()
            running_loss += minibatch_loss.item()

        train_loss = running_loss/len(trainloader)
        running_loss = 0
        with torch.no_grad():
            for x, y in testloader:
                y_hat = netGRU(x)
                running_loss += L1Loss(y, y_hat[:, -120:])

        val_loss = (running_loss / len(testloader))
        scheduler.step(train_loss)
        val_loss = val_loss.item()

        hist_loss[idx_epoch] = train_loss
        hist_loss_val[idx_epoch] = val_loss

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(netGRU.state_dict(), valid_model_path)
        
        
        # tqdm.tqdm.write('Epoch: [{}/{}]'
        #                 .format(idx_epoch + 1, EPOCHS))
    
    netGRU.load_state_dict(torch.load(valid_model_path))
    with torch.no_grad():
        for k, (x, y) in enumerate(trainloader, 1):
            output = netGRU(x)
            if k == 1:
                result_pred_train = output
                result_train = y
            else:
                result_pred_train = torch.cat((result_pred_train, output), 0)
                result_train = torch.cat((result_train, y), 0)

    with torch.no_grad():
        for k, (x, y) in enumerate(testloader, 1):
            output = netGRU(x)
            if k == 1:
                result_pred_test = output
                result_test = y
            else:
                result_pred_test = torch.cat((result_pred_test, output), 0)
                result_test = torch.cat((result_test, y), 0)
    
    shapeOutputTest = result_test.shape
    shapeOutputTrain = result_train.shape
    
    result_pred_train = train_.scalerY.inverse_transform(result_pred_train.cpu().numpy()[:, -120:].reshape((shapeOutputTrain[0], 120)))
    result_train = train_.scalerY.inverse_transform(result_train.cpu().numpy().reshape((shapeOutputTrain[0], 120)))
    
    result_pred_test = train_.scalerY.inverse_transform(result_pred_test.cpu().numpy()[:, -120:].reshape((shapeOutputTest[0], 120)))
    result_test = train_.scalerY.inverse_transform(result_test.cpu().numpy().reshape((shapeOutputTest[0], 120)))
    
    NSEs_train = []
    NSEs_test = []
    KGEs_train = []
    KGEs_test = []
    
    for i in range(120):
        TRAIN_NSE = get_Nash_efficiency_coefficient( result_pred_train[:, i], result_train[:, i])
        TEST_NSE =  get_Nash_efficiency_coefficient(result_pred_test[:, i], result_test[:, i])
        TRAIN_KGE = get_Kling_Gupta_efficiency( result_pred_train[:, i], result_train[:, i])
        TEST_KGE = get_Kling_Gupta_efficiency(result_pred_test[:, i], result_test[:, i])
        NSEs_train.append(TRAIN_NSE)
        NSEs_test.append(TEST_NSE)
        KGEs_train.append(TRAIN_KGE)
        KGEs_test.append(TEST_KGE)
        print('RANGE: {}, TRAIN_NSE: {:.3f}, TEST_NSE: {:.3f}, TRAIN_KGE: {:.3f}, TEST_KGE: {:.3f} '
                        .format(i, TRAIN_NSE, TEST_NSE, TRAIN_KGE, TEST_KGE))
        # tqdm.tqdm.write('RANGE: {}, TRAIN_NSE: {:.3f}, TEST_NSE: {:.3f}, TRAIN_KGE: {:.3f}, TEST_KGE: {:.3f} '
        #                 .format(i, TRAIN_NSE, TEST_NSE, TRAIN_KGE, TEST_KGE))
    
    updateJSON(sensorID, NSEs_train, NSEs_test, KGEs_train, KGEs_test, summaryFile)
    
    KGE_train = pd.DataFrame(KGEs_train)
    KGE_test = pd.DataFrame(KGEs_test)
    KGE_train.columns = ["KGEsTrain"]
    KGE_test.columns = ["KGEsTest"]
    NSE_train = pd.DataFrame(NSEs_train)
    NSE_test = pd.DataFrame(NSEs_test)
    NSE_train.columns = ["NSEsTrain"]
    NSE_test.columns = ["NSEsTest"]
    
    
    combined = pd.concat([NSE_train, NSE_test, KGE_train, KGE_test], axis=1)
    combined.to_csv("%s/%s.csv" % (sensorDirectory, str(sensorID)), index=True)
    

if __name__ == '__main__':
    summaryFilePath = "summary"
    if os.path.exists(summaryFilePath):
        os.remove(summaryFilePath)
    createSummaryFile(summaryFilePath)
    datasetPath = "dataset/data_ready/" # unzip the files in originalData folder and gave as dataset path
    resultsPath = "result/"
    os.mkdir(resultsPath)
    l = os.listdir(datasetPath)
    sensors = [i.split("_")[0] for i in l]
    sensors = list(set(sensors))
    for sensor in sensors:
        GRU_main(int(sensor), datasetPath, resultsPath, summaryFilePath)


