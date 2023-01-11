# 🎉HHForecasting

HHU 3209 flood forecasting codebase.
**paper：**[FloodDAN: Unsupervised Flood Forecasting based on Adversarial Domain Adaptation | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/9862723)


## Install

- Clone this repo:

    ```bash
    git clone https://github.com/ChenDelong1999/HHForecasting.git
    cd HHForecasting
    ```
  
- Create a conda virtual environment and activate it:

    ```bash
    conda create -n HHForecasting python=3.6 -y
    conda activate HHForecasting
    ```

- Install PyTorch [(official website)](https://pytorch.org/get-started/locally/).

- Install other requirements:
    ```bash
    conda install tqdm pandas seaborn matplotlib scikit-learn tensorboard -y
    ```

## Data Preparation

See https://www.yuque.com/bgh8fr/wh55rz/asa9wm for data descriptions.

| DATASET        | LINK                                                         |
| -------------- | ------------------------------------------------------------ |
| ChangHua&TunXi | https://pan.baidu.com/s/1Pp9Lm9fYs7su8K34SnTv2w <br/>access code：private|
| WaterBench     | https://pan.baidu.com/s/1Q_uiDNwLipFS50D-8I_YiQ <br/>access code：03l0 |



```bash
$ tree dataset /f

.. HHForecasting\dataset
├── ChangHua
│   └── data.csv
│
├── TunXi
│   ├── data.csv
│   └── documentation
│       
└── WaterBench
    ├── 1609_data.csv
    ├── 521_data.csv
    ├── 536_data.csv
    ├── 539_data.csv
    ├── 552_data.csv
    ├── 557_data.csv
    ├── 562_data.csv
    ├── 563_data.csv
    ├── 566_data.csv
    ├── 569_data.csv
    ├── 587_data.csv
    ├── 588_data.csv
    ├── 608_data.csv
    ├── 611_data.csv
    ├── 613_data.csv
    ├── 624_data.csv
    ├── 626_data.csv
    ├── 637_data.csv
    ├── 638_data.csv
    ├── 649_data.csv
    ├── 657_data.csv
    ├── 660_data.csv
    ├── 663_data.csv
    ├── 668_data.csv
    └── 671_data.csv
```



## Experimental repetition (https://www.yuque.com/bgh8fr/wh55rz/sw64fp)

- Fully supervised (Deep learning):
  ```bash
  python train_stage1.py --dataset ChangHua --structure residual --backbone TCN --head conv1d
  ```
  - `--structure`:`'residual'` or `'direct'` or `'joint'`

- Fully supervised (Machine learning):
  ```bash
  python sklearn_baselines.py
  ```

- Few-shot supervised:
  ```bash
  python train_stage1.py --dataset ChangHua --structure residual --backbone TCN --head conv1d --few_shot_num 20 --batch_size 16 --N_EPOCH 1000
  ```
  'few_shot_num' denotes the number of experiments on each training set scale. 
  
- Unsupervised learning:
  Stage 1, pretraining with TunXi dataset
  ```bash
  python train_stage1.py --dataset TunXi --structure residual --backbone TCN --head conv1d
  ```
  - `--structure`:`'residual'` or `'direct'`
  - `--backbone`:  `'TCN'` or `'ANN'` or `'LSTM'` or `'GRU'` or `'RNN'` or `'STGCN'`
  - `--head`:  `'linear'` or `'conv1d'`

  Stage 2, Adversarial domain adaptation
  ```bash
  python train_stage2.py --backbone TCN --pre_structure residual --pre_backbone TCN --pre_head conv1d --pretrained_weights runs/<your pretraining run log dir>/last.pt
  ```
  - `--backbone`:  `'TCN'` or `'ANN'` or `'LSTM'` or `'GRU'` or `'RNN'` or `'STGCN'`
  - `--pre_structure`:`'residual'` or `'direct'`
  - `--pre_backbone`:  `'TCN'` or `'ANN'` or `'LSTM'` or `'GRU'` or `'RNN'` or `'STGCN'`
  - `--pre_head`:  `'linear'` or `'conv1d'`
  - `--pretrained_weights`:  runs/<your stage 1 run log dir>/last.pt

- Monitoring training procedure from tensorboard:
  
  ```bash
  tensorboard --logdir runs
  ```

