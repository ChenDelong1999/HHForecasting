# HH💦Forecasting

Welcome to Hohai University (河海大学) [多模态人工智能实验室 (Artificial Intelligence of Multi-modality Group, AIM Group)](https://multimodality.group/) time-series forecasting codebase! 

This codebase is under active development. If you find any bugs or have any suggestions for code improvement, please raise an issue, thanks🎈


## Installation

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

See [YuQue Doc](https://www.yuque.com/bgh8fr/wh55rz/asa9wm) for data descriptions.

| Dataset        | Link                                                         |
| -------------- | ------------------------------------------------------------ |
| 屯溪、昌化 (Tunxi, Changhua) | [BaiduPan](https://pan.baidu.com/s/1Pp9Lm9fYs7su8K34SnTv2w ) (access code: private*)|
| [WaterBench](https://eartharxiv.org/repository/view/2988/)                      | [BaiduPan](https://pan.baidu.com/s/1Q_uiDNwLipFS50D-8I_YiQ) (access code: 03l0) |

> *Currently we do not plan to make these two datasets to be public. If you are a member of Prof. Fan Liu's lab, contact Prof. Liu (fanliu@hhu.edu.cn) or Delong Chen (chendelong@hhu.edu.cn) for the access code.

Download the dataset and put it to the `/dataset` folder as follows:

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
    ├── ...
    ├── 668_data.csv
    └── 671_data.csv
```

## Flood Forecasting

### Machine learning models:

```bash
python sklearn_baselines.py
```
  
### Deep learning models

```bash
python train_stage1.py --dataset ChangHua --structure residual --backbone TCN --head conv1d
```
- `--structure`: `'residual'` or `'direct'` or `'joint'`


### Few-shot learning

```bash
python train_stage1.py --dataset ChangHua --structure residual --backbone TCN --head conv1d --few_shot_num 20 --batch_size 16 --N_EPOCH 1000
  ```
'few_shot_num' denotes the number of experiments on each training set scale. 


## FloodDAN Re-implementation

See our FloodDAN paper below and [YuQue Doc](https://www.yuque.com/bgh8fr/wh55rz/sw64fp) for details of this implementation.

> [Delong Chen](https://chendelong.world/), [Ruizhi Zhou](https://www.researchgate.net/scientific-contributions/Ruizhi-Zhou-2223957483), [Yanling Pan](https://www.linkedin.com/in/yanling-pan-2399821a1/?originalSubdomain=cn), [Fan Liu](https://cies.hhu.edu.cn/2013/0508/c4122a54931/page.htm): [**A Simple Baseline for Adversarial Domain Adaptation-based Unsupervised Flood Forecasting**](https://arxiv.org/abs/2206.08105). _Technical Report, ArXiv, CoRR abs/2206.08105 (2022)_.
  
### **Stage 1**, pretraining with TunXi dataset

  ```bash
  python train_stage1.py --dataset TunXi --structure residual --backbone TCN --head conv1d
   ```

- `--structure`: `'residual'` or `'direct'`
- `--backbone`: `'TCN'` or `'ANN'` or `'LSTM'` or `'GRU'` or `'RNN'` or `'STGCN'`
- `--head`: `'linear'` or `'conv1d'`


###  **Stage 2,** Adversarial domain adaptation

  ```bash
  python train_stage2.py --backbone TCN --pre_structure residual --pre_backbone TCN --pre_head conv1d --pretrained_weights runs/<your pretraining run log dir>/last.pt
   ```

- `--backbone`: `'TCN'` or `'ANN'` or `'LSTM'` or `'GRU'` or `'RNN'` or `'STGCN'`
- `--pre_structure`: `'residual'` or `'direct'`
- `--pre_backbone`: `'TCN'` or `'ANN'` or `'LSTM'` or `'GRU'` or `'RNN'` or `'STGCN'`
- `--pre_head`: `'linear'` or `'conv1d'`
- `--pretrained_weights`:  runs/<your stage 1 run log dir>/last.pt


Monitoring training procedure from tensorboard:
  
  ```bash
  tensorboard --logdir runs
  ```


## Papers

- [Delong Chen](https://chendelong.world/), [Ruizhi Zhou](https://www.researchgate.net/scientific-contributions/Ruizhi-Zhou-2223957483), [Yanling Pan](https://www.linkedin.com/in/yanling-pan-2399821a1/?originalSubdomain=cn), [Fan Liu](https://cies.hhu.edu.cn/2013/0508/c4122a54931/page.htm): [**A Simple Baseline for Adversarial Domain Adaptation-based Unsupervised Flood Forecasting**](https://arxiv.org/abs/2206.08105). _Technical Report, ArXiv, CoRR abs/2206.08105 (2022)_.

- [Delong Chen](https://chendelong.world/), [Fan Liu](https://cies.hhu.edu.cn/2013/0508/c4122a54931/page.htm), Zheqi Zhang, Xiaomin Lu, [Zewen Li](https://zewenli.cn/): [**Significant Wave Height Prediction based on Wavelet Graph Neural Network**](https://arxiv.org/abs/2107.09483). _2021 IEEE 4th International Conference on Big Data and Artificial Intelligence (BDAI)_.

- [Fan Liu](https://cies.hhu.edu.cn/2013/0508/c4122a54931/page.htm), Xiaomin Lu, Dan Xu, [Wenwen Dai](https://www.researchgate.net/profile/Dai-Wenwen), Huizhou Li: [**Research progress of ocean waves forecasting method**](https://jour.hhu.edu.cn/hhdxxbzren/article/abstract/202105001). _Journal of Hohai University (Natural Sciences)_.

- [Fan Liu](https://cies.hhu.edu.cn/2013/0508/c4122a54931/page.htm), [Feng Xu](https://cies.hhu.edu.cn/2013/0507/c4122a54830/page.psp), [Sai Yang](https://dqxy.ntu.edu.cn/2019/0904/c1290a48382/page.htm): [**A Flood Forecasting Model Based on Deep Learning Algorithm via Integrating Stacked Autoencoders with BP Neural Network**](https://ieeexplore.ieee.org/document/7966716). _2017 IEEE Third International Conference on Multimedia Big Data (BigMM)_.
