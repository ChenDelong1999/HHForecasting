# HHForecasting

HHU 3209 flood forecasting codebase.


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

Ask Delong Chen for a copy of data.zip, extract and replace the `/dataset` folder. **PLEASE KEEP IT SECRET !!!**

```bash
$ tree dataset /f

.. HHForecasting\dataset
├───ChangHua
│       data.csv
│
└───TunXi
    │   data.csv
    │
    └───documentation
            ...
```
  
## Experimental repetition (https://www.yuque.com/bgh8fr/wh55rz/aqify1)
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
