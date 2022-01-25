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
├───Toy
│       data.csv
│
└───TunXi
    │   data.csv
    │
    └───documentation
            ...
```

## Training

- Train deep learning models:
  ```bash
  python main.py --dataset TunXi --model TCN
  ```
  - `--dataset`:`'TunXi'` or `'ChangHua'` or `'Toy'`
  - `--model`:  `'TCN'` or `'ANN'` or `'LSTM'` or `'GRU'` or `'RNN'`



- Monitoring training procedure from tensorboard:
  
  ```bash
  tensorboard --logdir runs
  ```
  

- Train machine learning models:
  ```bash
  python sklearn_baselines.py
  ```