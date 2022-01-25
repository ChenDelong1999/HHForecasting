# HHForecasting

HHU 3209 flood forecasting codebase.


## Install
- Create a conda virtual environment and activate it:

    ```bash
    conda create -n HHForecasting python=3.6 -y
    conda activate HHForecasting
    ```

- Install CUDA Toolkit 11.3 (link) and cudnn==8.2.1 (link), then install PyTorch==1.10.1:

    ```bash
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
    # if you prefer other cuda versions, please choose suitable pytorch versions
    # see: https://pytorch.org/get-started/locally/
    ```

- Install other requirements:
    ```bash
    conda install tqdm pandas seaborn matplotlib scikit-learn tensorboard -y
    ```

## Data Preparation

See https://www.yuque.com/bgh8fr/wh55rz/asa9wm

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

- Train deep learning models
  ```bash
  python main.py --dataset TunXi --model TCN
  ```
  - `--dataset`:`'TunXi'` or `'ChangHua'` or `'Toy'`
  - `--model`:  `'TCN'` or `'ANN'` or `'LSTM'` or `'GRU'` or `'RNN'`

  see `main.py` for more details.


- Monitoring training procedure from tensorboard
  
  ```bash
  tensorboard --logdir runs
  ```
  

- Train machin learning models
  ```bash
  python sklearn_baselines.py
  ```