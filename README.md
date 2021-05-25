# Info
This repository contains the deep learning model which achieved [the 3rd position](https://mofc.unic.ac.cy/m5-competition-winners/) in the [M5 Accuracy competition](https://mofc.unic.ac.cy/m5-competition/). This is the same model that submitted the results to the competition (Last model update: 2020-05-31).

This model is developed by [mofl](https://mofl.ai/)



# Prerequisite
## Libraries
     Python 3.7.4
     CUDA 10.1
     CUDNN 7.6.5
     nvidia drivers 435.21
     PyTorch 1.4

## Install
     * Initialize docker setup for the GPU (pytorch/pytorch:1.4-cuda10.1-cudnn7-devel)
     * run sh ./install.sh


# How-to
## Data Setup
     1. make data directory to [DATA_PATH], i.e. /data/m5
     2. copy 4 files (calendar.csv, sales_train_evaluation.csv, sample_submission.csv, sell_prices.csv) to [DATA_PATH]

## Data Preprocessing
The train/predict code will also call this script if it has not already been run on the relevant data. This takes about 10 minutes.

    python ./examples/m5/preprocessing.py --data_path [DATA_PATH]

## Reproduce
a) train 8 models (can be excuted parallelly using multiple gpus). 
One model takes about 6 hour to train

     python examples/m5/training.py --trial t1 --data_path [DATA_PATH]
     python examples/m5/training.py --trial t2 --data_path [DATA_PATH]
     python examples/m5/training.py --trial t3 --data_path [DATA_PATH]
     python examples/m5/training.py --trial t4 --data_path [DATA_PATH]
     python examples/m5/training.py --trial t5 --data_path [DATA_PATH]
     python examples/m5/training.py --trial t6 --data_path [DATA_PATH]
     python examples/m5/training.py --trial t7 --data_path [DATA_PATH]
     python examples/m5/training.py --trial t8 --data_path [DATA_PATH]

b) make 14 predictions of 10 epochs per each trained models (can be excuted parallelly using multiple gpus). 
One model takes about 1 hour to generate predictions. 
If gpu memory is not enough, adjust batch size with --bs option. e.g. --bs 640 (default: 6400)

     python examples/m5/make_predictions.py --trial t1 --data_path [DATA_PATH]
     python examples/m5/make_predictions.py --trial t2 --data_path [DATA_PATH]
     python examples/m5/make_predictions.py --trial t3 --data_path [DATA_PATH]
     python examples/m5/make_predictions.py --trial t4 --data_path [DATA_PATH]
     python examples/m5/make_predictions.py --trial t5 --data_path [DATA_PATH]
     python examples/m5/make_predictions.py --trial t6 --data_path [DATA_PATH]
     python examples/m5/make_predictions.py --trial t7 --data_path [DATA_PATH]
     python examples/m5/make_predictions.py --trial t8 --data_path [DATA_PATH]

c) make final ensemble model (this takes less than 5 minutes). 
Final csv file is ./logs/m5_submission/rolled_deepAR/drop1/submission_final.csv

     python examples/m5/ensemble.py --data_path [DATA_PATH]   
     
# Auxiliary
## Directories & Files
|File                                   |Description                                    | Entry points |
|---|---|:-:|
|./examples/m5/accuracy_evaluator.py    | caulating wrmsse                                  |   |
|./examples/m5/ensemble.py              | making final prediction                           | v |
|./examples/m5/load_dataset.py          | generating features and dataset                   |   |
|./examples/m5/make_predictions.py      | make predictions from saved model                 | v |
|./examples/m5/preprocessing.py         | preprocessing raw data                            | v |
|./examples/m5/training.py              | training model from scratch                       | v |
|./examples/m5/utils.py                 | auxiliary code                                    |   |
|./pts                                  | code for backbone engine                          |   |
|./test                                 | test script for backbone engine                   |   |
     

## Outputs
|File                                                             |Description                        |
|---|---|
|./logs/m5_submission                                             | root path of logs                                 |
|./logs/m5_submission/rolled_deepAR/drop1                         | root path of rolled deepAR model without dropout  |
|./logs/m5_submission/rolled_deepAR/drop1/t1~t8                   | logs of each trials                 |
|./logs/m5_submission/rolled_deepAR/drop1/tx/CV/train_net_xx      | 14 predictions of xx epoch          |
|./logs/m5_submission/rolled_deepAR/drop1/tx/predictor            | saved predictor configurations      |
|./logs/m5_submission/rolled_deepAR/drop1/tx/trained_model        | saved model parameter per each epoch|
|./logs/m5_submission/rolled_deepAR/drop1/tx/events.out...        | event file for tensorboard          |
|./logs/m5_submission/rolled_deepAR/drop1/tx/log.txt              | txt log of training                 |
|./logs/m5_submission/rolled_deepAR/drop1/ensemble.log            | txt log of ensemble                 |
|./logs/m5_submission/rolled_deepAR/drop1/submission_final.csv    | final prediction result             |

     
# Reference

This model is developed based on [PyTorchTS](https://github.com/zalandoresearch/pytorch-ts).

