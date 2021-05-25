
import numpy as np

import torch
import os
from tqdm import tqdm
from pathlib import Path

import logging
from pts.core.logging import get_log_path
from pts.model import Predictor

from load_dataset import make_m5_dataset

from pts.evaluation.backtest import make_evaluation_predictions, make_validation_data

#test_start : Start index of the final possible data chunk. For example, for M5 dataset, correct value is 1942
TEST_START = 1942
PREDICTION_LEN = 28

def make_predictions(predictor, ds, num_samples=30, n_iter = 15, start_offset=0, log_path=None, show_progress=True):

    for i in tqdm(range(start_offset, n_iter+start_offset), disable=not show_progress):
        start_this = TEST_START - PREDICTION_LEN * i

        #make prediction
        forecast_it, ts_it = make_evaluation_predictions(
        dataset=make_validation_data(ds, val_start=start_this, val_start_final=TEST_START - PREDICTION_LEN),
        predictor=predictor,
        num_samples=100 if start_this==1942 else num_samples)
        
        forecasts = list(forecast_it)

        #[TODO]
        #is this loop necessary?
        prediction = np.zeros((len(forecasts), PREDICTION_LEN))
        for n in range(len(forecasts)):
            prediction[n] = np.mean(forecasts[n].samples, axis=0)

        # save result
        if log_path is not None:
            np.save(log_path / f'prediction_{start_this}.npy', prediction)

    return prediction   #return last prediction

def run_prediction(args, trial_path, model_idx, ds, predictor):
    cv_log_path = Path(os.path.join(trial_path, 'CV', model_idx))
    cv_log_path.mkdir(parents=True, exist_ok=True)

    # load trained model
    trained_model_path = Path(os.path.join(trial_path, 'trained_model'))
    predictor.prediction_net.load_state_dict(torch.load(trained_model_path / model_idx))

    if args.bs is not None:
        predictor.batch_size = args.bs
    predictor.prediction_net.num_parallel_samples = args.n_par
    
    make_predictions(predictor, ds, num_samples=args.n_samples, log_path=cv_log_path)
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    ###[Important argument]
    parser.add_argument(
        "--data_path",
        default='/data/m5'
    )
    parser.add_argument(
        "--comment",
        type=str,
        default='drop1'
    )
    parser.add_argument(
        "--trial",
        type=str,
        default='t0'
    )

    ###[Important argument]
    parser.add_argument(
        "--bs",
        type=int,
        default=6400
    )
    parser.add_argument(
        "--n_par",
        default=30
    )
    parser.add_argument(
        "--n_samples",
        default=30
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
    
    # load model
    model_name = 'rolled_deepAR'
    trial_path, _, _ = get_log_path(f"m5_submission/{model_name}", log_comment=args.comment, trial=args.trial, mkdir=False)
    print(f"Make predictions for {trial_path}")

    pretrained_model_path = Path(os.path.join(trial_path, 'predictor'))
    if not pretrained_model_path.exists():
        assert False, "Error: Pretrained model not exist!"

    predictor = Predictor.deserialize(pretrained_model_path, device)

    # load data
    test_ds = make_m5_dataset(m5_input_path=args.data_path, exclude_no_sales=False, ds_split=False, prediction_start=1942)

    # generate predictions
    for epoch in range(200,300,10):
        model_idx = f"train_net_{epoch}"
        run_prediction(args, trial_path, model_idx, test_ds, predictor)