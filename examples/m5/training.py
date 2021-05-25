import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
import logging
import os

from pts.model import Predictor
from pts.model.deepar import DeepAREstimator

from pts.modules import TweedieOutput
from pts.trainer import Trainer
from pts.core.logging import get_log_path, set_logger

from load_dataset import make_m5_dataset

from pts.feature.time_feature import *

logger = logging.getLogger("mofl").getChild("training")


prediction_length = 28

def get_rolled_deepAR_estimator(stat_cat_cardinalities, device, log_path):
    batch_size = 64
    num_batches_per_epoch = 30490 // batch_size + 1    

    return DeepAREstimator(
        input_size=102,        
        num_cells=120,
        prediction_length=prediction_length,
        dropout_rate=0.1,
        freq="D",
        time_features=[DayOfWeek(), DayOfMonth(), MonthOfYear(), WeekOfYear(), Year()],
        distr_output = TweedieOutput(1.2),
        lags_seq=[1], 
        moving_avg_windows=[7, 28],
        scaling=False,
        use_feat_dynamic_real=True,
        use_feat_static_cat=True,
        use_feat_dynamic_cat=True,
        cardinality=stat_cat_cardinalities,
        dc_cardinality=[5, 5, 31, 31],   #event_type1,2 / event_name1,2
        dc_embedding_dimension=[2, 2, 15, 2],
        pick_incomplete=True,
        trainer=Trainer(
            learning_rate=1e-3,
            epochs=300,
            num_batches_per_epoch=num_batches_per_epoch,
            betas=(0.9, 0.98),
            use_lr_scheduler=True,
            lr_warmup_period=num_batches_per_epoch*5,
            batch_size=batch_size,
            device=device,
            log_path=log_path,
            num_workers=4,
        )
    )

def get_estimator(model_name, stat_cat_cardinalities, device, base_log_path, full_log_path):
    estimator = globals()["get_" + model_name + "_estimator"](stat_cat_cardinalities, device, full_log_path)

    return estimator

def main(args):
    # parameters
    comment     = args.comment
    model_name  = args.model
    data_path = args.data_path
    trial = args.trial

    # set default config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_log_path, base_log_path, trial_path = get_log_path(f"m5_submission/{model_name}", comment, trial)

    # set logger
    set_logger(full_log_path)
    
    # make dataset
    train_ds, val_ds, stat_cat_cardinalities = make_m5_dataset(m5_input_path=data_path, exclude_no_sales=True)

    # get estimator
    logger.info(f"Using {model_name} model...")
    estimator = get_estimator(model_name, stat_cat_cardinalities, device, base_log_path, full_log_path)

    # path for trained model
    model_path = Path(full_log_path+"/trained_model")
    model_path.mkdir()

    # prediction
    predictor = estimator.train(train_ds, validation_period=10)

    # save model
    if args.save_model:
        logger.info(f"Save {model_name} model...")
        model_path = Path(full_log_path+"/predictor")        
        model_path.mkdir()
        predictor.serialize(model_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
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
    
    args = parser.parse_args()
    args.save_model = True  # always save model
    args.model = 'rolled_deepAR'

    main(args)