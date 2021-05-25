import numpy as np 
import pandas as pd
from glob import glob
import os
from tqdm import tqdm
import logging

from accuracy_evaluator import load_precalculated_data
from pts.core.logging import get_log_path, set_logger

logger = logging.getLogger("mofl").getChild("ensemble")

class Ensembler(object):
    def __init__(self, data_path, base_path):
        self.base_path = base_path
        self.data_path = data_path

        # load data
        _, _, self.sw, self.roll_mat_csr = load_precalculated_data(data_path=data_path, prediction_start = 1942)
        self.sales_train_evaluation = pd.read_csv(f'{data_path}/sales_train_evaluation.csv')

    def wrmsse(self, error):
        return np.sum(
                np.sqrt(
                    np.mean(
                        np.square(self.roll_mat_csr * error), axis=1)) * self.sw)/12

    def calc_wrmsse(self, prediction, prediction_start):
        dayCols = ["d_{}".format(i) for i in range(prediction_start, prediction_start+28)]
        y_true = self.sales_train_evaluation[dayCols]

        error = prediction - y_true.values

        return self.wrmsse(error)

    def calc_wrmsse_list(self, model_path, cv_path, corr_factor=1):
        wrmsse_list = []

        for prediction_start in range(1914, 1522, -28):
            cv_file = os.path.join(self.base_path, model_path, cv_path, f'prediction_{prediction_start}.npy')
            prediction = np.load(cv_file) * corr_factor

            wrmsse_list.append(self.calc_wrmsse(prediction, prediction_start))

        return wrmsse_list 

    def load_all_predictions(self, model, choosed_epoch):
        all_predictions = []

        for prediction_start in tqdm(range(1942, 1522, -28)):

            model_predictions = []        
            for model, epochs in zip(models, choosed_epoch):

                epoch_predictions = []
                for epoch in epochs:
                    cv_path = f'CV/train_net_{epoch}/'
                    cv_file = os.path.join(self.base_path, model, cv_path, f'prediction_{prediction_start}.npy')            
                    prediction = np.load(cv_file)
                    
                    epoch_predictions.append(prediction)
            
                model_predictions.append(epoch_predictions)
            
            all_predictions.append(model_predictions)

        # all_predictions.shape : period * model * epoch * predictions
        return np.array(all_predictions)

    def get_topK_epochs(self, models, K=3):
        epoch_list = np.arange(200,300,10)

        choosed_epoch = []
        ens_results = []

        for model in tqdm(models):
            epoch_results = []
            
            for epoch in epoch_list:
                cv_path = f'CV/train_net_{epoch}/'
                epoch_results.append(self.calc_wrmsse_list(model, cv_path))

            epoch_results = np.array(epoch_results)
            
            # select top K epoch
            criteria = epoch_results.mean(axis=1)
            topK = sorted(list(zip(criteria, range(0,20))))[:K]
            topK = np.array(topK)
            topK_index = np.int32(topK[:,1]) # topK epoch index
            #topK_wrmsse = epoch_results[topK_index] # topK wrmsse list
            topK_epoch = epoch_list[topK_index]
            
            # ensemble best K epoch
            choosed_epoch.append(topK_epoch)

        return choosed_epoch

    def export_final_csv(self, prediction_1914, prediction_1942, result_path):
        sample_submission = pd.read_csv(f'{self.data_path}/sample_submission.csv')

        sample_submission.iloc[:30490,1:] = prediction_1914
        sample_submission.iloc[30490:,1:] = prediction_1942

        sample_submission.to_csv(result_path + '/submission_final.csv', index=False)
        
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
    args = parser.parse_args()
    
    args.model = 'rolled_deepAR'
    

    # initialize
    base_path = 'logs/m5_submission/'
    model_path = args.model + '/' + args.comment
    ens = Ensembler(args.data_path, base_path)

    # set logger
    full_log_path = base_path + model_path
    set_logger(full_log_path, text_log_file='ensemble.log')

    # choose top-K epoch    
    models = [model_path  + '/' + f.name for f in os.scandir(base_path + model_path) if f.is_dir()]
    choosed_epoch = ens.get_topK_epochs(models)

    # logging
    logger.info(f"Selected Epochs...")
    for m,e in zip(models,choosed_epoch):
        logger.info(f"{m}: {e}")

    # get all predictions
    all_predictions = ens.load_all_predictions(models, choosed_epoch)

    # ensemble (mean of predictions)
    mean_predictions = np.mean(all_predictions, axis=(1,2)) # shape: period * (30490, 28)

    # check final wrmsse
    logger.info("Final WRMSSEs...")
    ensemble_wrmsse = []
    mean_predictions_1914 = mean_predictions[1:]
    for p, prediction_start in enumerate(range(1914, 1522, -28)):
        prediction = mean_predictions_1914[p]
        wrmsse_val = ens.calc_wrmsse(prediction, prediction_start)        
        ensemble_wrmsse.append(wrmsse_val)

        # logging
        logger.info(f"{prediction_start}: {wrmsse_val}")

    # export_final_csv
    prediction_1942 = mean_predictions[0]
    prediction_1914 = mean_predictions[1]

    ens.export_final_csv(prediction_1914, prediction_1942, result_path=full_log_path)
