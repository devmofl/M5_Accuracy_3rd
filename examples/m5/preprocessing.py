import numpy as np
import pandas as pd
import copy
import os

from pathlib import Path

from utils import convert_price_file
from accuracy_evaluator import calculate_and_save_data

def main(m5_input_path):
    # make price file
    converted_price_file = Path(f'{m5_input_path}/converted_price_evaluation.csv')
    if not converted_price_file.exists():
        convert_price_file(m5_input_path)
    converted_price = pd.read_csv(converted_price_file)

    # make rolling matrix for wrmsse
    _ = calculate_and_save_data(data_path=m5_input_path, prediction_start = 1942)
    

    # normalized sell prices
    normalized_price_file = Path(f'{m5_input_path}/normalized_price_evaluation.npz')
    if not normalized_price_file.exists():
        # normalized sell prices per each item
        price_feature = converted_price.drop(["id","item_id","dept_id","cat_id","store_id","state_id"], axis=1).values
        price_mean_per_item = np.nanmean(price_feature, axis=1, keepdims=True)
        price_std_per_item = np.nanstd(price_feature, axis=1, keepdims=True)
        normalized_price_per_item = (price_feature - price_mean_per_item) / (price_std_per_item + 1e-6)
        
        # normalized sell prices per day within the same dept
        dept_groups = converted_price.groupby('dept_id')
        price_mean_per_dept = dept_groups.transform(np.nanmean)
        price_std_per_dept = dept_groups.transform(np.nanstd)
        normalized_price_per_group_pd = (converted_price[price_mean_per_dept.columns] - price_mean_per_dept) / (price_std_per_dept + 1e-6)

        normalized_price_per_group = normalized_price_per_group_pd.values
        np.savez(normalized_price_file, per_item = normalized_price_per_item, per_group = normalized_price_per_group)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    ###[Important argument]
    parser.add_argument(
        "--data_path",
        default='/data/m5'
    )
    args = parser.parse_args()

    main(args.data_path)