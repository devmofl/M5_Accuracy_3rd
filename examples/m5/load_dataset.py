import numpy as np
import pandas as pd
import copy
import os

from pathlib import Path

from utils import convert_price_file
from pts.dataset import ListDataset, FieldName

from accuracy_evaluator import calculate_and_save_data

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def get_second_sale_idx(target_values):
    first_sale = first_nonzero(target_values, axis=1)    
    
    target_values_copy = copy.deepcopy(target_values)
    for i in range(30490):
        target_values_copy[i,first_sale[i]] = 0

    second_sale = first_nonzero(target_values_copy, axis=1)
    
    return second_sale

def make_m5_dummy_features(m5_input_path):
    target_values = np.array([list(range(1969))] * 30490)
    
    dynamic_cat = np.zeros([30490, 4, 1969])
    dynamic_real = np.zeros([30490, 6, 1969])

    stat_cat = np.zeros([30490, 5])
    stat_cat_cardinalities = [3049, 7, 3, 10, 3]
    dynamic_past = np.zeros([30490, 1, 1969])

    return target_values, dynamic_real, dynamic_cat, dynamic_past, stat_cat, stat_cat_cardinalities

def make_m5_features(m5_input_path):
    # First we need to convert the provided M5 data into a format that is readable by GluonTS. 
    # At this point we assume that the M5 data, which can be downloaded from Kaggle, is present under m5_input_path.
    calendar = pd.read_csv(f'{m5_input_path}/calendar.csv')
    sales_train_evaluation = pd.read_csv(f'{m5_input_path}/sales_train_evaluation.csv')
    sample_submission = pd.read_csv(f'{m5_input_path}/sample_submission.csv')

    # append dummy for expanding all period
    for i in range(1942, 1970):
        sales_train_evaluation[f"d_{i}"] = np.nan   # d_1 ~ d1969
    
    converted_price_file = Path(f'{m5_input_path}/converted_price_evaluation.csv')
    if not converted_price_file.exists():
        convert_price_file(m5_input_path)
    converted_price = pd.read_csv(converted_price_file)
    
    # target_value
    train_df = sales_train_evaluation.drop(["id","item_id","dept_id","cat_id","store_id","state_id"], axis=1)    # d_1 ~ d_1969
    target_values = train_df.values
    
    #################################
    # FEAT_DYNAMIC_CAT
    
    # Event type
    event_type_to_idx = {"nan":0, "Cultural":1, "National":2, "Religious":3, "Sporting":4}
    event_type1 = np.array([event_type_to_idx[str(x)] for x in calendar['event_type_1'].values])
    event_type2 = np.array([event_type_to_idx[str(x)] for x in calendar['event_type_2'].values])

    # Event name
    event_name_to_idx = {'nan':0, 'Chanukah End':1, 'Christmas':2, 'Cinco De Mayo':3, 'ColumbusDay':4, 'Easter':5,
                        'Eid al-Fitr':6, 'EidAlAdha':7, "Father's day":8, 'Halloween':9, 'IndependenceDay':10, 'LaborDay':11,
                        'LentStart':12, 'LentWeek2':13, 'MartinLutherKingDay':14, 'MemorialDay':15, "Mother's day":16, 'NBAFinalsEnd':17,
                        'NBAFinalsStart':18, 'NewYear':19, 'OrthodoxChristmas':20, 'OrthodoxEaster':21, 'Pesach End':22, 'PresidentsDay':23,
                        'Purim End':24, 'Ramadan starts':25, 'StPatricksDay':26, 'SuperBowl':27, 'Thanksgiving':28, 'ValentinesDay':29, 'VeteransDay':30}

    event_name1 = np.array([event_name_to_idx[str(x)] for x in calendar['event_name_1'].values])
    event_name2 = np.array([event_name_to_idx[str(x)] for x in calendar['event_name_2'].values])

    event_features = np.stack([event_type1, event_type2, event_name1, event_name2])
    dynamic_cat = [event_features] * len(sales_train_evaluation)


    #################################
    # FEAT_DYNAMIC_REAL        
    # SNAP_CA, TX, WI
    snap_features = calendar[['snap_CA', 'snap_TX', 'snap_WI']]
    snap_features = snap_features.values.T
    snap_features_expand = np.array([snap_features] * len(sales_train_evaluation))    # 30490 * 3 * T

    # sell_prices
    price_feature = converted_price.drop(["id","item_id","dept_id","cat_id","store_id","state_id"], axis=1).values

    # normalized sell prices
    normalized_price_file = Path(f'{m5_input_path}/normalized_price_evaluation.npz')
    if not normalized_price_file.exists():
        # normalized sell prices per each item
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
    else:
        normalized_price = np.load(normalized_price_file)

        normalized_price_per_item = normalized_price['per_item']
        normalized_price_per_group = normalized_price['per_group']

    price_feature = np.nan_to_num(price_feature)
    normalized_price_per_item = np.nan_to_num(normalized_price_per_item)
    normalized_price_per_group = np.nan_to_num(normalized_price_per_group)

    all_price_features = np.stack([price_feature, normalized_price_per_item, normalized_price_per_group], axis=1)   # 30490 * 3 * T
    dynamic_real = np.concatenate([snap_features_expand, all_price_features], axis=1)    # 30490 * 6 * T
    
    #################################
    # FEAT_STATIC_CAT
    # We then go on to build static features (features which are constant and series-specific). 
    # Here, we make use of all categorical features that are provided to us as part of the M5 data.
    state_ids = sales_train_evaluation["state_id"].astype('category').cat.codes.values
    state_ids_un , state_ids_counts = np.unique(state_ids, return_counts=True)

    store_ids = sales_train_evaluation["store_id"].astype('category').cat.codes.values
    store_ids_un , store_ids_counts = np.unique(store_ids, return_counts=True)

    cat_ids = sales_train_evaluation["cat_id"].astype('category').cat.codes.values
    cat_ids_un , cat_ids_counts = np.unique(cat_ids, return_counts=True)

    dept_ids = sales_train_evaluation["dept_id"].astype('category').cat.codes.values
    dept_ids_un , dept_ids_counts = np.unique(dept_ids, return_counts=True)

    item_ids = sales_train_evaluation["item_id"].astype('category').cat.codes.values
    item_ids_un , item_ids_counts = np.unique(item_ids, return_counts=True)

    stat_cat_list = [item_ids, dept_ids, cat_ids, store_ids, state_ids]

    stat_cat = np.concatenate(stat_cat_list)
    stat_cat = stat_cat.reshape(len(stat_cat_list), len(item_ids)).T

    stat_cat_cardinalities = [len(item_ids_un), len(dept_ids_un), len(cat_ids_un), len(store_ids_un), len(state_ids_un)]
    # [3049, 7, 3, 10, 3]


    #################################
    # FEAT_STATIC_REAL
    # None

    #################################
    # FEAT_DYNAMIC_PAST
    # 미래에는 사용 불가능한 feature임
    
    # zero-sale period : 오늘까지 연속적으로 판매량이 0이였던 기간    
    sales_zero_period = np.zeros_like(target_values)
    sales_zero_period[:, 0] = 1

    for i in range(1, 1969):    
        sales_zero_period[:,i] = sales_zero_period[:, i-1] + 1
        sales_zero_period[target_values[:,i]!=0, i] = 0
    
    dynamic_past = np.expand_dims(sales_zero_period, 1) # 30490 * 1 * T 

    return target_values, dynamic_real, dynamic_cat, dynamic_past, stat_cat, stat_cat_cardinalities

def make_m5_dataset(m5_input_path="/data/m5", exclude_no_sales=False, ds_split=True, prediction_start=1942):    

    # make features
    target_values, dynamic_real, dynamic_cat, dynamic_past, stat_cat, stat_cat_cardinalities = make_m5_features(m5_input_path)

    #################################
    # ACCUMULATED TARGET
    # for online moving average caculation
    acc_target_values = np.cumsum(target_values, dtype=float, axis=1)

    #################################
    # TARGET
    # This is for evaluation set
    # D1 ~ 1941: train , D1942 ~ 1969: test
    PREDICTION_START = 1942

    # exclude no sale periods
    if exclude_no_sales:
        second_sale = get_second_sale_idx(target_values)
        second_sale = np.clip(second_sale, None, PREDICTION_START-28-1)
    else:
        second_sale = np.zeros(30490, dtype=np.int32)

    start_date = pd.Timestamp("2011-01-29", freq='1D')
    m5_dates = [start_date + pd.DateOffset(days=int(d)) for d in second_sale]


    ##########################################################
    #  Mode                 1886~         1914~          1942~    NaN    1969
    #                              train    /    val       |      test

    if ds_split==True:
        #
        idx_train_end = PREDICTION_START - 1    # index from 0
        idx_val_end = 1914 - 1

        

        ### Train Set
        train_set = [
            {
                FieldName.TARGET: target[first:idx_train_end],
                FieldName.START: start,
                FieldName.ACC_TARGET_SUM: acc_target[first:idx_train_end],
                FieldName.FEAT_DYNAMIC_REAL: fdr[...,first:idx_train_end],
                FieldName.FEAT_DYNAMIC_CAT: fdc[...,first:idx_train_end],
                FieldName.FEAT_DYNAMIC_PAST: fdp[...,first:idx_train_end],
                FieldName.FEAT_STATIC_REAL: None,
                FieldName.FEAT_STATIC_CAT: fsc
            }
            for i, (target, first, start, acc_target, fdr, fdc, fdp, fsc) in enumerate(zip(target_values, second_sale, 
                                                m5_dates,
                                                acc_target_values,
                                                dynamic_real,
                                                dynamic_cat,
                                                dynamic_past,
                                                stat_cat))
        ]
        #train_set = train_set[:20]
        train_ds = ListDataset(train_set, freq="D", shuffle=False)


        # reset to first day
        second_sale = np.zeros(30490, dtype=np.int32)
        m5_dates = [start_date + pd.DateOffset(days=int(d)) for d in second_sale]

        ### Validation Set
        val_set = [
            {
                FieldName.TARGET: target[first:idx_val_end],
                FieldName.START: start,
                FieldName.ACC_TARGET_SUM: acc_target[first:idx_val_end],
                FieldName.FEAT_DYNAMIC_REAL: fdr[...,first:idx_val_end],
                FieldName.FEAT_DYNAMIC_CAT: fdc[...,first:idx_val_end],
                FieldName.FEAT_DYNAMIC_PAST: fdp[...,first:idx_val_end],
                FieldName.FEAT_STATIC_REAL: None,
                FieldName.FEAT_STATIC_CAT: fsc
            }
            for i, (target, first, start, acc_target, fdr, fdc, fdp, fsc) in enumerate(zip(target_values, second_sale, 
                                                m5_dates,
                                                acc_target_values,
                                                dynamic_real,
                                                dynamic_cat,
                                                dynamic_past,
                                                stat_cat))
        ]
        #val_set = val_set[:20]
        val_ds = ListDataset(val_set, freq="D")

        return train_ds, val_ds, stat_cat_cardinalities

    else:
        idx_end = prediction_start-1+28
        ### Test Set
        test_set = [
            {
                FieldName.TARGET: target[first:idx_end],
                FieldName.START: start,
                FieldName.ACC_TARGET_SUM: acc_target[first:idx_end],
                FieldName.FEAT_DYNAMIC_REAL: fdr[...,first:idx_end],
                FieldName.FEAT_DYNAMIC_CAT: fdc[...,first:idx_end],
                FieldName.FEAT_DYNAMIC_PAST: fdp[...,first:idx_end],
                FieldName.FEAT_STATIC_REAL: None,
                FieldName.FEAT_STATIC_CAT: fsc
            }
            for i, (target, first, start, acc_target, fdr, fdc, fdp, fsc) in enumerate(zip(target_values, second_sale, 
                                                m5_dates,
                                                acc_target_values,
                                                dynamic_real,
                                                dynamic_cat,
                                                dynamic_past,
                                                stat_cat))
        ]
        #test_set = test_set[:20]
        test_ds = ListDataset(test_set, freq="D")       
       
        return test_ds