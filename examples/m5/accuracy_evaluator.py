import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import gc
import os
from pprint import pprint
from typing import Union
from tqdm.notebook import tqdm_notebook as tqdm

prediction_length = 28

# Memory reduction helper function:
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns
        col_type = df[col].dtypes
        if col_type in numerics: #numerics
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# Fucntion to calculate S weights:
def get_s(roll_mat_csr, sales, prediction_start):
    # Rollup sales:
    d_name = ['d_' + str(i) for i in range(1, prediction_start)]
    sales_train_val = roll_mat_csr * sales[d_name].values

    no_sales = np.cumsum(sales_train_val, axis=1) == 0

    # Denominator of RMSSE / RMSSE
    diff = np.diff(sales_train_val,axis=1)
    diff = np.where(no_sales[:,1:], np.nan, diff)

    weight1 = np.nanmean(diff**2,axis=1)
    weight1[np.isnan(weight1)] = 1e-9
    
    return weight1

# Functinon to calculate weights:
def get_w(roll_mat_csr, sale_usd):
    """
    """
    # Calculate the total sales in USD for each item id:
    total_sales_usd = sale_usd.groupby(
        ['id'], sort=False)['sale_usd'].apply(np.sum).values
    
    # Roll up total sales by ids to higher levels:
    weight2 = roll_mat_csr * total_sales_usd
    
    return 12*weight2/np.sum(weight2) # weight2/(np.sum(weight2) / 12) : np.sum(weight2)은 모든 합의 12배임

# Function to do quick rollups:
def rollup(roll_mat_csr, v):
    '''
    v - np.array of size (30490 rows, n day columns)
    v_rolledup - array of size (n, 42840)
    '''
    return roll_mat_csr*v #(v.T*roll_mat_csr.T).T

# Function to calculate WRMSSE:
def wrmsse(error, score_only, roll_mat_csr, s, w, sw):
    '''
    preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)
    y_true - True values: pd.DataFrame of size (30490 rows, N day columns)
    sequence_length - np.array of size (42840,)
    sales_weight - sales weights based on last 28 days: np.array (42840,)
    '''
    
    if score_only:
        return np.sum(
                np.sqrt(
                    np.mean(
                        np.square(rollup(roll_mat_csr, error))
                            ,axis=1)) * sw)/12 #<-used to be mistake here
    else: 
        score_matrix = (np.square(rollup(roll_mat_csr, error)) * np.square(w)[:, None])/ s[:, None]
        wrmsse_i = np.sqrt(np.mean(score_matrix,axis=1))
        wrmsse_raw = np.sqrt(score_matrix)

        aggregation_count = [1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490]

        idx = 0
        aggregated_wrmsse = np.zeros(12)
        aggregated_wrmsse_per_day = np.zeros([12, prediction_length])
        for i, count in enumerate(aggregation_count):
            endIdx = idx+count
            aggregated_wrmsse[i] = wrmsse_i[idx:endIdx].sum()            
            aggregated_wrmsse_per_day[i] = wrmsse_raw[idx:endIdx, :].sum(axis=0)
            idx = endIdx

        # score == aggregated_wrmsse.mean()
        wrmsse = np.sum(wrmsse_i)/12 #<-used to be mistake here

        return wrmsse, aggregated_wrmsse, aggregated_wrmsse_per_day, score_matrix 

def calculate_and_save_data(data_path, prediction_start):
    # Sales quantities:
    sales = pd.read_csv(data_path+'/sales_train_evaluation.csv')

    # Calendar to get week number to join sell prices:
    calendar = pd.read_csv(data_path+'/calendar.csv')
    calendar = reduce_mem_usage(calendar)

    # Sell prices to calculate sales in USD:
    sell_prices = pd.read_csv(data_path+'/sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)

    # Dataframe with only last 28 days:
    cols = ["d_{}".format(i) for i in range(prediction_start-28, prediction_start)]
    data = sales[["id", 'store_id', 'item_id'] + cols]

    # To long form:
    data = data.melt(id_vars=["id", 'store_id', 'item_id'], 
                     var_name="d", value_name="sale")

    # Add week of year column from 'calendar':
    data = pd.merge(data, calendar, how = 'left', 
                    left_on = ['d'], right_on = ['d'])

    data = data[["id", 'store_id', 'item_id', "sale", "d", "wm_yr_wk"]]

    # Add weekly price from 'sell_prices':
    data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
    data.drop(columns = ['wm_yr_wk'], inplace=True)

    # Calculate daily sales in USD:
    data['sale_usd'] = data['sale'] * data['sell_price']

    # List of categories combinations for aggregations as defined in docs:
    dummies_list = [sales.state_id, sales.store_id, 
                    sales.cat_id, sales.dept_id, 
                    sales.state_id +'_'+ sales.cat_id, sales.state_id +'_'+ sales.dept_id,
                    sales.store_id +'_'+ sales.cat_id, sales.store_id +'_'+ sales.dept_id, 
                    sales.item_id, sales.state_id +'_'+ sales.item_id, sales.id]

    ## First element Level_0 aggregation 'all_sales':
    dummies_df_list =[pd.DataFrame(np.ones(sales.shape[0]).astype(np.int8), 
                                   index=sales.index, columns=['all']).T]

    # List of dummy dataframes:
    for i, cats in enumerate(dummies_list):
        cat_dtype = pd.api.types.CategoricalDtype(categories=pd.unique(cats.values), ordered=True)
        ordered_cat = cats.astype(cat_dtype)
        dummies_df_list +=[pd.get_dummies(ordered_cat, drop_first=False, dtype=np.int8).T]
        
    #[1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490]
    # Concat dummy dataframes in one go:
    ## Level is constructed for free.
    roll_mat_df = pd.concat(dummies_df_list, keys=list(range(12)), 
                            names=['level','id'])#.astype(np.int8, copy=False)

    # Save values as sparse matrix & save index for future reference:
    roll_index = roll_mat_df.index
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    roll_mat_csr.shape

    roll_mat_df.to_pickle(data_path + '/ordered_roll_mat_df.pkl')

    del dummies_df_list, roll_mat_df
    gc.collect()

    S = get_s(roll_mat_csr, sales, prediction_start)
    W = get_w(roll_mat_csr, data[['id','sale_usd']])
    SW = W/np.sqrt(S)

    sw_df = pd.DataFrame(np.stack((S, W, SW), axis=-1),index = roll_index,columns=['s','w','sw'])
    sw_df.to_pickle(data_path + f'/ordered_sw_df_p{prediction_start}.pkl')

    return sales, S, W, SW, roll_mat_csr

def load_precalculated_data(data_path, prediction_start):
    # Load S and W weights for WRMSSE calcualtions:
    if not os.path.exists(data_path+f'/ordered_sw_df_p{prediction_start}.pkl'):
        calculate_and_save_data(data_path, prediction_start)    
    sw_df = pd.read_pickle(data_path+f'/ordered_sw_df_p{prediction_start}.pkl')
    S = sw_df.s.values
    W = sw_df.w.values
    SW = sw_df.sw.values

    # Load roll up matrix to calcualte aggreagates:
    roll_mat_df = pd.read_pickle(data_path+'/ordered_roll_mat_df.pkl')
    roll_index = roll_mat_df.index
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    del roll_mat_df

    return S, W, SW, roll_mat_csr

def evaluate_wrmsse(data_path, prediction, prediction_start, score_only=True):
    # Loading data in two ways:
    # if S, W, SW are calculated in advance, load from pickle files
    # otherwise, calculate from scratch
    if os.path.isfile(data_path + f'/ordered_sw_df_p{prediction_start}.pkl') and \
        os.path.isfile(data_path + '/ordered_roll_mat_df.pkl'):
        print('load precalculated data')
        # Sales quantities:
        sales = pd.read_csv(data_path+'/sales_train_evaluation.csv')		
        S, W, SW, roll_mat_csr = load_precalculated_data(data_path, prediction_start)
    else:
        print('load data from scratch')
        sales, S, W, SW, roll_mat_csr = calculate_and_save_data(data_path, prediction_start)

    # Ground truth:
    dayCols = ["d_{}".format(i) for i in range(prediction_start, prediction_start+prediction_length)]
    y_true = sales[dayCols]

    
    error = prediction - y_true.values
    results = wrmsse(error, score_only, roll_mat_csr, S, W, SW)
    
    return results


class WRMSSEEvaluator(object):

    def __init__(self, data_path, prediction_start):
        # Load Dataset        
        sales = pd.read_csv(data_path + 'sales_train_evaluation.csv')

        # append dummy
        for i in range(1942, 1970):
            sales[f"d_{i}"] = 0

        calendar = pd.read_csv(data_path + 'calendar.csv',
                            dtype={'wm_yr_wk': np.int32, 'wday': np.int32, 
                                    'month': np.int32, 'year': np.int32, 
                                    'snap_CA': np.int32, 'snap_TX': np.int32,
                                    'snap_WI': np.int32})

        prices = pd.read_csv(data_path + 'sell_prices.csv',
                                dtype={'wm_yr_wk': np.int32, 
                                        'sell_price': np.float32})

        prediction_start = prediction_start + 6 - 1 # num of heads
        train_df = sales.iloc[:, :prediction_start]
        valid_df = sales.iloc[:, prediction_start:prediction_start+prediction_length]


        # 
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 'all'  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')]\
                     .columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')]\
                               .columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], 
                                 axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)\
                    [valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns]\
                    .set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index()\
                   .rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left',
                                    on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd'])\
                    .unstack(level=2)['value']\
                    .loc[zip(self.train_df.item_id, self.train_df.store_id), :]\
                    .reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns],
                               weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score_raw = ((valid_y - valid_preds) ** 2)
        score = score_raw.mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt), np.sqrt(score_raw / np.expand_dims(scale, 1))

    def score(self, valid_preds: Union[pd.DataFrame, 
                                       np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape \
               == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, 
                                       columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], 
                                 valid_preds], axis=1, sort=False)

        all_scores = []
        all_scores_day = np.zeros([12,28])
        for i, group_id in enumerate(self.group_ids):

            valid_preds_grp = valid_preds.groupby(group_id)[self.valid_target_columns].sum()
            setattr(self, f'lv{i + 1}_valid_preds', valid_preds_grp)
            
            lv_rmsse, lv_rmsse_raw = self.rmsse(valid_preds_grp, i + 1)
            setattr(self, f'lv{i + 1}_rmsse', lv_rmsse)
            
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_rmsse], axis=1, 
                                  sort=False).prod(axis=1)

            lv_scores_raw = lv_rmsse_raw * np.expand_dims(weight,1)
            lv_scores_raw = lv_scores_raw.sum(axis=0)
            
            all_scores.append(lv_scores.sum())
            all_scores_day[i] = lv_scores_raw
            
        self.all_scores = all_scores
        self.all_scores_day = all_scores_day
        self.wrmsse = np.mean(all_scores)

        return self.wrmsse

if __name__ == '__main__':
    DATA_DIR = '/data/m5/'
    PREDICTION_START = 1886 #1886:offline val, 1914:validation, 1942:evaluation
    
    prediction_pd = pd.read_csv('logs/m5/csv_test/submission_v1.csv')
    prediction = np.array(prediction_pd.values[:30490,1:], dtype=np.float32)

    # First Evaluator
    wrmsse, aggregated_wrmsse, _, _ = evaluate_wrmsse(data_path=DATA_DIR, prediction=prediction, prediction_start=PREDICTION_START, score_only=False)
    print('---------------------------------------------------')
    print('First Evaluator')
    print('WRMSSE:', wrmsse)
    for i, val in enumerate(aggregated_wrmsse):
        print(f'WRMSSE level #{i+1}: {val}')


    # Second Evaluator
    print('---------------------------------------------------')
    print('Second Evaluator')

    evaluator = WRMSSEEvaluator(data_path=DATA_DIR, prediction_start=PREDICTION_START)
    wrmsse2 = evaluator.score(prediction)
    print('WRMSSE:', wrmsse2)
    for i, val in enumerate(evaluator.all_scores):
        print(f'WRMSSE level #{i+1}: {val}')


    # Show difference
    print('---------------------------------------------------')
    print('Difference')
    
    print('WRMSSE diff:', wrmsse - wrmsse2)
    for i, val in enumerate(zip(aggregated_wrmsse, evaluator.all_scores)):
        print(f'WRMSSE level #{i+1}: {val[0] - val[1]}')

    '''
    First Evaluator
    WRMSSE: 0.6539945520603164
    WRMSSE level #1: 0.5264150554353765
    WRMSSE level #2: 0.5840861168973064
    WRMSSE level #3: 0.6257473379469497
    WRMSSE level #4: 0.5463039489698954
    WRMSSE level #5: 0.6173161892685839
    WRMSSE level #6: 0.6024604778476885
    WRMSSE level #7: 0.6600226507063979
    WRMSSE level #8: 0.6515105177255677
    WRMSSE level #9: 0.6978254036803149
    WRMSSE level #10: 0.7706896289293005
    WRMSSE level #11: 0.7848556693564849
    WRMSSE level #12: 0.7807016279599293
    '''