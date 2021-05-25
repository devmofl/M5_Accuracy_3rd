
import numpy as np
from pts.evaluation import Evaluator
import accuracy_evaluator

class M5Evaluator(Evaluator):            
    def __init__(self, prediction_length, **kwargs):            
        super().__init__(**kwargs)
        self.prediction_length = prediction_length    

    def evaluate_wrmsse(self, prediction, prediction_start=1886, score_only=False, data_path='/data/m5'):
        return accuracy_evaluator.evaluate_wrmsse(data_path=data_path, prediction=prediction, prediction_start=prediction_start, score_only=score_only)


def convert_price_file(m5_input_path):
    # 주 단위로 되어 있는 가격정보를 sales 데이터와 동일하게 각 아이템의 매일 가격정보를 나타내는 형태로 변환
    import numpy as np
    import pandas as pd

    # load data
    calendar = pd.read_csv(f'{m5_input_path}/calendar.csv')
    sales_train_evaluation = pd.read_csv(f'{m5_input_path}/sales_train_evaluation.csv')
    sell_prices = pd.read_csv(f'{m5_input_path}/sell_prices.csv')

    # assign price for all days
    week_and_day = calendar[['wm_yr_wk', 'd']]

    price_all_days_items = pd.merge(week_and_day, sell_prices, on=['wm_yr_wk'], how='left') # join on week number
    price_all_days_items = price_all_days_items.drop(['wm_yr_wk'], axis=1)

    # convert days to column
    price_all_items = price_all_days_items.pivot_table(values='sell_price', index=['store_id', 'item_id'], columns='d') 
    price_all_items.reset_index(drop=False, inplace=True)

    # reorder column
    price_all_items = price_all_items.reindex(['store_id','item_id'] + ['d_%d' % x for x in range(1,1969+1)], axis=1) 

    sales_keys = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_keys_pd = sales_train_evaluation[sales_keys]

    # join with sales data
    price_converted = pd.merge(sales_keys_pd, price_all_items, on=['store_id','item_id'], how='left')


    # save file
    price_converted.to_csv(f'{m5_input_path}/converted_price_evaluation.csv', index=False)
    




