import numpy as np
import pandas as pd
import os

tickers = ['MGC', 'VONE', 'VOO', 'RSP', 'SPY', 'VV', 'IWV', 'FEX', 'IVV', 'ITOT', 'IYY', 'IWB', 'VTI', 'SCHB']

def get_data(dir):
    file_name = "/sp500_etf.csv"
    data = pd.read_csv(dir + file_name, index_col = 0)
    data.index = pd.to_datetime(data.index)
    return data

def get_covariate(data_dir):
    sp500_ret = pd.read_csv(os.path.join(data_dir, 'sp500.csv'), parse_dates=['Date'], index_col='Date')['Price'].sort_index().str.replace(',','').astype(float).pct_change()
    vix_diff = pd.read_csv(os.path.join(data_dir, 'vix.csv'), parse_dates=['Date'], index_col='Date')['Price'].sort_index().diff()
    covariate = pd.concat([sp500_ret, vix_diff], axis=1).dropna()
    covariate.columns = ['sp500_ret', 'vix_diff']
    return covariate

def yearly_sample(y, data):
    y_start, y_train, y_end = y-2, y, y+1
    temp = data.loc[str(y_start):str(y_end)]
    # .pivot_table(index='date', columns='ticker', values='adj_close')
    temp_in = temp.loc[str(y_start):str(y_train-1)]
    temp_out = temp.loc[str(y_train):str(y_end-1)]
    
    ret_in = np.log(temp_in)
    ret_out = np.log(temp_out)
    return temp_in, temp_out, ret_in, ret_out

def get_pair_data(y, data, pair):
    if isinstance(pair, str) and pair[0] == '(' and pair[-1] == ')':
        pair = eval(pair)
    
    temp_in, temp_out, ret_in, ret_out = yearly_sample(y, data)
    price_in = temp_in[list(pair)]
    price_out = temp_out[list(pair)]
    ret_in = ret_in[list(pair)]
    ret_out = ret_out[list(pair)]
    return price_in, price_out, ret_in, ret_out