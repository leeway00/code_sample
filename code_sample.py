import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import combinations
import os

from nonlin_coint.data_process import *
from nonlin_coint.FuncCoint import *
from nonlin_coint.VECM import *

import warnings
warnings.filterwarnings('ignore')


def adf_valid_test(y):
    adf_res = ts.adfuller(y, regression = 'n')
    res = list(adf_res[0:2]) + list(adf_res[4].values())
    return res

def adf_table(covariate):
    a = adf_valid_test(covariate['sp500_ret'])
    b = adf_valid_test(covariate['vix_diff'])
    adf_result = pd.DataFrame([a,b], columns=['ADF', 'p-value', '1%', '5%', '10%'], index=['S&P500', 'VIX'])
    
    print("---- Table 1: ADF test result for Covariates ----")
    print(adf_result)
    
    return adf_result
    
    
def adf_plot(covariate):
    fig, ax = plt.subplots(1,2, figsize=(10,3))
    covariate['sp500_ret']['2017':'2019'].plot(ax = ax[0], title = 'S&P500 Return')
    covariate['vix_diff']['2017':'2019'].plot(ax = ax[1], title = 'Difference of VIX')
    fig.suptitle('Figure 1: Covariates')
    plt.show()
    pass
    
def kernel_plot():
    u = np.linspace(-1, 1, int(1e5))
    prob = normal_pdf(u).sum() * (2/len(u))
    
    fig = plt.plot(u, normal_pdf(u))
    plt.title('Figure 1a: Truncated Gaussian Kernel')
    plt.xlabel('u')
    plt.ylabel('Density')
    plt.show()

    print("Probability of u in [-1, 1]:", prob)
    return fig
    
def funccoef_test(x_train, y_train, z_train):
    stab_res = stability_test(x_train, y_train, z_train)
    coint_res = coint_test(x_train, y_train, z_train)
    a = list(stab_res[0:2]) + stab_res[2]
    b = list(coint_res[0:2]) + list(coint_res[2])
    res = pd.DataFrame([a,b], columns=['ADF', 'p-value', '1%', '5%', '10%'], index=['Stability', 'Cointegration'])
    return res

if __name__ == "__main__":    
    dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir, 'data')

    data = get_data(data_dir)
    tickers = data.columns.to_list()
    years = data.index.year.unique()
    
    covariate = get_covariate(data_dir)
    
    kernel_plot()
    
    y = 2019
    pair = ['VV', 'SPY']
    price_in, price_out, ret_in, ret_out = get_pair_data(y, data, pair)

    x_train, y_train = price_in[pair[0]], price_in[pair[1]]
    x_test, y_test = price_out[pair[0]], price_out[pair[1]]
    z_train, z_test = covariate.loc[price_in.index], covariate.loc[price_out.index]
    

    print("---- Table 1: Stability and Cointegration test from Xiao (2009) ----")
    
    funccoef_res = pd.concat([funccoef_test(x_train, y_train, z_train.sp500_ret), 
                     funccoef_test(x_train, y_train, z_train.vix_diff)], axis=0, keys=['VIX', 'S&P500'])
    funccoef_res.index.names = ['Covariate', 'Type']
    funccoef_res = funccoef_res.round(2)
    print(funccoef_res)
    

    # This plots the common trend using VECM of in and out of sample data
    common_trend_vecm(price_in, price_out, adf = False, title = 'Figure 1b: Common Trend using VECM')