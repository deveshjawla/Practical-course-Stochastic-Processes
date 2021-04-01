from scipy.stats.distributions import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
import numpy as np

time_delay_list = [11, 15, 22, 33, 48, 71]
number_of_samples = [37, 27, 18, 12, 8, 5]

assets = ["UKX Index", "USDEUR Curncy"]

for asset in assets:
    df = pd.read_csv(f'./data/{asset}.csv',delimiter='\t',index_col=0, parse_dates=True,header=None, names=['Time', 'Tr', 'Price', 'q','w','t'],low_memory=False)
    df=df.drop(['Tr','q','w','t'],axis=1)
    for delay in time_delay_list:
        close=df['Price'].resample(f'{delay}S').ohlc()['close'].dropna()
        close.to_csv(f'./{asset}/prices_delay={delay}s.csv', header=['Price'], index_label='Time')