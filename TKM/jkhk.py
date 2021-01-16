import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm
from sklearn.preprocessing import MinMaxScaler

assets = ["SPX Index","CCMP Index", "UKX Index"]
intervals=[30, 79, 204, 529, 1375, 3577, 9304, 24206, 62975, 163840]
#intervals=[529, 9304, 163840]
for asset in assets:
    for interval in intervals:
        df=pd.read_csv(f'./{interval}_seconds_OHLC/{asset}_plot_ready.csv')
        sns.lineplot(df['price_change'],df['Probability'],label=f'{interval} s')
    #plt.xlim(-0.01, 0.01)
    plt.legend(prop={'size': 8}, title = asset)
    plt.title('Probability distribution for different time delays $\Delta t$')
    plt.xlabel('$\Delta x$')
    plt.ylabel('$P_{\Delta t}(\Delta x)$')
    plt.savefig(f'dsitribution_{asset}.png',format='png')
    plt.close()
