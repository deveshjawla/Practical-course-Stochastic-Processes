import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm
from sklearn.preprocessing import MinMaxScaler

assets = ["SPX Index","CCMP Index", "UKX Index"]
intervals=[30, 79, 204, 529, 1375, 3577, 9304, 24206, 62975, 163840]
#intervals=[529, 9304,163840]

for asset in assets:
    sigmas=[]
    for interval in intervals:
        df=pd.read_csv(f'./{interval}_seconds_OHLC/{asset}_pct_change.csv')
        #scaler = MinMaxScaler()
        sns.distplot(df['Price'], hist = False, kde = True, fit=lognorm,kde_kws = {'linewidth': 1},label = f'{interval} s')
        shape, loc, scale = lognorm.fit(df['Price'])
        mu,sigma=np.log(scale),shape**2
        sigmas.append(sigma)
        print("mu={0}, sigma={1}".format(mu, sigma))
        # counts, bins, bars = plt.hist(np.array(df['Price']))
        # print( len(counts), len(bins), bars)
        # counts=df['Price'].value_counts()
        # price_change=np.array(pd.DataFrame(counts).index)
        # counts=np.array(pd.DataFrame(counts).Price)
        # counts=np.hstack(scaler.fit_transform(counts.reshape(-1,1)))
        # plt.close()
        # df = pd.DataFrame({'price_change':bins[1:], 'Probability':np.hstack(scaler.fit_transform(counts.reshape(-1,1)))})
        # df.to_csv(f'./{interval}_seconds_OHLC/{asset}_plot_ready.csv',index=False)
        #print(np.hstack(scaler.fit_transform(counts.reshape(-1,1))))
        # plt.scatter(bins[1:],np.hstack(scaler.fit_transform(counts.reshape(-1,1))),s=5,label=interval,marker='.')
        # plt.scatter(price_change,counts,label=interval)
    # plt.xlim(-0.01, 0.01)
    plt.legend(prop={'size': 8}, title = asset)
    plt.title('Kernel density estimation for different time delays $\Delta t$ and LogNorm fit')
    plt.xlabel('$\Delta x$')
    plt.ylabel('$P_{\Delta t}(\Delta x)$')
    plt.savefig(f'distribution_{asset}.png',format='png')
    plt.close()

    sns.scatterplot(np.log(intervals), sigmas)
    plt.title(f'Dependence of $\lambda^{2}$ on $ln(\Delta t)$ for {asset}')
    plt.xlabel('$ln(\Delta t)$')
    plt.ylabel('$\lambda^{2}$')
    plt.savefig(f'lambda_{asset}.png',format='png')
    plt.close()
