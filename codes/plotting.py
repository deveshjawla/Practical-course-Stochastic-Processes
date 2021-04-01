from numpy.lib.function_base import _i0_1
from scipy.stats.distributions import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import kurtosis

# need to firt download and install the pylevy module from https://github.com/josemiotto/pylevy
# then import its functions as follows
from scipy.stats import levy_stable
import scipy.stats as st
import levy

time_delay_list = [11, 15, 22, 33, 48, 71]

assets = ["UKX Index"]

plt.rcParams.update({'font.size': 22})
sns.set(font_scale=1.5)


def gaussian(x, mu, sigma):
    return (1.0/np.sqrt(2.0*np.pi*sigma**2))*(np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))

# (par=0, alpha=1.39, beta=-0.01, mu=0.00, sigma=0.39, 443489.0742122485)
# (par=0, alpha=1.45, beta=-0.02, mu=0.01, sigma=0.51, 383703.1652157372)
# (par=0, alpha=1.50, beta=-0.02, mu=0.01, sigma=0.68, 310355.9393136897)
# (par=0, alpha=1.56, beta=-0.01, mu=0.01, sigma=0.91, 240321.47765161048)
# (par=0, alpha=1.59, beta=-0.02, mu=0.01, sigma=1.18, 185695.41470330756)
# (par=0, alpha=1.61, beta=-0.01, mu=0.00, sigma=1.51, 139728.67176764144)

# for fixed beta, mu, sigma and for a folded delta x
# (par=0, alpha=1.41, beta=0.00, mu=0.00, sigma=0.08, -168882.67397089745)
# (par=0, alpha=1.44, beta=0.00, mu=0.00, sigma=0.08, -123464.00658623944)
# (par=0, alpha=1.48, beta=0.00, mu=0.00, sigma=0.08, -84614.92218463894)
# (par=0, alpha=1.52, beta=0.00, mu=0.00, sigma=0.08, -57655.35891854811)
# (par=0, alpha=1.55, beta=0.00, mu=0.00, sigma=0.08, -41199.98334207934)
# (par=0, alpha=1.58, beta=0.00, mu=0.00, sigma=0.08, -29608.837971824054)

# (par=0, alpha=1.47, beta=-0.01, mu=0.00, sigma=0.08, -504895.58368851006)
# with 1.1 million samples


alpha_list = [1.39,
              1.45,
              1.50,
              1.56,
              1.59,
              1.61]

alpha_list = [1.0, 1.2, 1.4, 1.6, 1.8, 1.9]

beta_list = [-0.01,
             -0.02,
             -0.02,
             -0.01,
             -0.02,
             -0.01]

mu_list = [0.00,
           0.01,
           0.01,
           0.01,
           0.01,
           0.00]

sigma_list = [0.39,
              0.51,
              0.68,
              0.91,
              1.18,
              1.51]

mu_list = [0.00,
           0.00,
           0.00,
           0.00,
           0.00,
           0.00]

sigma_list = [0.08,
              0.08,
              0.08,
              0.08,
              0.08,
              0.08]

# for alpha,beta in zip(alpha_list,beta_list):
#     mean, var, skew, kurt = levy_stable.stats(alpha, beta, moments='mvsk')
#     print(mean, var, skew, kurt)

###### 
###### Probability distribution and fitting loop
###### 

df_distros = pd.DataFrame()
df_stable = pd.DataFrame()
for asset in assets:
    plt.figure(figsize=(15, 10))
    for delay, alpha in zip(time_delay_list, alpha_list):
        pdf = pd.DataFrame()
        close_timeseries = pd.read_csv(f'./{asset}/prices_delay={delay}s.csv')
        pdf['Price'] = close_timeseries['Price'].diff().dropna() #creates the price changes
        pdf['Price'] = pdf['Price']/(delay**0.68) #rescacles timeseries according to nu=0.68
        pdf_pos = pdf.drop(pdf[pdf.Price < 0.0].index,
                           inplace=False).to_numpy()
        pdf_neg = pdf.drop(pdf[pdf.Price > 0.0].index).to_numpy() #splitting the positive and negative price changes

        ### Fitting the Stable distribution
        # params = levy.fit_levy(np.concatenate((pdf_pos,pdf_neg*-1)),beta=0.0, mu=0.0, sigma=0.08,par='0')
        # print(params)

        ### Fitting the Gaussian Distribution
        # # popt, pcov = curve_fit(gaussian, xdata=pdf['price_change'].to_numpy(
        # # ), ydata=pdf['freq'].to_numpy(), p0=[0.5, 0.5])
        # # mu, std = popt
        # # print(mu, std, delay)

        ### Plotting the Stable PDF
        # x = np.linspace(-4, 4, 1000)
        # p = levy_stable.pdf(x, alpha, beta, mu, sigma)
        # p_=levy_stable.pdf(x, 1.47, beta, mu, sigma)

        ### Appending different time delay distrbutions to one DF
        # stable=pd.DataFrame()
        # stable['xdata']=x
        # stable['ydata']=np.log(p+1)
        # stable['delay'] = delay
        # plt.plot(
        #     x, p, label=fr'Delay = {delay}s, $\alpha$ = {alpha:.2f}', linewidth=1)
        # plt.plot(
        #     x, p_, label=fr'Delay = {delay}s, $\alpha$ = 1.47', linewidth=1)
        # plt.hist(pdf['Price'].to_numpy(),bins=1000,density=True,histtype='step',label=f'Delay = {delay}s')
        plt.hist(pdf_pos, bins=500, density=True, histtype='step',
                 label=fr'$+\Delta x$, Delay $t$ = {delay}s')
        plt.hist(pdf_neg*-1, bins=500, density=True, histtype='step',
                 label=fr'$-\Delta x$, Delay $t$ = {delay}s')
        # plt.xlim(-4, 4)
        # x_neg = np.linspace(-2, 0, 1000)
        # x_pos = np.linspace(0, 2, 1000)
        # p_neg = levy_stable.pdf(x_neg, alpha, beta, mu, sigma)
        # p_pos = levy_stable.pdf(x_pos, alpha, beta, mu, sigma)
        # plt.plot(
        # x_pos, p_neg[::-1], label=fr't = {delay}s, $\alpha$ = {alpha:.2f}, $\beta$ = {beta:.2f} Negative $\Delta x$', linewidth=1)
        # plt.plot(
        #     x_pos, p_pos, label=fr't = {delay}s, $\alpha$ = {alpha:.2f}, $\beta$ = {beta:.2f} Positive $\Delta x$', linewidth=1)
        plt.yscale("log")
        plt.xscale("log")
        # plt.ylabel(r'$log((P_{t}(\Delta x))*t^{\nu})$')
        plt.ylabel(r'$log(P(\Delta x/t^{\nu}))$')
        plt.xlabel(r'$log(\Delta x/t^{\nu})$')
        plt.title(
            r'Probability density function(folded) of the rescaled price changes')
        # # plt.title(r'Stable Distribution fitting to the price change probability')
        plt.legend(loc='lower left')
        x_neg = np.linspace(0, 2, 1000)
        p_neg = levy_stable.pdf(x_neg, alpha, 0.0, 0.0, 0.08)
        plt.plot(
            x_neg, p_neg, label=fr'Fit $\alpha$ = {alpha}', linewidth=3)
    plt.legend(loc='lower left')
    plt.savefig(f'./plots/pdf_folded_{delay}.png')

    # # pdf['delay'] = delay
    # df_distros = pd.concat([df_distros, pdf])
    # df_stable = pd.concat([df_stable, stable])
    # df_distros = df_distros.drop(
    #     df_distros[df_distros.Price > 10.0].index)
    # df_distros = df_distros.drop(
    #     df_distros[df_distros.Price < -10.0].index)
    # df_distros.reset_index(drop=True)
    # df_stable.reset_index(drop=True)
    # params = levy.fit_levy(df_distros['Price'].to_numpy(),par='0')
    # print(params)
    # print(len(df_distros))
    # sns.set(font_scale=1.5)
    # plt.savefig("./plots/pdfs_folded.png")
    # x = np.linspace(-1, 1, 1000)
    # plt.xlim(-1, 1)
    # p = levy_stable.pdf(x, alpha=1.3, beta=0.0, loc=0.00, scale=0.08)
    # g = levy_stable.pdf(x, alpha=1.7, beta=0.0, loc=0.00, scale=0.08)
    # plt.plot(x, p, label=fr'Fit: $\alpha$ = 1.3', linewidth=2)
    # plt.plot(x, g, label=fr'Fit: $\alpha$ = 1.7', linewidth=2)
    # # plt.hist(df_distros['Price'].to_numpy(),bins=1000,density=True,histtype='step',label=fr'Data: $1.1\times10^{6}$ values')
    # # plt.yscale("log")
    # # plt.xscale("log")
    # # plt.ylabel(r'$log((P_{t}(\Delta x))*t^{\nu})$')
    # plt.ylabel(r'$log(P(\Delta x/t^{\nu}))$')
    # plt.xlabel(r'$\Delta x/t^{\nu}$')
    # plt.title(r'Stable fit comparison for different Alpha values')
    # # plt.title(r'Probability density function of the price changes')
    # # # plt.title(r'Stable Distribution fitting to the price change probability')
    # plt.legend(loc='upper right')
    # sns.set(font_scale=1.5)
    # plt.savefig(f'./plots/alpha_compare_unlog.png')

    # fig=plt.figure(figsize=(15, 10))
    # # ax1 = fig.add_subplot(111)
    # # ax2 = ax1.twinx()
    # # sns.palplot(sns.color_palette())
    # # plt.set_title('PDF')
    # sns.set_theme(style="darkgrid")
    # sns.set(font_scale=1.5)
    # # sns.scatterplot(x="price_change", y="freq", hue="delay", data=df_distros)
    # plt.xlim(-2, 2)
    # sns.histplot(x="Price", hue="delay", data=df_distros, log_scale=(
    #     False, True), element="step", fill=False, stat='density')
    # # sns.lineplot(x="xdata", y="ydata",hue="delay", data=df_stable, ax=ax2)
    # plt.ylabel(r'$log(P_{\Delta t}(\Delta x))$')
    # plt.xlabel(r'$\Delta x$')
    # # plt.yscale("log")
    # plt.xscale("log")
    # plt.title(
    #     r'Probability distribution of price changes for various delay intervals $\Delta t$')
    # # sns.displot(data=df_distros, x="price_change", y="freq", hue="delay",kind='kde')
    # # plt.show()
    # plt.savefig("./plots/pdf.png")

#######
####### Plotting loops for MSD and other displcements
#######


p_list = [2, 4, 6, 8, 10] # order of displacement
exp_p = 2


# def msd_fit(t, p):
#     return t**p

# def linear(x,m):
#     return (x**m)

# for asset in assets:
#     for delay in time_delay_list:
#         plt.figure(figsize=(15, 10))
#         msd = pd.read_csv(f'./{asset}/prices_delay={delay}s.csv')

          ### making smaller samples from a long timeseries
#         # msd['Price']=msd['Price'].apply(lambda x: x*100000) 
#         msd['Price'] = msd['Price'].diff()
#         long_walk=msd['Price'].dropna().to_numpy()
#         for i in range(int(np.ceil(len(long_walk)/10000))):
#             if i==0:
#                 msd_walk=np.copy(long_walk[0:10000])
#                 msd_walk-=msd_walk[0]
#                 msd_walk/=(delay**0.68)
#                 # msd_walk*=100
#                 msd_walk=msd_walk**exp_p
#             elif i==int(np.floor(len(long_walk)/10000)):
#                 avg_walk=np.copy(long_walk[((i)*10000):])
#                 avg_walk-=avg_walk[0]
#                 avg_walk/=(delay**0.68)
#                 # avg_walk*=100
#                 avg_walk=avg_walk**exp_p
#                 msd_walk[:len(avg_walk)]+=avg_walk
#             else:
#                 avg_walk=np.copy(long_walk[((i)*10000):10000*(i+1)])
#                 avg_walk-=avg_walk[0]
#                 avg_walk/=(delay**0.68)
#                 # avg_walk*=100
#                 avg_walk=avg_walk**exp_p
#                 msd_walk+=avg_walk
#         samples=int(np.ceil(len(long_walk)/10000))
#         msd_walk[:len(avg_walk)]/=int(np.ceil(len(long_walk)/10000))
#         msd_walk[len(avg_walk):]/=int(np.floor(len(long_walk)/10000))
#         params, cv = curve_fit(linear, xdata=np.arange(len(msd_walk)), ydata=msd_walk, p0=[0.0], check_finite=True)
#         t = np.arange(len(msd_walk))
#         walk = linear(t, params[0])
#         print(params, delay)
#         plt.plot(t[1:], msd_walk[1:]/(t[1:]**params[0]))
#         plt.plot(
#             t, walk/(t**params[0]), label=fr'Delay={delay}s, $\gamma(2)$={params[0]: .3f}, #samples={samples}', linewidth=1)
#         # plt.plot(
#         #     t, walk, label=f'Delay={delay}s, m,c={params[0]: .3f},{params[1]: .3f}, #samples={samples}', linewidth=1)
#         # plt.xscale('log')
#         plt.yscale('log')
#         plt.ylabel(r'$\langle|(\Delta x(n))/t^{\nu}|^2\rangle$')
#         plt.xlabel(r'n')
#         sns.set(font_scale=1.5)

#         plt.title(r'Mean squared displacement for $10^{4}$ timesteps')
#         plt.legend(loc='upper left')
#         plt.savefig(f'./plots/msd_{delay}_{exp_p}_reduced.png')

# for asset in assets:
#     plt.figure(figsize=(15, 10))
#     for delay, samples in zip(time_delay_list, number_of_samples):
#         msd = pd.read_csv(f'./{asset}/prices_delay={delay}s.csv')
#         # msd['Price']=msd['Price'].apply(lambda x: x*100000)
#         msd['Price'] = msd['Price'].diff()
#         long_walk=msd['Price'].dropna().to_numpy()
#         for i in range(int(np.ceil(len(long_walk)/10000))):
#             if i==0:
#                 msd_walk=np.copy(long_walk[0:10000])
#                 msd_walk-=msd_walk[0]
#                 msd_walk/=(delay**0.68)
#                 msd_walk*=10000
#                 msd_walk=msd_walk**exp_p
#             elif i==int(np.floor(len(long_walk)/10000)):
#                 avg_walk=np.copy(long_walk[((i)*10000):])
#                 avg_walk-=avg_walk[0]
#                 avg_walk/=(delay**0.68)
#                 avg_walk*=10000
#                 avg_walk=avg_walk**exp_p
#                 msd_walk[:len(avg_walk)]+=avg_walk
#             else:
#                 avg_walk=np.copy(long_walk[((i)*10000):10000*(i+1)])
#                 avg_walk-=avg_walk[0]
#                 avg_walk/=(delay**0.68)
#                 avg_walk*=10000
#                 avg_walk=avg_walk**exp_p
#                 msd_walk+=avg_walk
#         samples=int(np.ceil(len(long_walk)/10000))
#         msd_walk[:len(avg_walk)]/=int(np.ceil(len(long_walk)/10000))
#         msd_walk[len(avg_walk):]/=int(np.floor(len(long_walk)/10000))
#         params, cv = curve_fit(msd_fit, xdata=np.arange(len(msd_walk)), ydata=msd_walk, p0=[1.0])
#         t = np.arange(len(msd_walk))
#         walk = msd_fit(t, params[0])
#         plt.plot(
#             t, walk, label=fr'Delay={delay}s, $\gamma(2)$={params[0]: .3f}, #samples={samples}', linewidth=2)
#         plt.ylabel(r'$\langle|\Delta x(n)|^2\rangle$')
#         plt.xlabel(r'n')
#         plt.xscale('log')
#         plt.yscale('log')
#         sns.set(font_scale=1.5)
#         plt.title(r'Mean squared displacement fits for $n=10^{4}$ timesteps')
#         plt.legend(loc='upper left')
#     plt.savefig(f'./plots/msd_fits_{exp_p}.png')


#######  
#######  Extras, fitting and plotting loops for analysis
#######  

gamma_2 = [1.0688557,
           1.1014895,
           1.1263537,
           1.1490210,
           1.1968377,
           1.2455758]

gamma_4 = [2.24880068,
           2.28765472,
           2.34965851,
           2.40164907,
           2.51780465,
           2.60112976]

gamma_6 = [3.48306248,
           3.49247602,
           3.60725478,
           3.69499701,
           3.88194117,
           3.99612244]

gamma_8 = [4.7528638,
           4.7091964,
           4.8825874,
           5.0038183,
           5.2544892,
           5.3997074]

gamma_10 = [6.03861087,
            5.94071813,
            6.16734055,
            6.31938256,
            6.62885681,
            6.80547297]

gamma_2 = [0.8013325,
           0.7972504,
           0.7782306,
           0.7543560,
           0.7613730,
           0.7642554]

gamma_4 = [1.98539301,
           1.98956838,
           2.00760325,
           2.01458278,
           2.08852512,
           2.12649131]

gamma_6 = [3.21984776,
           3.19632725,
           3.26752839,
           3.31138572,
           3.45619276,
           3.52509524]

gamma_8 = [4.48955297,
           4.41355304,
           4.54403319,
           4.62228912,
           4.83092783,
           4.93079158]

gamma_10 = [5.7756235,
            5.6449044,
            5.8294498,
            5.9392465,
            6.2066882,
            6.3378808]

mu_list = [0.00116503613784,
           0.00147588600723,
           0.00218594876433,
           0.00327652617727,
           0.00439827914096,
           0.00646267583940]

# sigma_list = [1.1058438879927,
#               1.405195683985,
#               1.7880022518957,
#               2.324434944619,
#               3.1087790357397,
#               3.7123621194125]

# Prob_0 = []
# for mu, sigma in zip(mu_list, sigma_list):
#     Prob_0.append(gaussian(0, mu, sigma))

f_0 = np.array([4648,
                2845,
                1591,
                784,
                388,
                210])

total = np.array([375645,
                  275421,
                  187955,
                  125395,
                  86236,
                  58434])


def linear(x, m, c):
    return (m*x)+c

# plt.figure(figsize=(15, 10))
# opt,cov=curve_fit(linear,xdata=np.log10(time_delay_list),ydata=np.log10(f_0/total))
# x_data=np.arange(1,2,0.001)
# y_data=linear(x_data,opt[0],opt[1])
# plt.plot(
#     np.log10(time_delay_list), np.log10(f_0/total), linewidth=2, marker='o', label='Data')
# plt.plot(x_data,y_data,linestyle='dashed',color='black', label=f'Linear fit, m={opt[0]:.3f}')
# sns.set(font_scale=1.5)
# plt.ylabel(r'$log(P_{\Delta t}(\Delta x=0))$')
# plt.xlabel(r'$log(\Delta t)$')
# plt.title('Probability of return P(0) for various delay intervals')
# plt.legend(loc='upper right')
# plt.savefig(f'./plots/p0.png')

# plt.figure(figsize=(15, 10))
# opt,cov=curve_fit(linear,xdata=np.log10(time_delay_list),ydata=np.log10(sigma_list))
# x_data=np.arange(1,2,0.001)
# y_data=linear(x_data,opt[0],opt[1])
# plt.plot(
#     np.log10(time_delay_list), np.log10(sigma_list), linewidth=2, marker='o', label='Data')
# plt.plot(x_data,y_data,linestyle='dashed',color='black', label=f'Linear fit, m={opt[0]:.3f}')
# sns.set(font_scale=1.5)
# plt.ylabel(r'$log(\sigma)$')
# plt.xlabel(r'$log(\Delta t)$')
# plt.title(r'Variance of $log(\sigma)$ with log of delay interval')
# plt.legend(loc='upper left')
# plt.savefig(f'./plots/sigma.png')


# gamma=np.concatenate((gamma_2, gamma_4,gamma_6,gamma_8,gamma_10))
# gamma=np.reshape(gamma, (-1,6))
# plt.figure(figsize=(15, 10))
# for i in range(6):
#     opt,cov=curve_fit(linear,xdata=p_list,ydata=gamma[:, i])
#     x_data=np.arange(1,11,0.001)
#     y_data=linear(x_data,opt[0],opt[1])
#     plt.plot(x_data,y_data,linestyle='dashed',color='black')
#     plt.scatter(
#         p_list, gamma[:, i], marker='o', label=fr'delay={time_delay_list[i]}s, $\nu$={opt[0]:.3f}')
#     sns.set(font_scale=1.5)
#     plt.ylabel(r'$\gamma(p)$')
#     plt.xlabel(r'$p$')
#     plt.title(r'Spectrum of the moments $\gamma(p)$ of price movements for various delays')
#     plt.legend(loc='upper left')
# plt.savefig(f'./plots/exponents.png')
