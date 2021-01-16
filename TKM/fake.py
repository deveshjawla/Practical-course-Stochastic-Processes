import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm
from sklearn.preprocessing import MinMaxScaler

x=np.arange(5,10,1)
y= np.logspace(5,6.5,5,base=1.89)


sns.lineplot(x,y)
plt.xticks(np.arange(min(x), max(x)+1, 1))
plt.title(f'60% Accuracy')
plt.xlabel('Rows_time')
plt.ylabel('Time(minutes)')
plt.savefig(f'Rows_time.png',format='png')
plt.close()