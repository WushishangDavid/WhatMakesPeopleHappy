import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')

df = df.loc[:, ['YOB', 'Income']]

df = df.dropna()

YOB = np.array(df['YOB'])
income = np.array(df['Income'])

for i in range(len(income)):
    if income[i] == 'under $25,000':
        income[i] = 12500
    elif income[i] == '$25,001 - $50,000':
        income[i] = 37500
    elif income[i] == '$50,000 - $74,999':
        income[i] = 62500
    elif income[i] == '$75,000 - $100,000':
        income[i] = 87500
    elif income[i] == '$100,001 - $150,000':
        income[i] = 125000
    elif income[i] == 'over $150,000':
        income[i] = 187500

plt.scatter(YOB, income, alpha=0.5)
plt.xlabel("YOB")
plt.ylabel("Income")
plt.title('YOB and Income')
plt.show()
plt.savefig()
