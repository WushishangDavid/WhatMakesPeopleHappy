import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')

YOB = df['YOB']

YOB = YOB.dropna()

plt.hist(YOB, normed=True)
plt.xlabel("YOB")
plt.ylabel("Probability Density")
plt.title('Histogram of YOB')

plt.show()
plt.savefig()
