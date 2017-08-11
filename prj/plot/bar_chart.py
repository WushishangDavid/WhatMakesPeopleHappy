import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')

df = df.loc[:, ['Income', 'Happy']]

df = df.dropna()

income_happy = np.array(df)

incomes = 6
income_tuple = [np.where(income_happy == "under $25,000")[0], np.where(income_happy == "$25,001 - $50,000")[0],
                np.where(income_happy == "$50,000 - $74,999")[0], np.where(income_happy == "$75,000 - $100,000")[0],
                np.where(income_happy == "$100,001 - $150,000")[0], np.where(income_happy == "over $150,000")[0]]

happy_idx = np.where(income_happy == 1)[0]
unhappy_idx = np.where(income_happy == 0)[0]

happy_income_count = []
unhappy_income_count = []

for i in range(incomes):
	happy_income_count.append(np.intersect1d(income_tuple[i],happy_idx).shape[0])
	unhappy_income_count.append(np.intersect1d(income_tuple[i],unhappy_idx).shape[0])

total_count = []
for i in range(incomes):
	total_count.append(happy_income_count[i]+unhappy_income_count[i])

for i in range(incomes):
    happy_income_count[i]=float(happy_income_count[i])/total_count[i]
    unhappy_income_count[i] = float(unhappy_income_count[i])/total_count[i]


bar_width = 0.40
index = np.arange(incomes)
bar1 = plt.bar(index, happy_income_count, bar_width, color='r', label="Happy")
bar2 = plt.bar(index + bar_width, unhappy_income_count, bar_width, color='b', label="Unhappy")
plt.title("Bar Chart of Income and Happiness")
plt.xlabel("Income Levels")
plt.ylabel("Fraction of Happy/Unhappy")
plt.xticks(index + bar_width, ("under $25,000", "$25,001 - $50,000", "$50,000 - $74,999",
			"$75,000 - $100,000", "$100,001 - $150,000", "over $150,000"), rotation=10)
plt.legend()

# attach some text labels
count = 0
for bar in bar1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height, "%.2f" % happy_income_count[count], ha='center', va='bottom')
    count += 1

count = 0
for bar in bar2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height, "%.2f" % unhappy_income_count[count], ha='center', va='bottom')
    count += 1

plt.show()
plt.savefig()
