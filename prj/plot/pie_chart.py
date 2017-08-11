import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')

gender = df.loc[:, ['Gender', 'Happy']]

gender = gender.dropna()

men_happy = gender.loc[gender['Gender'] == "Male", :]
women_happy = gender.loc[gender['Gender'] == "Female", :]

plt.figure(figsize=(18, 9))
plt.subplots_adjust(bottom = 0, left = .01, right = .99, top = .90, hspace = .2)
plt.subplot(1,2,1)
plt.pie(x=[men_happy.loc[men_happy['Happy'] == 1, :].shape[0], men_happy.loc[men_happy['Happy'] == 0, :].shape[0]],
        labels=['Happy', 'Unhappy'], colors=['r', 'b'], shadow=True, autopct='%1.1f%%')
plt.title('Fraction of Happy Men')
plt.subplot(1,2,2)
plt.pie(x=[women_happy.loc[women_happy['Happy'] == 1, :].shape[0], women_happy.loc[women_happy['Happy'] == 0, :].shape[0]],
        labels=['Happy', 'Unhappy'], colors=['r', 'b'], shadow=True, autopct='%1.1f%%')
plt.title('Fraction of Happy Women')

plt.show()
