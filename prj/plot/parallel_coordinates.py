import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')
data = df[['Gender', 'Income', 'Happy', 'HouseholdStatus', 'EducationLevel', 'Party']].dropna()[:100]

incomes = [12500,37500,62500,87500,125000,187500]
house = [0,1,2,3,4,5]
Edu = [0,1,2,3,4,5,6]
party = [0,1,2,3,4]
data = data.replace(["Male","Female"],[0,1])
data = data.replace(['under $25,000','$25,001 - $50,000','$50,000 - $74,999','$75,000 - $100,000','$100,001 - $150,000',
            'over $150,000'], [float(i)/max(incomes) for i in incomes])
data = data.replace(['Single (no kids)','Single (w/kids)','Married (w/kids)','Married (no kids)',
                     'Domestic Partners (no kids)','Domestic Partners (w/kids)'], [float(i)/max(house) for i in house])
data = data.replace(['Current K-12','High School Diploma', "Associate's Degree", "Current Undergraduate",
                     "Bachelor's Degree", "Master's Degree","Doctoral Degree"], [float(i)/max(Edu) for i in Edu])
data = data.replace(["Independent", "Republican", "Libertarian", "Democrat","Other"],[float(i)/max(party) for i in party])

plt.figure()
parallel_coordinates(data, 'Happy', color=['r','b'])
plt.title("Parallel Coordinates Plot")
plt.show()
plt.savefig()
