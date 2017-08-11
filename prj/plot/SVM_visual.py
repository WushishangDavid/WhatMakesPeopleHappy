import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd


def map_income(income):
    if(income == "under $25,000"):
        return 0
    elif(income == "$25,001 - $50,000"):
        return 1
    elif(income == "$50,000 - $74,999"):
        return 2
    elif(income == "$75,000 - $100,000"):
        return 3
    elif(income == "$100,001 - $150,000"):
        return 4
    elif(income == "over $150,000"):
        return 5

def map_education(edu):
    if(edu == "Current K-12"):
        return 0
    elif(edu == "High School Diploma"):
        return 1
    elif(edu == "Associate's Degree"):
        return 2
    elif(edu == "Current Undergraduate"):
        return 3
    elif(edu == "Bachelor's Degree"):
        return 4 
    elif(edu == "Master's Degree"):
        return 5
    elif(edu == "Doctoral Degree"):
        return 6
df = pd.read_csv('../data/train.csv',header=0,na_values='NaN')
df = df.dropna()
print(df)
df['Income'] = df['Income'].map(map_income, na_action='ignore')
df['EducationLevel'] = df['EducationLevel'].map(map_education, na_action='ignore')
y=df['Happy'].as_matrix()
a=df['Income'].as_matrix().reshape(503,1)
b=df['EducationLevel'].as_matrix().reshape(503,1)
X=np.concatenate((a,b),axis=1)
print(X)

h=.02
C=1.0
svc=svm.SVC(kernel='linear',C=C).fit(X,y)
rbf_svc=svm.SVC(kernel='rbf',gamma=0.7, C=C).fit(X,y)
poly_svc=svm.SVC(kernel='poly',degree=3, C=C).fit(X,y)
lin_svc=svm.SVC(C=C).fit(X,y)

x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

titles=['SVC with linear kernel','LinearSVC','SVC with RBF kernel','SVC with polynomial (degree 3) kernel']
for i, clf in enumerate((svc,lin_svc,rbf_svc,poly_svc)):
     plt.subplot(2,2,i+1)
     plt.subplots_adjust(wspace=0.4,hspace=0.4)
     Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
     Z=Z.reshape(xx.shape)
     plt.contourf(xx,yy,Z,cmap=plt.cm.coolwarm,alpha=0.8)
     plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.coolwarm)
     plt.xlim(xx.min(),xx.max())
     plt.ylim(yy.min(),yy.max())
     plt.xlabel('Income')
     plt.ylabel('Education level')
     plt.xticks(())
     plt.yticks(())
     plt.title(titles[i])
plt.show()


