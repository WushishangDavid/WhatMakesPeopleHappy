import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn
from sklearn.preprocessing import StandardScaler
seaborn.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor



    # df = df.select_dtypes(include=[object])
    # le = LabelEncoder()
    # temp=pd.DataFrame()
    # for feature in df:
        # value = le.fit_transform(df[feature].values)
        # temp[feature] = pd.Series(value)
    # df=temp
    # enc = OneHotEncoder()
    # enc.fit(df)
    # df= enc.transform(df)
def map_yob(yob):
    if(yob < 1940):
        return 1
    elif(yob<1960):
        return 2
    elif(yob<1970):
        return 3
    elif(yob<1980):
        return 4
    elif(yob<1990):
        return 5
    elif(yob<2000):
        return 6
    elif(yob<2020):
        return 7
# def set_missing_yob(df):
     # df_full=pd.DataFrame()
     # process_df=df
     # known=process_df[process_df.YOB>0].as_matrix()
     # unknown=process_df[process_df.YOB==0].as_matrix()
     # X=known[:,1:]
     # y=known[:,0]
     # print(X)
     # print(y)
     # rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
     # rfr.fit(X,y)
     # print(rfr)
     # predicted=rfr.predict(unknown[:,1:])
     # df.loc[(df.YOB==0),'YOB']=predicted
     # df_full['YOB']=df['YOB']
     # print(df.columns)
     # print(df.index)
     # for col_name in df.columns:
          # process_df=df
          # temp=process_df.pop(col_name)
          # process_df.insert(0,col_name,temp)
          # known=process_df[process_df[col_name]>0].as_matrix()
          # unknown=process_df[process_df[col_name]==0].as_matrix()
          # X=known[:,1:]
          # y=known[:,0]
          # print(X)
          # print(y)
          # rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
          # rfr.fit(X,y)
          # predicted=rfr.predict(unknown[:,1:])
          # df.loc[(df[col_name]==0),col_name]=predicted
          # df_full[col_name]=df[col_name]
     # print(df_full)
     # return df_full
id=np.array([[]*3176])
def transform(filename):
    global id
    df = pd.read_csv(filename,header=0,na_values='NaN')
    df= df[df.votes>30]
    temp1=df['Happy']
    id=df['UserID'].as_matrix().reshape(3176,1)
    df.drop('votes',axis=1, inplace=True)
    df.drop('UserID',axis=1, inplace=True)
    df.drop('Q124742',axis=1, inplace=True)
    df.drop('Happy',axis=1, inplace=True)
    df['YOB'] = df['YOB'].map(map_yob, na_action='ignore')
    for col_name, values in df.iteritems():
        if(col_name in ["YOB","Happy", "votes"]):
            continue
        else:
            unique_value_list = df[col_name].unique()
            unique_value_list = [value for value in unique_value_list if not pd.isnull(value)]
            unique_value_list = sorted(unique_value_list)
            num_unique_value = len(unique_value_list)
            tmp_dict={}
            i=1
            for value in unique_value_list:
                if(pd.isnull(value)):
                    continue
                else:
                    tmp_dict[value] = i
                    i += 1
        df[col_name] = df[col_name].map(tmp_dict,na_action='ignore')
    df=df.fillna(0)
    #print(df.isnull().sum(axis=0)[0:50])
    #print(df.isnull().sum(axis=0)[51:110])
    #print((df.isnull().sum(axis=1)+df['votes'])[100:150])
    return {'data':df,'target':temp1.as_matrix()} 

	
def getMean(col):
    col_sum = np.sum(np.nan_to_num(col))
    col_count = np.sum(np.isnan(col))
    if col_count != 0:
        return 1.0*col_sum/col_count
    else:
        return 0

def getMedian(col):
    if col.shape[0] <= 0:
        return 0
    else:
        col=col[~np.isnan(col)]
        return np.median(col)

def getMostFrequent(col):
    col=col[~np.isnan(col)]
    if col.shape[0] <= 0:
        return 0
    # Count and record occurance of all the values
    counter = dict()
    for entry in col:
        if entry not in counter.keys():
            counter[entry] = 0
        else:
            counter[entry] += 1
    # Find the most frequent value
    vals = counter.keys()
    occur = 0
    fre_val = 0
    for val in vals:
        if(val>0):
            if counter[val] > occur:
                fre_val = val
                occur = counter[val]
    return fre_val

	
	
def fill_missing(df, strategy, isClassified):
    """
     @X: input matrix with missing data filled by nan
     @strategy: string, 'median', 'mean', 'most_frequent'
     @isclassfied: boolean value, if isclassfied == true, then you need build a
     decision tree to classify users into different classes and use the
     median/mean/mode values of different classes to fill in the missing data;
     otherwise, just take the median/mean/most_frequent values of input data to
     fill in the missing data
    """
    """ your code here """
    for col_name in ['YOB','Gender','Income','HouseholdStatus','EducationLevel','Q101162']:
          print("Filling missing data in column %s" %(col_name))
          process_df=df
          temp=process_df.pop(col_name)
          process_df.insert(0,col_name,temp)
          known=process_df[process_df[col_name]>0].as_matrix()
          unknown=process_df[process_df[col_name]==0].as_matrix()
          X=known[:,1:]
          y=known[:,0]
          rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
          rfr.fit(X,y)
          predicted=rfr.predict(unknown[:,1:])
          #print(predicted)
          df.loc[(df[col_name]==0),col_name]=predicted
    X=df.as_matrix()
    (n, m) = X.shape
    if isClassified == False:
        discarded_rows = []
        for i in range(0,m):
            col = X[:,i]
            sub_val = getMostFrequent(col)
            row_id = np.where((X[:,i]==0))
            X[row_id,i] = sub_val
    for i in range(0,m):
         for j in range(0,n):
            X[j,i]=round(X[j,i])
    enc=OneHotEncoder()
    enc.fit(X)
    X=enc.transform(X).toarray()
    global id
    X=np.concatenate((id,X),axis=1)
    print(X.shape)
    return X

def test_transform(filename):
    global id2
    df = pd.read_csv(filename,header=0,na_values='NaN')
    id2=df['UserID'].as_matrix().reshape(924,1)
    df.drop('votes',axis=1, inplace=True)
    df.drop('UserID',axis=1, inplace=True)
    df.drop('Q124742',axis=1, inplace=True)
    df['YOB'] = df['YOB'].map(map_yob, na_action='ignore')
    for col_name, values in df.iteritems():
        if(col_name in ["YOB", "votes"]):
            continue
        else:
            unique_value_list = df[col_name].unique()
            unique_value_list = [value for value in unique_value_list if not pd.isnull(value)]
            unique_value_list = sorted(unique_value_list)
            num_unique_value = len(unique_value_list)
            tmp_dict={}
            i=1
            for value in unique_value_list:
                if(pd.isnull(value)):
                    continue
                else:
                    tmp_dict[value] = i
                    i += 1
        df[col_name] = df[col_name].map(tmp_dict,na_action='ignore')
    df=df.fillna(0)
    #print(df.isnull().sum(axis=0)[0:50])
    #print(df.isnull().sum(axis=0)[51:110])
    #print((df.isnull().sum(axis=1)+df['votes'])[100:150])
    return df

def test_fill_missing(df, strategy, isClassified):
    """
     @X: input matrix with missing data filled by nan
     @strategy: string, 'median', 'mean', 'most_frequent'
     @isclassfied: boolean value, if isclassfied == true, then you need build a
     decision tree to classify users into different classes and use the
     median/mean/mode values of different classes to fill in the missing data;
     otherwise, just take the median/mean/most_frequent values of input data to
     fill in the missing data
    """
    """ your code here """
    for col_name in ['YOB','Gender','Income','HouseholdStatus','EducationLevel','Q101162']:
          print("Filling missing data in column %s" %(col_name))
          process_df=df
          temp=process_df.pop(col_name)
          process_df.insert(0,col_name,temp)
          known=process_df[process_df[col_name]>0].as_matrix()
          unknown=process_df[process_df[col_name]==0].as_matrix()
          X=known[:,1:]
          y=known[:,0]
          rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
          rfr.fit(X,y)
          predicted=rfr.predict(unknown[:,1:])
          #print(predicted)
          df.loc[(df[col_name]==0),col_name]=predicted
    X=df.as_matrix()
    (n, m) = X.shape
    if isClassified == False:
        discarded_rows = []
        for i in range(0,m):
            col = X[:,i]
            sub_val = getMostFrequent(col)
            row_id = np.where((X[:,i]==0))
            X[row_id,i] = sub_val
    for i in range(0,m):
         for j in range(0,n):
            X[j,i]=round(X[j,i])
    enc=OneHotEncoder()
    enc.fit(X)
    X=enc.transform(X).toarray()
    global id2
    X=np.concatenate((id2,X),axis=1)
    print(X.shape)
    return X