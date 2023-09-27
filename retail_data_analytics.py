#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
for dirname, _, filenames in os.walk('Features data set.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


get_ipython().system('pip install --upgrade storyscience')


# In[5]:


import storyscience
# storyscience.Shree()


# In[7]:


import numpy as np 
import pandas as pd
a=pd.read_csv("sales data-set.csv")
fea=pd.read_csv("stores data-set.csv")
abc=pd.read_csv("Features data set.csv")


# In[8]:


#<-----Count----->
#<-----Subbhashit----->
from collections import Counter
def Count(x):
    dictionary = dict()
    array = list(x)
    countArray = dict(Counter(array).most_common(1))
    return countArray
b=Count(a['IsHoliday'])
b,list(b.keys())[0]


# In[9]:


import storyscience as ss


# In[10]:


a.iloc[0,1]=np.nan
print(a.head())


from collections import Counter

def impute(array,method='mean'):
    arr = list(array)
    pos = []
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            pos.append(i)
    for i in pos:
        arr.remove(arr[i])
    #<-----mean----->
    if method=='mean':
        for i in pos:
            key = int(sum(arr)/len(arr))
            arr.insert(i,key)
     #<-----mode----->
    elif method=='mode':
        for i in pos:
            dictionary = dict(Counter(arr).most_common(1))
            key = int(list(dictionary.keys())[0])
            arr.insert(i,key)
    return arr     
b=impute(a['Dept'],'mode')
print("b------>",b[:5])


# In[11]:


def zscore(data,threshold=1):
    threshold = 3
    outliers = []
    arr = list(data)
    mean = np.mean(arr)
    std = np.std(arr)
    for i in arr:
        #zscore formula
        z = (i-mean)/std
        if z > threshold:
            outliers.append(i)
    return outliers

b=zscore(a['Weekly_Sales'],15981.258123467243)    
print(b[:5])


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

def SinglePlot(arr):
    #Plots initialization
    fig, ax =plt.subplots(2,2)
    fig.set_size_inches(12.7, 10.27)
    
    #Pie plot
    plt.subplot(2,2,1)
    arr.value_counts().tail().plot(kind='pie',figsize=(15,10))
    
    #Histogram
    sns.distplot(arr,ax=ax[0,1])
    
    #Bar plot
    plt.subplot(2, 2,3)
    arr.value_counts().tail().plot(kind='bar',color=['c','y','r'],figsize=(15,10))
    
    #Box plot
    sns.boxplot(arr,ax=ax[1,1])
    
    
    fig.show()
SinglePlot(a['Dept'])


# In[13]:


import numpy as np
def IQR(data,arg1=75,arg2=25):
    q3, q1 = np.percentile(data, [arg1 ,arg2])
    iqr = q3 - q1
    return iqr

ran = IQR(a['Store'])
ran


# In[14]:


def Describe(data):
    l = list(data.columns)
    length = []
    mini = []
    maxi =[]
    mean = []
    median = []
    mode = []
    typ =[]
    std =[]
    std=[]
    types = ['float64','int64']
    for  i in l:
        typ.append(data[i].dtype)
        length.append(len(data[i]))
        mini.append(min(data[i]))
        maxi.append(max(data[i]))
        if data[i].dtype in types:
            mean.append(data[i].mean())
            median.append(data[i].median())
            mode.append(data[i].mode()[0])
            std.append(np.std(data[i]))
            
        else:
            mean.append(np.nan)
            median.append(np.nan)
            mode.append(np.nan)
            std.append(np.nan)
            
        
    df = pd.DataFrame([typ,length,mini,maxi,mean,median,mode,std], index=['Type','Length','Minimum','Maximum','Mean','Median','Mode','STD'] ,columns = l)
    return df
        
    
df=Describe(abc)
df.head(10)


# In[15]:


get_ipython().system('pip install tabulate ')


# In[16]:


from tabulate import tabulate as tb


# In[17]:


def suggest_cats(data, th=40):
    dtb = []
    print('Following columns might be considered to be changed as categorical\nTaking', th, 
          '% as Threshold for uniqueness percentage determination\nLength of the dataset is:', len(data))
    ln = len(data)
    
    for i in data.columns:
        unique_vals = data[i].nunique()
        total_percent = (unique_vals/ln) * 100
        eff_percent = (data[i].dropna().nunique()/ln) * 100
        avg_percent = (total_percent + eff_percent)/2
        if avg_percent <= th:
            dtb.append([i, round(unique_vals,5), round(total_percent,5), round(eff_percent,5), round(avg_percent,5)])
            
    print(tb(dtb, headers=['Column name', 'Number of unique values', 'Total uniqueness percent', 
                           'Effective uniqueness percent', 'Average uniqueness percentage'], 
            tablefmt="fancy_grid"))

suggest_cats(abc, 10)


# In[18]:


def suggest_drops(data, th=60):
    dtb = []
    print('Following columns might be considered to be dropped as percent of missing values are greater than the threshold-',th, 
          '%\nLength of the dataset is:', len(data))
    ln = len(data)
    
    for i in data.columns:
        nans = data[i].isna().sum()
        nan_percent = (nans/ln)*100
        if nan_percent >= th:
            dtb.append([i, round(nans, 5), round(nan_percent, 5)])
    
    print(tb(dtb, headers=['Column name', 'Number of nulls', 'Percent of null values'],
             tablefmt="fancy_grid"))
    
suggest_drops(abc)


# In[19]:


def suggest_fillers(data, th=40):
    dtb = []
    print('Following columns might be considered to be imputed as percent of missing values are less than the threshold-',th, 
          '%\nLength of the dataset is:', len(data))
    ln = len(data)
    
    for i in data.columns:
        nans = data[i].isna().sum()
        nan_percent = (nans/ln)*100
        if nan_percent <= th and nan_percent != 0:
            dtb.append([i, round(nans, 5), round(nan_percent, 5)])
    
    print(tb(dtb, headers=['Column name', 'Number of nulls', 'Percent of null values'],
             tablefmt="fancy_grid"))
    
suggest_fillers(abc)


# In[20]:


def suggest_quants(data, th=60):
    dtb = []
    print('Following columns might be considered to be converted as categorical as \nthe column is numerical and the uniqueness percent is greater than the threshold-',th, 
          '%\nLength of the dataset is:', len(data))
    ln = len(data)
    numer = data.select_dtypes(include=np.number).columns.tolist()

    for i in numer:
        unique_vals = data[i].nunique()
        total_percent = (unique_vals/ln) * 100
        if total_percent >= 60:
            dtb.append([i])
            
    
    print(tb(dtb, headers=['Column name'],
             tablefmt="fancy_grid"))
    
suggest_quants(a)


# In[21]:


def create_quants(data, cols):
    dtb = []
    print('Creating Quantile columns...')

    for col in cols:
        low = np.percentile(data[col], 25)
        mid = np.percentile(data[col], 50)
        high = np.percentile(data[col], 75)
        data[col + '_quant'] = data[col].apply(
            lambda i: 0 if low > i else (1 if mid > i else (2 if high > i else 3)))
        print(col + '_quant'+' has been created using column '+col)

            
    
    print('completed!')
    
create_quants(a, ['Weekly_Sales'])


# In[24]:


from datetime import date
import pandas as pd

def extract_date_features(data, date_cols):
    today = date.today()

    for col in date_cols:
        data[col + '_age'] = today.year - data[col].dt.year
        data[col + '_months'] = data[col + '_age'] * 12 + data[col].dt.month
        data[col + '_days'] = data[col + '_months'] * 30 + data[col].dt.day
        data[col + '_season'] = data[col + '_months'].apply(lambda i:
            'Winter' if i in [1, 2, 12] else (
            'Spring' if i in [3, 4, 5] else (
            'Summer' if i in [6, 7, 8] else 'Autumn')))
        data[col + '_weekday'] = data[col].dt.day_name()

    print('Features extracted from columns:', ', '.join(date_cols))

# Convert the 'Date' column to datetime format if not already done
a['Date'] = pd.to_datetime(a['Date'])

# Example usage:
# Assuming you have a DataFrame 'a' with a 'Date' column.
extract_date_features(a, ['Date'])

