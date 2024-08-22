
import pandas as pd
import numpy as np

import seaborn as sns

#DATA CLEANING PART

df= pd.read_csv("laptop_data.csv")

# print(df.head())

# print(df.columns)
# 'Company', 'TypeName', 'Inches', 'ScreenResolution',    
# 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price'
df.drop(['Unnamed: 0'], inplace=True, axis=1)

# print(df.info())

print(df.shape)

df['Weight'] = df['Weight'].str.replace("kg", "")
df['Ram'] = df['Ram'].str.replace("GB", "")
# print(df['Ram'].head(2))
 
df['Weight'] = df['Weight'].astype('float64')
df['Ram'] = df['Ram'].astype('int32')

# print("CPU values: ")
# print(df['Cpu'].unique())

# print("Memory values: ")
# print(df['Memory'].unique())

# print("Gpu values: ")
# print(df['Gpu'].unique())

# print("TypeName values: ")
# print(df['TypeName'].unique())

# TypeNames :
# ['Ultrabook' 'Notebook' 'Netbook' 'Gaming' '2 in 1 Convertible'
#  'Workstation']

# print("ScreenResolution values: ")
# print(df['ScreenResolution'].unique())

#check for TouchScreen in the ScreenResolution

ts = []

for i in range(df.shape[0]):

    if('Touchscreen' in df['ScreenResolution'][i]):
        ts.append(1)

    else:
        ts.append(0)


df['TouchScreen'] = ts

# print(df['TouchScreen'].unique())

# print(df.head())

# s = "IPS Panel Full HD 1366x768"

#creating IPS Panel column

df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)


def findScreenResolution(str):
    t= ""
    for i in str:

        if(i.isnumeric() or i=='x' or i=='X'):
            t+= i

    return t


for i in range(df.shape[0]):

    df['ScreenResolution'][i] = findScreenResolution(df['ScreenResolution'][i])

df['ScreenResolution'].astype('str')

# print(df['ScreenResolution'])

df['Gpu brand'] = df['Gpu'].apply(lambda x: list(x.split(" "))[0])

# print(df['Gpu brand'].value_counts())

df = df[df['Gpu brand'] != 'ARM']

print(df['Gpu brand'].value_counts())

print(df.columns)

df.drop(['Gpu'],inplace=True, axis=1)

#cpu

# s =  'Intel Core i7 7700HQ 2.8GHz'
# s1 = 'Intel Core i5 2.0GHz'

#CPU Brand

df['Cpu name'] = df['Cpu'].apply(lambda x: " ".join(list(x.split(" "))[0:3]))
 
def findCpuProc(s):

    if(s== "Intel Core i7" or s== "Intel Core i5"  or s== "Intel Core i3"):
        return s

    elif("Intel" in s):

        return "Other Intel Processor"

    else:
        return "AMD Processor"  
    

df['CpuBrand'] = df['Cpu name'].apply(findCpuProc)
    
df.drop(['Cpu name'], inplace=True, axis=1)

def findCpu(s):

    i = len(s) - 4

    ans = ""

    while(s[i].isnumeric() or s[i]=='.'):
        ans += s[i]
        i-=1

    ans = ans[::-1]
    return float(ans)

def findMem(s):

    i = 0

    ans = ""

    while(s[i].isnumeric() or s[i]=='.'):
        ans += s[i]
        i+=1

    return float(ans)

# print(findCpu(s))
# print(findCpu(s1))

for i in range(df.shape[0]):

    df['Cpu'][i] = findCpu(df['Cpu'][i])

df['Cpu'] = df['Cpu'].astype('float64')
# print(df['Cpu'])

#memory values

# '128GB SSD' '128GB Flash Storage' '256GB SSD' '512GB SSD' '500GB HDD'


for i in range(df.shape[0]):

    # df['Memory'].str.replace('TB','000GB')

    val = findMem(df['Memory'][i])

    if('TB' in df['Memory'][i]):

        val *= 1000
    
    df['Memory'][i] = val

df['Memory'] = df['Memory'].astype('float64')
# print(df['Memory'])

print(df['ScreenResolution'].head())

df['X_dim'] = df['ScreenResolution'].apply(lambda x:  list(x.split('x'))[0])
df['Y_dim'] = df['ScreenResolution'].apply(lambda x:  list(x.split('x'))[1])

df['X_dim']= df['X_dim'].astype('int32')
df['Y_dim']= df['Y_dim'].astype('int32')

# print(df['X_dim'].head())
# print(df['Y_dim'].head())

df['PPI'] = (((df['X_dim']**2 + df['Y_dim']**2)**0.5)/df['Inches']).astype('float')

df.drop(['X_dim','Y_dim','ScreenResolution'],inplace=True, axis=1)

print(df.head())


print("Updated info of Dataset is:")
print(df.info())

#creating new dataset file
df.to_csv("cleanedData.csv", index= False)
