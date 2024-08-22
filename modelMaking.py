import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split

df = pd.read_csv("laptops.csv")

df.drop(['Cpu'], axis=1, inplace=True)

x = df.drop(columns= ['Price'])
y= np.log(df['Price'])

X_train , X_test, y_train, y_test  = train_test_split(x,y,test_size = 0.15,
    random_state= 2)

# print(x.head())
# print(x.shape)
# print(y[1])

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.ensemble import RandomForestClassifier

step1 = ColumnTransformer(transformers= 
                  [('col_tnf',OneHotEncoder(sparse=False, drop= 'first'),[0,1,5,9,10])],
                  remainder= 'passthrough')

step2 = LinearRegression()

#Random Forest

# step1 = ColumnTransformer(transformers= 
#                   [('col_tnf',OneHotEncoder(sparse=False, drop= 'first'),[0,1,6,10,11])],
#                   remainder= 'passthrough')

# step2 = RandomForestClassifier(
#     n_estimators= 100,
#     random_state= 1,
#     max_samples= 0.5,
#     max_depth= 15,
#     max_features= 0.75
# )


pipe = Pipeline([('step1', step1),
                 ('step2', step2)
])

print(df['Company'].value_counts())

# scaler=StandardScaler()

# X_train=scaler.fit_transform(X_train)
# X_test=scaler.transform(X_test)

pipe.fit(X_train, y_train)

pred_Prices = pipe.predict(X_test)

# print(pred_Prices[0:5])

from sklearn.metrics import r2_score

acc =r2_score(y_test,pred_Prices)

print("The accuracy of the ML model is:  ", acc*100, '%')

import pickle

pickle.dump(pipe, open('laptopPricePred.pkl','wb'))
pickle.dump(df,open('dataframe.pkl','wb'))

myModel= pickle.load(open('laptopPricePred.pkl','rb'))

op = myModel.predict(X_test)

# print("PREDICTION MADE BY THE MODEL IS: ")

# print(y_test[1:10])
# print(op[1:10])

# print(np.exp(y_test[1:10]))
# print(np.exp(op[1:10]))

# print(df.iloc[693])

# print(df.apply(lambda x: df['Price'] > 100000 and df['Company'] == 'Lenovo').loc[:,['Price','OpSys','TypeName','TouchScreen']])



