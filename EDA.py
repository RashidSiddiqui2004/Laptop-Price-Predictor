
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#EDA-> ENGG. DATA ANALYTICS

df = pd.read_csv("cleanedData.csv")

print(df.info())

df.drop(['Gpu'], axis=1, inplace=True)

# print(df.columns)
# 'Company', 'TypeName', 'Inches', 'ScreenResolution',    
# 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price'

# sns.displot(df['Price'],rug = True)

# plt.show()

# df['Company'].value_counts().plot(kind= 'bar')

# plt.show()

# df['TypeName'].value_counts().plot(kind= 'bar')

# plt.show()

# sns.barplot(x= df['Company'], y= df['Price'])

# plt.xticks(rotation= 'vertical')

# plt.show()

# sns.barplot(x= df['TypeName'], y= df['Price'])

# plt.xticks(rotation= 'vertical')

# plt.show()


# sns.barplot(x= df['TouchScreen'], y= df['Price'])
# plt.xticks(rotation= 'vertical')
# plt.show()

df.to_csv('laptops.csv', index=False)




