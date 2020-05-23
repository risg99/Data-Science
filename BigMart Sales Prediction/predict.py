import pandas as pd

df = pd.read_csv('train.csv')
print(df.head())
print(df.columns)

print(df['Outlet_Type'].unique())