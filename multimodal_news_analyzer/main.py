import pandas as pd

df=pd.read_json('data/News_Category_Dataset_v3.json', lines=True)

print("\n First 5 rows of the data:")
print(df.head())

print("\n DataFrame info")
print(df.info())

#Check for missing values

print("\n Missing values per column:")
print(df.isnull().sum())

#Check distribution of categories

print(df['category'].value_counts())

print("\n Example headlines and descriptions:")
print(df[['headline', 'short_description', 'category']].sample(5))

