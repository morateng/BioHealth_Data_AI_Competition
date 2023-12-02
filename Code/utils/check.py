import pandas as pd

df = pd.read_csv(f'/DATA/train/train.csv')

print(df['risk'].value_counts())