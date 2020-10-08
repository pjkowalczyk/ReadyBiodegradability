# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:36:52 2019

@author: US16120 / PJKowalczyk
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# read data
df = pd.read_csv('data/processed/alles.csv', header = 0)
df = df.drop(['Unnamed: 0'], axis=1)
# df['Result'] = pd.factorize(df.EndPt)[0]
df.sample(5).head()

def ReadyBiodeg (row):
   if row['EndPt'] == 'RB' :
      return 1
   return 0
df['ReadyBiodeg'] = df.apply (lambda row: ReadyBiodeg(row), axis=1)

df.sample(5).head()

train, test = train_test_split(df, test_size = 0.2, random_state = 350,
                               stratify=df[['ReadyBiodeg']])
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
