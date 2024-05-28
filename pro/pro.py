import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#used for removing the sp char and any link from the text
import re
import string

#to read the dataset
df_fake = pd.read_csv("E:/GIETU/4TH SEM/project/Fake_news/Fake.csv")
df_true = pd.read_csv("E:/GIETU/4TH SEM/project/Fake_news/True.csv")

#data look like
print(df_fake.head(10))
print(df_true.head(10))

df_fake["class"] = 0
df_true["class"] = 1

print(df_fake.shape, df_true.shape)

df_fake_manual_testing = df_fake.tail(10)

for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)

df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("E:/GIETU/4TH SEM/project/Fake_news/manual_testing1.csv")

df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(10)
