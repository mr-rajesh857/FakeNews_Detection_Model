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
df_fake.head(5)
df_true.head(5)

