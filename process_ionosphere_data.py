import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv("ionosphere_data.csv")
#print(df.head)
#print(df.shape)

train, test = np.split(df, [int(len(df)*0.80)])
#print(train.shape, test.shape)

# labels
train_label = train.iloc[:, -1:].to_numpy()
test_label = test.iloc[:, -1:].to_numpy()

# encode labels
le = preprocessing.LabelEncoder()
train_label = le.fit_transform(train_label)
test_label = le.transform(test_label)
#train_label = train_label.reshape(train_label.shape[0], -1, 1)
#test_label = test_label.reshape(test_label.shape[0], -1, 1)
#print(train_label.shape, test_label.shape)

train.drop(train.iloc[:, -1:], axis=1, inplace=True)
test.drop(test.iloc[:, -1:], axis=1, inplace=True)
#print(train.shape, test.shape)

train = train.to_numpy()
#train = train.reshape(train.shape[0], -1, 1)
test = test.to_numpy()
#test = test.reshape(test.shape[0], -1, 1)
#print(train.shape, test.shape) # (280, 34, 1) (70, 34, 1)

def get_train_data():
    return train

def get_test_data():
    return test

def get_train_label():
    return train_label

def get_test_label():
    return test_label
