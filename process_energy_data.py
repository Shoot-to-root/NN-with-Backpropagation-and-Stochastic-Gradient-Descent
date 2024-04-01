import pandas as pd
import numpy as np

df = pd.read_csv("energy_efficiency_data.csv")
#print(df.head)
#print(df.shape)

train, test = np.split(df, [int(len(df)*0.75)])
#print(train.shape, test.shape)

# labels
train_label = train['Heating Load'].to_numpy()
test_label = test['Heating Load'].to_numpy()
#train_label = train_label.reshape(train_label.shape[0], -1, 1)
#test_label = test_label.reshape(test_label.shape[0], -1, 1)
#print(train_label.shape, test_label.shape)

train.drop(['Heating Load'], axis=1, inplace=True)
test.drop(['Heating Load'], axis=1, inplace=True)
#print(train.shape, test.shape)

train_categorical_features = train[['Orientation', 'Glazing Area Distribution']]
test_categorical_features = test[['Orientation', 'Glazing Area Distribution']]
#print(train_categorical_features.shape, test_categorical_features.shape)

train_numerical_features = train.drop(['Orientation', 'Glazing Area Distribution'], axis=1)
test_numerical_features = test.drop(['Orientation', 'Glazing Area Distribution'], axis=1)
#print(train_numerical_features.shape, test_numerical_features.shape)

# one-hot encoding
orient_train_dummies = pd.get_dummies(train_categorical_features['Orientation'])
glazing_train_dummies = pd.get_dummies(train_categorical_features['Glazing Area Distribution'])
encoded_train_categorical_features = pd.concat([train_categorical_features, orient_train_dummies, glazing_train_dummies], axis=1)
encoded_train_categorical_features = encoded_train_categorical_features.drop(['Orientation', 'Glazing Area Distribution'], axis=1)
print(encoded_train_categorical_features.shape)

orient_test_dummies = pd.get_dummies(test_categorical_features['Orientation'])
glazing_test_dummies = pd.get_dummies(test_categorical_features['Glazing Area Distribution'])
encoded_test_categorical_features = pd.concat([test_categorical_features, orient_test_dummies, glazing_test_dummies], axis=1)
encoded_test_categorical_features = encoded_test_categorical_features.drop(['Orientation', 'Glazing Area Distribution'], axis=1)

# combine numerical and categorical data
#combined_train_data = zip((train_numerical_features.values, encoded_train_categorical_features.values))
#print(list(combined_train_data))
combined_train_data = pd.concat([train_numerical_features, encoded_train_categorical_features], axis=1)
#combined_train_data.drop(['Cooling Load'], axis=1, inplace=True)
#print(combined_train_data.shape)
combined_train_data = combined_train_data.to_numpy()
#combined_train_data = combined_train_data.reshape(combined_train_data.shape[0], -1, 1)
#print(combined_train_data.shape) # (576, 17)
#combined_train_data = zip((test_numerical_features.values, encoded_train_categorical_features.values))
#print(list(combined_train_data))
combined_test_data = pd.concat([test_numerical_features, encoded_test_categorical_features], axis=1)
combined_test_data = combined_test_data.to_numpy()
#combined_test_data = combined_test_data.reshape(combined_test_data.shape[0], -1, 1)
#print(combined_test_data.shape) # (192, 17)

# check shape
#print(combined_train_data.shape, train_label.shape)
#print(combined_test_data.shape, test_label.shape)

def get_train_data():
    return combined_train_data

def get_test_data():
    return combined_test_data

def get_train_label():
    return train_label

def get_test_label():
    return test_label


