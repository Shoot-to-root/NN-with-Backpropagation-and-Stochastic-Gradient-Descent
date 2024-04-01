import process_energy_data
import process_ionosphere_data
import classifier, regressor
import numpy as np

# classification (34, 280) (1, 280) (34, 70) (1, 70)
train_X = process_ionosphere_data.get_train_data()
train_Y = process_ionosphere_data.get_train_label().reshape(-1, 1)
test_X = process_ionosphere_data.get_test_data()
test_Y = process_ionosphere_data.get_test_label().reshape(-1, 1)
"""
# regression (17, 192) (1, 576) (17, 192) (1, 192)
train_X = process_energy_data.get_train_data()
train_Y = process_energy_data.get_train_label().reshape(-1, 1)
test_X = process_energy_data.get_test_data()
test_Y = process_energy_data.get_test_label().reshape(-1, 1)
"""
# reshape to fit into model
train_X = np.transpose(train_X, (1, 0))
train_Y = np.transpose(train_Y, (1, 0))
test_X = np.transpose(test_X, (1, 0))
test_Y = np.transpose(test_Y, (1, 0))

params = classifier.train(test_X, test_Y)
predictions = classifier.predict(test_X, test_Y, params)
"""
params = regressor.train(train_X, train_Y)
#predictions = []
#predictions.append(regressor.predict(train_X, train_Y, params))
"""
