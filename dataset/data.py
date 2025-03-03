import numpy as np

x_train = np.load(r'\dataset\x_train_flattened.npy') /255
y_train = np.load(r'\dataset\y_train.npy')

x_test = np.load(r'\dataset\x_test_flattened.npy') /255
y_test = np.load(r'\dataset\y_test.npy')

#print(x_train[0])
