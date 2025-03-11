import numpy as np
import os

current_dir = os.path.dirname(__file__)

x_train = np.load(os.path.join(current_dir, "x_test_flattened.npy")) /255
y_train = np.load(os.path.join(current_dir, "y_train.npy"))

x_test = np.load(os.path.join(current_dir, "x_test_flattened.npy")) / 255
y_test = np.load(os.path.join(current_dir, "y_test.npy"))
#print(x_train[0])