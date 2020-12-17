import os, sys
sys.path.append(os.path.abspath('..'))
import JITtrees
import numpy as np


model = JITtrees.Boosting(16, 3)
x_train = np.array([[1], [2], [4], [5], [7], [8], [10]], dtype='float')
y_train = np.array([1, 2, 4, 5, 7, 8, 10], dtype='float')

x_valid = np.array([[2], [3], [6]], dtype='float')
y_valid = np.array([2, 3, 6], dtype='float')

history = model.fit(x_train, y_train, x_valid, y_valid, 3, 1, 0.5)
x_test = np.array([9], dtype='float')
print(model.predict(x_test))

print(f"Trees learnt: {history.trees_number()}")

print("Train losses:")
print(history.train_losses())

print("Validation losses:")
print(history.valid_losses())
