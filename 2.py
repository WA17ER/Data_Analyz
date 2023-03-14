# Найти ошибку и испавить
import numpy as np

X = np.array([[ 1,  1],
              [ 1,  1],
              [ 1,  2],
              [ 1,  5],
              [ 1,  3],
              [ 1,  0],
              [ 1,  5],
              [ 1, 10],
              [ 1,  1],
              [ 1,  2]])

y = [45, 55, 50, 55, 60, 35, 75, 80, 50, 60]

def MAE(y, y_pred):
    mae = np.mean(np.abs(y - y_pred))
    return mae
def MSE(y, y_pred):
    mse = np.mean((y - y_pred)**2)
    return mse


n = X.shape[0]

eta = 1e-2
n_iter = 100

W = np.array([1, 0.5])
print(f'Number of objects = {n} \
       \nLearning rate = {eta} \
       \nInitial weights = {W} \n')

for i in range(n_iter):
    y_pred = np.dot(X, W)
    err = MSE(y, y_pred)
    for k in range(W.shape[0]):
        W -= eta * (1/n * 2 * np.dot(X.T, y_pred - y))
    if i % 10 == 0:
        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err,2)}')