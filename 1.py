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



for learning_rate in np.linspace(0.05, 0.01, 50):
    n = X.shape[0]
    W = np.array([1, 0.5])
    result = []
    W = np.array([1, 0.5])
    list_mse__err = [0]
    list_mae_err =[]
    while True:
        # проверяем пока разница между вычисленной ошибкой и предыдущей не достигнет определённого значения в джанном случае .00005
        y_pred = np.dot(X, W)
        mse_err = MSE(y, y_pred)
        mae_err = MAE(y, y_pred)
        if abs(list_mse__err[-1] - mse_err) < .005:
            break
            del list_err[0]
        else:
            list_mse__err.append(mse_err)
            list_mae_err.append(mae_err)
            for k in range(W.shape[0]):
                W[k] -= learning_rate * (1/n * 2 * X[:, k] @ (y_pred - y))
        alpha_dict={'MSE':list_mse__err, 'MAE':list_mae_err, 'n_iter':len(list_mae_err)}
    print(f"Альфа {learning_rate}, кол-во итераций {alpha_dict['n_iter']}, достигнутая MSE {min(alpha_dict['MSE'][1:])},  достигнутая MAE  {min(alpha_dict['MAE'])}")

# Лучше всего справлется модель со скоростью 0,005 получает минимальную ошибку  и наименьшее кол-во итераций


