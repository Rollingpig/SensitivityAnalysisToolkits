import numpy as np

from Sensitivity.ModelBasedIndex import ModelBasedIndex


class SRC(ModelBasedIndex):
    label_y_axis = 'Squared SRC'

    def evaluate_once(self, x_train, x_val, y_train, y_val, model, r2_train, r2_val) -> np.ndarray:
        r = model.coef_
        x_std = np.array(x_train).std(axis=0)
        y_std = np.array(y_train).std(axis=0)
        result = r * x_std / y_std
        result = np.square(result)
        result = np.append(result, [r2_train, r2_val])
        return result
