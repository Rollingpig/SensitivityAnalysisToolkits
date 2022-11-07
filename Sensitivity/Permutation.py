from sklearn.inspection import permutation_importance
import numpy as np

from Sensitivity.ModelBasedIndex import ModelBasedIndex


class Permutation(ModelBasedIndex):
    label_y_axis = 'Permutation'

    def evaluate_once(self, x_train, x_val, y_train, y_val, model, r2_train, r2_val) -> np.ndarray:
        r = permutation_importance(model, x_val, y_val,
                                   n_repeats=10, )
        result = r.importances_mean
        # result = np.sqrt(result)
        result = np.append(result, [r2_train, r2_val])
        return result
