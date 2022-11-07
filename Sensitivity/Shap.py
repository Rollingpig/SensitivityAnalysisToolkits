import shap
import numpy as np

from Sensitivity.ModelBasedIndex import ModelBasedIndex


class Shap(ModelBasedIndex):
    label_y_axis = 'Squared SHAP Value'

    def evaluate_once(self, x_train, x_val, y_train, y_val, model, r2_train, r2_val):
        X100 = shap.utils.sample(x_val, 100)  # 100 instances for use as the background distribution
        explainer = shap.Explainer(model.predict, X100)
        shap_values = explainer(x_val)
        # shap.plots.bar(shap_values)

        shap_values_abs = np.average(np.absolute(shap_values.values), axis=0)
        result = shap_values_abs / np.array(y_val).std()
        result = np.square(result)
        result = np.append(result, [r2_train, r2_val])

        return result



