from Sensitivity.IndicesBySampleNum import IndicesBySampleNum

import numpy as np
import scipy.stats as st


class ModelBasedIndex(IndicesBySampleNum):
    def prepare_dataset(self, sample_num=200):
        train_num = int(sample_num * 3 / 4)
        new_data = self.data.sample(sample_num, replace=False)

        x = new_data[self.x_labels]
        x_train, x_val = x[:train_num], x[train_num:]
        y = new_data[self.y_label]
        y_train, y_val = y[:train_num], y[train_num:]

        return x_train, x_val, y_train, y_val

    def evaluate_once(self, x_train, x_val, y_train, y_val, model, r2_train, r2_val) -> np.ndarray:
        return np.array([])

    def get_importance(self, sample_num=200, print_result=False, bootstrapping_num=15):
        rs = None
        # Bootstrapping: datasets are randomly resampled from the population
        for i in range(bootstrapping_num):
            x_train, x_val, y_train, y_val = self.prepare_dataset(sample_num)
            model = self.fit_model.fit(x_train, y_train)
            r2_train = model.score(x_train, y_train)
            r2_val = model.score(x_val, y_val)
            result = self.evaluate_once(x_train, x_val, y_train, y_val, model, r2_train, r2_val)
            if i > 0:
                rs = np.row_stack((rs, result))
            else:
                rs = result

        mean = rs.mean(axis=0)
        scale = st.sem(rs, axis=0)
        bottom, upper = st.t.interval(alpha=0.95, df=len(rs) - 1, loc=mean, scale=scale)

        return mean, upper, bottom
