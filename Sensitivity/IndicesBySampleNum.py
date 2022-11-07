import numpy as np
import pandas as pd
import math


class IndicesBySampleNum:
    sample_num_list = None

    input_colors = ['red', 'darkorange', 'gold', 'green', 'darkturquoise', 'dodgerblue', 'purple']
    input_markers = ['o', 'x', '^', 's', '+', 'H', '.', 'v']

    data = None
    y_label: str = None
    x_labels = None
    fit_model = None

    label_y_axis = None

    def __init__(self, file_path: str, y_label: str, x_labels: list, sample_num_list: list, model):
        self.data = pd.read_csv(file_path)
        self.y_label = y_label
        self.x_labels = x_labels
        self.sample_num_list = sample_num_list
        self.fit_model = model

    def get_importance(self, sample_num=200):
        return None, None, None

    def indices_by_sample_num(self):
        m, u, b = None, None, None
        for i, sample_n in enumerate(self.sample_num_list):
            m0, u0, b0 = self.get_importance(sample_n)
            if i > 0:
                m = np.row_stack((m, m0))
                u = np.row_stack((u, u0))
                b = np.row_stack((b, b0))
            else:
                m, u, b = m0, u0, b0
        return m, u, b

    def run(self, axe, axe_r2):
        mean, upper, bottom = self.indices_by_sample_num()
        mean_r2, upper_r2, bottom_r2 = mean[:, -2:], upper[:, -2:], bottom[:, -2:]
        mean, upper, bottom = mean[:, :-2], upper[:, :-2], bottom[:, :-2]

        x = [math.log(s) for s in self.sample_num_list]
        for i, input_label in enumerate(self.x_labels):
            y = mean[:, i]
            axe.plot(x, y, color=self.input_colors[i % len(self.input_colors)], linestyle='-', linewidth=1,
                     label=input_label, marker=self.input_markers[i % len(self.input_markers)], markersize=4, )

            y1s = bottom[:, i]
            y2s = upper[:, i]
            axe.fill_between(x=x, y1=y1s, y2=y2s,
                             facecolor=self.input_colors[i % len(self.input_colors)], alpha=0.2)

        # r2_train
        y = mean_r2[:, -2]
        axe_r2.plot(x, y, color=self.input_colors[0], linestyle='-', linewidth=1,
                    label='R2 Training', marker=self.input_markers[0], markersize=4, )
        y1s = bottom_r2[:, -2]
        y2s = upper_r2[:, -2]
        axe_r2.fill_between(x=x, y1=y1s, y2=y2s,
                            facecolor=self.input_colors[0], alpha=0.2)

        # r2_validation
        y = mean_r2[:, -1]
        axe_r2.plot(x, y, color=self.input_colors[1], linestyle='-', linewidth=1,
                    label='R2 Validation', marker=self.input_markers[1], markersize=4, )
        y1s = bottom_r2[:, -1]
        y2s = upper_r2[:, -1]
        axe_r2.fill_between(x=x, y1=y1s, y2=y2s,
                            facecolor=self.input_colors[1], alpha=0.2)

        axe.set_xlim([math.log(self.sample_num_list[0]), math.log(self.sample_num_list[-1])])
        axe.set_ylim([0, None])
        axe.set_xticks(x)
        axe.set_xticklabels(self.sample_num_list)
        axe.set_xlabel('number of samples')
        axe.set_ylabel(self.label_y_axis)

        axe_r2.set_xlim([math.log(self.sample_num_list[0]), math.log(self.sample_num_list[-1])])
        axe_r2.set_ylim([0, None])
        axe_r2.set_xticks(x)
        axe_r2.set_xticklabels(self.sample_num_list)
        axe_r2.set_xlabel('number of samples')
