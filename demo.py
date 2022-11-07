from Sensitivity import SRC, Permutation, Shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

if __name__ == '__main__':
    grid_dict = {
        'bottom': 0.32,
        'left': 0.085,
        'right': 0.98,
        'top': 0.87,
        'wspace': 0.15,
    }
    fig, ax = plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=(10, 5),
                           sharex='col')

    param = {
        'file_path': 'example.csv',
        'y_label': 'y1',
        'x_labels': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'],
        'sample_num_list': [25, 50, 100, 200],
    }

    print('run SRC')
    SRC(**param, model=LinearRegression()).run(ax[1][0], ax[0][0])
    ax[0][0].set_ylim([0, 1.1])
    ax[0][0].set_title('Squared SRC')

    print('run Permutation')
    Permutation(**param, model=GradientBoostingRegressor()).run(ax[1][1], ax[0][1])
    ax[0][1].set_ylim([0, 1.1])
    ax[0][1].set_title('GBR-based Permutation')

    print('run SHAP')
    Shap(**param, model=GradientBoostingRegressor()).run(ax[1][2], ax[0][2])
    ax[1][2].legend(loc=(1.2, 0))
    ax[0][2].set_ylim([0, 1.1])
    ax[0][2].set_title('GBR-based Squared SHAP')
    ax[0][2].legend(loc=(1.2, 0))
    ax[0][0].set_ylabel('$R^2$')
    ax[1][0].set_ylabel('sensitivity metric')

    plt.savefig('Indices Compare.png', dpi=200)
