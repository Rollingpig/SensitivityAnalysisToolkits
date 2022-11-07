# SensitivityAnalysisToolkits
Integrate various sensitivity indices into one tool.


Easily set configurations like this:
```
param = {
      'file_path': 'example.csv',
      'y_label': 'y1',
      'x_labels': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'],
      'sample_num_list': [25, 50, 100, 200],
}
```

Train models and draw figures with one line of code:
```
from Sensitivity import SRC
from sklearn.linear_model import LinearRegression

SRC(**param, model=LinearRegression()).run(axe1, axe2)
```

Output using example dataset
<div>
    <p><img src="Indices Compare.png" alt="Example" width="700"/></p>
</div>
