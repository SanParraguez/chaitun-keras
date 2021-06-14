# Chaitun Keras
```python
import chaitunkeras as cht
```

Chaitun-Keras is a Python module to evaluate multiple Keras models using a grid search or a random search. It allows store the results in text files and provides functions to easily inspect these results.

----------
#### Dependencies

- Python 3.7
- Numpy 1.18.5
- Pandas 1.2.3

More compatibility testing will be done in the future

----------

To use both GridSearch and RandomSearch it is necessary to write a ``create_model`` function that receives a dictionary with parameters to create the model as input and returns a keras model already compiled.

### KerasGridSearch

```python
param_grid = {
    'epochs':       100,
    'batch_size':   32,
    'learn_rate':   [0.005, 0.001, 0.0005],
    'optimizer':    ['Adam', 'RMSprop']
}

kgs = cht.KerasGridSearch(create_model, param_grid, monitor='val_loss', mode='min', verbose=1)
kgs.search(x_train, y_train, validation_data=(x_val, y_val), verbose=0)
kgs.save_trials('grid_search_trials.txt')
```

### KerasRandomSearch

```python
param_grid = {
    'epochs':       100,
    'batch_size':   32,
    'learn_rate':   [0.005, 0.001, 0.0005],
    'optimizer':    ['Adam', 'RMSprop']
}

krs = cht.KerasRandomSearch(create_model, param_grid, n_trials=2, monitor='val_loss', mode='min', verbose=1)
krs.search(x_train, y_train, validation_data=(x_val, y_val), verbose=0)
krs.save_trials('random_search_trials.txt')
```

### TrialsInspector

```python
stats = {
    'score_mean':  ('score', 'mean'),
    'score_std':   ('score', 'std'),
    'score_min':   ('score', 'min'),
    'score_max':   ('score', 'max'),
    'epochs_mean': ('epochs', 'mean'),
    'count':       ('score', 'count')
}

trials = cht.TrialsInspector('grid_search_trials.txt')
df = trials.get_uniques(stats)
```

### Main references
* [Scikit-Learn](https://github.com/scikit-learn/scikit-learn)
* [Keras-Hypetune](https://github.com/cerlymarco/keras-hypetune)
