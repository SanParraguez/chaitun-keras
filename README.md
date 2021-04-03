# Chaitun Keras v0.1.1
```python
import chaitunkeras as cht
```

Chaitun-Keras is a Python module to evaluate multiple Keras models using a grid search or a random search. It allows store the results in text files and provides functions to easily inspect these results.

### KerasGridSearch

```python
param_grid = {
    'epochs':       100,
    'batch_size':   32,
    'learn_rate':   [0.005, 0.001, 0.0005],
    'optimizer':    ['Adam', 'RMSprop']
}

kgs = cht.KerasGridSearch(create_model, param_grid, monitor='val_loss', greater=False)
kgs.search(x_train, y_train, validation_data=(x_val, y_val))
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

krs = cht.KerasRandomSearch(create_model, param_grid, n_trials=2, monitor='val_loss', greater=False)
krs.search(x_train, y_train, validation_data=(x_val, y_val))
krs.save_trials('random_search_trials.txt')
```

## Main references
* [Scikit-Learn](https://github.com/scikit-learn/scikit-learn)
* [Keras-Hypetune](https://github.com/cerlymarco/keras-hypetune)
