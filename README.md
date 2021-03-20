# Chaitun Keras
Hyperparameter searcher for models created in keras.

## KerasGridSearch

```python
import chaitunkeras as cht

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

## KerasRandomSearch

## Main references
* Sklearn
* Kerashypetune
