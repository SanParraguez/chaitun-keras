# -*- coding: utf-8 -*-
"""
Santiago Parraguez Cerda
Universidad de Chile - 2021
mail: santiago.parraguez@ug.uchile.cl

=============
KERAS CHAITUN
=============
  - TUNERS

Provides different searchers to perform an hyperparameter tune.
"""
__all__ = ["KerasGridSearch", "KerasRandomSearch"]
# ============= IMPORTS ===============
import time
import logging
import numpy as np
from itertools import product
from .tools import _check_param, _save_trials, ParameterSampler
# =================================================================================
class KerasGridSearch(object):
    """
    Class which provides the functions to implement a Grid Search over a parameter
    grid and save the trials.

    Pass a function which returns a Keras model and a parameter grid for the exploration.
    It supports arguments to the model.fit() function.

    Parameters
    ----------
    hypermodel : function
        Callable function that takes parameters in dict format and returns a tf.keras.Model instance.
    param_grid : dict
        Hyperparameter options to use, must have the keys to create the model with the hypermodel function.
    monitor : str, default val_loss
        Variable to monitor when searching for the best model.
    mode : str, default 'auto'
        'auto', 'min' or 'max'. Indicates if the quantity to monitor must be maximized or minimized to be better.
    verbose : int, default 1
        0 or 1. Indicates if the tuner prints information during the training.

    """
    def __init__(self,
                 hypermodel,
                 param_grid,
                 monitor='val_loss',
                 mode='auto',
                 verbose=1):

        self.hypermodel = hypermodel
        self.param_grid = param_grid
        self.monitor = monitor
        self.verbose = verbose

        if mode not in ['auto', 'min', 'max']:
            logging.warning('Monitor mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        self.trials = []
        self.scores = []
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.best_history = None

# ===============================================
    def search(self,
               x,
               y=None,
               validation_data=None,
               validation_split=0.0,
               **fitargs):
        """
        Performs a search across the parameters grid creating
        all the possible trials and evaluating on the validation set provided.

        Parameters
        ----------
        x : multi types
            Input data. All the input format supported by Keras model are accepted.
        y : multi types, default None
            Target data. All the target format supported by Keras model are accepted.
        validation_data : multi types, default None
            Data on which to evaluate the loss and any model metrics at the end of each epoch.
            All the validation_data format supported by Keras model are accepted.
        validation_split : float, default 0.0
            Float between 0 and 1. Fraction of the training data to be used as validation data.
        **fitargs :
            Additional fitting arguments, the same accepted in Keras model.fit(...).
        """

        if validation_data is None and validation_split == 0.0:
            raise ValueError("Must pass either validation data or a validation split")
        if not isinstance(self.param_grid, dict):
            raise ValueError("Param_grid must be in dict format")

        self.param_grid = self.param_grid.copy()
        for p_key, p_value in self.param_grid.items():
            self.param_grid[p_key] = _check_param(p_value)

        self.best_score = np.inf if self.monitor_op == np.less else -np.inf
        eval_epoch = np.argmin if self.monitor_op == np.less else np.argmax

        n_trials = np.prod([len(p) for p in self.param_grid.values()])
        if self.verbose == 1:
            print(f"============ Starting grid search ============\n"
                  f"> {n_trials} trials detected from parameter grid\n")

        tunable_fitargs = ['batch_size', 'epochs', 'steps_per_epoch', 'class_weight']
        early_stop = next((c for c in fitargs.get('callbacks', []) if hasattr(c, 'patience')), None)
        succ = 0
        # ----- Initiate search -----
        for trial, param in enumerate(product(*self.param_grid.values())):

            param = dict(zip(self.param_grid.keys(), param))
            fit_param = {k: v for k, v in param.items() if k in tunable_fitargs}
            all_fitargs = {**fitargs, **fit_param}

            if self.verbose == 1:
                print(f"===== Trial {trial+1}/{n_trials} =====")

            try:
                timit = time.time()
                model = self.hypermodel(param)
                history = model.fit(x=x, y=y,
                                    validation_split=validation_split,
                                    validation_data=validation_data,
                                    **all_fitargs)
                train_time = (time.time() - timit)/60
                succ += 1
            except Exception as err:
                print(f"Exception raised: {err}")
                continue

            if early_stop:
                if early_stop.restore_best_weights:
                    if early_stop.stopped_epoch != 0:
                        epoch = early_stop.stopped_epoch - early_stop.patience
                    else:
                        epoch = eval_epoch(history.history[early_stop.monitor])
                        model.set_weights(early_stop.best_weights)
                else:
                    epoch = len(history.history[self.monitor]) - 1
            else:
                epoch = len(history.history[self.monitor]) - 1

            score = np.float32(history.history[self.monitor][epoch])
            param['score'] = score
            param['epochs'] = epoch + 1
            param['loss'] = np.float32(history.history['val_loss'][epoch])
            param['time'] = train_time

            if self.monitor_op(score, self.best_score):
                self.best_params = param
                self.best_model = model
                self.best_history = history.history
                self.best_score = score

            self.trials.append(param)
            self.scores.append(score)

        if self.verbose == 1:
            print(f"========== Search completed ==========\n"
                  f"Done {succ}/{n_trials} trainings successfully\n"
                  f"Best score: {self.best_score}")

        return None

# ===============================================
    def save_trials(self, path):
        """
        Save the trials params and score of the search method into a text file.

        Parameters
        ----------
        path: str
            Indicates the path of the text file.
        """
        _save_trials(self.trials, path)
        return None
# =================================================================================
class KerasRandomSearch(object):
    """
    Class which provides the functions to implement a Random Search over a parameter
    grid and save the trials.

    Pass a function which returns a Keras model, a parameter grid and the number of trials for the exploration.
    It supports arguments to the model.fit() function.

    Parameters
    ----------
    hypermodel : function
        Callable function that takes parameters in dict format and returns a tf.keras.Model instance.
    param_grid : dict
        Hyperparameter options to use, must have the keys to create the model with the hypermodel function.
    n_trials : int
        Number of trials to do from the grid.
    monitor : str, default val_loss
        Variable to monitor when searching for the best model.
    mode : str, default 'auto'
        'auto', 'min' or 'max'. Indicates if the quantity to monitor must be maximized or minimized to be better.
    verbose : int, default 1
        0 or 1. Indicates if the tuner prints information during the training.

    """
    def __init__(self,
                 hypermodel,
                 param_grid,
                 n_trials,
                 monitor='val_loss',
                 mode='auto',
                 verbose=1):

        self.hypermodel = hypermodel
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.monitor = monitor
        self.verbose = verbose

        if mode not in ['auto', 'min', 'max']:
            logging.warning('Monitor mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        self.trials = []
        self.scores = []
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.best_history = None

# ===============================================
    def search(self,
               x,
               y=None,
               validation_data=None,
               validation_split=0.0,
               **fitargs):
        """
        Performs a search across a random selection of the parameters of the grid, creating
        a certain number of trials and evaluating on the validation set provided.

        Parameters
        ----------
        x : multi types
            Input data. All the input format supported by Keras model are accepted.
        y : multi types, default None
            Target data. All the target format supported by Keras model are accepted.
        validation_data : multi types, default None
            Data on which to evaluate the loss and any model metrics at the end of each epoch.
            All the validation_data format supported by Keras model are accepted.
        validation_split : float, default 0.0
            Float between 0 and 1. Fraction of the training data to be used as validation data.
        **fitargs :
            Additional fitting arguments, the same accepted in Keras model.fit(...).
        """
        if validation_data is None and validation_split == 0.0:
            raise ValueError("Must pass either validation data or a validation split")
        if not isinstance(self.param_grid, dict):
            raise ValueError("Param_grid must be in dict format")

        self.param_grid = self.param_grid.copy()
        for p_key, p_value in self.param_grid.items():
            self.param_grid[p_key] = _check_param(p_value)

        self.best_score = np.inf if self.monitor_op == np.less else -np.inf
        eval_epoch = np.argmin if self.monitor_op == np.less else np.argmax

        max_trials = np.prod([len(p) for p in self.param_grid.values()])
        if self.verbose == 1:
            print(f"============ Starting random search ============\n"
                  f"{self.n_trials} trials to do from {max_trials} options in the parameter grid")

        tunable_fitargs = ['batch_size', 'epochs', 'steps_per_epoch', 'class_weight']
        early_stop = next((c for c in fitargs.get('callbacks', []) if hasattr(c, 'patience')), None)
        succ = 0

        rs = ParameterSampler(param_distributions=self.param_grid,
                              n_iter=self.n_trials)
        sampled_params = rs.sample()

        # ----- Initiate search -----
        for trial, param in enumerate(sampled_params):

            param = dict(zip(self.param_grid.keys(), param))
            fit_param = {k: v for k, v in param.items() if k in tunable_fitargs}
            all_fitargs = {**fitargs, **fit_param}

            if self.verbose == 1:
                print(f"===== Trial {trial + 1}/{self.n_trials} =====")

            try:
                timit = time.time()
                model = self.hypermodel(param)
                history = model.fit(x=x, y=y,
                                    validation_split=validation_split,
                                    validation_data=validation_data,
                                    **all_fitargs)
                train_time = (time.time() - timit) / 60
                succ += 1
            except Exception as err:
                print(f"Exception raised: {err}")
                continue

            if early_stop:
                if early_stop.restore_best_weights:
                    if early_stop.stopped_epoch != 0:
                        epoch = early_stop.stopped_epoch - early_stop.patience
                    else:
                        epoch = eval_epoch(history.history[early_stop.monitor])
                        model.set_weights(early_stop.best_weights)
                else:
                    epoch = len(history.history[self.monitor]) - 1
            else:
                epoch = len(history.history[self.monitor]) - 1

            score = np.float32(history.history[self.monitor][epoch])
            param['score'] = score
            param['epochs'] = epoch + 1
            param['loss'] = np.float32(history.history['val_loss'][epoch])
            param['time'] = train_time

            if self.monitor_op(score, self.best_score):
                self.best_params = param
                self.best_model = model
                self.best_history = history.history
                self.best_score = score

            self.trials.append(param)
            self.scores.append(score)

        if self.verbose == 1:
            print(f"========== Search completed ==========\n"
                  f"Done {succ}/{self.n_trials} trainings successfully\n"
                  f"Best score: {self.best_score}")

        return None

# ===============================================
    def save_trials(self, path):
        """
        Save the trials params and score of the search method into a text file.

        Parameters
        ----------
        path: str
            Indicates the path of the text file.
        """
        _save_trials(self.trials, path)
        return None
# =================================================================================
