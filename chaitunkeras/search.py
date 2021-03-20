# -*- coding: utf-8 -*-
"""
Santiago Parraguez Cerda
Universidad de Chile - 2021
mail: santiago.parraguez@ug.uchile.cl

=============
KERAS CHAITUN
=============
  - SEARCH
"""
# ============= IMPORTS ===============
import os
import numpy as np
import pandas as pd

from itertools import product
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from ._tools import _check_param

# =================================================================================
class KerasGridSearch(object):
    """


    """
    def __init__(self,
                 hypermodel,
                 param_grid,
                 monitor='val_loss',
                 greater=False,
                 tuner_verbose=1):

        self.hypermodel = hypermodel
        self.param_grid = param_grid
        self.monitor = monitor
        self.greater = greater
        self.verbose = tuner_verbose
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
        Performs a search across de parameters grid creating
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

        for p_key, p_value in self.param_grid.items():
            self.param_grid[p_key] = _check_param(p_value)
        n_trials = np.prod([len(p) for p in self.param_grid.values()])

        start_score = -np.inf if self.greater else np.inf
        self.best_score = start_score

        eval_epoch = np.argmax if self.greater else np.argmin
        eval_score = np.max if self.greater else np.min

        if self.verbose == 1:
            print(f"{n_trials} trials detected for parameter grid")
            verbose = fitargs['verbose'] if 'verbose' in fitargs.keys() else 1
        else:
            verbose = 0

        fitargs['verbose'] = verbose
        tunable_fitargs = ['batch_size', 'epochs', 'steps_per_epoch', 'class_weight']

        for trial, param in enumerate(product(*self.param_grid.values())):

            param = dict(zip(self.param_grid.keys(), param))
            model = self.hypermodel(param)

            fit_param = {k: v for k, v in param.items() if k in tunable_fitargs}
            all_fitargs = {**fitargs, **fit_param}

            if self.verbose == 1:
                print(f"===== Trial {trial+1}/{n_trials} =====")

            try:
                history = model.fit(x=x, y=y,
                                    validation_split=validation_split,
                                    validation_data=validation_data,
                                    **all_fitargs)
            except ResourceExhaustedError as err:
                print(f"Resource Exhausted Error: {err}")
                continue

            epoch = eval_epoch(history.history[self.monitor])
            score = np.float32(history.history[self.monitor][epoch])
            param['epochs'] = epoch + 1
            param['score'] = score

            best_score = eval_score([self.best_score, score])

            if self.best_score != best_score:
                self.best_params = param
                self.best_model = model
                self.best_history = history.history
                self.best_score = best_score

            self.trials.append(param)
            self.scores.append(score)

            if self.verbose == 1:
                print(f"Score: {score} at epoch {epoch+1}")

        if self.verbose == 1:
            print(f"---- Search completed ----\nBest score: {self.best_score}")

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

        if not self.trials:
            print(f"No trials to save. Use the method 'search' before.")
            return None

        df = pd.DataFrame(self.trials)

        if os.path.isfile(path):
            cols = pd.read_csv(path, index_col=None, nrows=0, sep=' ').columns.tolist()

            if sorted(df.columns.to_list()) == sorted(cols):
                df[cols].to_csv(path, sep=' ', header=None, mode='a', index=False)
            else:
                new_cols = cols + [key for key in df.columns.to_list() if key not in cols]
                new_df = pd.read_csv(path, index_col=None, sep=' ')
                new_df = new_df.reindex(columns=new_cols)
                new_df = new_df.append(df.reindex(columns=new_cols))
                new_df.to_csv(path, sep=' ', index=False)
        else:
            df.to_csv(path, sep=' ', index=False)

        return None
# =================================================================================
# class KerasRandomSearch(object):
#     """
#
#     """
#     def __init__(self,
#                  hypermodel,
#                  param_grid,
#                  n_trials,
#                  monitor='val_loss',
#                  greater=False,
#                  tuner_verbose=1):
#
#         self.hypermodel = hypermodel
#         self.param_grid = param_grid
#         self.n_trials = n_trials
#         self.monitor = monitor
#         self.greater = greater
#         self.verbose = tuner_verbose
#         self.trials = []
#         self.scores = []
#         self.best_params = None
#         self.best_score = None
#         self.best_model = None
#         self.best_history = None

# ===============================================
