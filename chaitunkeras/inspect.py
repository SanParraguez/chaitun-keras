# -*- coding: utf-8 -*-
"""
Santiago Parraguez Cerda
Universidad de Chile - 2021
mail: santiago.parraguez@ug.uchile.cl

=============
CHAITUN KERAS
=============
  - INSPECT

Provides inspector to manage trials.
"""
__all__ = ["TrialsInspector"]
# ============= IMPORTS ===============
import pandas as pd

# =================================================================================
class TrialsInspector(object):
    """
    Provides functionality to easily inspect trials results loaded from path.

    Parameters
    ----------
    file_path : str
        Indicates path to file with trials.

    """
    def __init__(self, file_path):

        self.path = file_path
        self.df = pd.read_csv(file_path, sep=' ', index_col=False)
        self.df = self.df.sort_values('score', ascending=True)
        self.columns = self.df.columns.to_list()

        pd.set_option('display.max_columns', None)

# ===============================================
    def sort(self, var='score', ascending=True) -> pd.DataFrame:
        """
        Sort the data frame by ascending or descending order across a variable given.

        Parameters
        ----------
        var : str
            Variable used to sorting de data frame.
        ascending : bool
            Order by ascending or descending.

        Returns
        -------
        Trials data frame in order by the variable specified.

        """
        if var not in self.columns:
            raise AssertionError('Variable used to sort must be in dataframe.')

        return self.df.sort_values(var, ascending=ascending)

# ===============================================
    def get_uniques(self, stats=None) -> pd.DataFrame:
        """
        Returns a pd.DataFrame with all the unique trials found.
        Allows to indicate stats to been calculated.

        Parameters
        ----------
        stats : dict
            dictionary with the new column name as key and a tuple as value with
            the column and the function to be calculated.

            e.g.: 'score_mean': ('score', 'mean')

        Returns
        -------
        pd.DataFrame with unique values and new stats calculated.

        """
        heads = self.df.columns.to_list()
        to_remove = ['score', 'epochs', 'time'] + heads[heads.index('loss'):]

        if stats is None:
            stats = {
                'score_mean':  ('score', 'mean'),
                'score_std':   ('score', 'std'),
                'score_min':   ('score', 'min'),
                'score_max':   ('score', 'max'),
                'epochs_mean': ('epochs', 'mean'),
                'count':       ('score', 'count'),
                'time_mean':   ('time', 'mean')
            }
            stats.update({m + '_mean': (m, 'mean') for m in heads[heads.index('loss'):]})
        assert type(stats) is dict

        heads = [key for key in heads if key not in to_remove]
        df_uniques = self.df.groupby(heads, as_index=False).agg(**stats)

        if 'score_mean' in df_uniques.keys():
            df_uniques = df_uniques.sort_values('score_mean', ascending=True)

        return df_uniques[[*stats, *heads]]

# ===============================================
