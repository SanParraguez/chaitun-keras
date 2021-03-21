# -*- coding: utf-8 -*-
"""
Santiago Parraguez Cerda
Universidad de Chile - 2021
mail: santiago.parraguez@ug.uchile.cl

=============
KERAS CHAITUN
=============
  - TOOLS
"""
# ============= IMPORTS ===============
import random
import numpy as np

from itertools import product

# =================================================================================
def _check_param(values) -> object:
    """
    Check the parameter boundaries passed in dict values.

    Parameters
    ----------
    values

    Returns
    -------
    list of checked parameters.
    """

    if isinstance(values, (list, tuple, np.ndarray)):
        return list(set(values))
    elif hasattr(values, 'rvs'):
        return values
    else:
        return [values]

# =================================================================================
class ParameterSampler(object):
    # modified from scikit-learn and kerashypetune ParameterSampler
    """
    Generator on parameters sampled from given distributions.
    Non-deterministic iterable over random candidate combinations for hyper-
    parameter search. If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Parameters
    ----------
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.
    n_iter : integer
        Number of parameter settings that are produced.
    random_state : int, default None
        Pass an int for reproducible output across multiple
        function calls.
    """
    def __init__(self, param_distributions, n_iter, random_state=None):

        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = param_distributions

    # ===============================================
    # noinspection PyStatementEffect,PyUnresolvedReferences,PyProtectedMember,PyCallingNonCallable
    def sample(self):
        """
        Perform the sampling from the data grid provided previously.

        Returns
        -------
        param_combi : list of tuple
            list of sampled parameter combination
        """
        self.param_distributions = self.param_distributions.copy()

        for p_k, p_v in self.param_distributions.items():
            self.param_distributions[p_k] = _check_param(p_v)

        all_lists = all(not hasattr(p, "rvs")
                        for p in self.param_distributions.values())

        seed = (random.randint(1, 100) if self.random_state is None
                else self.random_state + 1)
        random.seed(seed)

        if all_lists:
            grid_size = np.prod([len(i) for i in self.param_distributions.values()])
            if grid_size < self.n_iter:
                raise ValueError(
                    f"The total space of parameters {grid_size} is smaller "
                    f"than n_iter={self.n_iter}. Try with KerasGridSearch.")

        param_combi = []
        k = self.n_iter
        for i in range(self.n_iter):
            dist = self.param_distributions
            params = []
            for j, v in enumerate(dist.values()):
                if hasattr(v, "rvs"):
                    params.append(v.rvs(random_state=seed * (k + j)))
                else:
                    params.append(v[random.randint(0, len(v) - 1)])
                k += i + j
            param_combi.append(tuple(params))

        # reset seed
        np.random.mtrand._rand

        return param_combi
