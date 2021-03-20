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
import numpy as np

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
