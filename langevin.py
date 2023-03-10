"""
Implementation of the Langevin function and related functions
that appear in the statistical mechanics of dipolar media
"""
import numpy as np

def langevin_x_over_x(x): # pylint: disable=invalid-name
    """
    Returns L(x)/x, where L(x) is the Langevin function:
    L(x) = 1/tanh(x) - 1/x
    For small x, the function value is 1/3
    """
    ret = np.zeros(np.atleast_1d(x).shape)
    select = np.abs(x) > 1e-4
    ret[select] = (1/np.tanh(x[select]) - 1/x[select])/x[select]
    ret[~select] = 1/3
    return ret

def langevin_x(x): #pylint: disable=invalid-name
    """
    Returns L(x), where L(x) is the Langevin function:
    L(x) = 1/tanh(x) - 1/x
    For small x, the function value is zero
    """
    ret = np.zeros(np.atleast_1d(x.shape))
    select = np.abs(x) > 1e-9
    ret[select] = 1/np.tanh(x[select]) - 1/x[select]
    return ret

def d_langevin_x(x): #pylint: disable=invalid-name
    """
    Returns the derivative of the Langevin function to x:
    dL/dx = 1/x^2 - 1/sinh^2 x
    For small x, the function value is 1/3
    """
    ret = np.zeros(np.atleast_1d(x.shape))

    select_small = np.abs(x) < 1e-4
    select_big = np.abs(x) > 1e2
    select_normal = ~select_small * ~select_big

    ret[select_small] = 1/3
    ret[select_normal] = 1/x[select_normal]**2 - 1/np.sinh(x[select_normal])**2
    ret[select_big] = 0
    return ret

def sinh_x_over_x(x): #pylint: disable=invalid-name
    """
    Returns sinh(x)/x. For small x, the function value is one
    """
    ret = np.zeros(x.shape)

    select_small = np.abs(x) < 1e-9
    select_big = np.abs(x) > 1e2
    select_normal = ~select_small * ~select_big

    ret[select_small] = 1
    ret[select_normal] = np.sinh(x[select_normal])/(x[select_normal])
    ret[select_big] = 2e41
    return ret

def sinh_x1_over_x1_times_exp_x2(x1, x2): #pylint: disable=invalid-name
    """
    Returns sinh(x1)/x1 * exp(x2).
    """
    select = np.abs(x1) < 1e-9
    ret = np.zeros(x1.shape)
    exp_diff = np.exp(np.minimum(x1[~select] + x2[~select], 1e2)) \
        - np.exp(np.minimum(-x1[~select] + x2[~select], 1e2))
    ret[~select] = exp_diff/2/x1[~select]
    ret[select] = np.exp(x2[select])
    return ret
