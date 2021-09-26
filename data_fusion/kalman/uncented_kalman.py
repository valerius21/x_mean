from functools import reduce

import numpy as np

from data_fusion.kalman.extended_kalman import h_x
from data_fusion.utils.data_parsing import result

"""
Constants
"""

k = 1
alpha = 10e-3
beta = 2

"""
Functions
"""


def calc_sigma_points(x_pred, c_pred) -> np.ndarray:
    """
    Calculate the sigma points.

    :param c_pred:
    :param x_pred: EKF point prediction.
    :return: Matrix 4x9
    """
    A = calc_a_matrix(c_pred)
    sigma = np.eye(4, 9).T
    xx = x_pred.reshape(4)
    sigma[0] = xx

    for i in range(1, 5):
        sigma[i] = xx + alpha * np.sqrt(k) * A[i - 1]

    for i in range(5, 9):
        j = i - 5
        sigma[i] = xx - alpha * np.sqrt(k) * A[j]

    return sigma.T


def calc_weights_wa():
    """
    Calculate __first__ order weights. s.th. $\sum \eq 1$

    :return:
    """
    wa = list()

    wa.append((alpha ** 2 * k - 4) / (alpha ** 2 * k))  # 4 is a dimension
    for i in range(1, 9):
        wa.append(1. / 2 * alpha ** 2 * k)
    return np.array(wa, dtype=float)


def calc_weights_wc():
    """
    Calculate __second__ order weights. s.th. $\sum \eq 1$

    :return:
    """
    wa_zero = calc_weights_wa()[0]  # TODO: check if stackable

    wc = list()

    wc.append(wa_zero + 1. - alpha ** 2 + beta)
    for i in range(1, 9):
        wc.append(1. / 2 * alpha ** 2 * k)
    return np.array(wc, dtype=float)


def calc_a_matrix(c_pred):
    """
    Cholesky decomposition from C-prediction.

    :param c_pred:
    :return:
    """
    return np.linalg.cholesky(c_pred)


def _r_fn(index: int, sigma: np.ndarray):
    row = result[index]
    ego_x = row[6][0]  # ego pose
    ego_y = row[6][1]

    lst = list()
    for sig in sigma.T:
        [s_x, s_y, _, _] = sig
        lst.append(np.array([s_x - ego_x, s_y - ego_y], dtype=float))

    return np.array(lst, dtype=float)


def calc_z(index: int, sigma: np.ndarray):
    """
    Calculate transformed points.

    :return:
    """
    rfn = _r_fn(index, sigma)
    zts = []
    for i, sig in enumerate(sigma.T):
        zts.append(
            np.asarray(
                h_x(sig[0], sig[1], rfn[i][0], rfn[i][1]),
                dtype=float)
        )

    return np.array(zts).reshape(9, 4)


def calc_z_mean(index: int, sigma: np.ndarray):
    """
    Empirical mean.

    :return:
    """
    z_mean = []
    wa = calc_weights_wa()
    z = calc_z(index, sigma)
    for i, row in enumerate(z):
        z_mean.append(wa[i] * row)

    return sum(z_mean)


def calc_cross_covariance():
    """
    Calculate cross covariance matrix

    :return:
    """
    raise Exception("Not implemented")


def kalman_gain():
    """
    Calculate Kalman gain matrix

    :return:
    """
    raise Exception("Not implemented")


def x_update():
    """
    Update mean estimate of coordinates.

    :return:
    """
    raise Exception("Not implemented")


def c_update():
    """
    Update covariance estimate.

    :return:
    """

    raise Exception("Not implemented")


def s_covariance():
    """
    Covariance of transformed points

    :return:
    """

    raise Exception("Not implemented")
