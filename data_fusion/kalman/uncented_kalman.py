import numpy as np

"""
Constants
"""

k = 1
alpha = 10e-3
beta = 2

"""
Functions
"""


def calc_sigma_points(x_pred, a_j) -> np.ndarray:
    """
    Calculate the sigma points.

    :param x_pred: EKF point prediction.
    :param a_j: j-th column of matrix A.
    :return: Matrix 4x9
    """
    raise Exception("Not implemented")


def calc_weights_wa():
    """
    Calculate __first__ order weights. s.th. $\sum \eq 1$

    :return:
    """
    raise Exception("Not implemented")


def calc_weights_wc():
    """
    Calculate __second__ order weights. s.th. $\sum \eq 1$

    :return:
    """
    raise Exception("Not implemented")


def calc_a_matrix(c_pred):
    """
    Cholesky decomposition from C-prediction.

    :param c_pred:
    :return:
    """
    raise Exception("Not implemented")


def x_prediction_fn():
    """
    Prediction for $x_1, x_2$
    :return:
    """
    raise Exception("Not implemented")


def c_prediction_fn():
    """
    Covariance matrix of the prediction.

    :return:
    """
    raise Exception("Not implemented")


def calc_z():
    """
    Calculate transformed points.

    :return:
    """
    raise Exception("Not implemented")


def calc_z_mean():
    """
    Empirical mean.

    :return:
    """

    raise Exception("Not implemented")


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
