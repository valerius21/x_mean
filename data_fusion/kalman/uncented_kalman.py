import numpy as np

from data_fusion.kalman.extended_kalman import h_x, a_x
from data_fusion.utils.data_parsing import result

"""
Constants
"""

k = 1
alpha = 10e-3
beta = 2
R_K = np.eye(4)

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
    z_calc = []
    wa = calc_weights_wa()
    z = calc_z(index, sigma)
    for i, row in enumerate(z):
        z_calc.append(wa[i] * row)

    return sum(z_calc)


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


def x_prepare(sigma: np.ndarray):
    x_acc = []
    for s_row in sigma.T:
        [xx, yy, _, _] = s_row
        x_acc.append(a_x(xx, yy).reshape(1, 4))

    return np.array(x_acc, dtype=float)


def x_prediction(x_prep: np.ndarray):
    wa = calc_weights_wa()
    x_acc = np.zeros((4, 1))

    for i, prep in enumerate(x_prep):
        x_acc += wa[i] * prep.reshape(4, 1)

    return x_acc.reshape(4, 1)


def c_prediction():
    pass


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


def s_covariance(z: np.ndarray, z_mean: np.ndarray):
    """
    Covariance of transformed points

    :return:
    """

    wc = calc_weights_wc()
    acc = np.zeros((4, 4))

    for i, zz in enumerate(z):
        nz = zz.reshape(4, 1)
        nz_mean = z_mean.reshape(4, 1)
        rz = nz - nz_mean
        acc += wc[i] * (rz @ rz.T)

    return acc + R_K
