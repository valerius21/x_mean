import numpy as np

from data_fusion.kalman.extended_kalman import h_x, a_x, cww as Q, cxx_init, ekf_state
from data_fusion.utils.data import base_data
from data_fusion.utils.data_parsing import result
from data_fusion.utils.helpers import get_missing

"""
Exercise 7 A
"""

"""
Constants
"""

k = 1
alpha = 0.001
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


def calc_weights_wa() -> np.ndarray:
    """
    Calculate __first__ order weights. s.th. $\sum \eq 1$

    :return:
    """
    f_wa = list()

    f_wa.append((((alpha ** 2) * k) - 4) / ((alpha ** 2) * k))  # 4 is a dimension
    for i in range(1, 9):
        f_wa.append(1. / (2 * (alpha ** 2) * k))
    return np.array(f_wa, dtype=float)
    # return np.array([1.0/9 for _ in range(9)])


wa = calc_weights_wa()


def calc_weights_wc():
    """
    Calculate __second__ order weights. s.th. $\sum \eq 1$

    :return:
    """
    # f_wc = list()
    #
    # f_wc.append(wa[0] + 1. - (alpha ** 2) + beta)
    # for i in range(1, 9):
    #     f_wc.append(1. / (2 * (alpha ** 2) * k))
    # return np.array(f_wc, dtype=float)
    # return np.array([1.0/9 for _ in range(9)])


wc = calc_weights_wc()


def calc_a_matrix(c_pred) -> np.ndarray:
    """
    Cholesky decomposition from C-prediction.

    :param c_pred:
    :return:
    """
    return np.linalg.cholesky(c_pred)


def _r_fn(index: int, sigma: np.ndarray) -> np.ndarray:
    row = result[index]
    ego_x = row[6][0]  # ego pose
    ego_y = row[6][1]

    lst = list()
    for sig in sigma.T:
        [s_x, s_y, _, _] = sig
        lst.append(np.array([s_x - ego_x, s_y - ego_y], dtype=float))

    return np.array(lst, dtype=float)


def calc_z(index: int, sigma: np.ndarray) -> np.ndarray:
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


def calc_z_mean(index: int, sigma: np.ndarray) -> np.ndarray:
    """
    Empirical mean.

    :return:
    """
    z_calc = []
    z = calc_z(index, sigma)
    for i, row in enumerate(z):
        z_calc.append(wa[i] * row)

    return sum(z_calc)


def calc_cross_covariance(sigma: np.ndarray, x_pred: np.ndarray, z: np.ndarray, z_mean: np.ndarray) -> np.ndarray:
    """
    Calculate cross covariance matrix

    :return:
    """

    acc = np.zeros((4, 4))
    zm = z_mean.reshape(4, 1)

    for i, sig in enumerate(sigma.T):
        zz = z[i].reshape(4, 1)
        zz = zz - zm
        m = sig.reshape(4, 1) - x_pred
        acc += wc[i] * (m @ zz.T)

    return acc


def kalman_gain(cross_cov: np.ndarray, s_cov: np.ndarray) -> np.ndarray:
    """
    Calculate Kalman gain matrix

    :return:
    """
    return cross_cov @ np.linalg.inv(s_cov)


def x_prepare(sigma: np.ndarray) -> np.ndarray:
    x_acc = []
    for s_row in sigma.T:
        [xx, yy, _, _] = s_row
        a = a_x(xx, yy)
        x_acc.append(a)

    return np.array(x_acc, dtype=float)  # .reshape(4, 9)


def x_prediction(x_prep: np.ndarray) -> np.ndarray:
    x_acc = np.zeros((4, 1))

    for i, prep in enumerate(x_prep):
        x_acc += wa[i] * prep

    return x_acc.reshape(4, 1)


def c_prediction(x_prep: np.ndarray, x_predicts: np.ndarray) -> np.ndarray:
    acc = np.zeros((4, 4))

    for i, prep in enumerate(x_prep):
        part1 = (prep.reshape(4, 1) - x_predicts)
        part2 = part1.T
        part3 = part1 @ part2
        acc += wc[i] * part3

    return acc + Q


def x_update(x_pred: np.ndarray, kalman: np.ndarray, z: np.ndarray, z_mean: np.ndarray) -> np.ndarray:
    """
    Update mean estimate of coordinates.

    :return:
    """
    return x_pred + (kalman @ (z - z_mean).T)


def c_update(kalman: np.ndarray, s_cov: np.ndarray, c_pred: np.ndarray):
    """
    Update covariance estimate.

    :return:
    """

    return c_pred - (kalman @ s_cov @ kalman.T)


def s_covariance(z: np.ndarray, z_mean: np.ndarray) -> np.ndarray:
    """
    Covariance of transformed points

    :return:
    """

    acc = np.zeros((4, 4))

    for i, zz in enumerate(z):
        nz = zz.reshape(4, 1)
        nz_mean = z_mean.reshape(4, 1)
        rz = nz - nz_mean
        acc += wc[i] * (rz @ rz.T)

    return acc + R_K


def update_predictions():
    missing_samples = get_missing()
    c_meas = cxx_init
    x_init = ekf_state(0)

    collector = []

    x_meas = calc_sigma_points(x_init, c_meas)

    for i in range(27):
        # time
        x_prep = x_prepare(x_meas)

        x_predicts = x_prediction(x_prep)
        c_predicts = c_prediction(c_meas, x_predicts)

        if i in missing_samples:
            continue

        # update
        sigma = calc_sigma_points(x_predicts, c_predicts)
        zs = calc_z(i, sigma)
        z_means = calc_z_mean(i, sigma)
        s_cov = s_covariance(zs, z_means)
        cross_cov = calc_cross_covariance(sigma, x_predicts, zs, z_means)
        kk = kalman_gain(cross_cov, s_cov)
        x_meas = x_update(x_predicts, kk, zs, z_means)
        c_meas = c_update(kk, s_cov, c_predicts)

        est = np.array([x_meas[0][0], x_meas[1][0]])
        collector.append(est)

    return np.array(collector, dtype=float)


if __name__ == '__main__':
    predictions = update_predictions()
    print(predictions)
