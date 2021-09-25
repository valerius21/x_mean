from numpy import np
from sympy import Matrix, sin, cos, acos, simplify, Inverse
from sympy.abc import x, y, r, q, v, alpha
import math

from data_fusion.utils.basic_kalman import sigma
from data_fusion.utils.data import base_data
from data_fusion.utils.data_parsing import result

"""
Init / Base values
"""

vx_init = base_data[0]['vx']
vy_init = base_data[0]['vy']
v_val = np.sqrt(vx_init ** 2 + vy_init ** 2)
alpha_val = acos(vx_init / v)
q_ekf = np.array([[.125], [.125], [.5], [.5]])
cww = q_ekf @ ((sigma ** 2) * q_ekf.T)

# initial state -> x = [x, y, v, alpha]
# x_init = ekf_state(0)

# init the "pick"-parameters

# initialise Cxx aka error covariance
cxx_init = np.eye(4)

# initialise sigma
sigma = 1

# initialise Cvv aka R
cvv = np.eye(4)

# define / initialise H aka the measurement matrix
# H = np.eye(2,4)

"""
Functions
"""


def ekf_state(index: int):
    """
    given an index, return the state $x = [x_1 x_2 v_h \alpha]^T$.
    """
    current_data = base_data[index]
    return np.array([
        # X1
        current_data['x'],
        # X2
        current_data['y'],
        # v_h
        v_val,
        # alpha_h
        alpha_val
    ]).T


def ekf_y(index: int):
    """

    :param index:
    :return:
    """
    cd = base_data[index]
    return np.array([
        cd['x'],
        cd['y'],
        cd['vx'],
        cd['vy']
    ])


def ekf_prediction_x(state):
    """

    :param state:
    :return:
    """
    xx = state[0]
    yy = state[1]
    res = a.subs({'x': xx,
                  'y': yy,
                  'v': v_val,
                  'alpha': alpha_val}).T
    return np.asarray(
        res
    )


def ekf_prediction_c(state, cxx):
    """
    :param state: TODO(bianca):???
    :param cxx:
    :return:
    """
    return A_x @ cxx @ A_x.T + cww


def measurement_y_k(state, rr, qq):
    """

    :param state:
    :param rr:
    :param qq:
    :return:
    """
    xx = state[0]
    yy = state[1]

    return h.subs({'x': xx, 'y': yy, 'r': rr, 'q': qq})


def K(cxx, H):
    print('[K] cxx', cxx)
    m = Matrix(H @ cxx @ H.T + cvv)

    return cxx @ H.T @ Inverse(m)


def ekf_measurement_x(x_predict, y_k, y_k_mean, K):
    # [[x, y, v, alpha]] = x_predict.T
    return x_predict + K @ (y_k - y_k_mean)


def ekf_measurement_c(cxx, K):
    return cxx - K @ H @ cxx


def r_fn(index: int):
    [x_r, y_r, _, _] = ekf_state(index)
    row = result[index]
    s_x = row[6][0]  # ego pose
    s_y = row[6][1]
    return np.array([x_r - s_x, y_r - s_y])


def H_x(r_h, q_h, v_h, alpha_h):
    """
    TODO(valerius): refactor
    :param r_h:
    :param q_h:
    :param v_h:
    :param alpha_h:
    :return:
    """
    return Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0,
         q_h * v_h * (q_h ** 2 * math.sin(alpha_h) + 2 * q_h * r_h * math.cos(alpha_h)
                      - r_h ** 2 * math.sin(alpha_h)) / (q_h ** 4 + 2 * q_h ** 2 * r_h ** 2 + r_h ** 4),
         r_h * v_h * (-q_h ** 2 * math.sin(alpha_h) - 2 * q_h * r_h * math.cos(alpha_h)
                      + r_h ** 2 * math.sin(alpha_h)) / (q_h ** 2 + r_h ** 2) ** 2],
        [0, 0, -2 * q_h * r_h * v_h * (q_h * math.sin(alpha_h) + math.cos(alpha_h)) / (q_h ** 2 + r_h ** 2) ** 2,
         v_h * (-q_h ** 2 * math.cos(alpha_h) + 2 * q_h * r_h ** 2 * math.sin(alpha_h) + r_h ** 2 * math.cos(
             alpha_h)) / (
                 q_h ** 2 + r_h ** 2) ** 2]
    ])


"""
A Matrix
"""

a = Matrix([
    [x + .5 * v * cos(alpha)],
    [y + .5 * v * sin(alpha)],
    [v],
    [alpha]
]).T

a.subs({'x': ekf_state(0)[0], 'y': ekf_state(0)[1], 'v_h': v_val, 'alpha_h': alpha_val})

derivatives_a = [x, y, v, alpha]

A = a.jacobian(derivatives_a)

# https://stackoverflow.com/questions/39753260/sympy-to-numpy-causes-the-attributeerror-symbol-object-has-no-attribute-cos
# float is not a solution cuz same thing happens
A_x = A.subs({'v_h': v_val, 'alpha_h': alpha_val})

"""
H matrix
"""

h = Matrix([
    x,
    y,
    r * ((v * cos(alpha) * r + v * sin(alpha) * q) / (r ** 2 + q ** 2)),
    q * ((v * cos(alpha) + v * sin(alpha) * q) / (r ** 2 + q ** 2))
])

# h_x = lambdify((x,y,r_h,q_h) , h)

derivatives_h = [x, y, r, q]
H = simplify(h.jacobian(derivatives_h))
